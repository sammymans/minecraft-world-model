"""Full-data training and held-out evaluation for V2 latent diffusion."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mcwm.dynamics import _file_sha256
from mcwm.latent_data_v2 import CachedTemporalLatentDataset, LatentEpisodeCache
from mcwm.latent_diffusion_v2 import (
    LATENT_DIFFUSION_V2_ARCHITECTURE,
    TemporalActionUNet,
    _render_frame,
    autoregressive_action_rollout,
    counterfactual_action_scripts,
    diffusion_loss,
    load_latent_diffusion_v2_checkpoint,
    most_structured_seed,
)
from mcwm.manifest import DatasetManifest
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything


@dataclass(frozen=True)
class ActionConditionMetrics:
    examples: int
    correct_denoising_loss: float
    previous_final_action_loss: float
    shuffled_final_action_loss: float
    zero_final_action_loss: float
    previous_action_penalty_percent: float
    shuffled_action_penalty_percent: float
    zero_action_penalty_percent: float


@dataclass(frozen=True)
class SampleMetrics:
    examples: int
    sample_pixel_mse_to_oracle: float
    sample_pixel_psnr_to_oracle_db: float
    copy_pixel_mse_to_oracle: float
    copy_improvement_percent: float
    shuffled_sample_penalty_percent: float
    sample_edge_ratio_to_oracle: float
    sampled_latents_finite: bool


@dataclass(frozen=True)
class V2TrainingResult:
    checkpoint: Path
    latest_checkpoint: Path
    metrics_path: Path
    training_curve: Path
    samples: Path
    action_comparison: Path
    training_windows: int
    validation_windows: int
    action_change_windows: int
    parameter_count: int
    completed_steps: int
    device: str


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _variant_actions(actions: torch.Tensor, variant: str) -> torch.Tensor:
    changed = actions.clone()
    if variant == "correct":
        return changed
    if variant == "previous":
        changed[:, -1] = changed[:, -2]
    elif variant == "shuffled":
        changed[:, -1] = changed[:, -1].roll(1, dims=0)
    elif variant == "zero":
        changed[:, -1] = 0
    else:
        raise ValueError(f"unknown action variant: {variant}")
    return changed


@torch.no_grad()
def evaluate_action_conditions(
    model: TemporalActionUNet,
    dataset: Dataset[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    batch_size: int,
    seed: int,
) -> ActionConditionMetrics:
    """Compare only the target-driving action while holding noise fixed."""
    model.eval()
    variants = ("correct", "previous", "shuffled", "zero")
    totals = {variant: 0.0 for variant in variants}
    examples = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        count = len(batch["actions"])
        for variant in variants:
            candidate = dict(batch)
            candidate["actions"] = _variant_actions(batch["actions"], variant)
            loss = diffusion_loss(model, candidate, seed=seed + index)
            totals[variant] += float(loss) * count
        examples += count
    if not examples:
        raise ValueError("cannot evaluate an empty V2 dataset")
    averaged = {name: total / examples for name, total in totals.items()}
    correct = max(averaged["correct"], 1e-12)
    return ActionConditionMetrics(
        examples=examples,
        correct_denoising_loss=averaged["correct"],
        previous_final_action_loss=averaged["previous"],
        shuffled_final_action_loss=averaged["shuffled"],
        zero_final_action_loss=averaged["zero"],
        previous_action_penalty_percent=(averaged["previous"] / correct - 1) * 100,
        shuffled_action_penalty_percent=(averaged["shuffled"] / correct - 1) * 100,
        zero_action_penalty_percent=(averaged["zero"] / correct - 1) * 100,
    )


def _stack_items(
    dataset: Dataset[dict[str, torch.Tensor]], indices: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    items = [dataset[index] for index in indices]
    return {
        name: torch.stack([item[name] for item in items]).to(device)
        for name in ("context_latents", "actions", "target_latent")
    }


@torch.no_grad()
def evaluate_samples(
    model: TemporalActionUNet,
    autoencoder: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    maximum_samples: int = 16,
) -> SampleMetrics:
    model.eval()
    autoencoder.eval()
    count = min(maximum_samples, len(dataset))
    if count < 2:
        raise ValueError("sample evaluation needs at least two examples")
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    batch = _stack_items(dataset, indices, device)
    sampled = model.sample(
        batch["context_latents"], batch["actions"], steps=sampling_steps, seed=seed
    )
    shuffled_actions = _variant_actions(batch["actions"], "shuffled")
    shuffled = model.sample(
        batch["context_latents"], shuffled_actions, steps=sampling_steps, seed=seed
    )
    sampled_frames = autoencoder.decode(sampled).clamp(0, 1)
    shuffled_frames = autoencoder.decode(shuffled).clamp(0, 1)
    copy_frames = autoencoder.decode(batch["context_latents"][:, -1]).clamp(0, 1)
    oracle_frames = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
    sample_mse = nn.functional.mse_loss(sampled_frames, oracle_frames).item()
    shuffled_mse = nn.functional.mse_loss(shuffled_frames, oracle_frames).item()
    copy_mse = nn.functional.mse_loss(copy_frames, oracle_frames).item()
    oracle_dx, oracle_dy = image_gradients(oracle_frames)
    sample_dx, sample_dy = image_gradients(sampled_frames)
    oracle_edges = oracle_dx.abs().sum() + oracle_dy.abs().sum()
    sample_edges = sample_dx.abs().sum() + sample_dy.abs().sum()
    return SampleMetrics(
        examples=count,
        sample_pixel_mse_to_oracle=sample_mse,
        sample_pixel_psnr_to_oracle_db=10 * math.log10(1 / max(sample_mse, 1e-12)),
        copy_pixel_mse_to_oracle=copy_mse,
        copy_improvement_percent=(1 - sample_mse / max(copy_mse, 1e-12)) * 100,
        shuffled_sample_penalty_percent=(shuffled_mse / max(sample_mse, 1e-12) - 1) * 100,
        sample_edge_ratio_to_oracle=float(sample_edges / oracle_edges.clamp_min(1e-12)),
        sampled_latents_finite=bool(torch.isfinite(sampled).all()),
    )


def _action_aware_epoch_indices(
    dataset: CachedTemporalLatentDataset,
    target_action_change_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Include every natural window once, then add action-change examples."""
    if not 0 <= target_action_change_fraction < 1:
        raise ValueError("target action-change fraction must be in [0, 1)")
    natural = torch.randperm(len(dataset), generator=generator)
    current_changes = dataset.action_change_count
    current_fraction = current_changes / len(dataset)
    if current_changes == 0 or target_action_change_fraction <= current_fraction:
        return natural
    extra_count = math.ceil(
        (target_action_change_fraction * len(dataset) - current_changes)
        / (1 - target_action_change_fraction)
    )
    change_indices = torch.from_numpy(np.flatnonzero(dataset.action_changes).astype(np.int64))
    selected = change_indices[
        torch.randint(len(change_indices), (extra_count,), generator=generator)
    ]
    combined = torch.cat((natural, selected))
    return combined[torch.randperm(len(combined), generator=generator)]


def _training_loader(
    dataset: CachedTemporalLatentDataset,
    *,
    batch_size: int,
    target_action_change_fraction: float,
    generator: torch.Generator,
) -> DataLoader:
    indices = _action_aware_epoch_indices(dataset, target_action_change_fraction, generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices.tolist(),
        num_workers=0,
        drop_last=True,
    )


def _checkpoint_payload(
    model: TemporalActionUNet,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    target_steps: int,
    history: dict[str, list],
    validation: dict,
    data_generator: torch.Generator,
    training: CachedTemporalLatentDataset,
    validation_references: list,
    autoencoder_checkpoint: Path,
    manifest_path: Path,
    cache_root: Path,
    sampling_steps: int,
    maximum_context_noise: float,
    target_action_change_fraction: float,
    seed: int,
) -> dict:
    return {
        "format_version": 3,
        "model_type": "temporal_latent_diffusion_v2",
        "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "latent_channels": model.latent_channels,
        "action_dim": model.action_dim,
        "context_frames": model.context_frames,
        "base_channels": model.base_channels,
        "attention_heads": model.attention_heads,
        "diffusion_steps": model.schedule.diffusion_steps,
        "noise_schedule": "linear_beta",
        "beta_start": model.schedule.beta_start,
        "beta_end": model.schedule.beta_end,
        "sampling_steps": sampling_steps,
        "training_steps": step,
        "target_training_steps": target_steps,
        "maximum_context_noise": maximum_context_noise,
        "target_action_change_fraction": target_action_change_fraction,
        "seed": seed,
        "history": history,
        "validation_metrics": validation,
        "data_generator_state": data_generator.get_state(),
        "fixed_sequence_references": [asdict(reference) for reference in validation_references],
        "selection_policy": "full_v4_with_action_change_oversampling",
        "action_bucket_counts": training.action_bucket_counts,
        "action_change_windows": training.action_change_count,
        "training_windows": len(training),
        "source_episodes": [record.episode for record in training.cache.metadata.episodes],
        "cache_root": str(cache_root),
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
        "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _file_sha256(manifest_path),
    }


def _save_training_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_training_curve(history: dict[str, list], path: Path) -> None:
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["train_loss"], label="training velocity MSE")
    axes[0].plot(
        history["validation_step"],
        history["natural_correct_loss"],
        marker="o",
        label="held-out velocity MSE",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        history["validation_step"],
        history["natural_previous_penalty"],
        marker="o",
        label="natural: previous action",
    )
    axes[1].plot(
        history["validation_step"],
        history["change_previous_penalty"],
        marker="o",
        label="action changes: previous action",
    )
    axes[1].axhline(10, color="black", linestyle="--", alpha=0.5, label="10% gate")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("wrong-action penalty (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("V2 full-data latent diffusion")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_validation_visuals(
    model: TemporalActionUNet,
    autoencoder: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    sample_path: Path,
    action_path: Path,
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    count: int = 6,
) -> None:
    model.eval()
    autoencoder.eval()
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    batch = _stack_items(dataset, indices, device)
    sampled = model.sample(
        batch["context_latents"], batch["actions"], steps=sampling_steps, seed=seed
    )
    last_context = autoencoder.decode(batch["context_latents"][:, -1]).clamp(0, 1)
    oracle = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
    generated = autoencoder.decode(sampled).clamp(0, 1)
    columns = (last_context, oracle, generated)
    labels = ("last context", "target (decoder oracle)", f"V2 sample ({sampling_steps} DDIM)")
    tile, header = 160, 26
    canvas = np.full((count * (tile + header), len(columns) * tile, 3), 24, np.uint8)
    for row in range(count):
        for column, (frames, label) in enumerate(zip(columns, labels, strict=True)):
            x, y = column * tile, row * (tile + header)
            cv2.putText(
                canvas,
                label,
                (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            frame = frames[row].cpu().permute(1, 2, 0).numpy()
            canvas[y + header : y + header + tile, x : x + tile] = _render_frame(frame, tile)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(sample_path), canvas):
        raise ValueError(f"could not write {sample_path}")

    rollout_steps = 5
    seed_index = most_structured_seed(autoencoder, batch["context_latents"])
    context = batch["context_latents"][seed_index : seed_index + 1].repeat(3, 1, 1, 1, 1)
    history = batch["actions"][seed_index : seed_index + 1].repeat(3, 1, 1)
    scripts = counterfactual_action_scripts(rollout_steps, device, dtype=history.dtype)
    variants = autoregressive_action_rollout(
        model,
        context,
        history,
        scripts,
        sampling_steps=sampling_steps,
        seed=seed + 10_000,
        shared_noise_across_batch=True,
    )
    decoded = autoencoder.decode(variants.flatten(0, 1)).clamp(0, 1)
    variant_frames = decoded.reshape(3, rollout_steps, *decoded.shape[1:])
    seed_frame = autoencoder.decode(context[:1, -1]).clamp(0, 1)[0]
    row_labels = ("forward + sprint", "look left", "look right")
    action_tile = 128
    action_canvas = np.full(
        (3 * (action_tile + header), (rollout_steps + 1) * action_tile, 3), 24, np.uint8
    )
    for row, row_label in enumerate(row_labels):
        frames = torch.cat((seed_frame[None], variant_frames[row]), dim=0)
        for column, frame_tensor in enumerate(frames):
            x, y = column * action_tile, row * (action_tile + header)
            label = f"{row_label} | seed" if column == 0 else f"t+{column}"
            cv2.putText(
                action_canvas,
                label,
                (x + 5, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            frame = frame_tensor.cpu().permute(1, 2, 0).numpy()
            action_canvas[y + header : y + header + action_tile, x : x + action_tile] = (
                _render_frame(frame, action_tile)
            )
    if not cv2.imwrite(str(action_path), action_canvas):
        raise ValueError(f"could not write {action_path}")


def train_latent_diffusion_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    training_steps: int = 60_000,
    batch_size: int = 8,
    encode_batch_size: int = 128,
    base_channels: int = 112,
    attention_heads: int = 8,
    diffusion_steps: int = 1_000,
    sampling_steps: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    maximum_context_noise: float = 0.2,
    target_action_change_fraction: float = 0.35,
    evaluation_every: int = 2_000,
    maximum_validation_sequences: int = 512,
    sample_count: int = 16,
    resume_checkpoint: Path | None = None,
    force_cache: bool = False,
    seed: int = 7,
    requested_device: str = "auto",
) -> V2TrainingResult:
    if min(
        training_steps,
        batch_size,
        encode_batch_size,
        sampling_steps,
        evaluation_every,
        maximum_validation_sequences,
        sample_count,
    ) < 1:
        raise ValueError("V2 training sizes must be positive")
    if not 0 <= maximum_context_noise < 1:
        raise ValueError("maximum context noise must be in [0, 1)")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha = _file_sha256(autoencoder_checkpoint)
    manifest_sha = _file_sha256(manifest_path)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    manifest = DatasetManifest.load(manifest_path)
    train_paths, validation_paths = manifest.processed_splits(processed_dir)
    print("preparing the full V4 training latent cache...")
    train_cache = LatentEpisodeCache.build(
        train_paths,
        autoencoder,
        device,
        cache_root,
        autoencoder_sha256=autoencoder_sha,
        manifest_sha256=manifest_sha,
        split="training",
        encode_batch_size=encode_batch_size,
        force=force_cache,
    )
    print("preparing the held-out V4 validation latent cache...")
    validation_cache = LatentEpisodeCache.build(
        validation_paths,
        autoencoder,
        device,
        cache_root,
        autoencoder_sha256=autoencoder_sha,
        manifest_sha256=manifest_sha,
        split="validation",
        encode_batch_size=encode_batch_size,
        force=force_cache,
    )
    training = CachedTemporalLatentDataset(train_cache)
    validation = CachedTemporalLatentDataset(validation_cache)
    natural_indices = validation.subset_indices(maximum_validation_sequences, seed=seed)
    change_indices = validation.subset_indices(
        maximum_validation_sequences, action_changes_only=True, seed=seed + 1
    )
    natural_validation: Dataset[dict[str, torch.Tensor]] = Subset(validation, natural_indices)
    change_validation: Dataset[dict[str, torch.Tensor]] = Subset(validation, change_indices)
    print(f"full training windows: {len(training):,}")
    print(
        f"training action changes: {training.action_change_count:,} "
        f"({training.action_change_count / len(training):.1%})"
    )
    print(f"training action buckets: {training.action_bucket_counts}")
    print(
        f"fixed validation: {len(natural_validation):,} natural + "
        f"{len(change_validation):,} action-change windows"
    )

    data_generator = torch.Generator().manual_seed(seed)
    history: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "validation_step": [],
        "natural_correct_loss": [],
        "natural_previous_penalty": [],
        "change_correct_loss": [],
        "change_previous_penalty": [],
        "sample_psnr_to_oracle_db": [],
    }
    start_step = 0
    if resume_checkpoint is None:
        statistics = training.normalization_statistics()
        model = TemporalActionUNet(
            latent_channels=training.latent_shape[0],
            action_dim=9,
            context_frames=8,
            base_channels=base_channels,
            attention_heads=attention_heads,
            diffusion_steps=diffusion_steps,
            latent_mean=statistics[0],
            latent_std=statistics[1],
            action_mean=statistics[2],
            action_std=statistics[3],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        model, resume = load_latent_diffusion_v2_checkpoint(resume_checkpoint, device)
        if resume.get("autoencoder_sha256") != autoencoder_sha:
            raise ValueError("resume checkpoint uses a different autoencoder")
        if resume.get("dataset_manifest_sha256") != manifest_sha:
            raise ValueError("resume checkpoint uses a different V4 manifest")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if "optimizer_state" not in resume:
            raise ValueError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(resume["optimizer_state"])
        start_step = int(resume["training_steps"])
        history = resume["history"]
        if "data_generator_state" in resume:
            data_generator.set_state(resume["data_generator_state"].cpu())
        print(f"resuming {resume_checkpoint} from step {start_step:,}")
    if start_step >= training_steps:
        raise ValueError("resume checkpoint already reached the requested training steps")
    print(f"V2 parameters: {model.parameter_count:,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    metrics_path = output_dir / "metrics.json"
    curve_path = output_dir / "training-curve.png"
    samples_path = output_dir / "validation-samples.png"
    actions_path = output_dir / "validation-action-rollout.png"
    best_validation = float("inf")
    if best_path.exists():
        existing = torch.load(best_path, map_location="cpu", weights_only=True)
        best_validation = float(
            existing.get("validation_metrics", {})
            .get("natural", {})
            .get("correct_denoising_loss", float("inf"))
        )
    validation_record: dict = {}
    loader = _training_loader(
        training,
        batch_size=batch_size,
        target_action_change_fraction=target_action_change_fraction,
        generator=data_generator,
    )
    iterator = iter(loader)
    running_loss = 0.0
    running_examples = 0
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, training_steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            loader = _training_loader(
                training,
                batch_size=batch_size,
                target_action_change_fraction=target_action_change_fraction,
                generator=data_generator,
            )
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(
            model, batch, maximum_context_noise=maximum_context_noise
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(loss.detach()) * len(batch["actions"])
        running_examples += len(batch["actions"])
        if step == start_step + 1 or step % 100 == 0:
            averaged_loss = running_loss / running_examples
            history["step"].append(step)
            history["train_loss"].append(averaged_loss)
            elapsed = time.perf_counter() - started
            print(
                f"step {step:6,d}/{training_steps:,}: train={averaged_loss:.6f}  "
                f"{(step - start_step) / max(elapsed, 1e-9):.2f} steps/s"
            )
            running_loss = 0.0
            running_examples = 0

        should_evaluate = step % evaluation_every == 0 or step == training_steps
        if not should_evaluate:
            continue
        natural_metrics = evaluate_action_conditions(
            model, natural_validation, device, batch_size=batch_size, seed=seed
        )
        change_metrics = evaluate_action_conditions(
            model, change_validation, device, batch_size=batch_size, seed=seed + 20_000
        )
        sample_metrics = evaluate_samples(
            model,
            autoencoder,
            natural_validation,
            device,
            sampling_steps=sampling_steps,
            seed=seed,
            maximum_samples=sample_count,
        )
        validation_record = {
            "step": step,
            "natural": asdict(natural_metrics),
            "action_changes": asdict(change_metrics),
            "samples": asdict(sample_metrics),
        }
        history["validation_step"].append(step)
        history["natural_correct_loss"].append(natural_metrics.correct_denoising_loss)
        history["natural_previous_penalty"].append(
            natural_metrics.previous_action_penalty_percent
        )
        history["change_correct_loss"].append(change_metrics.correct_denoising_loss)
        history["change_previous_penalty"].append(
            change_metrics.previous_action_penalty_percent
        )
        history["sample_psnr_to_oracle_db"].append(
            sample_metrics.sample_pixel_psnr_to_oracle_db
        )
        references = [validation.reference(index) for index in natural_indices[:32]]
        payload = _checkpoint_payload(
            model,
            optimizer,
            step=step,
            target_steps=training_steps,
            history=history,
            validation=validation_record,
            data_generator=data_generator,
            training=training,
            validation_references=references,
            autoencoder_checkpoint=autoencoder_checkpoint,
            manifest_path=manifest_path,
            cache_root=cache_root,
            sampling_steps=sampling_steps,
            maximum_context_noise=maximum_context_noise,
            target_action_change_fraction=target_action_change_fraction,
            seed=seed,
        )
        _save_training_checkpoint(latest_path, payload)
        selection_payload = {
            name: value
            for name, value in payload.items()
            if name not in {"optimizer_state", "data_generator_state"}
        }
        _save_training_checkpoint(
            output_dir / "checkpoints" / f"step-{step:06d}.pt", selection_payload
        )
        if natural_metrics.correct_denoising_loss < best_validation:
            best_validation = natural_metrics.correct_denoising_loss
            _save_training_checkpoint(best_path, payload)
        _write_metrics(
            metrics_path,
            {
                "stage": "Stage 3 - full V4 held-out training",
                "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
                "parameters": model.parameter_count,
                "training_windows": len(training),
                "validation_windows": len(validation),
                "action_change_windows": training.action_change_count,
                "action_bucket_counts": training.action_bucket_counts,
                "target_action_change_fraction": target_action_change_fraction,
                "training_steps": step,
                "target_training_steps": training_steps,
                "batch_size": batch_size,
                "maximum_context_noise": maximum_context_noise,
                "sampling_steps": sampling_steps,
                "device": str(device),
                "validation": validation_record,
                "history": history,
            },
        )
        _save_training_curve(history, curve_path)
        save_validation_visuals(
            model,
            autoencoder,
            natural_validation,
            samples_path,
            actions_path,
            device,
            sampling_steps=sampling_steps,
            seed=seed,
        )
        print(
            f"validation step {step:,}: loss={natural_metrics.correct_denoising_loss:.6f}, "
            f"previous-action penalty={natural_metrics.previous_action_penalty_percent:+.1f}%, "
            f"change penalty={change_metrics.previous_action_penalty_percent:+.1f}%, "
            f"sample PSNR={sample_metrics.sample_pixel_psnr_to_oracle_db:.2f} dB"
        )
        model.train()

    return V2TrainingResult(
        checkpoint=best_path,
        latest_checkpoint=latest_path,
        metrics_path=metrics_path,
        training_curve=curve_path,
        samples=samples_path,
        action_comparison=actions_path,
        training_windows=len(training),
        validation_windows=len(validation),
        action_change_windows=training.action_change_count,
        parameter_count=model.parameter_count,
        completed_steps=training_steps,
        device=str(device),
    )
