"""Rollout fine-tuning for V2 latent diffusion.

Stage 3 trains one step at a time on eight real encoded frames, but inference
feeds the model seven of its own samples.  Those samples come back slightly
smoother than real latents, so recursive use converges on a smooth fixed point.

This module closes that gap the way V1's recursive fine-tune did: build the
conditioning context out of the model's own samples, then apply the ordinary
velocity objective against the real next latent.  Generation runs without
gradients, so the extra cost is forward passes rather than a longer graph.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mcwm.dynamics import _file_sha256
from mcwm.latent_data_v2 import CachedTemporalLatentDataset, LatentEpisodeCache
from mcwm.latent_diffusion_v2 import (
    LATENT_DIFFUSION_V2_ARCHITECTURE,
    TemporalActionUNet,
    diffusion_loss,
    load_latent_diffusion_v2_checkpoint,
)
from mcwm.latent_training_v2 import (
    _move_batch,
    _save_training_checkpoint,
    _write_metrics,
    evaluate_action_conditions,
    save_validation_visuals,
)
from mcwm.manifest import DatasetManifest
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

MAXIMUM_ROLLOUT_DEPTH = 3
CONTEXT_FRAMES = 8


@torch.no_grad()
def self_generated_context(
    model: TemporalActionUNet,
    context: torch.Tensor,
    actions: torch.Tensor,
    depth: int,
    *,
    sampling_steps: int,
    seed: int,
) -> torch.Tensor:
    """Replace the last ``depth`` context frames with the model's own samples.

    ``context`` holds ``CONTEXT_FRAMES + MAXIMUM_ROLLOUT_DEPTH`` real latents.
    Generation starts far enough back that, after ``depth`` samples, the window
    lands on exactly the frames an ordinary training context would have used.
    """
    extra = context.shape[1] - CONTEXT_FRAMES
    if extra != MAXIMUM_ROLLOUT_DEPTH:
        raise ValueError("rollout windows need exactly MAXIMUM_ROLLOUT_DEPTH extra frames")
    if not 0 <= depth <= extra:
        raise ValueError(f"depth must be in [0, {extra}]")
    start = extra - depth
    window = context[:, start : start + CONTEXT_FRAMES]
    for offset in range(depth):
        step_actions = actions[:, start + offset : start + offset + CONTEXT_FRAMES]
        sampled = model.sample(window, step_actions, steps=sampling_steps, seed=seed + offset)
        window = torch.cat((window[:, 1:], sampled[:, None]), dim=1)
    return window


def rollout_diffusion_loss(
    model: TemporalActionUNet,
    batch: dict[str, torch.Tensor],
    *,
    depth: int,
    generation_steps: int,
    maximum_context_noise: float,
    seed: int,
) -> torch.Tensor:
    """Ordinary velocity loss against a context the model generated itself."""
    context = self_generated_context(
        model,
        batch["context_latents"],
        batch["actions"],
        depth,
        sampling_steps=generation_steps,
        seed=seed,
    )
    extra = MAXIMUM_ROLLOUT_DEPTH
    return diffusion_loss(
        model,
        {
            "context_latents": context,
            "actions": batch["actions"][:, extra : extra + CONTEXT_FRAMES],
            "target_latent": batch["target_latent"],
        },
        maximum_context_noise=maximum_context_noise,
    )


@dataclass(frozen=True)
class RolloutMetrics:
    examples: int
    horizon: int
    sampling_steps: int
    edge_ratio_by_step: tuple[float, ...]
    latent_std_by_step: tuple[float, ...]
    pixel_mse_by_step: tuple[float, ...]
    real_latent_std: float
    edge_ratio_final: float
    edge_ratio_mean: float
    pixel_mse_mean: float
    rollout_score: float


@torch.no_grad()
def evaluate_rollout(
    model: TemporalActionUNet,
    autoencoder: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    horizon: int,
    sampling_steps: int,
    seed: int,
    examples: int = 32,
) -> RolloutMetrics:
    """Recursively imagine ``horizon`` frames under the real held-out actions.

    Edge energy is reported as a ratio against the decoded real frame at the
    same step, so a value near one means the rollout keeps as much structure as
    the observation it is trying to predict.
    """
    model.eval()
    autoencoder.eval()
    count = min(examples, len(dataset))
    if count < 1:
        raise ValueError("rollout evaluation needs at least one window")
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    items = [dataset[index] for index in indices]
    context = torch.stack([item["context_latents"] for item in items]).to(device)
    actions = torch.stack([item["actions"] for item in items]).to(device)
    if context.shape[1] < CONTEXT_FRAMES + horizon:
        raise ValueError("rollout windows are shorter than the requested horizon")

    def edge_energy(frames: torch.Tensor) -> torch.Tensor:
        dx, dy = image_gradients(frames)
        return dx.abs().mean() + dy.abs().mean()

    window = context[:, :CONTEXT_FRAMES].clone()
    edges: list[float] = []
    stds: list[float] = []
    errors: list[float] = []
    for step in range(horizon):
        step_actions = actions[:, step : step + CONTEXT_FRAMES]
        sampled = model.sample(window, step_actions, steps=sampling_steps, seed=seed + step)
        real = context[:, CONTEXT_FRAMES + step]
        predicted_frames = autoencoder.decode(sampled).clamp(0, 1)
        real_frames = autoencoder.decode(real).clamp(0, 1)
        edges.append(
            float(edge_energy(predicted_frames) / edge_energy(real_frames).clamp_min(1e-12))
        )
        stds.append(float(sampled.std()))
        errors.append(float(nn.functional.mse_loss(predicted_frames, real_frames)))
        window = torch.cat((window[:, 1:], sampled[:, None]), dim=1)
    return RolloutMetrics(
        examples=count,
        horizon=horizon,
        sampling_steps=sampling_steps,
        edge_ratio_by_step=tuple(edges),
        latent_std_by_step=tuple(stds),
        pixel_mse_by_step=tuple(errors),
        real_latent_std=float(context[:, :CONTEXT_FRAMES].std()),
        edge_ratio_final=edges[-1],
        edge_ratio_mean=float(np.mean(edges)),
        pixel_mse_mean=float(np.mean(errors)),
        # Accuracy alone rewards blur, and edge energy alone rewards grain, so
        # score accuracy and penalise sharpness that misses real edge energy in
        # either direction.
        rollout_score=float(np.mean(errors) * (1 + abs(np.mean(edges) - 1))),
    )


@dataclass(frozen=True)
class RolloutFinetuneResult:
    checkpoint: Path
    latest_checkpoint: Path
    metrics_path: Path
    samples: Path
    action_comparison: Path
    parameter_count: int
    completed_steps: int
    training_windows: int
    device: str


def _rollout_checkpoint_payload(
    model: TemporalActionUNet,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    history: dict[str, list],
    validation: dict,
    autoencoder_checkpoint: Path,
    manifest_path: Path,
    initial_checkpoint: Path,
    sampling_steps: int,
    generation_steps: int,
    maximum_context_noise: float,
    seed: int,
) -> dict:
    return {
        "format_version": 4,
        "model_type": "temporal_latent_diffusion_v2",
        "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
        "training_mode": "rollout_finetune",
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
        "generation_steps": generation_steps,
        "maximum_rollout_depth": MAXIMUM_ROLLOUT_DEPTH,
        "training_steps": step,
        "maximum_context_noise": maximum_context_noise,
        "seed": seed,
        "history": history,
        "validation_metrics": validation,
        "initial_checkpoint": str(initial_checkpoint),
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
        "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _file_sha256(manifest_path),
    }


def finetune_rollout_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    cache_root: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    *,
    training_steps: int = 6_000,
    batch_size: int = 8,
    encode_batch_size: int = 128,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
    generation_steps: int = 8,
    sampling_steps: int = 32,
    rollout_horizon: int = 8,
    maximum_context_noise: float = 0.2,
    evaluation_every: int = 1_000,
    maximum_validation_sequences: int = 256,
    rollout_examples: int = 24,
    seed: int = 7,
    requested_device: str = "auto",
) -> RolloutFinetuneResult:
    if min(training_steps, batch_size, generation_steps, sampling_steps, rollout_horizon) < 1:
        raise ValueError("rollout fine-tune sizes must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha = _file_sha256(autoencoder_checkpoint)
    manifest_sha = _file_sha256(manifest_path)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    manifest = DatasetManifest.load(manifest_path)
    train_paths, validation_paths = manifest.processed_splits(processed_dir)

    caches = {}
    for split, paths in (("training", train_paths), ("validation", validation_paths)):
        caches[split] = LatentEpisodeCache.build(
            paths,
            autoencoder,
            device,
            cache_root,
            autoencoder_sha256=autoencoder_sha,
            manifest_sha256=manifest_sha,
            split=split,
            encode_batch_size=encode_batch_size,
        )

    training = CachedTemporalLatentDataset(
        caches["training"], context_frames=CONTEXT_FRAMES + MAXIMUM_ROLLOUT_DEPTH
    )
    validation = CachedTemporalLatentDataset(caches["validation"])
    rollout_validation = CachedTemporalLatentDataset(
        caches["validation"], context_frames=CONTEXT_FRAMES + rollout_horizon
    )
    natural = Subset(validation, validation.subset_indices(maximum_validation_sequences, seed=seed))
    changes = Subset(
        validation,
        validation.subset_indices(
            maximum_validation_sequences, action_changes_only=True, seed=seed + 1
        ),
    )
    print(f"rollout training windows: {len(training):,}")
    print(f"held-out rollout windows: {len(rollout_validation):,}")

    model, metadata = load_latent_diffusion_v2_checkpoint(initial_checkpoint, device)
    if metadata.get("autoencoder_sha256") != autoencoder_sha:
        raise ValueError("the initial checkpoint uses a different autoencoder")
    if metadata.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("the initial checkpoint uses a different V4 manifest")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"fine-tuning {initial_checkpoint} ({model.parameter_count:,} parameters)")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "validation-samples.png"
    actions_path = output_dir / "validation-action-rollout.png"
    history: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "validation_step": [],
        "edge_ratio_final": [],
        "edge_ratio_mean": [],
        "pixel_mse_mean": [],
        "rollout_score": [],
        "change_previous_penalty": [],
    }

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True
    )
    iterator = iter(loader)
    best_score = float("inf")
    validation_record: dict = {}
    running, seen = 0.0, 0
    started = time.perf_counter()
    for step in range(1, training_steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = _move_batch(raw, device)
        depth = int(torch.randint(MAXIMUM_ROLLOUT_DEPTH + 1, (1,), generator=generator))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rollout_diffusion_loss(
            model,
            batch,
            depth=depth,
            generation_steps=generation_steps,
            maximum_context_noise=maximum_context_noise,
            seed=seed + step * 17,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += float(loss.detach()) * len(batch["actions"])
        seen += len(batch["actions"])
        if step == 1 or step % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step {step:5,d}/{training_steps:,}: train={running / seen:.6f}  "
                f"{step / max(elapsed, 1e-9):.2f} steps/s"
            )
            history["step"].append(step)
            history["train_loss"].append(running / seen)
            running, seen = 0.0, 0

        if step % evaluation_every and step != training_steps:
            continue
        rollout = evaluate_rollout(
            model,
            autoencoder,
            rollout_validation,
            device,
            horizon=rollout_horizon,
            sampling_steps=sampling_steps,
            seed=seed,
            examples=rollout_examples,
        )
        change_metrics = evaluate_action_conditions(
            model, changes, device, batch_size=batch_size, seed=seed + 20_000
        )
        validation_record = {
            "step": step,
            "rollout": asdict(rollout),
            "action_changes": asdict(change_metrics),
        }
        history["validation_step"].append(step)
        history["edge_ratio_final"].append(rollout.edge_ratio_final)
        history["edge_ratio_mean"].append(rollout.edge_ratio_mean)
        history["pixel_mse_mean"].append(rollout.pixel_mse_mean)
        history["rollout_score"].append(rollout.rollout_score)
        history["change_previous_penalty"].append(change_metrics.previous_action_penalty_percent)
        payload = _rollout_checkpoint_payload(
            model,
            optimizer,
            step=step,
            history=history,
            validation=validation_record,
            autoencoder_checkpoint=autoencoder_checkpoint,
            manifest_path=manifest_path,
            initial_checkpoint=initial_checkpoint,
            sampling_steps=sampling_steps,
            generation_steps=generation_steps,
            maximum_context_noise=maximum_context_noise,
            seed=seed,
        )
        _save_training_checkpoint(latest_path, payload)
        _save_training_checkpoint(
            output_dir / "checkpoints" / f"step-{step:06d}.pt",
            {k: v for k, v in payload.items() if k != "optimizer_state"},
        )
        if rollout.rollout_score < best_score:
            best_score = rollout.rollout_score
            _save_training_checkpoint(best_path, payload)
        _write_metrics(
            metrics_path,
            {
                "stage": "Stage 4 - rollout fine-tune",
                "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
                "initial_checkpoint": str(initial_checkpoint),
                "training_windows": len(training),
                "maximum_rollout_depth": MAXIMUM_ROLLOUT_DEPTH,
                "generation_steps": generation_steps,
                "sampling_steps": sampling_steps,
                "training_steps": step,
                "device": str(device),
                "validation": validation_record,
                "history": history,
            },
        )
        save_validation_visuals(
            model,
            autoencoder,
            natural,
            samples_path,
            actions_path,
            device,
            sampling_steps=sampling_steps,
            seed=seed,
        )
        edges = " ".join(f"{v:.3f}" for v in rollout.edge_ratio_by_step)
        print(
            f"validation step {step:,}: edge {edges} "
            f"| mse {rollout.pixel_mse_mean:.4f} | score {rollout.rollout_score:.4f} "
            f"| change penalty {change_metrics.previous_action_penalty_percent:+.1f}%"
        )

    return RolloutFinetuneResult(
        checkpoint=best_path,
        latest_checkpoint=latest_path,
        metrics_path=metrics_path,
        samples=samples_path,
        action_comparison=actions_path,
        parameter_count=model.parameter_count,
        completed_steps=training_steps,
        training_windows=len(training),
        device=str(device),
    )
