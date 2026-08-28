"""Anchored EDM training for stable, action-conditioned spatial latent forecasts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mcwm.dynamics import _file_sha256
from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics, SpatialLatentEDM
from mcwm.spatial_dynamics import (
    SpatialEncodedDynamicsDataset,
    _move_batch,
    load_spatial_dynamics_checkpoint,
)
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SPATIAL_EDM_ARCHITECTURE = "anchored_edm_unet_v2"


@dataclass(frozen=True)
class EDMSampleMetrics:
    examples: int
    denoising_mse: float
    sample_latent_mse: float
    anchor_latent_mse: float
    copy_latent_mse: float
    shuffled_action_latent_mse: float
    zero_action_latent_mse: float
    action_effect_latent_mse: float
    sample_pixel_mse: float
    anchor_pixel_mse: float
    sample_pixel_psnr_db: float
    sample_edge_ratio: float
    oracle_edge_ratio: float
    sample_gradient_cosine: float

    @property
    def beats_anchor_latent(self) -> bool:
        return self.sample_latent_mse < self.anchor_latent_mse

    @property
    def shuffled_action_penalty_percent(self) -> float:
        return 100 * (self.shuffled_action_latent_mse / self.sample_latent_mse - 1)


@dataclass(frozen=True)
class SpatialEDMTrainingResult:
    checkpoint: Path
    training_curve: Path
    comparison_grid: Path
    metrics_path: Path
    validation_metrics: EDMSampleMetrics
    parameter_count: int
    training_transitions: int
    latent_shape: tuple[int, int, int]
    device: str


class SpatialEncodedContextDataset(Dataset[dict[str, torch.Tensor]]):
    """Latent transitions with several clean historical frames and aligned actions."""

    def __init__(
        self,
        encoded: SpatialEncodedDynamicsDataset,
        *,
        context_steps: int = 4,
        maximum_transitions: int | None = None,
        seed: int = 7,
    ):
        if context_steps < 2:
            raise ValueError("context_steps must be at least two")
        if maximum_transitions is not None and maximum_transitions < 1:
            raise ValueError("maximum_transitions must be positive when supplied")
        self.episodes = encoded.episodes
        self.latents = encoded.latents
        self.latent_shape = encoded.latent_shape
        self.context_steps = context_steps
        self.index = [
            (episode_index, current_index)
            for episode_index, current_index in encoded.index
            if current_index >= context_steps - 1
        ]
        if maximum_transitions is not None and len(self.index) > maximum_transitions:
            selected = torch.randperm(
                len(self.index), generator=torch.Generator().manual_seed(seed)
            )[:maximum_transitions]
            self.index = [self.index[int(item)] for item in selected]
        if not self.index:
            raise ValueError("no spatial transitions have enough context")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        start = current_index - self.context_steps + 1
        episode = self.episodes[episode_index]
        return {
            "context_latents": self.latents[episode_index][start : current_index + 1].float(),
            "actions": torch.from_numpy(
                episode.actions[start : current_index + 1].astype(np.float32, copy=False)
            ),
            "target_latent": self.latents[episode_index][current_index + 1].float(),
            "target_frame": _frame_tensor(episode.frames[current_index + 1]),
        }


def _frame_tensor(frame: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(frame.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).float().div_(255)


def _anchor_prediction(
    base: SpatialLatentDynamics | None,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if base is None:
        return context_latents[:, -1]
    return base(context_latents[:, -2], context_latents[:, -1], actions[:, -1])


def _cpu_random(
    shape: torch.Size | tuple[int, ...],
    reference: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(shape, dtype=reference.dtype, generator=generator).to(reference.device)


def _corrupt_context(
    model: SpatialLatentEDM,
    context: torch.Tensor,
    maximum_noise: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if maximum_noise < 0:
        raise ValueError("maximum context noise must be non-negative")
    if maximum_noise == 0:
        return context, context.new_zeros(len(context))
    if generator is None:
        levels = torch.rand(len(context), device=context.device) * maximum_noise
        noise = torch.randn_like(context)
    else:
        levels = torch.rand(len(context), generator=generator).to(context.device) * maximum_noise
        noise = _cpu_random(context.shape, context, generator)
    scale = model.latent_std * levels[:, None, None, None, None]
    return context + noise * scale, levels


def _edm_loss(
    model: SpatialLatentEDM,
    base: SpatialLatentDynamics | None,
    batch: dict[str, torch.Tensor],
    *,
    context_noise: float,
    p_mean: float = -0.4,
    p_std: float = 1.2,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    context, context_levels = _corrupt_context(
        model, batch["context_latents"], context_noise, generator=generator
    )
    with torch.no_grad():
        anchor = _anchor_prediction(base, context, batch["actions"])
        clean = model.normalize_correction(anchor, batch["target_latent"])
    if generator is None:
        sigmas = (torch.randn(len(clean), device=clean.device) * p_std + p_mean).exp()
        noise = torch.randn_like(clean)
    else:
        sigmas = (
            torch.randn(len(clean), generator=generator).to(clean.device) * p_std + p_mean
        ).exp()
        noise = _cpu_random(clean.shape, clean, generator)
    noisy = clean + noise * sigmas[:, None, None, None]
    denoised = model.denoise(
        noisy, sigmas, anchor, context, batch["actions"], context_levels
    )
    weights = (
        sigmas.square() + model.sigma_data**2
    ) / (sigmas * model.sigma_data).square()
    per_example = (denoised - clean).square().flatten(start_dim=1).mean(dim=1)
    return (weights * per_example).mean()


@torch.no_grad()
def _correction_statistics(
    base: SpatialLatentDynamics | None,
    dataset: SpatialEncodedContextDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = dataset.latent_shape[0]
    total = torch.zeros(channels, dtype=torch.float64)
    squared = torch.zeros(channels, dtype=torch.float64)
    values = 0
    for raw_batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch = _move_batch(raw_batch, device)
        anchor = _anchor_prediction(base, batch["context_latents"], batch["actions"])
        correction = batch["target_latent"] - anchor
        # MPS has no float64 reductions; transfer the small per-channel sums
        # before promoting them for stable dataset-wide accumulation.
        total += correction.sum(dim=(0, 2, 3)).cpu().double()
        squared += correction.square().sum(dim=(0, 2, 3)).cpu().double()
        values += len(correction) * correction.shape[2] * correction.shape[3]
    mean = total / values
    variance = squared / values - mean.square()
    return mean.float(), variance.clamp_min(1e-8).sqrt().float()


@torch.no_grad()
def evaluate_edm(
    model: SpatialLatentEDM,
    base: SpatialLatentDynamics | None,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedContextDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    sampling_steps: int = 8,
    sigma_max: float = 0.25,
    maximum_examples: int = 512,
    context_noise: float = 0.0,
    seed: int = 7,
) -> EDMSampleMetrics:
    if maximum_examples < 1:
        raise ValueError("maximum_examples must be positive")
    model.eval()
    autoencoder.eval()
    totals = {name: 0.0 for name in (
        "loss", "sample_latent", "anchor_latent", "copy_latent", "shuffled_latent",
        "zero_latent", "action_effect", "sample_pixel", "anchor_pixel", "sample_edge",
        "oracle_edge", "target_edge", "gradient_dot", "sample_gradient_squared",
        "target_gradient_squared",
    )}
    latent_values = 0
    pixel_values = 0
    examples = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        remaining = maximum_examples - examples
        if len(batch["actions"]) > remaining:
            batch = {name: value[:remaining] for name, value in batch.items()}
        count = len(batch["actions"])
        totals["loss"] += float(
            _edm_loss(model, base, batch, context_noise=context_noise, seed=seed + index)
        ) * count
        context = batch["context_latents"]
        actions = batch["actions"]
        anchor = _anchor_prediction(base, context, actions)
        shuffled_actions = torch.roll(actions, shifts=1, dims=0) if count > 1 else actions.flip(1)
        shuffled_anchor = _anchor_prediction(base, context, shuffled_actions)
        zero_actions = torch.zeros_like(actions)
        zero_anchor = _anchor_prediction(base, context, zero_actions)
        sample_seed = seed + 100_000 + index

        def sample(
            with_anchor: torch.Tensor,
            with_actions: torch.Tensor,
            *,
            fixed_context: torch.Tensor = context,
            fixed_seed: int = sample_seed,
        ) -> torch.Tensor:
            return model.sample(
                with_anchor,
                fixed_context,
                with_actions,
                steps=sampling_steps,
                sigma_max=sigma_max,
                generator=torch.Generator().manual_seed(fixed_seed),
            )

        sampled = sample(anchor, actions)
        shuffled_sampled = sample(shuffled_anchor, shuffled_actions)
        zero_sampled = sample(zero_anchor, zero_actions)
        target = batch["target_latent"]
        current = context[:, -1]
        sampled_frame = autoencoder.decode(sampled).clamp(0, 1)
        anchor_frame = autoencoder.decode(anchor).clamp(0, 1)
        oracle_frame = autoencoder.decode(target).clamp(0, 1)
        target_frame = batch["target_frame"]

        for name, prediction in (
            ("sample_latent", sampled),
            ("anchor_latent", anchor),
            ("copy_latent", current),
            ("shuffled_latent", shuffled_sampled),
            ("zero_latent", zero_sampled),
        ):
            totals[name] += (prediction - target).square().sum().item()
        totals["action_effect"] += (sampled - shuffled_sampled).square().sum().item()
        totals["sample_pixel"] += (sampled_frame - target_frame).square().sum().item()
        totals["anchor_pixel"] += (anchor_frame - target_frame).square().sum().item()
        sample_h, sample_v = image_gradients(sampled_frame)
        oracle_h, oracle_v = image_gradients(oracle_frame)
        target_h, target_v = image_gradients(target_frame)
        totals["sample_edge"] += sample_h.abs().sum().item() + sample_v.abs().sum().item()
        totals["oracle_edge"] += oracle_h.abs().sum().item() + oracle_v.abs().sum().item()
        totals["target_edge"] += target_h.abs().sum().item() + target_v.abs().sum().item()
        totals["gradient_dot"] += (
            (sample_h * target_h).sum().item() + (sample_v * target_v).sum().item()
        )
        totals["sample_gradient_squared"] += (
            sample_h.square().sum().item() + sample_v.square().sum().item()
        )
        totals["target_gradient_squared"] += (
            target_h.square().sum().item() + target_v.square().sum().item()
        )
        latent_values += target.numel()
        pixel_values += target_frame.numel()
        examples += count
        if examples >= maximum_examples:
            break
    pixel_mse = totals["sample_pixel"] / pixel_values
    return EDMSampleMetrics(
        examples=examples,
        denoising_mse=totals["loss"] / examples,
        sample_latent_mse=totals["sample_latent"] / latent_values,
        anchor_latent_mse=totals["anchor_latent"] / latent_values,
        copy_latent_mse=totals["copy_latent"] / latent_values,
        shuffled_action_latent_mse=totals["shuffled_latent"] / latent_values,
        zero_action_latent_mse=totals["zero_latent"] / latent_values,
        action_effect_latent_mse=totals["action_effect"] / latent_values,
        sample_pixel_mse=pixel_mse,
        anchor_pixel_mse=totals["anchor_pixel"] / pixel_values,
        sample_pixel_psnr_db=10 * np.log10(1 / max(pixel_mse, 1e-12)),
        sample_edge_ratio=totals["sample_edge"] / max(totals["target_edge"], 1e-12),
        oracle_edge_ratio=totals["oracle_edge"] / max(totals["target_edge"], 1e-12),
        sample_gradient_cosine=totals["gradient_dot"]
        / max(
            (totals["sample_gradient_squared"] * totals["target_gradient_squared"]) ** 0.5,
            1e-12,
        ),
    )


def _save_comparison(
    model: SpatialLatentEDM,
    base: SpatialLatentDynamics | None,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedContextDataset,
    device: torch.device,
    path: Path,
    *,
    sampling_steps: int,
    sigma_max: float,
    count: int = 6,
    seed: int = 7,
) -> None:
    items = [dataset[index] for index in range(min(count, len(dataset)))]
    batch = _move_batch(
        {name: torch.stack([item[name] for item in items]) for name in items[0]}, device
    )
    with torch.no_grad():
        anchor = _anchor_prediction(base, batch["context_latents"], batch["actions"])
        sample = model.sample(
            anchor,
            batch["context_latents"],
            batch["actions"],
            steps=sampling_steps,
            sigma_max=sigma_max,
            generator=torch.Generator().manual_seed(seed),
        )
        images = {
            "current": autoencoder.decode(batch["context_latents"][:, -1]).clamp(0, 1),
            "anchor V1": autoencoder.decode(anchor).clamp(0, 1),
            "EDM sample": autoencoder.decode(sample).clamp(0, 1),
            "target": batch["target_frame"],
        }
    tile = 192
    header = 28
    canvas = np.full((len(images) * (tile + header), len(items) * tile, 3), 20, np.uint8)
    for row, (label, frames) in enumerate(images.items()):
        cv2.putText(
            canvas,
            label,
            (5, row * (tile + header) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        for column, frame in enumerate(frames):
            rgb = frame.mul(255).byte().permute(1, 2, 0).cpu().numpy()
            canvas[
                row * (tile + header) + header : (row + 1) * (tile + header),
                column * tile : (column + 1) * tile,
            ] = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                (tile, tile),
                interpolation=cv2.INTER_NEAREST,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write EDM comparison to {path}")


def _save_curve(history: dict[str, list[float]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train"], label="train")
    axes[0].plot(history["validation"], label="validation")
    axes[0].set_title("EDM objective")
    axes[0].legend()
    axes[1].plot(history["sample_latent_mse"], label="EDM sample")
    axes[1].plot(history["anchor_latent_mse"], label="V1 anchor")
    axes[1].set_title("latent MSE")
    axes[1].legend()
    axes[2].plot(history["gradient_cosine"], label="gradient cosine")
    axes[2].plot(history["edge_ratio"], label="edge ratio")
    axes[2].set_title("structure")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("epoch")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_checkpoint(
    path: Path,
    model: SpatialLatentEDM,
    *,
    history: dict[str, list[float]],
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path | None,
    manifest_path: Path,
    sampling_steps: int,
    sigma_max: float,
    context_noise: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "spatial_latent_edm",
            "architecture": SPATIAL_EDM_ARCHITECTURE,
            "model_state": model.state_dict(),
            "latent_channels": model.latent_channels,
            "action_dim": model.action_dim,
            "context_steps": model.context_steps,
            "hidden_channels": model.hidden_channels,
            "blocks_per_level": model.blocks_per_level,
            "sigma_data": model.sigma_data,
            "sampling_steps": sampling_steps,
            "sigma_max": sigma_max,
            "context_noise": context_noise,
            "history": history,
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
            "dynamics_checkpoint": (
                None if dynamics_checkpoint is None else str(dynamics_checkpoint)
            ),
            "dynamics_sha256": (
                None if dynamics_checkpoint is None else _file_sha256(dynamics_checkpoint)
            ),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
        },
        path,
    )


def load_spatial_edm_checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialLatentEDM, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "spatial_latent_edm":
        raise ValueError("checkpoint is not a spatial latent EDM")
    if checkpoint.get("architecture") != SPATIAL_EDM_ARCHITECTURE:
        raise ValueError("spatial EDM checkpoint uses an incompatible architecture")
    state = checkpoint["model_state"]
    model = SpatialLatentEDM(
        latent_channels=int(checkpoint["latent_channels"]),
        action_dim=int(checkpoint["action_dim"]),
        context_steps=int(checkpoint["context_steps"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        blocks_per_level=int(checkpoint["blocks_per_level"]),
        sigma_data=float(checkpoint["sigma_data"]),
        action_mean=state["action_mean"],
        action_std=state["action_std"],
        latent_mean=state["latent_mean"].flatten(),
        latent_std=state["latent_std"].flatten(),
        correction_mean=state["correction_mean"].flatten(),
        correction_std=state["correction_std"].flatten(),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _load_base(
    checkpoint: Path | None, device: torch.device
) -> SpatialLatentDynamics | None:
    if checkpoint is None:
        return None
    base, _ = load_spatial_dynamics_checkpoint(checkpoint, device)
    base.requires_grad_(False)
    base.eval()
    return base


def train_spatial_edm(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    dynamics_checkpoint: Path | None,
    epochs: int = 10,
    batch_size: int = 32,
    encode_batch_size: int = 128,
    maximum_transitions: int = 20_000,
    evaluation_examples: int = 512,
    context_steps: int = 4,
    hidden_channels: int = 64,
    blocks_per_level: int = 2,
    sampling_steps: int = 8,
    sigma_max: float = 0.25,
    context_noise: float = 0.1,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialEDMTrainingResult:
    if min(epochs, batch_size, maximum_transitions, evaluation_examples, patience) < 1:
        raise ValueError("training sizes must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    base = _load_base(dynamics_checkpoint, device)
    manifest = DatasetManifest.load(manifest_path)
    train_paths, validation_paths = manifest.processed_splits(processed_dir)
    print(f"encoding up to {maximum_transitions:,} EDM training transitions...")
    encoded_training = SpatialEncodedDynamicsDataset.from_paths(
        train_paths,
        autoencoder,
        device,
        maximum_transitions=maximum_transitions * 2,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    training = SpatialEncodedContextDataset(
        encoded_training,
        context_steps=context_steps,
        maximum_transitions=maximum_transitions,
        seed=seed,
    )
    print("encoding frozen EDM validation transitions...")
    encoded_validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths,
        autoencoder,
        device,
        maximum_transitions=evaluation_examples * 4,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    validation = SpatialEncodedContextDataset(
        encoded_validation,
        context_steps=context_steps,
        maximum_transitions=evaluation_examples,
        seed=seed,
    )
    statistics = encoded_training.normalization_statistics()
    correction_mean, correction_std = _correction_statistics(
        base, training, device, batch_size=batch_size
    )
    model = SpatialLatentEDM(
        latent_channels=training.latent_shape[0],
        action_dim=9,
        context_steps=context_steps,
        hidden_channels=hidden_channels,
        blocks_per_level=blocks_per_level,
        action_mean=statistics[0],
        action_std=statistics[1],
        latent_mean=statistics[2],
        latent_std=statistics[3],
        correction_mean=correction_mean,
        correction_std=correction_std,
    ).to(device)
    print(f"EDM parameters: {model.parameter_count:,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    history = {
        "train": [], "validation": [], "sample_latent_mse": [],
        "anchor_latent_mse": [], "gradient_cosine": [], "edge_ratio": [],
        "action_penalty_percent": [],
    }
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _edm_loss(model, base, batch, context_noise=context_noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch["actions"])
            total += float(loss.detach()) * count
            examples += count
        metrics = evaluate_edm(
            model,
            base,
            autoencoder,
            validation,
            device,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            sigma_max=sigma_max,
            maximum_examples=evaluation_examples,
            seed=seed,
        )
        train_loss = total / examples
        for name, value in (
            ("train", train_loss),
            ("validation", metrics.denoising_mse),
            ("sample_latent_mse", metrics.sample_latent_mse),
            ("anchor_latent_mse", metrics.anchor_latent_mse),
            ("gradient_cosine", metrics.sample_gradient_cosine),
            ("edge_ratio", metrics.sample_edge_ratio),
            ("action_penalty_percent", metrics.shuffled_action_penalty_percent),
        ):
            history[name].append(value)
        print(
            f"epoch {epoch + 1:3d}/{epochs}: train={train_loss:.6f}  "
            f"validation={metrics.denoising_mse:.6f}  "
            f"latent={metrics.sample_latent_mse:.6f}  "
            f"anchor={metrics.anchor_latent_mse:.6f}  "
            f"grad={metrics.sample_gradient_cosine:.3f}  "
            f"action={metrics.shuffled_action_penalty_percent:+.1f}%"
        )
        score = metrics.sample_latent_mse
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {epoch + 1} epochs")
                break
    if best_state is None:
        raise RuntimeError("spatial EDM training produced no checkpoint")
    model.load_state_dict(best_state)
    final = evaluate_edm(
        model,
        base,
        autoencoder,
        validation,
        device,
        batch_size=batch_size,
        sampling_steps=sampling_steps,
        sigma_max=sigma_max,
        maximum_examples=evaluation_examples,
        seed=seed,
    )
    checkpoint = output_dir / "best.pt"
    curve = output_dir / "training-curve.png"
    comparison = output_dir / "one-step-comparison.png"
    metrics_path = output_dir / "metrics.json"
    _save_checkpoint(
        checkpoint,
        model,
        history=history,
        autoencoder_checkpoint=autoencoder_checkpoint,
        dynamics_checkpoint=dynamics_checkpoint,
        manifest_path=manifest_path,
        sampling_steps=sampling_steps,
        sigma_max=sigma_max,
        context_noise=context_noise,
    )
    _save_curve(history, curve)
    _save_comparison(
        model,
        base,
        autoencoder,
        validation,
        device,
        comparison,
        sampling_steps=sampling_steps,
        sigma_max=sigma_max,
        seed=seed,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "anchored spatial latent EDM",
                "training_transitions": len(training),
                "validation_transitions": len(validation),
                "parameters": model.parameter_count,
                "context_steps": context_steps,
                "sampling_steps": sampling_steps,
                "sigma_max": sigma_max,
                "context_noise": context_noise,
                "dynamics_checkpoint": (
                    None if dynamics_checkpoint is None else str(dynamics_checkpoint)
                ),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "dataset_manifest": str(manifest_path),
                "device": str(device),
                "validation": final.__dict__,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SpatialEDMTrainingResult(
        checkpoint=checkpoint,
        training_curve=curve,
        comparison_grid=comparison,
        metrics_path=metrics_path,
        validation_metrics=final,
        parameter_count=model.parameter_count,
        training_transitions=len(training),
        latent_shape=training.latent_shape,
        device=str(device),
    )


@torch.no_grad()
def evaluate_saved_spatial_edm(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    edm_checkpoint: Path,
    *,
    batch_size: int = 32,
    encode_batch_size: int = 128,
    sampling_steps: int | None = None,
    sigma_max: float | None = None,
    maximum_examples: int = 512,
    split: DatasetSplit = "validation",
    seed: int = 7,
    requested_device: str = "auto",
) -> EDMSampleMetrics:
    device = choose_device(requested_device)
    model, metadata = load_spatial_edm_checkpoint(edm_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial EDM belongs to a different autoencoder checkpoint")
    dynamics_path = metadata["dynamics_checkpoint"]
    if dynamics_path is not None and metadata["dynamics_sha256"] != _file_sha256(
        Path(dynamics_path)
    ):
        raise ValueError("spatial EDM anchor checkpoint changed")
    base = _load_base(None if dynamics_path is None else Path(dynamics_path), device)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    paths = DatasetManifest.load(manifest_path).processed_paths(processed_dir, split)
    encoded = SpatialEncodedDynamicsDataset.from_paths(
        paths,
        autoencoder,
        device,
        maximum_transitions=maximum_examples * 4,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    dataset = SpatialEncodedContextDataset(
        encoded,
        context_steps=model.context_steps,
        maximum_transitions=maximum_examples,
        seed=seed,
    )
    return evaluate_edm(
        model,
        base,
        autoencoder,
        dataset,
        device,
        batch_size=batch_size,
        sampling_steps=sampling_steps or int(metadata["sampling_steps"]),
        sigma_max=sigma_max or float(metadata.get("sigma_max", 5.0)),
        maximum_examples=maximum_examples,
        seed=seed,
    )
