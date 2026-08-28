"""Train and evaluate conditional latent diffusion for the spatial dynamics."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mcwm.dynamics import _file_sha256
from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import SpatialAutoencoder, SpatialLatentDiffusion
from mcwm.spatial_dynamics import SpatialEncodedDynamicsDataset, _move_batch
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SPATIAL_DIFFUSION_ARCHITECTURE = "conditional_residual_diffusion_v1"


@dataclass(frozen=True)
class DiffusionSampleMetrics:
    """Quality of one sampled step, scored against the real next frame."""

    examples: int
    denoising_mse: float
    sample_latent_mse: float
    sample_pixel_mse: float
    sample_pixel_psnr_db: float
    sample_edge_ratio: float
    oracle_edge_ratio: float
    copy_latent_mse: float


@dataclass(frozen=True)
class SpatialDiffusionTrainingResult:
    checkpoint: Path
    training_curve: Path
    metrics_path: Path
    validation_metrics: DiffusionSampleMetrics
    parameter_count: int
    training_transitions: int
    latent_shape: tuple[int, int, int]
    device: str


def _noised_batch(
    model: SpatialLatentDiffusion,
    batch: dict[str, torch.Tensor],
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a timestep and noise, and return the noisy residual to denoise."""
    current = batch["current_latent"]
    start = model.normalize_residual(current, batch["target_latent"])
    if seed is None:
        timesteps = torch.randint(
            0, model.diffusion_steps, (len(current),), device=current.device
        )
        noise = torch.randn_like(start)
    else:
        # Validation must not move because the noise draw moved.
        generator = torch.Generator().manual_seed(seed)
        timesteps = torch.randint(
            0, model.diffusion_steps, (len(current),), generator=generator
        ).to(current.device)
        noise = torch.randn(start.shape, generator=generator).to(current.device)
    alpha_bar = model.alpha_bars[timesteps][:, None, None, None]
    noisy = alpha_bar.sqrt() * start + (1 - alpha_bar).sqrt() * noise
    return noisy, noise, timesteps


def _denoising_loss(
    model: SpatialLatentDiffusion,
    batch: dict[str, torch.Tensor],
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """The standard denoising objective: predict the noise that was added."""
    noisy, noise, timesteps = _noised_batch(model, batch, seed=seed)
    predicted = model(
        batch["previous_latent"], batch["current_latent"], batch["action"], noisy, timesteps
    )
    return nn.functional.mse_loss(predicted, noise)


@torch.no_grad()
def evaluate_diffusion(
    model: SpatialLatentDiffusion,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedDynamicsDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    sampling_steps: int = 20,
    maximum_examples: int = 2_000,
    seed: int = 7,
) -> DiffusionSampleMetrics:
    """Score the denoising objective and the quality of one sampled step."""
    if not len(dataset):
        raise ValueError("cannot evaluate an empty diffusion dataset")
    model.eval()
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    denoising_total = 0.0
    latent_squared = 0.0
    copy_squared = 0.0
    pixel_squared = 0.0
    sample_edge = 0.0
    oracle_edge = 0.0
    target_edge = 0.0
    latent_values = 0
    pixel_values = 0
    examples = 0
    for index, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        count = len(batch["action"])
        denoising_total += float(_denoising_loss(model, batch, seed=seed + index)) * count

        sampled = model.sample(
            batch["previous_latent"],
            batch["current_latent"],
            batch["action"],
            steps=sampling_steps,
            generator=torch.Generator().manual_seed(seed + index),
        )
        target_latent = batch["target_latent"]
        target_frame = batch["target_frame"]
        sampled_frame = autoencoder.decode(sampled).clamp(0, 1)
        oracle_frame = autoencoder.decode(target_latent).clamp(0, 1)

        latent_squared += (sampled - target_latent).square().sum().item()
        copy_squared += (batch["current_latent"] - target_latent).square().sum().item()
        pixel_squared += (sampled_frame - target_frame).square().sum().item()
        for total, image in (
            ("sample", sampled_frame),
            ("oracle", oracle_frame),
            ("target", target_frame),
        ):
            horizontal, vertical = image_gradients(image)
            energy = horizontal.abs().sum().item() + vertical.abs().sum().item()
            if total == "sample":
                sample_edge += energy
            elif total == "oracle":
                oracle_edge += energy
            else:
                target_edge += energy
        latent_values += target_latent.numel()
        pixel_values += target_frame.numel()
        examples += count
        if examples >= maximum_examples:
            break

    pixel_mse = pixel_squared / pixel_values
    return DiffusionSampleMetrics(
        examples=examples,
        denoising_mse=denoising_total / examples,
        sample_latent_mse=latent_squared / latent_values,
        sample_pixel_mse=pixel_mse,
        sample_pixel_psnr_db=10.0 * np.log10(1.0 / max(pixel_mse, 1e-12)),
        sample_edge_ratio=sample_edge / max(target_edge, 1e-12),
        oracle_edge_ratio=oracle_edge / max(target_edge, 1e-12),
        copy_latent_mse=copy_squared / latent_values,
    )


class SampledDynamics(nn.Module):
    """Present a diffusion model through the deterministic dynamics interface.

    The rollout evaluator and the interactive engine both call the dynamics as
    ``dynamics(previous, current, action)``. Wrapping the sampler here keeps a
    single recursive rollout implementation rather than a parallel one.
    """

    def __init__(
        self,
        model: SpatialLatentDiffusion,
        *,
        sampling_steps: int = 50,
        seed: int = 7,
    ):
        super().__init__()
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        self.model = model
        self.sampling_steps = sampling_steps
        self.latent_channels = model.latent_channels
        self.action_dim = model.action_dim
        self._seed = seed
        self._calls = 0

    def reset_sampling(self, seed: int | None = None) -> None:
        """Restart the noise stream so paired rollouts can share exact noise."""
        if seed is not None:
            self._seed = seed
        self._calls = 0

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # Each step draws fresh noise, but from a stream that a caller can
        # rewind, so a rollout stays reproducible without being frozen.
        generator = torch.Generator().manual_seed(self._seed + self._calls)
        self._calls += 1
        return self.model.sample(
            previous_latent,
            current_latent,
            action,
            steps=self.sampling_steps,
            generator=generator,
        )


def _save_checkpoint(
    path: Path,
    model: SpatialLatentDiffusion,
    *,
    history: dict[str, list[float]],
    autoencoder_checkpoint: Path,
    autoencoder_sha256: str,
    manifest_path: Path,
    sampling_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "spatial_latent_diffusion",
            "architecture": SPATIAL_DIFFUSION_ARCHITECTURE,
            "model_state": model.state_dict(),
            "latent_channels": model.latent_channels,
            "action_dim": model.action_dim,
            "hidden_channels": model.hidden_channels,
            "blocks": model.blocks,
            "diffusion_steps": model.diffusion_steps,
            "sampling_steps": sampling_steps,
            "history": history,
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": autoencoder_sha256,
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
        },
        path,
    )


def load_spatial_diffusion_checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialLatentDiffusion, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "spatial_latent_diffusion":
        raise ValueError("checkpoint is not spatial latent diffusion")
    if checkpoint.get("architecture") != SPATIAL_DIFFUSION_ARCHITECTURE:
        raise ValueError(
            "spatial diffusion checkpoint uses an incompatible or unversioned "
            "architecture; retrain it with the current code"
        )
    state = checkpoint["model_state"]
    model = SpatialLatentDiffusion(
        latent_channels=int(checkpoint["latent_channels"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        blocks=int(checkpoint["blocks"]),
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        action_mean=state["action_mean"],
        action_std=state["action_std"],
        latent_mean=state["latent_mean"].flatten(),
        latent_std=state["latent_std"].flatten(),
        motion_mean=state["motion_mean"].flatten(),
        motion_std=state["motion_std"].flatten(),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _save_curve(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (loss_axis, edge_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history["train"], label="training denoising MSE")
    loss_axis.plot(history["validation"], label="validation denoising MSE")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("noise prediction error")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    edge_axis.plot(history["validation_edge_ratio"], marker="o", label="sampled")
    edge_axis.axhline(0.584, linestyle="--", color="gray", label="deterministic V1")
    edge_axis.axhline(0.973, linestyle=":", color="green", label="decoder oracle")
    edge_axis.set_xlabel("epoch")
    edge_axis.set_ylabel("edge energy ratio")
    edge_axis.grid(alpha=0.25)
    edge_axis.legend()
    figure.suptitle("Conditional latent diffusion dynamics")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_spatial_diffusion(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    encode_batch_size: int = 128,
    maximum_transitions: int = 100_000,
    hidden_channels: int = 64,
    blocks: int = 3,
    diffusion_steps: int = 1_000,
    sampling_steps: int = 20,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    patience: int = 5,
    evaluation_examples: int = 2_000,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialDiffusionTrainingResult:
    if min(epochs, batch_size, maximum_transitions, patience, sampling_steps) < 1:
        raise ValueError("training sizes must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha256 = _file_sha256(autoencoder_checkpoint)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    manifest = DatasetManifest.load(manifest_path)
    train_paths, validation_paths = manifest.processed_splits(processed_dir)
    print(f"encoding up to {maximum_transitions:,} training transitions...")
    training = SpatialEncodedDynamicsDataset.from_paths(
        train_paths,
        autoencoder,
        device,
        maximum_transitions=maximum_transitions,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    print("encoding frozen validation transitions...")
    validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths,
        autoencoder,
        device,
        maximum_transitions=evaluation_examples * 4,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    statistics = training.normalization_statistics()
    model = SpatialLatentDiffusion(
        latent_channels=training.latent_shape[0],
        action_dim=9,
        hidden_channels=hidden_channels,
        blocks=blocks,
        diffusion_steps=diffusion_steps,
        action_mean=statistics[0],
        action_std=statistics[1],
        latent_mean=statistics[2],
        latent_std=statistics[3],
        motion_mean=statistics[4],
        motion_std=statistics[5],
    ).to(device)
    print(f"diffusion parameters: {model.parameter_count:,}")
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
    history: dict[str, list[float]] = {
        "train": [],
        "validation": [],
        "validation_edge_ratio": [],
        "validation_pixel_mse": [],
    }
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _denoising_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch["action"])
            total += float(loss.detach()) * count
            examples += count
        train_loss = total / examples
        metrics = evaluate_diffusion(
            model,
            autoencoder,
            validation,
            device,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            maximum_examples=evaluation_examples,
            seed=seed,
        )
        history["train"].append(train_loss)
        history["validation"].append(metrics.denoising_mse)
        history["validation_edge_ratio"].append(metrics.sample_edge_ratio)
        history["validation_pixel_mse"].append(metrics.sample_pixel_mse)
        print(
            f"epoch {epoch + 1:3d}/{epochs}: train={train_loss:.6f}  "
            f"validation={metrics.denoising_mse:.6f}  "
            f"edge={metrics.sample_edge_ratio:.3f}  "
            f"pixel={metrics.sample_pixel_mse:.6f}"
        )
        # The denoising objective is the honest selection signal; sample pixel
        # error would reward the averaging this model exists to avoid.
        if metrics.denoising_mse < best_validation:
            best_validation = metrics.denoising_mse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {epoch + 1} epochs")
                break
    if best_state is None:
        raise RuntimeError("spatial diffusion training produced no checkpoint")
    model.load_state_dict(best_state)
    if _file_sha256(autoencoder_checkpoint) != autoencoder_sha256:
        raise ValueError("the spatial autoencoder checkpoint changed during training")

    checkpoint = output_dir / "best.pt"
    _save_checkpoint(
        checkpoint,
        model,
        history=history,
        autoencoder_checkpoint=autoencoder_checkpoint,
        autoencoder_sha256=autoencoder_sha256,
        manifest_path=manifest_path,
        sampling_steps=sampling_steps,
    )
    final = evaluate_diffusion(
        model,
        autoencoder,
        validation,
        device,
        batch_size=batch_size,
        sampling_steps=sampling_steps,
        maximum_examples=evaluation_examples,
        seed=seed,
    )
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    _save_curve(history, curve)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "conditional latent diffusion dynamics",
                "training_transitions": len(training),
                "validation_transitions": len(validation),
                "latent_shape": training.latent_shape,
                "parameters": model.parameter_count,
                "diffusion_steps": diffusion_steps,
                "sampling_steps": sampling_steps,
                "device": str(device),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "autoencoder_sha256": autoencoder_sha256,
                "dataset_manifest": str(manifest_path),
                "validation": final.__dict__,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SpatialDiffusionTrainingResult(
        checkpoint=checkpoint,
        training_curve=curve,
        metrics_path=metrics_path,
        validation_metrics=final,
        parameter_count=model.parameter_count,
        training_transitions=len(training),
        latent_shape=training.latent_shape,
        device=str(device),
    )


def evaluate_saved_spatial_diffusion(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    diffusion_checkpoint: Path,
    *,
    batch_size: int = 64,
    encode_batch_size: int = 128,
    sampling_steps: int = 20,
    maximum_examples: int = 2_000,
    split: DatasetSplit = "validation",
    seed: int = 7,
    requested_device: str = "auto",
) -> DiffusionSampleMetrics:
    if split not in {"validation", "test"}:
        raise ValueError("diffusion evaluation split must be validation or test")
    device = choose_device(requested_device)
    model, metadata = load_spatial_diffusion_checkpoint(diffusion_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial diffusion belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    paths = DatasetManifest.load(manifest_path).processed_paths(processed_dir, split)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        paths,
        autoencoder,
        device,
        maximum_transitions=maximum_examples * 4,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    return evaluate_diffusion(
        model,
        autoencoder,
        dataset,
        device,
        batch_size=batch_size,
        sampling_steps=sampling_steps,
        maximum_examples=maximum_examples,
        seed=seed,
    )
