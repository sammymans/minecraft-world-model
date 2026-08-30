"""Training and evaluation for the spatial Minecraft autoencoder."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import SpatialAutoencoder
from mcwm.training import (
    FrameDataset,
    _dataset_metadata,
    choose_device,
    load_frame_splits,
    save_reconstruction_grid,
    seed_everything,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SpatialReconstructionMetrics:
    objective: float
    pixel_l1: float
    pixel_mse: float
    psnr_db: float
    gradient_l1: float
    gradient_energy_ratio: float


@dataclass(frozen=True)
class SpatialTrainingResult:
    checkpoint: Path
    reconstruction_grid: Path
    training_curve: Path
    metrics_path: Path
    train_metrics: SpatialReconstructionMetrics
    validation_metrics: SpatialReconstructionMetrics | None
    parameter_count: int
    latent_shape: tuple[int, int, int]
    device: str


@dataclass(frozen=True)
class SpatialEvaluationResult:
    metrics: SpatialReconstructionMetrics
    reconstruction_grid: Path
    frame_count: int
    latent_shape: tuple[int, int, int]
    device: str


def image_gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite RGB differences preserve Minecraft's horizontal and vertical edges."""
    horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
    vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
    return horizontal, vertical


def spatial_reconstruction_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_weight < 0:
        raise ValueError("edge_weight must be non-negative")
    pixel_l1 = torch.nn.functional.l1_loss(reconstructed, target)
    pixel_mse = torch.nn.functional.mse_loss(reconstructed, target)
    reconstructed_dx, reconstructed_dy = image_gradients(reconstructed)
    target_dx, target_dy = image_gradients(target)
    gradient_l1 = 0.5 * (
        torch.nn.functional.l1_loss(reconstructed_dx, target_dx)
        + torch.nn.functional.l1_loss(reconstructed_dy, target_dy)
    )
    return pixel_mse + 0.1 * pixel_l1 + edge_weight * gradient_l1, pixel_l1, gradient_l1


@torch.no_grad()
def evaluate_spatial_autoencoder(
    model: SpatialAutoencoder,
    dataset: Dataset[torch.Tensor],
    device: torch.device,
    *,
    batch_size: int = 64,
    edge_weight: float = 0.25,
) -> SpatialReconstructionMetrics:
    if not len(dataset):
        raise ValueError("cannot evaluate an empty frame dataset")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pixel_absolute = 0.0
    pixel_squared = 0.0
    pixel_count = 0
    gradient_absolute = 0.0
    gradient_count = 0
    reconstructed_gradient_energy = 0.0
    target_gradient_energy = 0.0
    for frames in loader:
        frames = frames.to(device)
        reconstructed = model(frames).clamp(0, 1)
        difference = reconstructed - frames
        pixel_absolute += difference.abs().sum().item()
        pixel_squared += difference.square().sum().item()
        pixel_count += frames.numel()
        reconstructed_dx, reconstructed_dy = image_gradients(reconstructed)
        target_dx, target_dy = image_gradients(frames)
        for reconstructed_gradient, target_gradient in (
            (reconstructed_dx, target_dx),
            (reconstructed_dy, target_dy),
        ):
            gradient_absolute += (reconstructed_gradient - target_gradient).abs().sum().item()
            gradient_count += target_gradient.numel()
            reconstructed_gradient_energy += reconstructed_gradient.abs().sum().item()
            target_gradient_energy += target_gradient.abs().sum().item()
    pixel_l1 = pixel_absolute / pixel_count
    pixel_mse = pixel_squared / pixel_count
    gradient_l1 = gradient_absolute / gradient_count
    return SpatialReconstructionMetrics(
        objective=pixel_mse + 0.1 * pixel_l1 + edge_weight * gradient_l1,
        pixel_l1=pixel_l1,
        pixel_mse=pixel_mse,
        psnr_db=10.0 * math.log10(1.0 / max(pixel_mse, 1e-12)),
        gradient_l1=gradient_l1,
        gradient_energy_ratio=(reconstructed_gradient_energy / max(target_gradient_energy, 1e-12)),
    )


def _save_checkpoint(
    path: Path,
    model: SpatialAutoencoder,
    *,
    history: dict[str, list[float]],
    edge_weight: float,
    train_episodes: list[str],
    validation_episodes: list[str],
    manifest_path: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": "spatial_autoencoder",
            "model_state": model.state_dict(),
            "latent_channels": model.latent_channels,
            "latent_shape": model.latent_shape,
            "base_channels": model.base_channels,
            "image_size": 64,
            "edge_weight": edge_weight,
            "history": history,
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "training_frame_policy": "non_gui",
            "validation_frame_policy": "valid_sequences",
            **_dataset_metadata(manifest_path),
        },
        path,
    )


def load_spatial_autoencoder_checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialAutoencoder, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "spatial_autoencoder":
        raise ValueError("checkpoint is not a spatial autoencoder")
    model = SpatialAutoencoder(
        latent_channels=int(checkpoint["latent_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _save_curve(history: dict[str, list[float]], path: Path, x_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_objective"], label="training objective")
    if history.get("validation_objective"):
        axis.plot(history["validation_objective"], label="validation objective")
    axis.set_xlabel(x_label)
    axis.set_ylabel("MSE + 0.1 L1 + edge objective")
    axis.set_title("Spatial autoencoder reconstruction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _manifest_paths(processed_dir: Path, manifest_path: Path) -> tuple[list[Path], list[Path]]:
    return DatasetManifest.load(manifest_path).processed_splits(processed_dir)


def _limited_dataset(
    dataset: Dataset[torch.Tensor], maximum: int | None, seed: int
) -> Dataset[torch.Tensor]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("max_training_frames must be positive when supplied")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def sanity_overfit_spatial_autoencoder(
    processed_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    frame_count: int = 32,
    steps: int = 500,
    latent_channels: int = 16,
    base_channels: int = 32,
    edge_weight: float = 0.25,
    learning_rate: float = 1e-3,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialTrainingResult:
    if frame_count < 1 or steps < 1:
        raise ValueError("frame_count and steps must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    train_paths, validation_paths = _manifest_paths(processed_dir, manifest_path)
    episode_frames = FrameDataset.from_paths(train_paths[:1], policy="non_gui")
    frame_count = min(frame_count, len(episode_frames))
    indices = np.linspace(0, len(episode_frames) - 1, frame_count, dtype=int).tolist()
    subset = Subset(episode_frames, indices)
    batch = torch.stack([subset[index] for index in range(len(subset))]).to(device)
    model = SpatialAutoencoder(latent_channels, base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {
        "train_objective": [],
        "validation_objective": [],
    }
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(batch)
        loss, pixel_l1, gradient_l1 = spatial_reconstruction_loss(
            reconstructed, batch, edge_weight=edge_weight
        )
        loss.backward()
        optimizer.step()
        history["train_objective"].append(float(loss.detach()))
        if step == 0 or (step + 1) % 100 == 0:
            print(
                f"step {step + 1:4d}/{steps}: objective={loss.item():.6f}  "
                f"pixel L1={pixel_l1.item():.6f}  edge L1={gradient_l1.item():.6f}"
            )
    metrics = evaluate_spatial_autoencoder(
        model, subset, device, batch_size=frame_count, edge_weight=edge_weight
    )
    checkpoint = output_dir / "checkpoint.pt"
    grid = output_dir / "reconstructions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    _save_checkpoint(
        checkpoint,
        model,
        history=history,
        edge_weight=edge_weight,
        train_episodes=[train_paths[0].stem],
        validation_episodes=[path.stem for path in validation_paths],
        manifest_path=manifest_path,
    )
    save_reconstruction_grid(
        model, subset, grid, device, count=8, original_label="memorized original"
    )
    _save_curve(history, curve, "optimization step")
    _write_metrics(
        metrics_path,
        {
            "mode": "spatial autoencoder sanity overfit",
            "frames": frame_count,
            "steps": steps,
            "latent_shape": model.latent_shape,
            "latent_values": model.latent_value_count,
            "base_channels": base_channels,
            "parameters": model.parameter_count,
            "edge_weight": edge_weight,
            "device": str(device),
            **_dataset_metadata(manifest_path),
            "train": asdict(metrics),
        },
    )
    return SpatialTrainingResult(
        checkpoint=checkpoint,
        reconstruction_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        train_metrics=metrics,
        validation_metrics=None,
        parameter_count=model.parameter_count,
        latent_shape=model.latent_shape,
        device=str(device),
    )


def train_spatial_autoencoder(
    processed_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    latent_channels: int = 16,
    base_channels: int = 32,
    edge_weight: float = 0.25,
    learning_rate: float = 1e-3,
    patience: int = 6,
    max_training_frames: int | None = None,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialTrainingResult:
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    training, validation, train_paths, validation_paths = load_frame_splits(
        processed_dir, manifest_path=manifest_path
    )
    selected_training = _limited_dataset(training, max_training_frames, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        selected_training,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = SpatialAutoencoder(latent_channels, base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history: dict[str, list[float]] = {
        "train_objective": [],
        "validation_objective": [],
        "validation_pixel_l1": [],
        "validation_gradient_ratio": [],
    }
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        objective_sum = 0.0
        frames_seen = 0
        for frames in loader:
            frames = frames.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = spatial_reconstruction_loss(model(frames), frames, edge_weight=edge_weight)
            loss.backward()
            optimizer.step()
            objective_sum += float(loss.detach()) * len(frames)
            frames_seen += len(frames)
        train_objective = objective_sum / frames_seen
        validation_metrics = evaluate_spatial_autoencoder(
            model,
            validation,
            device,
            batch_size=batch_size,
            edge_weight=edge_weight,
        )
        history["train_objective"].append(train_objective)
        history["validation_objective"].append(validation_metrics.objective)
        history["validation_pixel_l1"].append(validation_metrics.pixel_l1)
        history["validation_gradient_ratio"].append(validation_metrics.gradient_energy_ratio)
        print(
            f"epoch {epoch + 1:3d}/{epochs}: train={train_objective:.6f}  "
            f"validation={validation_metrics.objective:.6f}  "
            f"L1={validation_metrics.pixel_l1:.6f}  "
            f"PSNR={validation_metrics.psnr_db:.2f} dB  "
            f"edge ratio={validation_metrics.gradient_energy_ratio:.3f}"
        )
        if validation_metrics.objective < best_validation:
            best_validation = validation_metrics.objective
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {epoch + 1} epochs")
                break
    if best_state is None:
        raise RuntimeError("spatial training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output_dir / "best.pt"
    _save_checkpoint(
        checkpoint,
        model,
        history=history,
        edge_weight=edge_weight,
        train_episodes=[path.stem for path in train_paths],
        validation_episodes=[path.stem for path in validation_paths],
        manifest_path=manifest_path,
    )
    train_metrics = evaluate_spatial_autoencoder(
        model,
        selected_training,
        device,
        batch_size=batch_size,
        edge_weight=edge_weight,
    )
    validation_metrics = evaluate_spatial_autoencoder(
        model, validation, device, batch_size=batch_size, edge_weight=edge_weight
    )
    grid = output_dir / "held-out-reconstructions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    save_reconstruction_grid(model, validation, grid, device, count=8)
    _save_curve(history, curve, "epoch")
    _write_metrics(
        metrics_path,
        {
            "mode": "spatial autoencoder",
            "epochs_completed": len(history["train_objective"]),
            "training_frames": len(selected_training),
            "available_training_frames": len(training),
            "validation_frames": len(validation),
            "latent_shape": model.latent_shape,
            "latent_values": model.latent_value_count,
            "base_channels": base_channels,
            "parameters": model.parameter_count,
            "edge_weight": edge_weight,
            "device": str(device),
            "training_episodes": [path.stem for path in train_paths],
            "validation_episodes": [path.stem for path in validation_paths],
            **_dataset_metadata(manifest_path),
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
            "history": history,
        },
    )
    return SpatialTrainingResult(
        checkpoint=checkpoint,
        reconstruction_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        parameter_count=model.parameter_count,
        latent_shape=model.latent_shape,
        device=str(device),
    )


def evaluate_saved_spatial_autoencoder(
    processed_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    split: DatasetSplit = "validation",
    batch_size: int = 64,
    count: int = 8,
    requested_device: str = "auto",
) -> SpatialEvaluationResult:
    if split not in {"training", "validation", "test"}:
        raise ValueError("split must be training, validation, or test")
    device = choose_device(requested_device)
    manifest = DatasetManifest.load(manifest_path)
    paths = manifest.processed_paths(processed_dir, split)
    policy = "non_gui" if split == "training" else "valid_sequences"
    dataset = FrameDataset.from_paths(paths, policy=policy)
    model, checkpoint = load_spatial_autoencoder_checkpoint(checkpoint_path, device)
    edge_weight = float(checkpoint["edge_weight"])
    metrics = evaluate_spatial_autoencoder(
        model, dataset, device, batch_size=batch_size, edge_weight=edge_weight
    )
    grid = output_dir / f"{split}-reconstructions.png"
    save_reconstruction_grid(
        model,
        dataset,
        grid,
        device,
        count=count,
        original_label=f"{split} original",
    )
    return SpatialEvaluationResult(
        metrics=metrics,
        reconstruction_grid=grid,
        frame_count=len(dataset),
        latent_shape=model.latent_shape,
        device=str(device),
    )
