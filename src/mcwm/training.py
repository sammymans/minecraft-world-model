"""Training and evaluation for the visual autoencoder milestone."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mcwm.cleaning import RejectionReason
from mcwm.dataset import ProcessedEpisode, SequenceDataset, split_episode_paths
from mcwm.manifest import DatasetManifest
from mcwm.model import TinyAutoencoder

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _dataset_metadata(manifest_path: Path | None) -> dict[str, str | None]:
    if manifest_path is None:
        return {"dataset_manifest": None, "dataset_manifest_sha256": None}
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": digest,
    }


@dataclass(frozen=True)
class FrameReference:
    episode_index: int
    frame_index: int


class FrameDataset(Dataset[torch.Tensor]):
    """Minecraft frames selected for representation learning or evaluation."""

    def __init__(
        self,
        episodes: list[ProcessedEpisode],
        horizon: int = 8,
        policy: str = "valid_sequences",
    ):
        self.episodes = episodes
        references: set[FrameReference] = set()
        if policy == "valid_sequences":
            sequences = SequenceDataset(episodes, horizon=horizon)
            for episode_index, start in sequences.index:
                for frame_index in range(start - 1, start + horizon + 1):
                    references.add(FrameReference(episode_index, frame_index))
        elif policy == "non_gui":
            for episode_index, episode in enumerate(episodes):
                eligible = np.ones(len(episode.frames), dtype=bool)
                gui_transitions = np.flatnonzero(
                    episode.rejection_reasons == RejectionReason.GUI_OPEN
                )
                eligible[gui_transitions] = False
                eligible[gui_transitions + 1] = False
                for frame_index in np.flatnonzero(eligible):
                    references.add(FrameReference(episode_index, int(frame_index)))
        else:
            raise ValueError("frame policy must be 'valid_sequences' or 'non_gui'")
        self.references = sorted(references, key=lambda ref: (ref.episode_index, ref.frame_index))

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        horizon: int = 8,
        policy: str = "valid_sequences",
    ) -> FrameDataset:
        return cls(
            [ProcessedEpisode.load(path) for path in paths],
            horizon=horizon,
            policy=policy,
        )

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> torch.Tensor:
        reference = self.references[index]
        frame = self.episodes[reference.episode_index].frames[reference.frame_index]
        contiguous = np.ascontiguousarray(frame.transpose(2, 0, 1))
        return torch.from_numpy(contiguous).to(torch.float32).div_(255.0)


@dataclass(frozen=True)
class ReconstructionMetrics:
    l1: float
    mse: float
    psnr_db: float


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    reconstruction_grid: Path
    training_curve: Path
    metrics_path: Path
    train_metrics: ReconstructionMetrics
    validation_metrics: ReconstructionMetrics | None
    parameter_count: int
    device: str


@dataclass(frozen=True)
class EvaluationResult:
    metrics: ReconstructionMetrics
    reconstruction_grid: Path
    training_curve: Path
    frame_count: int
    device: str


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_frame_splits(
    processed_dir: Path,
    horizon: int = 8,
    manifest_path: Path | None = None,
) -> tuple[FrameDataset, FrameDataset, list[Path], list[Path]]:
    if manifest_path is not None:
        manifest = DatasetManifest.load(manifest_path)
        train_paths, validation_paths = manifest.processed_splits(processed_dir)
    else:
        paths = sorted(processed_dir.glob("*.npz"))
        if not paths:
            raise ValueError(f"No processed episodes in {processed_dir}")
        train_paths, validation_paths = split_episode_paths(paths)
    return (
        FrameDataset.from_paths(train_paths, horizon=horizon, policy="non_gui"),
        FrameDataset.from_paths(validation_paths, horizon=horizon),
        train_paths,
        validation_paths,
    )


@torch.no_grad()
def evaluate_autoencoder(
    model: TinyAutoencoder,
    dataset: Dataset[torch.Tensor],
    device: torch.device,
    batch_size: int = 64,
) -> ReconstructionMetrics:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    absolute_error = 0.0
    squared_error = 0.0
    value_count = 0
    for frames in loader:
        frames = frames.to(device)
        reconstructed = model(frames).clamp(0, 1)
        absolute_error += torch.abs(reconstructed - frames).sum().item()
        squared_error += torch.square(reconstructed - frames).sum().item()
        value_count += frames.numel()
    l1 = absolute_error / value_count
    mse = squared_error / value_count
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return ReconstructionMetrics(l1=l1, mse=mse, psnr_db=psnr)


def _save_checkpoint(
    path: Path,
    model: TinyAutoencoder,
    *,
    latent_dim: int,
    base_channels: int,
    history: dict[str, list[float]],
    train_episodes: list[str],
    validation_episodes: list[str],
    manifest_path: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "image_size": 64,
            "history": history,
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "training_frame_policy": "non_gui",
            "validation_frame_policy": "valid_sequences",
            **_dataset_metadata(manifest_path),
        },
        path,
    )


def load_autoencoder_checkpoint(path: Path, device: torch.device) -> tuple[TinyAutoencoder, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyAutoencoder(
        latent_dim=int(checkpoint["latent_dim"]),
        base_channels=int(checkpoint.get("base_channels", 16)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _save_curve(history: dict[str, list[float]], path: Path, x_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="training")
    if history.get("validation_loss"):
        axis.plot(history["validation_loss"], label="validation")
    axis.set_xlabel(x_label)
    axis.set_ylabel("mean squared pixel error")
    axis.set_title("Tiny autoencoder reconstruction loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_reconstruction_grid(
    model: TinyAutoencoder,
    dataset: Dataset[torch.Tensor],
    path: Path,
    device: torch.device,
    count: int = 8,
    original_label: str = "held-out original",
) -> None:
    if not len(dataset):
        raise ValueError("Cannot visualize an empty frame dataset")
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    originals = torch.stack([dataset[int(index)] for index in indices]).to(device)
    model.eval()
    reconstructions = model(originals).clamp(0, 1)
    originals_np = originals.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions_np = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

    tile_size = 192
    header = 28
    canvas = np.full((count * (tile_size + header), 3 * tile_size, 3), 24, dtype=np.uint8)
    labels = (original_label, "reconstruction", "absolute error x4")
    for row, (original, reconstruction) in enumerate(
        zip(originals_np, reconstructions_np, strict=True)
    ):
        error = np.abs(original - reconstruction).mean(axis=2)
        error_image = cv2.applyColorMap(
            np.clip(error * 4 * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )
        images = (
            cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((reconstruction * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            error_image,
        )
        y = row * (tile_size + header)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = column * tile_size
            cv2.putText(
                canvas,
                label,
                (x + 8, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            resized = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            canvas[y + header : y + header + tile_size, x : x + tile_size] = resized

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"Could not write reconstruction grid: {path}")


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sanity_overfit(
    processed_dir: Path,
    output_dir: Path,
    *,
    frame_count: int = 32,
    steps: int = 600,
    latent_dim: int = 256,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    horizon: int = 8,
    manifest_path: Path | None = None,
    seed: int = 7,
    requested_device: str = "auto",
) -> TrainingResult:
    """Intentionally memorize a tiny fixed set to validate the implementation."""
    seed_everything(seed)
    device = choose_device(requested_device)
    train_frames, _, train_paths, validation_paths = load_frame_splits(
        processed_dir, horizon=horizon, manifest_path=manifest_path
    )
    frame_count = min(frame_count, len(train_frames))
    indices = np.linspace(0, len(train_frames) - 1, frame_count, dtype=int).tolist()
    subset = Subset(train_frames, indices)
    batch = torch.stack([subset[index] for index in range(len(subset))]).to(device)

    model = TinyAutoencoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    history = {"train_loss": [], "validation_loss": []}

    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(batch)
        loss = loss_function(reconstruction, batch)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(float(loss.item()))
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step {step + 1:4d}/{steps}: MSE={loss.item():.6f}")

    train_metrics = evaluate_autoencoder(model, subset, device, batch_size=frame_count)
    checkpoint = output_dir / "checkpoint.pt"
    grid = output_dir / "reconstructions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    _save_checkpoint(
        checkpoint,
        model,
        latent_dim=latent_dim,
        base_channels=base_channels,
        history=history,
        train_episodes=[path.stem for path in train_paths],
        validation_episodes=[path.stem for path in validation_paths],
        manifest_path=manifest_path,
    )
    save_reconstruction_grid(
        model,
        subset,
        grid,
        device,
        count=8,
        original_label="memorized original",
    )
    _save_curve(history, curve, "optimization step")
    _write_metrics(
        metrics_path,
        {
            "mode": "32-frame sanity overfit",
            "frames": frame_count,
            "steps": steps,
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "parameters": model.parameter_count,
            "device": str(device),
            "training_frame_policy": "non_gui",
            **_dataset_metadata(manifest_path),
            "train": asdict(train_metrics),
        },
    )
    return TrainingResult(
        checkpoint=checkpoint,
        reconstruction_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        train_metrics=train_metrics,
        validation_metrics=None,
        parameter_count=model.parameter_count,
        device=str(device),
    )


def train_full_autoencoder(
    processed_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    latent_dim: int = 256,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    horizon: int = 8,
    manifest_path: Path | None = None,
    patience: int = 8,
    seed: int = 7,
    requested_device: str = "auto",
) -> TrainingResult:
    seed_everything(seed)
    device = choose_device(requested_device)
    train_frames, validation_frames, train_paths, validation_paths = load_frame_splits(
        processed_dir, horizon=horizon, manifest_path=manifest_path
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_frames,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = TinyAutoencoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    history = {"train_loss": [], "validation_loss": []}
    checkpoint = output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_frames = 0
        for frames in train_loader:
            frames = frames.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(frames)
            loss = loss_function(reconstruction, frames)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(frames)
            total_frames += len(frames)

        train_loss = total_loss / total_frames
        validation_metrics = evaluate_autoencoder(
            model, validation_frames, device, batch_size=batch_size
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_metrics.mse)
        print(
            f"epoch {epoch + 1:3d}/{epochs}: train MSE={train_loss:.6f}  "
            f"validation MSE={validation_metrics.mse:.6f}  "
            f"validation PSNR={validation_metrics.psnr_db:.2f} dB"
        )

        if validation_metrics.mse < best_validation:
            best_validation = validation_metrics.mse
            stale_epochs = 0
            _save_checkpoint(
                checkpoint,
                model,
                latent_dim=latent_dim,
                base_channels=base_channels,
                history=history,
                train_episodes=[path.stem for path in train_paths],
                validation_episodes=[path.stem for path in validation_paths],
                manifest_path=manifest_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {epoch + 1} epochs")
                break

    model, _ = load_autoencoder_checkpoint(checkpoint, device)
    train_metrics = evaluate_autoencoder(model, train_frames, device, batch_size=batch_size)
    validation_metrics = evaluate_autoencoder(
        model, validation_frames, device, batch_size=batch_size
    )
    grid = output_dir / "held-out-reconstructions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    save_reconstruction_grid(model, validation_frames, grid, device, count=8)
    _save_curve(history, curve, "epoch")
    _write_metrics(
        metrics_path,
        {
            "mode": "full autoencoder",
            "epochs_completed": len(history["train_loss"]),
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "parameters": model.parameter_count,
            "device": str(device),
            "training_frames": len(train_frames),
            "validation_frames": len(validation_frames),
            "training_episodes": [path.stem for path in train_paths],
            "validation_episodes": [path.stem for path in validation_paths],
            "training_frame_policy": "non_gui",
            "validation_frame_policy": "valid_sequences",
            **_dataset_metadata(manifest_path),
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
            "history": history,
        },
    )
    return TrainingResult(
        checkpoint=checkpoint,
        reconstruction_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        parameter_count=model.parameter_count,
        device=str(device),
    )


def evaluate_saved_autoencoder(
    processed_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    split: str = "validation",
    horizon: int = 8,
    manifest_path: Path | None = None,
    batch_size: int = 64,
    count: int = 8,
    requested_device: str = "auto",
) -> EvaluationResult:
    """Recreate metrics and visuals from a saved checkpoint."""
    if split not in {"training", "validation"}:
        raise ValueError("split must be 'training' or 'validation'")
    device = choose_device(requested_device)
    training, validation, _, _ = load_frame_splits(
        processed_dir, horizon=horizon, manifest_path=manifest_path
    )
    dataset = training if split == "training" else validation
    model, checkpoint = load_autoencoder_checkpoint(checkpoint_path, device)
    metrics = evaluate_autoencoder(model, dataset, device, batch_size=batch_size)
    grid = output_dir / f"{split}-reconstructions.png"
    curve = output_dir / "training-curve.png"
    save_reconstruction_grid(
        model,
        dataset,
        grid,
        device,
        count=count,
        original_label=f"{split} original",
    )
    _save_curve(checkpoint["history"], curve, "epoch")
    return EvaluationResult(
        metrics=metrics,
        reconstruction_grid=grid,
        training_curve=curve,
        frame_count=len(dataset),
        device=str(device),
    )
