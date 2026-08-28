"""Shared metrics and visualizations for spatial latent dynamics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mcwm.dataset import split_episode_paths
from mcwm.manifest import DatasetManifest


@dataclass(frozen=True)
class DynamicsMetrics:
    examples: int
    latent_mse: float
    pixel_l1: float
    pixel_mse: float
    pixel_psnr_db: float
    copy_latent_mse: float
    copy_pixel_l1: float
    copy_pixel_mse: float
    copy_pixel_psnr_db: float
    decoded_copy_pixel_l1: float
    decoded_copy_pixel_mse: float
    decoded_copy_pixel_psnr_db: float
    oracle_pixel_l1: float
    oracle_pixel_mse: float
    oracle_pixel_psnr_db: float
    shuffled_action_latent_mse: float
    shuffled_action_pixel_mse: float
    action_effect_latent_mse: float

    @property
    def shuffled_action_degradation(self) -> float:
        return self.shuffled_action_latent_mse - self.latent_mse


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_splits(
    processed_dir: Path,
    manifest_path: Path | None,
) -> tuple[list[Path], list[Path]]:
    if manifest_path is not None:
        return DatasetManifest.load(manifest_path).processed_splits(processed_dir)
    paths = sorted(processed_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No processed episodes in {processed_dir}")
    return split_episode_paths(paths)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def evaluate_dynamics(
    dynamics: nn.Module,
    autoencoder: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    batch_size: int = 64,
    seed: int = 7,
) -> DynamicsMetrics:
    """Measure one-step prediction against copy and mismatched-action controls."""
    if not len(dataset):
        raise ValueError("cannot evaluate an empty dynamics dataset")
    dynamics.eval()
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_actions = torch.stack([dataset[index]["action"] for index in range(len(dataset))])
    generator = torch.Generator().manual_seed(seed)
    shuffled_actions = all_actions[torch.randperm(len(dataset), generator=generator)]

    sums = {
        name: 0.0
        for name in (
            "latent",
            "pixel_abs",
            "pixel_sq",
            "copy_latent",
            "copy_pixel_abs",
            "copy_pixel_sq",
            "decoded_copy_abs",
            "decoded_copy_sq",
            "oracle_abs",
            "oracle_sq",
            "shuffled_latent",
            "shuffled_pixel",
            "action_effect",
        )
    }
    latent_values = pixel_values = examples = offset = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        count = len(batch["action"])
        shuffled = shuffled_actions[offset : offset + count].to(device)
        offset += count
        predicted_latent = dynamics(
            batch["previous_latent"], batch["current_latent"], batch["action"]
        )
        predicted_frame = autoencoder.decode(predicted_latent).clamp(0, 1)
        decoded_copy = autoencoder.decode(batch["current_latent"]).clamp(0, 1)
        oracle = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
        shuffled_latent = dynamics(batch["previous_latent"], batch["current_latent"], shuffled)
        shuffled_frame = autoencoder.decode(shuffled_latent).clamp(0, 1)

        sums["latent"] += (predicted_latent - batch["target_latent"]).square().sum().item()
        sums["pixel_abs"] += (predicted_frame - batch["target_frame"]).abs().sum().item()
        sums["pixel_sq"] += (predicted_frame - batch["target_frame"]).square().sum().item()
        sums["copy_latent"] += (
            (batch["current_latent"] - batch["target_latent"]).square().sum().item()
        )
        sums["copy_pixel_abs"] += (
            (batch["current_frame"] - batch["target_frame"]).abs().sum().item()
        )
        sums["copy_pixel_sq"] += (
            (batch["current_frame"] - batch["target_frame"]).square().sum().item()
        )
        sums["decoded_copy_abs"] += (decoded_copy - batch["target_frame"]).abs().sum().item()
        sums["decoded_copy_sq"] += (decoded_copy - batch["target_frame"]).square().sum().item()
        sums["oracle_abs"] += (oracle - batch["target_frame"]).abs().sum().item()
        sums["oracle_sq"] += (oracle - batch["target_frame"]).square().sum().item()
        sums["shuffled_latent"] += (shuffled_latent - batch["target_latent"]).square().sum().item()
        sums["shuffled_pixel"] += (shuffled_frame - batch["target_frame"]).square().sum().item()
        sums["action_effect"] += (predicted_latent - shuffled_latent).square().sum().item()
        latent_values += predicted_latent.numel()
        pixel_values += predicted_frame.numel()
        examples += count

    pixel_mse = sums["pixel_sq"] / pixel_values
    copy_pixel_mse = sums["copy_pixel_sq"] / pixel_values
    decoded_copy_mse = sums["decoded_copy_sq"] / pixel_values
    oracle_mse = sums["oracle_sq"] / pixel_values
    return DynamicsMetrics(
        examples=examples,
        latent_mse=sums["latent"] / latent_values,
        pixel_l1=sums["pixel_abs"] / pixel_values,
        pixel_mse=pixel_mse,
        pixel_psnr_db=_psnr(pixel_mse),
        copy_latent_mse=sums["copy_latent"] / latent_values,
        copy_pixel_l1=sums["copy_pixel_abs"] / pixel_values,
        copy_pixel_mse=copy_pixel_mse,
        copy_pixel_psnr_db=_psnr(copy_pixel_mse),
        decoded_copy_pixel_l1=sums["decoded_copy_abs"] / pixel_values,
        decoded_copy_pixel_mse=decoded_copy_mse,
        decoded_copy_pixel_psnr_db=_psnr(decoded_copy_mse),
        oracle_pixel_l1=sums["oracle_abs"] / pixel_values,
        oracle_pixel_mse=oracle_mse,
        oracle_pixel_psnr_db=_psnr(oracle_mse),
        shuffled_action_latent_mse=sums["shuffled_latent"] / latent_values,
        shuffled_action_pixel_mse=sums["shuffled_pixel"] / pixel_values,
        action_effect_latent_mse=sums["action_effect"] / latent_values,
    )


@torch.no_grad()
def save_prediction_grid(
    dynamics: nn.Module,
    autoencoder: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    path: Path,
    device: torch.device,
    *,
    count: int = 6,
) -> None:
    if not len(dataset):
        raise ValueError("cannot visualize an empty dynamics dataset")
    dynamics.eval()
    autoencoder.eval()
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    samples = [dataset[int(index)] for index in indices]
    previous = torch.stack([sample["previous_latent"] for sample in samples]).to(device)
    current = torch.stack([sample["current_latent"] for sample in samples]).to(device)
    actions = torch.stack([sample["action"] for sample in samples]).to(device)
    targets = torch.stack([sample["target_latent"] for sample in samples]).to(device)
    predicted = autoencoder.decode(dynamics(previous, current, actions)).clamp(0, 1)
    decoded_copy = autoencoder.decode(current).clamp(0, 1)
    oracle = autoencoder.decode(targets).clamp(0, 1)

    tile_size, header = 144, 28
    labels = (
        "previous",
        "current",
        "real next",
        "decoded copy",
        "predicted next",
        "decoder oracle",
        "error x4",
    )
    canvas = np.full((count * (tile_size + header), len(labels) * tile_size, 3), 24, dtype=np.uint8)
    rows = zip(samples, decoded_copy.cpu(), predicted.cpu(), oracle.cpu(), strict=True)
    for row, (sample, copied, prediction, reconstructed_target) in enumerate(rows):
        episode_index, current_index = dataset.index[int(indices[row])]  # type: ignore[attr-defined]
        previous_frame = dataset.episodes[episode_index].frames[current_index - 1]  # type: ignore[attr-defined]
        current_frame = sample["current_frame"].permute(1, 2, 0).numpy()
        target_frame = sample["target_frame"].permute(1, 2, 0).numpy()
        copied_np = copied.permute(1, 2, 0).numpy()
        prediction_np = prediction.permute(1, 2, 0).numpy()
        oracle_np = reconstructed_target.permute(1, 2, 0).numpy()
        images = (
            cv2.cvtColor(previous_frame, cv2.COLOR_RGB2BGR),
            cv2.cvtColor((current_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((target_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((copied_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((prediction_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((oracle_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.applyColorMap(
                np.clip(np.abs(prediction_np - target_frame).mean(axis=2) * 4 * 255, 0, 255).astype(
                    np.uint8
                ),
                cv2.COLORMAP_INFERNO,
            ),
        )
        y = row * (tile_size + header)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = column * tile_size
            cv2.putText(
                canvas,
                label,
                (x + 7, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            canvas[y + header : y + header + tile_size, x : x + tile_size] = cv2.resize(
                image, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write prediction grid: {path}")


def _metrics_payload(metrics: DynamicsMetrics) -> dict[str, float | int | bool]:
    payload: dict[str, float | int | bool] = asdict(metrics)
    payload.update(
        {
            "beats_copy_latent": metrics.latent_mse < metrics.copy_latent_mse,
            "beats_raw_copy_pixel": metrics.pixel_mse < metrics.copy_pixel_mse,
            "beats_decoded_copy_pixel": metrics.pixel_mse < metrics.decoded_copy_pixel_mse,
            "pixel_mse_above_decoder_oracle": metrics.pixel_mse - metrics.oracle_pixel_mse,
            "shuffled_action_degradation": metrics.shuffled_action_degradation,
            "shuffling_actions_is_worse": metrics.shuffled_action_degradation > 0,
        }
    )
    return payload
