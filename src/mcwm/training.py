"""Shared data, device, and visualization helpers for model training."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from mcwm.cleaning import RejectionReason
from mcwm.dataset import ProcessedEpisode, SequenceDataset, split_episode_paths
from mcwm.manifest import DatasetManifest


def _dataset_metadata(manifest_path: Path | None) -> dict[str, str | None]:
    if manifest_path is None:
        return {"dataset_manifest": None, "dataset_manifest_sha256": None}
    return {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
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
        train_paths, validation_paths = DatasetManifest.load(manifest_path).processed_splits(
            processed_dir
        )
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
def save_reconstruction_grid(
    model: nn.Module,
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
    labels = (original_label, "reconstruction", "absolute error x4")
    canvas = np.full((count * (tile_size + header), 3 * tile_size, 3), 24, dtype=np.uint8)
    for row, (original, reconstruction) in enumerate(
        zip(originals_np, reconstructions_np, strict=True)
    ):
        error = np.abs(original - reconstruction).mean(axis=2)
        images = (
            cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((reconstruction * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.applyColorMap(
                np.clip(error * 4 * 255, 0, 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO,
            ),
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
            canvas[y + header : y + header + tile_size, x : x + tile_size] = cv2.resize(
                image, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"Could not write reconstruction grid: {path}")
