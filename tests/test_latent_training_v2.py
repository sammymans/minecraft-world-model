from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mcwm.latent_data_v2 import CachedTemporalLatentDataset, LatentEpisodeCache
from mcwm.latent_training_v2 import _action_aware_epoch_indices, _variant_actions


class CountingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        pooled = nn.functional.adaptive_avg_pool2d(frames, (2, 2))
        return torch.cat((pooled, pooled[:, :1]), dim=1)


def _write_episode(path: Path, *, frames: int = 18) -> np.ndarray:
    images = np.arange(frames, dtype=np.uint8)[:, None, None, None]
    images = np.broadcast_to(images, (frames, 4, 4, 3)).copy()
    actions = np.zeros((frames - 1, 9), dtype=np.float32)
    for index in range(len(actions)):
        if index % 4 == 0:
            actions[index, 0] = 1
        elif index % 4 == 1:
            actions[index, -2] = -10
        elif index % 4 == 2:
            actions[index, -2] = 10
    metadata = json.dumps({"episode": path.stem, "model_fps": 10.0})
    np.savez_compressed(
        path,
        metadata=metadata,
        frames=images,
        actions=actions,
        rejection_reasons=np.zeros(frames - 1, dtype=np.int8),
        source_frame_indices=np.arange(frames, dtype=np.int32) * 2,
    )
    return actions


def test_disk_cache_reopens_and_temporal_windows_remain_aligned(tmp_path: Path) -> None:
    processed = tmp_path / "episode.npz"
    actions = _write_episode(processed)
    encoder = CountingEncoder()
    cache = LatentEpisodeCache.build(
        [processed],
        encoder,
        torch.device("cpu"),
        tmp_path / "cache",
        autoencoder_sha256="a" * 64,
        manifest_sha256="b" * 64,
        split="training",
        encode_batch_size=5,
    )
    dataset = CachedTemporalLatentDataset(cache, context_frames=8)
    sample = dataset[0]

    assert encoder.calls == 4
    assert len(dataset) == 10
    assert sample["context_latents"].shape == (8, 4, 2, 2)
    assert sample["target_latent"].shape == (4, 2, 2)
    assert torch.equal(sample["actions"], torch.from_numpy(actions[:8]))
    assert torch.allclose(
        sample["context_latents"][:, 0, 0, 0], torch.arange(8) / 255, atol=1e-4
    )
    assert torch.allclose(
        sample["target_latent"][0], torch.full((2, 2), 8 / 255), atol=1e-4
    )

    reopened_encoder = CountingEncoder()
    reopened = LatentEpisodeCache.build(
        [processed],
        reopened_encoder,
        torch.device("cpu"),
        tmp_path / "cache",
        autoencoder_sha256="a" * 64,
        manifest_sha256="b" * 64,
        split="training",
    )
    assert reopened_encoder.calls == 0
    assert reopened.metadata == cache.metadata


def test_action_aware_epoch_keeps_every_window_and_adds_switches(tmp_path: Path) -> None:
    processed = tmp_path / "episode.npz"
    _write_episode(processed, frames=40)
    cache = LatentEpisodeCache.build(
        [processed],
        CountingEncoder(),
        torch.device("cpu"),
        tmp_path / "cache",
        autoencoder_sha256="c" * 64,
        manifest_sha256="d" * 64,
        split="training",
    )
    dataset = CachedTemporalLatentDataset(cache)
    indices = _action_aware_epoch_indices(dataset, 0.8, torch.Generator().manual_seed(5))

    assert set(indices.tolist()) == set(range(len(dataset)))
    sampled_change_fraction = float(dataset.action_changes[indices.numpy()].mean())
    assert sampled_change_fraction >= 0.79


def test_action_variants_only_replace_the_target_driving_action() -> None:
    actions = torch.arange(4 * 8 * 9, dtype=torch.float32).reshape(4, 8, 9)

    previous = _variant_actions(actions, "previous")
    shuffled = _variant_actions(actions, "shuffled")
    zero = _variant_actions(actions, "zero")

    assert torch.equal(previous[:, :-1], actions[:, :-1])
    assert torch.equal(previous[:, -1], actions[:, -2])
    assert torch.equal(shuffled[:, :-1], actions[:, :-1])
    assert torch.equal(shuffled[:, -1], actions[:, -1].roll(1, dims=0))
    assert torch.equal(zero[:, :-1], actions[:, :-1])
    assert torch.count_nonzero(zero[:, -1]) == 0
