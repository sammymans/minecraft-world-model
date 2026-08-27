from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics
from mcwm.spatial_dynamics import (
    SpatialEncodedDynamicsDataset,
    _prediction_loss,
    _save_checkpoint,
    load_spatial_dynamics_checkpoint,
)


def _write_episode(path: Path, *, frames: int = 8) -> None:
    rng = np.random.default_rng(4)
    metadata = json.dumps({"episode": path.stem, "model_fps": 10.0})
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=metadata,
            frames=rng.integers(0, 256, (frames, 64, 64, 3), dtype=np.uint8),
            actions=rng.normal(size=(frames - 1, 9)).astype(np.float32),
            rejection_reasons=np.zeros(frames - 1, dtype=np.int8),
            source_frame_indices=np.arange(frames, dtype=np.int32) * 2,
        )


def test_spatial_dataset_bounds_transitions_and_calculates_statistics(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)

    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path],
        autoencoder,
        torch.device("cpu"),
        maximum_transitions=3,
        encode_batch_size=4,
    )
    statistics = dataset.normalization_statistics()

    assert len(dataset) == 3
    assert dataset.encoded_frames == 8
    assert dataset.latent_shape == (4, 16, 16)
    assert dataset[0]["current_latent"].shape == (4, 16, 16)
    assert [tuple(value.shape) for value in statistics] == [
        (9,),
        (9,),
        (4,),
        (4,),
        (4,),
        (4,),
    ]
    assert torch.all(statistics[1] > 0)
    assert torch.all(statistics[3] > 0)
    assert torch.all(statistics[5] > 0)


def test_spatial_prediction_loss_backpropagates_only_into_dynamics(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    autoencoder.requires_grad_(False)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), maximum_transitions=2
    )
    batch = {
        name: torch.stack([dataset[index][name] for index in range(2)])
        for name in dataset[0]
    }
    statistics = dataset.normalization_statistics()
    dynamics = SpatialLatentDynamics(
        latent_channels=4,
        hidden_channels=8,
        blocks=1,
        action_mean=statistics[0],
        action_std=statistics[1],
        latent_mean=statistics[2],
        latent_std=statistics[3],
        motion_mean=statistics[4],
        motion_std=statistics[5],
    )

    total, latent, pixel = _prediction_loss(
        dynamics, autoencoder, batch, latent_weight=1.0, pixel_weight=1.0
    )
    total.backward()

    assert total.item() == pytest.approx(latent.item() + pixel.item())
    assert any(parameter.grad is not None for parameter in dynamics.parameters())
    assert all(parameter.grad is None for parameter in autoencoder.parameters())


def test_spatial_dynamics_checkpoint_round_trip(tmp_path: Path) -> None:
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)
    autoencoder_path = tmp_path / "autoencoder.pt"
    autoencoder_path.write_bytes(b"stable checkpoint")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "dynamics.pt"
    _save_checkpoint(
        checkpoint,
        dynamics,
        history={"train": [1.0]},
        autoencoder_checkpoint=autoencoder_path,
        autoencoder_sha256="hash",
        manifest_path=manifest,
        latent_weight=1.0,
        pixel_weight=1.0,
    )

    loaded, metadata = load_spatial_dynamics_checkpoint(checkpoint, torch.device("cpu"))

    assert loaded.latent_channels == 4
    assert loaded.hidden_channels == 8
    assert metadata["model_type"] == "spatial_latent_dynamics"
    for expected, actual in zip(dynamics.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_spatial_edge_weight_penalizes_blur(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    autoencoder.requires_grad_(False)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), maximum_transitions=2
    )
    batch = {
        name: torch.stack([dataset[index][name] for index in range(2)])
        for name in dataset[0]
    }
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)

    plain, latent, pixel = _prediction_loss(
        dynamics, autoencoder, batch, latent_weight=1.0, pixel_weight=1.0
    )
    with_edges, _, _ = _prediction_loss(
        dynamics, autoencoder, batch, latent_weight=1.0, pixel_weight=1.0, edge_weight=10.0
    )

    assert plain.item() == pytest.approx(latent.item() + pixel.item())
    # The zero-initialized model predicts the copy, whose decoded gradients differ
    # from the oracle's, so the edge term must add cost rather than vanish.
    assert with_edges.item() > plain.item()


def test_spatial_dataset_horizon_yields_rollout_sequences(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path, frames=12)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)

    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), horizon=3, encode_batch_size=4
    )
    sample = dataset[0]

    # horizon + 2 latents seed the rollout and supply one target per step.
    assert sample["latent_sequence"].shape == (5, 4, 16, 16)
    assert sample["action_sequence"].shape == (3, 9)
    assert torch.equal(sample["latent_sequence"][1], sample["current_latent"])
    assert torch.equal(sample["latent_sequence"][2], sample["target_latent"])
    assert torch.equal(sample["action_sequence"][0], sample["action"])


def test_spatial_rollout_loss_feeds_predictions_back() -> None:
    """Each step must consume the previous prediction, not the real latent."""

    class ConstantDrift(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.step = nn.Parameter(torch.full((1, 4, 16, 16), 0.1))
            self.latent_std = torch.ones(1, 4, 1, 1)

        def forward(self, previous, current, action):
            return current + self.step

    # Every latent in the window is zero, so drift is the only source of error.
    batch = {
        "latent_sequence": torch.zeros(2, 5, 4, 16, 16),
        "action_sequence": torch.zeros(2, 3, 9),
    }

    total, latent, _ = _prediction_loss(
        ConstantDrift(), None, batch, latent_weight=1.0, pixel_weight=0.0, rollout_steps=3
    )
    total.backward()

    # Feeding predictions back makes the drift accumulate: 0.1, then 0.2, then 0.3.
    assert latent.item() == pytest.approx((0.01 + 0.04 + 0.09) / 3, rel=1e-5)


def test_spatial_rollout_loss_needs_a_horizon_dataset(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu")
    )
    batch = {
        name: torch.stack([dataset[index][name] for index in range(2)])
        for name in dataset[0]
    }
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)

    with pytest.raises(ValueError, match="horizon"):
        _prediction_loss(
            dynamics, autoencoder, batch, latent_weight=1.0, pixel_weight=0.0, rollout_steps=2
        )
