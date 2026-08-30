from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics
from mcwm.spatial_dynamics import (
    SpatialEncodedDynamicsDataset,
    SpatialEncodedSequenceDataset,
    _prediction_loss,
    _recursive_prediction_loss,
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


def test_nested_prefix_transition_subsets_are_deterministic_and_nested(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"episode-{index}.npz" for index in range(4)]
    for path in paths:
        _write_episode(path, frames=12)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)

    def selected(maximum: int) -> tuple[list[tuple[str, int]], str]:
        dataset = SpatialEncodedDynamicsDataset.from_paths(
            paths,
            autoencoder,
            torch.device("cpu"),
            maximum_transitions=maximum,
            encode_batch_size=4,
            selection_policy="nested_prefix",
            seed=7,
        )
        references = [
            (dataset.episodes[episode_index].episode, current_index)
            for episode_index, current_index in dataset.index
        ]
        return references, dataset.selection_sha256

    small, small_fingerprint = selected(8)
    large, _ = selected(24)

    assert len(small) == 8
    assert len(large) == 24
    assert small == large[: len(small)]
    assert (small, small_fingerprint) == selected(8)


def test_spatial_dataset_rejects_unknown_selection_policy(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)

    with pytest.raises(ValueError, match="selection_policy"):
        SpatialEncodedDynamicsDataset.from_paths(
            [path],
            autoencoder,
            torch.device("cpu"),
            selection_policy="unknown",  # type: ignore[arg-type]
        )


def test_spatial_prediction_loss_backpropagates_only_into_dynamics(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    autoencoder.requires_grad_(False)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), maximum_transitions=2
    )
    batch = {name: torch.stack([dataset[index][name] for index in range(2)]) for name in dataset[0]}
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


def test_spatial_dynamics_rejects_unversioned_checkpoint(tmp_path: Path) -> None:
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
    payload = torch.load(checkpoint, weights_only=True)
    del payload["architecture"]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="incompatible or unversioned"):
        load_spatial_dynamics_checkpoint(checkpoint, torch.device("cpu"))


def _recursive_batch(dataset: SpatialEncodedSequenceDataset, count: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([dataset[index][name] for index in range(count)]) for name in dataset[0]
    }


def test_spatial_sequence_dataset_aligns_latent_and_action_windows(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path, frames=12)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    encoded = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), count_horizon=5, encode_batch_size=4
    )

    windows = SpatialEncodedSequenceDataset(encoded, horizon=5)
    sample = windows[0]

    # A five-step window needs two seed latents plus five targets, and one
    # action per predicted step.
    assert sample["latents"].shape == (7, 4, 16, 16)
    assert sample["actions"].shape == (5, 9)
    episode_index, current_index = windows.index[0]
    timeline = encoded.latents[episode_index]
    assert torch.equal(sample["latents"][1], timeline[current_index].float())
    assert torch.equal(sample["latents"][-1], timeline[current_index + 5].float())


def test_spatial_sequence_dataset_bounds_and_rejects_bad_horizons(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path, frames=12)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    encoded = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), count_horizon=3
    )

    assert len(SpatialEncodedSequenceDataset(encoded, horizon=3, maximum_sequences=2)) == 2
    with pytest.raises(ValueError, match="horizon must be positive"):
        SpatialEncodedSequenceDataset(encoded, horizon=0)
    with pytest.raises(ValueError, match="no clean spatial sequences"):
        SpatialEncodedSequenceDataset(encoded, horizon=64)


def test_recursive_loss_consumes_its_own_predictions(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path, frames=12)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    autoencoder.requires_grad_(False)
    encoded = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), count_horizon=3
    )
    windows = SpatialEncodedSequenceDataset(encoded, horizon=3)
    batch = _recursive_batch(windows, 2)
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)
    seen: list[torch.Tensor] = []
    original_forward = dynamics.forward

    def record(previous, current, action):
        seen.append(current)
        return original_forward(previous, current, action)

    dynamics.forward = record  # type: ignore[method-assign]
    total, latent, pixel = _recursive_prediction_loss(
        dynamics,
        autoencoder,
        batch,
        latent_weight=1.0,
        pixel_weight=1.0,
        horizon_decay=0.8,
    )
    total.backward()

    assert len(seen) == 3
    # Steps two and three must run on predictions, never on the real latents.
    assert torch.equal(seen[0], batch["latents"][:, 1])
    for step in (1, 2):
        assert not torch.equal(seen[step], batch["latents"][:, step + 1])
    assert total.item() == pytest.approx(latent.item() + pixel.item())
    assert any(parameter.grad is not None for parameter in dynamics.parameters())
    assert all(parameter.grad is None for parameter in autoencoder.parameters())


def test_recursive_loss_matches_one_step_loss_at_horizon_one(tmp_path: Path) -> None:
    path = tmp_path / "episode.npz"
    _write_episode(path, frames=10)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    autoencoder.requires_grad_(False)
    encoded = SpatialEncodedDynamicsDataset.from_paths([path], autoencoder, torch.device("cpu"))
    windows = SpatialEncodedSequenceDataset(encoded, horizon=1)
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)

    recursive, _, _ = _recursive_prediction_loss(
        dynamics,
        autoencoder,
        _recursive_batch(windows, 2),
        latent_weight=1.0,
        pixel_weight=1.0,
        horizon_decay=0.8,
    )
    single_step = {
        "previous_latent": torch.stack([windows[index]["latents"][0] for index in range(2)]),
        "current_latent": torch.stack([windows[index]["latents"][1] for index in range(2)]),
        "target_latent": torch.stack([windows[index]["latents"][2] for index in range(2)]),
        "action": torch.stack([windows[index]["actions"][0] for index in range(2)]),
    }
    expected, _, _ = _prediction_loss(
        dynamics, autoencoder, single_step, latent_weight=1.0, pixel_weight=1.0
    )

    assert recursive.item() == pytest.approx(expected.item(), rel=1e-6)


def test_recursive_loss_rejects_misaligned_windows() -> None:
    dynamics = SpatialLatentDynamics(latent_channels=4, hidden_channels=8, blocks=1)
    autoencoder = SpatialAutoencoder(latent_channels=4, base_channels=4)
    batch = {
        "latents": torch.zeros(2, 4, 4, 16, 16),
        "actions": torch.zeros(2, 3, 9),
    }

    with pytest.raises(ValueError, match="misaligned"):
        _recursive_prediction_loss(
            dynamics,
            autoencoder,
            batch,
            latent_weight=1.0,
            pixel_weight=1.0,
            horizon_decay=0.8,
        )
    with pytest.raises(ValueError, match="horizon_decay"):
        _recursive_prediction_loss(
            dynamics,
            autoencoder,
            {"latents": torch.zeros(2, 5, 4, 16, 16), "actions": torch.zeros(2, 3, 9)},
            latent_weight=1.0,
            pixel_weight=1.0,
            horizon_decay=0.0,
        )


def test_multi_step_checkpoint_records_its_training_objective(tmp_path: Path) -> None:
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
        rollout_steps=5,
        horizon_decay=0.8,
        initial_checkpoint=tmp_path / "v0.pt",
    )

    _, metadata = load_spatial_dynamics_checkpoint(checkpoint, torch.device("cpu"))

    assert metadata["rollout_steps"] == 5
    assert metadata["horizon_decay"] == pytest.approx(0.8)
    assert metadata["initial_checkpoint"] == str(tmp_path / "v0.pt")
