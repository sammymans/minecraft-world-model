from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mcwm.model import SpatialAutoencoder, SpatialLatentEDM
from mcwm.spatial_dynamics import SpatialEncodedDynamicsDataset
from mcwm.spatial_edm import SpatialEncodedContextDataset, _edm_loss, evaluate_edm


def _write_episode(path: Path, *, frames: int = 9) -> None:
    rng = np.random.default_rng(8)
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


def _context_dataset(tmp_path: Path) -> tuple[SpatialEncodedContextDataset, SpatialAutoencoder]:
    path = tmp_path / "episode.npz"
    _write_episode(path)
    autoencoder = SpatialAutoencoder(latent_channels=3, base_channels=4)
    encoded = SpatialEncodedDynamicsDataset.from_paths(
        [path], autoencoder, torch.device("cpu"), encode_batch_size=4
    )
    return SpatialEncodedContextDataset(encoded, context_steps=4), autoencoder


def test_context_dataset_aligns_four_frames_actions_and_target(tmp_path: Path) -> None:
    dataset, _ = _context_dataset(tmp_path)

    first = dataset[0]

    assert len(dataset) == 5
    assert first["context_latents"].shape == (4, 3, 16, 16)
    assert first["actions"].shape == (4, 9)
    assert first["target_latent"].shape == (3, 16, 16)
    assert first["target_frame"].shape == (3, 64, 64)


def test_edm_objective_learns_on_a_tiny_fixed_batch(tmp_path: Path) -> None:
    torch.manual_seed(5)
    dataset, _ = _context_dataset(tmp_path)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    model = SpatialLatentEDM(
        latent_channels=3,
        context_steps=4,
        hidden_channels=8,
        blocks_per_level=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    initial = float(_edm_loss(model, None, batch, context_noise=0, seed=11).detach())

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = _edm_loss(model, None, batch, context_noise=0, seed=11)
        loss.backward()
        optimizer.step()

    final = float(_edm_loss(model, None, batch, context_noise=0, seed=11).detach())
    assert final < initial * 0.7


def test_edm_evaluation_uses_common_noise_for_action_ablation(tmp_path: Path) -> None:
    dataset, autoencoder = _context_dataset(tmp_path)
    model = SpatialLatentEDM(
        latent_channels=3,
        context_steps=4,
        hidden_channels=8,
        blocks_per_level=1,
    )

    metrics = evaluate_edm(
        model,
        None,
        autoencoder,
        dataset,
        torch.device("cpu"),
        batch_size=len(dataset),
        sampling_steps=2,
        maximum_examples=len(dataset),
    )

    assert metrics.examples == len(dataset)
    assert metrics.action_effect_latent_mse == 0
    assert metrics.shuffled_action_latent_mse == metrics.sample_latent_mse
    assert -1 <= metrics.sample_gradient_cosine <= 1
