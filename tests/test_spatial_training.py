from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mcwm.model import SpatialAutoencoder
from mcwm.spatial_training import (
    _save_checkpoint,
    evaluate_spatial_autoencoder,
    load_spatial_autoencoder_checkpoint,
    spatial_reconstruction_loss,
)


def test_spatial_loss_rewards_pixels_and_edges() -> None:
    target = torch.zeros(1, 3, 8, 8)
    target[:, :, :, 4:] = 1
    identical, _, identical_edge = spatial_reconstruction_loss(target, target)
    blurred = target.clone()
    blurred[:, :, :, 3:5] = 0.5
    objective, pixel_l1, edge_l1 = spatial_reconstruction_loss(blurred, target, edge_weight=0.25)

    assert identical.item() == pytest.approx(0)
    assert identical_edge.item() == pytest.approx(0)
    assert objective > 0
    assert pixel_l1 > 0
    assert edge_l1 > 0


def test_spatial_metrics_measure_gradient_energy() -> None:
    torch.manual_seed(4)
    model = SpatialAutoencoder(latent_channels=8, base_channels=8)
    frames = torch.rand(4, 3, 64, 64)

    metrics = evaluate_spatial_autoencoder(model, frames, torch.device("cpu"), batch_size=2)

    assert metrics.objective > metrics.pixel_mse
    assert metrics.pixel_mse > 0
    assert metrics.psnr_db > 0
    assert metrics.gradient_l1 > 0
    assert metrics.gradient_energy_ratio >= 0


def test_spatial_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = SpatialAutoencoder(latent_channels=8, base_channels=8)
    path = tmp_path / "spatial.pt"
    _save_checkpoint(
        path,
        model,
        history={"train_objective": [1.0], "validation_objective": []},
        edge_weight=0.25,
        train_episodes=["train"],
        validation_episodes=["validation"],
        manifest_path=None,
    )

    loaded, metadata = load_spatial_autoencoder_checkpoint(path, torch.device("cpu"))

    assert loaded.latent_shape == (8, 16, 16)
    assert metadata["model_type"] == "spatial_autoencoder"
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)
