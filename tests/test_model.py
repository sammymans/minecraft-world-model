from __future__ import annotations

import pytest
import torch

from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics


def test_spatial_autoencoder_preserves_a_16_by_16_feature_map() -> None:
    model = SpatialAutoencoder(latent_channels=16, base_channels=16)
    frames = torch.rand(3, 3, 64, 64)

    latents = model.encode(frames)
    reconstructed = model.decode(latents)

    assert latents.shape == (3, 16, 16, 16)
    assert reconstructed.shape == frames.shape
    assert model.latent_shape == (16, 16, 16)
    assert model.latent_value_count == 4096
    assert reconstructed.isfinite().all()
    assert model.parameter_count < 1_000_000


def test_spatial_dynamics_starts_as_copy_and_preserves_shape() -> None:
    model = SpatialLatentDynamics(latent_channels=8, hidden_channels=16, blocks=2)
    previous = torch.randn(4, 8, 16, 16)
    current = torch.randn(4, 8, 16, 16)
    actions = torch.randn(4, 9)

    predicted = model(previous, current, actions)

    assert predicted.shape == current.shape
    assert torch.equal(predicted, current)
    assert model.parameter_count < 1_000_000


def test_spatial_dynamics_validates_input_shapes() -> None:
    model = SpatialLatentDynamics(latent_channels=8, hidden_channels=16, blocks=1)
    previous = torch.randn(2, 8, 16, 16)
    current = torch.randn(2, 8, 16, 16)

    with pytest.raises(ValueError, match="actions"):
        model(previous, current, torch.randn(2, 8))
    with pytest.raises(ValueError, match="channels"):
        model(previous[:, :7], current[:, :7], torch.randn(2, 9))
