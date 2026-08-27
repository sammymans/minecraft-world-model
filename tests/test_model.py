from __future__ import annotations

import pytest
import torch
from torch import nn

from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics, TinyAutoencoder


def test_autoencoder_shapes_and_range() -> None:
    model = TinyAutoencoder(latent_dim=32)
    frames = torch.rand(4, 3, 64, 64)

    latents = model.encode(frames)
    reconstructed = model.decode(latents)

    assert latents.shape == (4, 32)
    assert reconstructed.shape == frames.shape
    assert reconstructed.isfinite().all()
    assert model.parameter_count < 1_000_000


def test_autoencoder_can_reduce_loss_on_a_tiny_fixed_batch() -> None:
    torch.manual_seed(3)
    model = TinyAutoencoder(latent_dim=16)
    frames = torch.zeros(2, 3, 64, 64)
    frames[1] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(frames), frames).item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(frames), frames)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(frames), frames).item()

    assert final < initial


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
    # The zero-initialized flow and residual heads make the model start as the
    # copy baseline; bilinear resampling costs a little float precision.
    torch.testing.assert_close(predicted, current, atol=1e-5, rtol=0)
    assert model.parameter_count < 1_000_000


def test_spatial_dynamics_validates_input_shapes() -> None:
    model = SpatialLatentDynamics(latent_channels=8, hidden_channels=16, blocks=1)
    previous = torch.randn(2, 8, 16, 16)
    current = torch.randn(2, 8, 16, 16)

    with pytest.raises(ValueError, match="actions"):
        model(previous, current, torch.randn(2, 8))
    with pytest.raises(ValueError, match="channels"):
        model(previous[:, :7], current[:, :7], torch.randn(2, 9))


def test_spatial_dynamics_warp_translates_content() -> None:
    model = SpatialLatentDynamics(latent_channels=1, hidden_channels=8, blocks=1)
    latent = torch.zeros(1, 1, 8, 8)
    latent[0, 0, 4, 4] = 1.0
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0  # sample two cells to the right, so content moves left

    warped = model.warp(latent, flow)

    assert warped[0, 0, 4, 2] == pytest.approx(1.0)
    assert warped[0, 0, 4, 4] == pytest.approx(0.0)


def test_spatial_dynamics_action_changes_the_prediction() -> None:
    torch.manual_seed(0)
    model = SpatialLatentDynamics(latent_channels=4, hidden_channels=16, blocks=2)
    # Break the copy initialization so the action pathways carry signal.
    for head in (model.local_flow, model.global_flow, model.residual):
        nn.init.normal_(head.weight, std=0.05)
    previous = torch.randn(3, 4, 16, 16)
    current = torch.randn(3, 4, 16, 16)

    first = model(previous, current, torch.zeros(3, 9))
    second = model(previous, current, torch.full((3, 9), 2.0))

    assert not torch.allclose(first, second, atol=1e-4)
