from __future__ import annotations

import torch

from mcwm.model import TinyAutoencoder


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
