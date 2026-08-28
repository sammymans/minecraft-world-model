from __future__ import annotations

import pytest
import torch

from mcwm.model import (
    SpatialAutoencoder,
    SpatialLatentDiffusion,
    SpatialLatentDynamics,
    SpatialLatentEDM,
    TinyAutoencoder,
    cosine_alpha_bars,
    timestep_embedding,
)


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


def test_cosine_schedule_decays_monotonically_inside_the_unit_interval() -> None:
    alpha_bars = cosine_alpha_bars(50)

    assert alpha_bars.shape == (50,)
    assert torch.all(alpha_bars > 0) and torch.all(alpha_bars < 1)
    # Later timesteps carry more noise, so cumulative alpha must fall.
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])


def test_timestep_embedding_is_deterministic_and_distinguishes_steps() -> None:
    steps = torch.tensor([0, 1, 500])

    embedding = timestep_embedding(steps, 16)

    assert embedding.shape == (3, 16)
    assert torch.equal(embedding, timestep_embedding(steps, 16))
    assert not torch.allclose(embedding[0], embedding[1])
    with pytest.raises(ValueError, match="one-dimensional"):
        timestep_embedding(steps.unsqueeze(0), 16)


def test_spatial_diffusion_predicts_noise_and_validates_shapes() -> None:
    model = SpatialLatentDiffusion(latent_channels=4, hidden_channels=8, blocks=2)
    latent = torch.randn(2, 4, 6, 6)
    action = torch.zeros(2, 9)
    timesteps = torch.tensor([0, 999])

    predicted = model(latent, latent, action, torch.randn_like(latent), timesteps)

    assert predicted.shape == latent.shape
    with pytest.raises(ValueError, match="noisy residual"):
        model(latent, latent, action, torch.randn(2, 4, 3, 3), timesteps)
    with pytest.raises(ValueError, match="timesteps must contain"):
        model(latent, latent, action, torch.randn_like(latent), torch.tensor([0]))


def test_residual_normalization_round_trips() -> None:
    model = SpatialLatentDiffusion(
        latent_channels=3,
        hidden_channels=8,
        blocks=1,
        motion_mean=torch.tensor([0.5, -1.0, 2.0]),
        motion_std=torch.tensor([2.0, 0.5, 4.0]),
    )
    current = torch.randn(2, 3, 5, 5)
    following = torch.randn(2, 3, 5, 5)

    normalized = model.normalize_residual(current, following)

    assert torch.allclose(model.denormalize_residual(current, normalized), following, atol=1e-5)


class _PerfectDenoiser(SpatialLatentDiffusion):
    """A denoiser that knows exactly which noise hides a fixed target."""

    def set_target(self, target: torch.Tensor) -> None:
        self._target = target

    def forward(self, previous_latent, current_latent, action, noisy_residual, timesteps):
        alpha_bar = self.alpha_bars[timesteps[0]]
        return (noisy_residual - alpha_bar.sqrt() * self._target) / (1 - alpha_bar).sqrt()


def test_ddim_sampling_recovers_the_target_a_perfect_denoiser_implies() -> None:
    # If the network predicts exactly the noise standing between the current
    # sample and a fixed clean residual, every DDIM update must leave that
    # residual unchanged and sampling must return it. This checks the sampler
    # arithmetic independently of anything the model learns.
    model = _PerfectDenoiser(latent_channels=3, hidden_channels=8, blocks=1)
    # Seeded so the target stays inside the sampler's clamp deterministically.
    target = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(3))
    model.set_target(target)
    current = torch.randn(2, 3, 4, 4)
    action = torch.zeros(2, 9)

    sampled = model.sample(current, current, action, steps=12)

    assert torch.allclose(sampled, model.denormalize_residual(current, target), atol=1e-4)


def test_sampling_draws_different_futures_from_different_noise() -> None:
    model = SpatialLatentDiffusion(latent_channels=3, hidden_channels=8, blocks=1)
    current = torch.zeros(1, 3, 4, 4)
    action = torch.zeros(1, 9)

    first = model.sample(
        current, current, action, steps=4, generator=torch.Generator().manual_seed(1)
    )
    second = model.sample(
        current, current, action, steps=4, generator=torch.Generator().manual_seed(2)
    )

    # Sampling is what replaces averaging; two draws must not coincide.
    assert not torch.allclose(first, second)
    with pytest.raises(ValueError, match="sampling steps must be positive"):
        model.sample(current, current, action, steps=0)
    with pytest.raises(ValueError, match="residual_limit must be positive"):
        model.sample(current, current, action, steps=4, residual_limit=0.0)


def test_sampling_clamps_the_recovered_residual() -> None:
    # An untrained model predicts zero noise, which at the noisiest timestep
    # implies a residual hundreds of units wide. The clamp must bound it.
    model = SpatialLatentDiffusion(latent_channels=3, hidden_channels=8, blocks=1)
    current = torch.zeros(1, 3, 4, 4)
    action = torch.zeros(1, 9)

    sampled = model.sample(current, current, action, steps=3, residual_limit=2.0)
    residual = model.normalize_residual(current, sampled)

    assert residual.abs().max().item() <= 2.0 + 1e-5


def test_spatial_edm_correction_round_trips_and_validates_context() -> None:
    model = SpatialLatentEDM(
        latent_channels=3,
        context_steps=4,
        hidden_channels=8,
        blocks_per_level=1,
        correction_mean=torch.tensor([0.1, -0.2, 0.3]),
        correction_std=torch.tensor([0.5, 2.0, 1.5]),
    )
    anchor = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 3, 4, 4)
    context = torch.randn(2, 4, 3, 4, 4)
    actions = torch.randn(2, 4, 9)
    sigmas = torch.tensor([0.1, 1.0])

    correction = model.normalize_correction(anchor, target)
    denoised = model.denoise(correction, sigmas, anchor, context, actions)

    assert torch.allclose(model.apply_correction(anchor, correction), target, atol=1e-5)
    assert denoised.shape == correction.shape
    assert 10_000 < model.parameter_count < 100_000
    with pytest.raises(ValueError, match="context latent"):
        model.denoise(correction, sigmas, anchor, context[:, :2], actions[:, :2])


class _PerfectEDM(SpatialLatentEDM):
    def set_target(self, target: torch.Tensor) -> None:
        self._target = target

    def denoise(
        self,
        noisy_correction,
        sigmas,
        anchor_latent,
        context_latents,
        actions,
        context_noise=None,
    ):
        del noisy_correction, sigmas, anchor_latent, context_latents, actions, context_noise
        return self._target


def test_edm_heun_sampling_recovers_a_perfect_denoisers_correction() -> None:
    model = _PerfectEDM(
        latent_channels=3, context_steps=2, hidden_channels=8, blocks_per_level=1
    )
    anchor = torch.randn(2, 3, 4, 4)
    context = torch.randn(2, 2, 3, 4, 4)
    actions = torch.randn(2, 2, 9)
    correction = torch.randn(2, 3, 4, 4)
    model.set_target(correction)

    sampled = model.sample(anchor, context, actions, steps=4)

    assert torch.allclose(sampled, model.apply_correction(anchor, correction), atol=1e-4)
