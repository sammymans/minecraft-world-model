"""Small neural-network components for the latent world model."""

from __future__ import annotations

import math

import torch
from torch import nn


class TinyAutoencoder(nn.Module):
    """Compress one 64x64 RGB frame into a small latent vector and reconstruct it."""

    def __init__(self, latent_dim: int = 256, base_channels: int = 16):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feature_channels = channels[3]
        flattened_features = self.feature_channels * 4 * 4
        self.to_latent = nn.Linear(flattened_features, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened_features)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                channels[3], channels[2], kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                channels[2], channels[1], kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                channels[1], channels[0], kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(frames)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latents).reshape(-1, self.feature_channels, 4, 4)
        return self.decoder_conv(features)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(frames))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SpatialAutoencoder(nn.Module):
    """Preserve Minecraft layout in a compact 16x16 latent feature map."""

    def __init__(self, latent_channels: int = 16, base_channels: int = 32):
        super().__init__()
        if latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        channels = (base_channels, base_channels * 2, base_channels * 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[2], latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels[2], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(channels[0], 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(frames))

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, 16, 16)

    @property
    def latent_value_count(self) -> int:
        return self.latent_channels * 16 * 16

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SpatialResidualBlock(nn.Module):
    """A small local update block for the spatial dynamics network."""

    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.network(features)


class SpatialLatentDynamics(nn.Module):
    """Predict a residual update to a spatial latent from motion and controls."""

    def __init__(
        self,
        latent_channels: int = 16,
        action_dim: int = 9,
        hidden_channels: int = 64,
        blocks: int = 3,
        *,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if min(latent_channels, action_dim, hidden_channels, blocks) < 1:
            raise ValueError("all spatial dynamics dimensions must be positive")
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.hidden_channels = hidden_channels
        self.blocks = blocks

        def statistic(value: torch.Tensor | None, size: int, fill: float) -> torch.Tensor:
            result = torch.full((size,), fill) if value is None else value.detach().float().clone()
            if result.shape != (size,):
                raise ValueError("normalization statistic has the wrong shape")
            return result

        action_mean = statistic(action_mean, action_dim, 0.0)
        action_std = statistic(action_std, action_dim, 1.0)
        latent_mean = statistic(latent_mean, latent_channels, 0.0)
        latent_std = statistic(latent_std, latent_channels, 1.0)
        motion_mean = statistic(motion_mean, latent_channels, 0.0)
        motion_std = statistic(motion_std, latent_channels, 1.0)
        if torch.any(action_std <= 0) or torch.any(latent_std <= 0) or torch.any(motion_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("latent_mean", latent_mean.reshape(1, -1, 1, 1))
        self.register_buffer("latent_std", latent_std.reshape(1, -1, 1, 1))
        self.register_buffer("motion_mean", motion_mean.reshape(1, -1, 1, 1))
        self.register_buffer("motion_std", motion_std.reshape(1, -1, 1, 1))

        input_channels = latent_channels * 2 + action_dim
        self.input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *(SpatialResidualBlock(hidden_channels) for _ in range(blocks))
        )
        self.output = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if previous_latent.shape != current_latent.shape or current_latent.ndim != 4:
            raise ValueError("spatial latents must have matching [batch, channels, height, width]")
        if current_latent.shape[1] != self.latent_channels:
            raise ValueError("latent channels do not match the spatial dynamics model")
        if action.shape != (len(current_latent), self.action_dim):
            raise ValueError("actions must have shape [batch, action_dim]")
        motion = current_latent - previous_latent
        normalized_current = (current_latent - self.latent_mean) / self.latent_std
        normalized_motion = (motion - self.motion_mean) / self.motion_std
        normalized_action = (action - self.action_mean) / self.action_std
        action_map = normalized_action[:, :, None, None].expand(
            -1, -1, current_latent.shape[2], current_latent.shape[3]
        )
        features = torch.cat((normalized_current, normalized_motion, action_map), dim=1)
        change = self.output(self.residual_blocks(self.input(features)))
        return current_latent + change * self.latent_std

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LatentDynamics(nn.Module):
    """Predict the next visual latent from motion context and one raw action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 9,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        *,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        if action_mean is None:
            action_mean = torch.zeros(action_dim)
        if action_std is None:
            action_std = torch.ones(action_dim)
        if action_mean.shape != (action_dim,) or action_std.shape != (action_dim,):
            raise ValueError("action statistics must match action_dim")
        if torch.any(action_std <= 0):
            raise ValueError("action standard deviations must be positive")
        self.register_buffer("action_mean", action_mean.detach().to(torch.float32).clone())
        self.register_buffer("action_std", action_std.detach().to(torch.float32).clone())

        self.current_norm = nn.LayerNorm(latent_dim)
        self.motion_norm = nn.LayerNorm(latent_dim)
        layers: list[nn.Module] = []
        input_dim = latent_dim * 2 + action_dim
        for layer_index in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.residual = nn.Sequential(*layers)

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if previous_latent.shape != current_latent.shape:
            raise ValueError("previous and current latents must have the same shape")
        if previous_latent.shape[-1] != self.latent_dim:
            raise ValueError("latent input does not match latent_dim")
        if action.shape[:-1] != current_latent.shape[:-1] or action.shape[-1] != self.action_dim:
            raise ValueError("action batch shape does not match latent batch shape")

        motion = current_latent - previous_latent
        normalized_action = (action - self.action_mean) / self.action_std
        features = torch.cat(
            (
                self.current_norm(current_latent),
                self.motion_norm(motion),
                normalized_action,
            ),
            dim=-1,
        )
        return current_latent + self.residual(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a diffusion timestep."""
    if dim < 2:
        raise ValueError("timestep embedding needs at least two dimensions")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a one-dimensional batch")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


def cosine_alpha_bars(steps: int, offset: float = 0.008) -> torch.Tensor:
    """Nichol and Dhariwal's cosine noise schedule as cumulative alphas."""
    if steps < 1:
        raise ValueError("diffusion steps must be positive")
    positions = torch.arange(steps + 1, dtype=torch.float64) / steps
    values = torch.cos((positions + offset) / (1 + offset) * math.pi / 2).square()
    alpha_bars = (values / values[0])[1:]
    return alpha_bars.clamp(1e-5, 0.9999).float()


class SpatialDiffusionBlock(nn.Module):
    """A local update block whose activations are shifted by the timestep."""

    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.input_norm = nn.GroupNorm(groups, channels)
        self.input_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.timestep = nn.Linear(embedding_dim, channels)
        self.output = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.input_conv(nn.functional.silu(self.input_norm(features)))
        hidden = hidden + self.timestep(embedding)[:, :, None, None]
        return features + self.output(hidden)


class SpatialLatentDiffusion(nn.Module):
    """Denoise the next spatial latent's residual, conditioned on motion and controls.

    The deterministic model regresses the next latent directly, so squared error
    drives it to the average over every plausible future and the result is
    smooth. This predicts the noise added to the normalized residual instead,
    which lets sampling draw one plausible future rather than their mean.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        action_dim: int = 9,
        hidden_channels: int = 64,
        blocks: int = 3,
        diffusion_steps: int = 1_000,
        *,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if min(latent_channels, action_dim, hidden_channels, blocks, diffusion_steps) < 1:
            raise ValueError("all spatial diffusion dimensions must be positive")
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.hidden_channels = hidden_channels
        self.blocks = blocks
        self.diffusion_steps = diffusion_steps

        def statistic(value: torch.Tensor | None, size: int, fill: float) -> torch.Tensor:
            result = torch.full((size,), fill) if value is None else value.detach().float().clone()
            if result.shape != (size,):
                raise ValueError("normalization statistic has the wrong shape")
            return result

        action_mean = statistic(action_mean, action_dim, 0.0)
        action_std = statistic(action_std, action_dim, 1.0)
        latent_mean = statistic(latent_mean, latent_channels, 0.0)
        latent_std = statistic(latent_std, latent_channels, 1.0)
        motion_mean = statistic(motion_mean, latent_channels, 0.0)
        motion_std = statistic(motion_std, latent_channels, 1.0)
        if torch.any(action_std <= 0) or torch.any(latent_std <= 0) or torch.any(motion_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("latent_mean", latent_mean.reshape(1, -1, 1, 1))
        self.register_buffer("latent_std", latent_std.reshape(1, -1, 1, 1))
        self.register_buffer("motion_mean", motion_mean.reshape(1, -1, 1, 1))
        self.register_buffer("motion_std", motion_std.reshape(1, -1, 1, 1))
        self.register_buffer("alpha_bars", cosine_alpha_bars(diffusion_steps))

        embedding_dim = hidden_channels * 4
        self.embedding = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # noisy residual, current latent, observed motion, broadcast action
        input_channels = latent_channels * 3 + action_dim
        self.input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList(
            SpatialDiffusionBlock(hidden_channels, embedding_dim) for _ in range(blocks)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def normalize_residual(
        self, current_latent: torch.Tensor, next_latent: torch.Tensor
    ) -> torch.Tensor:
        """Map an observed step to the unit-scale target the model diffuses."""
        return (next_latent - current_latent - self.motion_mean) / self.motion_std

    def denormalize_residual(
        self, current_latent: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return current_latent + self.motion_mean + residual * self.motion_std

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise that was added to the normalized residual."""
        if previous_latent.shape != current_latent.shape or current_latent.ndim != 4:
            raise ValueError("spatial latents must have matching [batch, channels, height, width]")
        if current_latent.shape[1] != self.latent_channels:
            raise ValueError("latent channels do not match the spatial diffusion model")
        if noisy_residual.shape != current_latent.shape:
            raise ValueError("the noisy residual must match the latent shape")
        if action.shape != (len(current_latent), self.action_dim):
            raise ValueError("actions must have shape [batch, action_dim]")
        if timesteps.shape != (len(current_latent),):
            raise ValueError("timesteps must contain one value per batch item")
        motion = current_latent - previous_latent
        normalized_current = (current_latent - self.latent_mean) / self.latent_std
        normalized_motion = (motion - self.motion_mean) / self.motion_std
        normalized_action = (action - self.action_mean) / self.action_std
        action_map = normalized_action[:, :, None, None].expand(
            -1, -1, current_latent.shape[2], current_latent.shape[3]
        )
        features = self.input(
            torch.cat(
                (noisy_residual, normalized_current, normalized_motion, action_map), dim=1
            )
        )
        embedding = self.embedding(timestep_embedding(timesteps, self.hidden_channels))
        for block in self.residual_blocks:
            features = block(features, embedding)
        return self.output(features)

    @torch.no_grad()
    def sample(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
        *,
        steps: int = 20,
        residual_limit: float = 4.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one plausible next latent with a deterministic DDIM schedule."""
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        if residual_limit <= 0:
            raise ValueError("residual_limit must be positive")
        steps = min(steps, self.diffusion_steps)
        schedule = torch.linspace(
            self.diffusion_steps - 1, 0, steps, device=current_latent.device
        ).round().long()
        if generator is not None and generator.device != current_latent.device:
            # A seeded CPU generator is the portable way to make sampling
            # reproducible, including on MPS where device generators differ.
            residual = torch.randn(
                current_latent.shape, dtype=current_latent.dtype, generator=generator
            ).to(current_latent.device)
        else:
            residual = torch.randn(
                current_latent.shape,
                device=current_latent.device,
                dtype=current_latent.dtype,
                generator=generator,
            )
        for position, timestep in enumerate(schedule):
            batch_timesteps = timestep.repeat(len(current_latent))
            alpha_bar = self.alpha_bars[timestep]
            predicted_noise = self.forward(
                previous_latent, current_latent, action, residual, batch_timesteps
            )
            # At the noisiest timestep alpha_bar is ~1e-5, so this division
            # amplifies any error in the predicted noise by about 300x. The
            # diffused target is a unit-scale normalized residual, so clamping
            # keeps an early mistake from dominating the whole trajectory.
            predicted_start = (
                (residual - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            ).clamp(-residual_limit, residual_limit)
            if position + 1 == len(schedule):
                residual = predicted_start
            else:
                next_alpha_bar = self.alpha_bars[schedule[position + 1]]
                residual = (
                    next_alpha_bar.sqrt() * predicted_start
                    + (1 - next_alpha_bar).sqrt() * predicted_noise
                )
        return self.denormalize_residual(current_latent, residual)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
