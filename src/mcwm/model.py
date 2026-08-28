"""Small neural-network components for the latent world model."""

from __future__ import annotations

import torch
from torch import nn


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
