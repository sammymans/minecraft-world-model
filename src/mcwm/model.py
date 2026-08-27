"""Small neural-network components for the latent world model."""

from __future__ import annotations

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
