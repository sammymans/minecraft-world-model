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


class ActionFilm(nn.Module):
    """Scale and shift every feature channel from the normalized action."""

    def __init__(self, action_dim: int, channels: int):
        super().__init__()
        self.projection = nn.Linear(action_dim, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(action).chunk(2, dim=1)
        return features * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class SpatialResidualBlock(nn.Module):
    """A small local update block whose middle activation is action-conditioned."""

    def __init__(self, channels: int, action_dim: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.first = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.film = ActionFilm(action_dim, channels)
        self.second = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return features + self.second(self.film(self.first(features), action))


def identity_sampling_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return the [1, height, width, 2] grid that `grid_sample` reads as identity."""
    rows = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    columns = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)[None]


class SpatialLatentDynamics(nn.Module):
    """Move a spatial latent with an action-conditioned warp, then correct it.

    Measured on `vpt_v3`, a single global warp read straight off the two mouse
    deltas explains 16% of the frame-to-frame latent change with no learned
    parameters, and the best possible global warp explains 35%. The median
    displacement is half a latent cell. A network that can only *add* a residual
    has to synthesize that sub-cell translation everywhere at once, and under a
    squared-error objective it settles for the blurry average instead. So the
    warp is explicit here: the network predicts where each cell moves, samples
    the current latent there, and only then adds a small residual for the
    content a translation cannot explain.
    """

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
        self.residual_blocks = nn.ModuleList(
            SpatialResidualBlock(hidden_channels, action_dim) for _ in range(blocks)
        )
        self.trunk_output = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(inplace=True),
        )
        # Where each latent cell moves, in cells.
        self.local_flow = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        # A direct action -> whole-frame displacement path, so camera motion does
        # not have to survive the convolutional trunk to reach the warp.
        self.global_flow = nn.Linear(action_dim, 2)
        # What a translation cannot explain: new sky, new terrain, lighting.
        self.residual = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        for layer in (self.local_flow, self.global_flow, self.residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

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
        height, width = current_latent.shape[2], current_latent.shape[3]
        motion = current_latent - previous_latent
        normalized_current = (current_latent - self.latent_mean) / self.latent_std
        normalized_motion = (motion - self.motion_mean) / self.motion_std
        normalized_action = (action - self.action_mean) / self.action_std
        action_map = normalized_action[:, :, None, None].expand(-1, -1, height, width)
        features = torch.cat((normalized_current, normalized_motion, action_map), dim=1)
        features = self.input(features)
        for block in self.residual_blocks:
            features = block(features, normalized_action)
        features = self.trunk_output(features)

        flow = self.local_flow(features) + self.global_flow(normalized_action)[:, :, None, None]
        warped = self.warp(current_latent, flow)
        return warped + self.residual(features) * self.latent_std

    def warp(self, latent: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Sample `latent` at a per-cell displacement given in latent cells."""
        height, width = latent.shape[2], latent.shape[3]
        if flow.shape != (len(latent), 2, height, width):
            raise ValueError("flow must have shape [batch, 2, height, width]")
        # grid_sample reads normalized coordinates, so one cell is 2/(size - 1).
        scale = flow.new_tensor(
            [2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)]
        ).view(1, 2, 1, 1)
        offset = (flow * scale).permute(0, 2, 3, 1)
        grid = identity_sampling_grid(
            height, width, device=latent.device, dtype=latent.dtype
        ) + offset
        return nn.functional.grid_sample(
            latent, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

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
