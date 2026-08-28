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


def _sinusoidal_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Encode continuous flow time without learning a timestep lookup table."""
    if timestep.ndim != 1:
        raise ValueError("flow time must have shape [batch]")
    half = width // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(10_000.0, device=timestep.device))
        * torch.arange(half, device=timestep.device)
        / max(half - 1, 1)
    )
    angles = timestep[:, None] * frequencies[None] * 1_000.0
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < width:
        embedding = nn.functional.pad(embedding, (0, width - embedding.shape[1]))
    return embedding


class FlowResidualBlock(nn.Module):
    """Residual image block modulated by flow time and the action plan."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, channels * 2)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm1(features)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(nn.functional.silu(hidden))
        hidden = self.conv2(nn.functional.silu(self.norm2(hidden)))
        return features + hidden


class SpatialLatentVideoFlow(nn.Module):
    """Generate a short action-conditioned future latent clip with rectified flow."""

    def __init__(
        self,
        latent_channels: int = 16,
        action_dim: int = 9,
        horizon: int = 8,
        hidden_channels: int = 128,
        condition_dim: int = 128,
        *,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if min(latent_channels, action_dim, horizon, hidden_channels, condition_dim) < 1:
            raise ValueError("all latent video flow dimensions must be positive")
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.condition_dim = condition_dim

        def statistic(value: torch.Tensor | None, size: int, fill: float) -> torch.Tensor:
            result = torch.full((size,), fill) if value is None else value.detach().float().clone()
            if result.shape != (size,):
                raise ValueError("normalization statistic has the wrong shape")
            return result

        action_mean = statistic(action_mean, action_dim, 0.0)
        action_std = statistic(action_std, action_dim, 1.0)
        latent_mean = statistic(latent_mean, latent_channels, 0.0)
        latent_std = statistic(latent_std, latent_channels, 1.0)
        if torch.any(action_std <= 0) or torch.any(latent_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("latent_mean", latent_mean.reshape(1, 1, -1, 1, 1))
        self.register_buffer("latent_std", latent_std.reshape(1, 1, -1, 1, 1))

        time_width = 64
        self.condition = nn.Sequential(
            nn.Linear(time_width + horizon * action_dim + 1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        input_channels = (
            2 * horizon * latent_channels + 2 * latent_channels + horizon * action_dim + 1
        )
        self.input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.high1 = FlowResidualBlock(hidden_channels, condition_dim)
        self.down = nn.Conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.low1 = FlowResidualBlock(hidden_channels * 2, condition_dim)
        self.low2 = FlowResidualBlock(hidden_channels * 2, condition_dim)
        self.up = nn.ConvTranspose2d(
            hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.high2 = FlowResidualBlock(hidden_channels, condition_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, horizon * latent_channels, kernel_size=3, padding=1),
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean) / self.latent_std

    def forward(
        self,
        noisy_future: torch.Tensor,
        context: torch.Tensor,
        base_future: torch.Tensor,
        actions: torch.Tensor,
        flow_time: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, horizon, channels, height, width = noisy_future.shape
        if (horizon, channels) != (self.horizon, self.latent_channels):
            raise ValueError("future clip does not match the configured horizon or channels")
        if context.shape != (batch, 2, channels, height, width):
            raise ValueError("context must have shape [batch, 2, channels, height, width]")
        if base_future.shape != noisy_future.shape:
            raise ValueError("base future must have the same shape as the noisy correction")
        if actions.shape != (batch, horizon, self.action_dim):
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        if flow_time.shape != (batch,):
            raise ValueError("flow time must have shape [batch]")
        if condition_mask is None:
            condition_mask = torch.ones(batch, device=actions.device, dtype=actions.dtype)
        if condition_mask.shape != (batch,):
            raise ValueError("condition mask must have shape [batch]")

        normalized_context = self.normalize_latents(context)
        normalized_actions = (actions - self.action_mean) / self.action_std
        normalized_actions = normalized_actions * condition_mask[:, None, None]
        action_map = normalized_actions.reshape(batch, -1, 1, 1).expand(-1, -1, height, width)
        mask_map = condition_mask[:, None, None, None].expand(-1, -1, height, width)
        features = torch.cat(
            (
                noisy_future.reshape(batch, -1, height, width),
                normalized_context.reshape(batch, -1, height, width),
                self.normalize_latents(base_future).reshape(batch, -1, height, width),
                action_map,
                mask_map,
            ),
            dim=1,
        )
        time_embedding = _sinusoidal_embedding(flow_time, 64)
        condition = self.condition(
            torch.cat(
                (time_embedding, normalized_actions.flatten(1), condition_mask[:, None]), dim=1
            )
        )
        high = self.high1(self.input(features), condition)
        low = self.low2(self.low1(self.down(high), condition), condition)
        decoded = self.up(low) + high
        velocity = self.output(self.high2(decoded, condition))
        return velocity.reshape(batch, horizon, channels, height, width)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        base_future: torch.Tensor,
        actions: torch.Tensor,
        *,
        steps: int = 8,
        guidance_scale: float = 2.0,
        refinement_strength: float = 1.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the learned velocity field from Gaussian noise to a future clip."""
        if steps < 1:
            raise ValueError("sampling steps must be positive")
        if not 0 <= refinement_strength <= 1:
            raise ValueError("refinement strength must be in [0, 1]")
        batch = len(context)
        shape = (batch, self.horizon, self.latent_channels, context.shape[-2], context.shape[-1])
        if initial_noise is None:
            if generator is not None and generator.device != context.device:
                state = torch.randn(shape, generator=generator, dtype=context.dtype).to(
                    context.device
                )
            else:
                state = torch.randn(
                    shape, device=context.device, dtype=context.dtype, generator=generator
                )
        else:
            if initial_noise.shape != shape:
                raise ValueError("initial noise has the wrong shape")
            state = initial_noise.to(device=context.device, dtype=context.dtype).clone()

        def guided_velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            conditional = self(value, context, base_future, actions, time)
            if guidance_scale == 1:
                return conditional
            unconditional = self(
                value,
                context,
                base_future,
                actions,
                time,
                torch.zeros(batch, device=context.device, dtype=context.dtype),
            )
            return unconditional + guidance_scale * (conditional - unconditional)

        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index * delta, device=context.device, dtype=context.dtype)
            velocity = guided_velocity(state, time)
            proposed = state + delta * velocity
            if index + 1 < steps:
                next_time = torch.full(
                    (batch,), (index + 1) * delta, device=context.device, dtype=context.dtype
                )
                next_velocity = guided_velocity(proposed, next_time)
                state = state + 0.5 * delta * (velocity + next_velocity)
            else:
                state = proposed
        return base_future + refinement_strength * state * self.latent_std

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
