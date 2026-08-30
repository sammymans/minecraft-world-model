"""V2 action-conditioned temporal latent diffusion.

This module is deliberately separate from the V1 dynamics path.  It uses the
frozen spatial autoencoder as its observation interface, but owns its dataset,
model, schedule, sampler, checkpoints, and tiny-set gate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mcwm.dataset import ProcessedEpisode
from mcwm.dynamics import _file_sha256
from mcwm.manifest import DatasetManifest
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LATENT_DIFFUSION_V2_ARCHITECTURE = "temporal_multiscale_action_unet_velocity_v2"
ACTION_BUCKETS = ("forward", "look_left", "look_right", "other")


def action_bucket(action: np.ndarray | torch.Tensor, camera_threshold: float = 2.0) -> str:
    """Assign the target-driving action to one deterministic control bucket."""
    mouse_dx = float(action[-2])
    if mouse_dx <= -camera_threshold:
        return "look_left"
    if mouse_dx >= camera_threshold:
        return "look_right"
    if float(action[0]) > 0:
        return "forward"
    return "other"


@dataclass(frozen=True)
class TemporalSequenceReference:
    episode: str
    context_start: int
    action_bucket: str


class TemporalLatentDataset(Dataset[dict[str, torch.Tensor]]):
    """Eight context latents, their transitions/actions, and one next latent."""

    def __init__(
        self,
        episodes: list[ProcessedEpisode],
        latents: list[torch.Tensor],
        *,
        context_frames: int = 8,
        maximum_sequences: int | None = None,
        selection_policy: Literal["random", "action_balanced"] = "random",
        seed: int = 7,
    ):
        if not episodes or len(episodes) != len(latents):
            raise ValueError("each temporal episode needs one latent timeline")
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        if maximum_sequences is not None and maximum_sequences < 1:
            raise ValueError("maximum_sequences must be positive when supplied")
        if selection_policy not in {"random", "action_balanced"}:
            raise ValueError("selection_policy must be random or action_balanced")
        latent_shapes: set[tuple[int, int, int]] = set()
        for episode, timeline in zip(episodes, latents, strict=True):
            if timeline.ndim != 4 or len(timeline) != len(episode.frames):
                raise ValueError("latent timelines must be [time, channels, height, width]")
            latent_shapes.add(tuple(int(value) for value in timeline.shape[1:]))
        if len(latent_shapes) != 1:
            raise ValueError("all temporal episodes must use the same latent shape")

        self.episodes = episodes
        self.latents = latents
        self.context_frames = context_frames
        self.selection_policy = selection_policy
        self.latent_shape = latent_shapes.pop()
        candidates: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(episodes):
            # start:start+context_frames are exactly the transitions from the
            # first context frame through the target frame.
            for start in range(0, len(episode.frames) - context_frames):
                if episode.valid[start : start + context_frames].all():
                    candidates.append((episode_index, start))
        if maximum_sequences is not None and len(candidates) > maximum_sequences:
            generator = torch.Generator().manual_seed(seed)
            if selection_policy == "random":
                selected = torch.randperm(len(candidates), generator=generator)[:maximum_sequences]
                candidates = [candidates[int(index)] for index in selected]
            else:
                by_bucket: dict[str, list[tuple[int, int]]] = {
                    bucket: [] for bucket in ACTION_BUCKETS
                }
                for episode_index, start in candidates:
                    target_action = episodes[episode_index].actions[start + context_frames - 1]
                    by_bucket[action_bucket(target_action)].append((episode_index, start))
                minimum_control_examples = max(1, maximum_sequences // 8)
                for bucket in ACTION_BUCKETS[:3]:
                    if len(by_bucket[bucket]) < minimum_control_examples:
                        raise ValueError(
                            f"action-balanced subset needs {minimum_control_examples} "
                            f"{bucket} examples but found {len(by_bucket[bucket])}"
                        )
                quota = maximum_sequences // len(ACTION_BUCKETS)
                chosen: list[tuple[int, int]] = []
                leftovers: list[tuple[int, int]] = []
                for bucket in ACTION_BUCKETS:
                    bucket_candidates = by_bucket[bucket]
                    order = torch.randperm(len(bucket_candidates), generator=generator).tolist()
                    shuffled = [bucket_candidates[index] for index in order]
                    chosen.extend(shuffled[:quota])
                    leftovers.extend(shuffled[quota:])
                remaining = maximum_sequences - len(chosen)
                if remaining:
                    order = torch.randperm(len(leftovers), generator=generator)[:remaining]
                    chosen.extend(leftovers[int(index)] for index in order)
                final_order = torch.randperm(len(chosen), generator=generator)
                candidates = [chosen[int(index)] for index in final_order]
        if not candidates:
            raise ValueError("no clean temporal latent sequences were found")
        self.index = candidates

    @classmethod
    @torch.no_grad()
    def from_paths(
        cls,
        paths: list[Path],
        autoencoder: nn.Module,
        device: torch.device,
        *,
        context_frames: int = 8,
        maximum_sequences: int | None = None,
        encode_batch_size: int = 128,
        selection_policy: Literal["random", "action_balanced"] = "random",
        minimum_episodes: int = 1,
        seed: int = 7,
    ) -> TemporalLatentDataset:
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        if minimum_episodes < 1:
            raise ValueError("minimum_episodes must be positive")
        ordered_paths = list(paths)
        if maximum_sequences is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(ordered_paths)
        autoencoder.eval()
        episodes: list[ProcessedEpisode] = []
        timelines: list[torch.Tensor] = []
        available = 0
        for path in ordered_paths:
            episode = ProcessedEpisode.load(path)
            valid = sum(
                bool(episode.valid[start : start + context_frames].all())
                for start in range(0, len(episode.frames) - context_frames)
            )
            if not valid:
                continue
            chunks: list[torch.Tensor] = []
            for start in range(0, len(episode.frames), encode_batch_size):
                frames = episode.frames[start : start + encode_batch_size]
                contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
                batch = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32)
                batch.div_(255.0)
                chunks.append(autoencoder.encode(batch).to(device="cpu", dtype=torch.float16))
            episodes.append(episode)
            timelines.append(torch.cat(chunks))
            available += valid
            if (
                maximum_sequences is not None
                and available >= maximum_sequences
                and len(episodes) >= minimum_episodes
            ):
                break
        if maximum_sequences is not None and available < maximum_sequences:
            raise ValueError(
                f"requested {maximum_sequences} fixed sequences but only found {available}"
            )
        return cls(
            episodes,
            timelines,
            context_frames=context_frames,
            maximum_sequences=maximum_sequences,
            selection_policy=selection_policy,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, start = self.index[item]
        stop = start + self.context_frames
        episode = self.episodes[episode_index]
        return {
            "context_latents": self.latents[episode_index][start:stop].float(),
            "actions": torch.from_numpy(episode.actions[start:stop].astype(np.float32, copy=False)),
            "target_latent": self.latents[episode_index][stop].float(),
            "target_frame": _frame_tensor(episode.frames[stop]),
        }

    @property
    def references(self) -> list[TemporalSequenceReference]:
        return [
            TemporalSequenceReference(
                self.episodes[episode_index].episode,
                start,
                action_bucket(
                    self.episodes[episode_index].actions[start + self.context_frames - 1]
                ),
            )
            for episode_index, start in self.index
        ]

    @property
    def action_bucket_counts(self) -> dict[str, int]:
        counts = {bucket: 0 for bucket in ACTION_BUCKETS}
        for reference in self.references:
            counts[reference.action_bucket] += 1
        return counts

    @property
    def encoded_frames(self) -> int:
        return sum(len(timeline) for timeline in self.latents)

    def normalization_statistics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-channel latent and per-action statistics over the fixed windows."""
        latent_batches: list[torch.Tensor] = []
        action_batches: list[torch.Tensor] = []
        for item in range(len(self)):
            sample = self[item]
            latent_batches.append(
                torch.cat((sample["context_latents"], sample["target_latent"][None]))
            )
            action_batches.append(sample["actions"])
        latents = torch.cat(latent_batches).float()
        actions = torch.cat(action_batches).float()
        latent_mean = latents.mean(dim=(0, 2, 3))
        latent_std = latents.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-4)
        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0, unbiased=False).clamp_min(0.05)
        return latent_mean, latent_std, action_mean, action_std


def _frame_tensor(frame: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(frame.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).float().div_(255.0)


class LinearVelocitySchedule(nn.Module):
    """GameNGen-style linear VP schedule and exact velocity identities."""

    def __init__(
        self,
        diffusion_steps: int = 1_000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("linear beta bounds must satisfy 0 < start < end < 1")
        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float64)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        # Training index zero is slightly noisy; sampling performs a final x0
        # projection after that denoiser call.
        self.diffusion_steps = diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.register_buffer("alpha", alpha_bar.sqrt().float())
        self.register_buffer("sigma", (1 - alpha_bar).sqrt().float())

    def _coefficients(
        self, timesteps: torch.Tensor, dimensions: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be [batch]")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.diffusion_steps):
            raise ValueError("diffusion timestep is out of range")
        shape = (len(timesteps),) + (1,) * (dimensions - 1)
        return self.alpha[timesteps].reshape(shape), self.sigma[timesteps].reshape(shape)

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean sample and noise must have matching shapes")
        alpha, sigma = self._coefficients(timesteps, clean.ndim)
        return alpha * clean + sigma * noise

    def velocity(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha, sigma = self._coefficients(timesteps, clean.ndim)
        return alpha * noise - sigma * clean

    def clean_from_velocity(
        self, noised: torch.Tensor, velocity: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha, sigma = self._coefficients(timesteps, noised.ndim)
        return alpha * noised - sigma * velocity

    def noise_from_velocity(
        self, noised: torch.Tensor, velocity: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha, sigma = self._coefficients(timesteps, noised.ndim)
        return sigma * noised + alpha * velocity

    def sampling_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        if not 1 <= steps <= self.diffusion_steps:
            raise ValueError("sampling steps must be between one and diffusion_steps")
        return (
            torch.linspace(self.diffusion_steps - 1, 0, steps, device=device, dtype=torch.float32)
            .round()
            .long()
        )


def _sinusoidal_embedding(values: torch.Tensor, dimensions: int) -> torch.Tensor:
    if dimensions < 2 or dimensions % 2:
        raise ValueError("embedding dimensions must be a positive even number")
    half = dimensions // 2
    scales = torch.exp(
        -math.log(10_000) * torch.arange(half, device=values.device) / max(half - 1, 1)
    )
    angles = values.float()[:, None] * scales[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConditionedResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, condition_dim: int):
        super().__init__()
        self.input_norm = nn.GroupNorm(_groups(input_channels), input_channels)
        self.input_conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.output_norm = nn.GroupNorm(_groups(output_channels), output_channels)
        self.condition = nn.Linear(condition_dim, output_channels * 2)
        self.output_conv = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_conv(torch.nn.functional.silu(self.input_norm(features)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.output_norm(hidden) * (1 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.output_conv(torch.nn.functional.silu(hidden))
        return self.skip(features) + hidden


class ActionCrossAttention2d(nn.Module):
    """Let one spatial U-Net resolution attend to the complete action history."""

    def __init__(self, channels: int, action_token_dim: int, attention_heads: int):
        super().__init__()
        if channels % attention_heads:
            raise ValueError("attention channels must divide evenly across heads")
        self.spatial_norm = nn.LayerNorm(channels)
        self.action_norm = nn.LayerNorm(action_token_dim)
        self.action_projection = nn.Linear(action_token_dim, channels)
        self.attention = nn.MultiheadAttention(channels, attention_heads, batch_first=True)

    def forward(self, features: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        spatial_shape = features.shape
        spatial_tokens = features.flatten(2).transpose(1, 2)
        projected_actions = self.action_projection(self.action_norm(action_tokens))
        attended, _ = self.attention(
            self.spatial_norm(spatial_tokens),
            projected_actions,
            projected_actions,
            need_weights=False,
        )
        return (spatial_tokens + attended).transpose(1, 2).reshape(spatial_shape)


class TemporalActionUNet(nn.Module):
    """16x16 latent U-Net with multi-resolution action cross-attention."""

    def __init__(
        self,
        latent_channels: int = 16,
        action_dim: int = 9,
        context_frames: int = 8,
        base_channels: int = 112,
        attention_heads: int = 8,
        diffusion_steps: int = 1_000,
        *,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if min(latent_channels, action_dim, context_frames, base_channels) < 1:
            raise ValueError("all temporal U-Net dimensions must be positive")
        bottleneck_channels = base_channels * 4
        if bottleneck_channels % attention_heads:
            raise ValueError("bottleneck channels must divide evenly across attention heads")
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.context_frames = context_frames
        self.base_channels = base_channels
        self.attention_heads = attention_heads
        self.schedule = LinearVelocitySchedule(diffusion_steps)

        def statistic(value: torch.Tensor | None, size: int, fill: float) -> torch.Tensor:
            result = torch.full((size,), fill) if value is None else value.detach().float().clone()
            if result.shape != (size,):
                raise ValueError("normalization statistic has the wrong shape")
            return result

        latent_mean = statistic(latent_mean, latent_channels, 0.0)
        latent_std = statistic(latent_std, latent_channels, 1.0)
        action_mean = statistic(action_mean, action_dim, 0.0)
        action_std = statistic(action_std, action_dim, 1.0)
        if torch.any(latent_std <= 0) or torch.any(action_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

        condition_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.context_noise_mlp = nn.Sequential(
            nn.Linear(base_channels, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        )
        self.action_positions = nn.Parameter(torch.zeros(1, context_frames, bottleneck_channels))

        input_channels = latent_channels * (context_frames + 1)
        self.input_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.down1a = ConditionedResidualBlock(base_channels, base_channels, condition_dim)
        self.down1b = ConditionedResidualBlock(base_channels, base_channels, condition_dim)
        self.action_attention1 = ActionCrossAttention2d(
            base_channels, bottleneck_channels, attention_heads
        )
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down2a = ConditionedResidualBlock(base_channels * 2, base_channels * 2, condition_dim)
        self.down2b = ConditionedResidualBlock(base_channels * 2, base_channels * 2, condition_dim)
        self.action_attention2 = ActionCrossAttention2d(
            base_channels * 2, bottleneck_channels, attention_heads
        )
        self.downsample2 = nn.Conv2d(base_channels * 2, bottleneck_channels, 4, stride=2, padding=1)
        self.middle1 = ConditionedResidualBlock(
            bottleneck_channels, bottleneck_channels, condition_dim
        )
        self.action_attention_middle = ActionCrossAttention2d(
            bottleneck_channels, bottleneck_channels, attention_heads
        )
        self.middle2 = ConditionedResidualBlock(
            bottleneck_channels, bottleneck_channels, condition_dim
        )
        self.upsample2 = nn.ConvTranspose2d(
            bottleneck_channels, base_channels * 2, 4, stride=2, padding=1
        )
        self.up2 = ConditionedResidualBlock(base_channels * 4, base_channels * 2, condition_dim)
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 4, stride=2, padding=1
        )
        self.up1 = ConditionedResidualBlock(base_channels * 2, base_channels, condition_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _latent_statistics(self, dimensions: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1,) * (dimensions - 3) + (-1, 1, 1)
        return self.latent_mean.reshape(shape), self.latent_std.reshape(shape)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_statistics(latent.ndim)
        return (latent - mean) / std

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_statistics(latent.ndim)
        return latent * std + mean

    def forward(
        self,
        noised_target: torch.Tensor,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        context_noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = noised_target.shape
        expected_context = (batch, self.context_frames, channels, height, width)
        if channels != self.latent_channels or context_latents.shape != expected_context:
            raise ValueError("temporal context has the wrong latent shape")
        if actions.shape != (batch, self.context_frames, self.action_dim):
            raise ValueError("actions must be [batch, context_frames, action_dim]")
        if timesteps.shape != (batch,) or context_noise_levels.shape != (batch,):
            raise ValueError("timestep and context noise levels must be [batch]")

        time_values = timesteps.float() / max(self.schedule.diffusion_steps - 1, 1) * 1_000
        condition = self.time_mlp(_sinusoidal_embedding(time_values, self.base_channels))
        condition = condition + self.context_noise_mlp(
            _sinusoidal_embedding(context_noise_levels * 1_000, self.base_channels)
        )
        normalized_actions = (actions - self.action_mean) / self.action_std
        action_tokens = self.action_mlp(normalized_actions) + self.action_positions

        flattened_context = context_latents.reshape(batch, -1, height, width)
        features = self.input_conv(torch.cat((noised_target, flattened_context), dim=1))
        skip1 = self.down1b(self.down1a(features, condition), condition)
        skip1 = self.action_attention1(skip1, action_tokens)
        features = self.downsample1(skip1)
        skip2 = self.down2b(self.down2a(features, condition), condition)
        skip2 = self.action_attention2(skip2, action_tokens)
        features = self.downsample2(skip2)
        features = self.middle1(features, condition)
        features = self.action_attention_middle(features, action_tokens)
        features = self.middle2(features, condition)
        features = self.upsample2(features)
        features = self.up2(torch.cat((features, skip2), dim=1), condition)
        features = self.upsample1(features)
        features = self.up1(torch.cat((features, skip1), dim=1), condition)
        return self.output(features)

    @torch.no_grad()
    def sample(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        steps: int = 8,
        seed: int = 7,
        context_noise_level: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling for a fixed seed (eta=0)."""
        if context_latents.ndim != 5:
            raise ValueError("context_latents must be [batch, time, channels, height, width]")
        device = context_latents.device
        normalized_context = self.normalize_latent(context_latents)
        batch = len(context_latents)
        sample_shape = (
            batch,
            self.latent_channels,
            context_latents.shape[-2],
            context_latents.shape[-1],
        )
        if initial_noise is None:
            generator = torch.Generator().manual_seed(seed)
            current = torch.randn(sample_shape, generator=generator).to(device)
        else:
            if initial_noise.shape != sample_shape:
                raise ValueError("initial_noise has the wrong latent shape")
            current = initial_noise.to(device=device, dtype=context_latents.dtype).clone()
        sampling_timesteps = self.schedule.sampling_timesteps(steps, device)
        context_levels = torch.full(
            (batch,), context_noise_level, device=device, dtype=context_latents.dtype
        )
        for index, timestep in enumerate(sampling_timesteps):
            timesteps = timestep.expand(batch)
            velocity = self(current, normalized_context, actions, timesteps, context_levels)
            clean = self.schedule.clean_from_velocity(current, velocity, timesteps)
            if index + 1 == len(sampling_timesteps):
                current = clean
                break
            noise = self.schedule.noise_from_velocity(current, velocity, timesteps)
            next_timestep = sampling_timesteps[index + 1].expand(batch)
            alpha, sigma = self.schedule._coefficients(next_timestep, current.ndim)
            current = alpha * clean + sigma * noise
        return self.denormalize_latent(current)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def counterfactual_action_scripts(
    steps: int,
    device: torch.device,
    *,
    action_dim: int = 9,
    camera_step: float = 30.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Constant forward, look-left, and look-right scripts for one shared seed."""
    if steps < 1:
        raise ValueError("counterfactual rollout steps must be positive")
    if action_dim != 9:
        raise ValueError("Minecraft counterfactual scripts require nine-value actions")
    scripts = torch.zeros(3, steps, action_dim, device=device, dtype=dtype)
    scripts[0, :, 0] = 1
    scripts[0, :, 5] = 1
    scripts[1, :, -2] = -camera_step
    scripts[2, :, -2] = camera_step
    return scripts


@torch.no_grad()
def most_structured_seed(autoencoder: nn.Module, context_latents: torch.Tensor) -> int:
    """Index of the context whose final frame carries the most visible structure.

    The counterfactual rollout is only readable when the seed scene has edges to
    move. Seeding from an arbitrary window can land on a night or cave frame
    where no camera motion could be seen regardless of model quality.
    """
    if context_latents.ndim != 5:
        raise ValueError("context_latents must be [batch, time, channels, height, width]")
    frames = autoencoder.decode(context_latents[:, -1]).clamp(0, 1)
    horizontal, vertical = image_gradients(frames)
    energy = horizontal.abs().mean(dim=(1, 2, 3)) + vertical.abs().mean(dim=(1, 2, 3))
    return int(energy.argmax())


@torch.no_grad()
def autoregressive_action_rollout(
    model: TemporalActionUNet,
    context_latents: torch.Tensor,
    action_history: torch.Tensor,
    future_actions: torch.Tensor,
    *,
    sampling_steps: int = 8,
    seed: int = 7,
    shared_noise_across_batch: bool = False,
    context_noise_level: float = 0.0,
) -> torch.Tensor:
    """Generate a future while shifting generated latents back into context.

    ``context_noise_level`` is the corruption level reported to the model. After
    the first step the context holds generated latents, so a model trained with
    context noise augmentation should be told the context is imperfect.
    """
    if context_latents.ndim != 5:
        raise ValueError("context_latents must be [batch, time, channels, height, width]")
    batch = len(context_latents)
    if action_history.shape != (batch, model.context_frames, model.action_dim):
        raise ValueError("action_history has the wrong temporal shape")
    if future_actions.ndim != 3 or future_actions.shape[:1] != (batch,):
        raise ValueError("future_actions must be [batch, steps, action_dim]")
    if future_actions.shape[2] != model.action_dim or future_actions.shape[1] < 1:
        raise ValueError("future_actions has the wrong action shape")

    context = context_latents.clone()
    history = action_history.clone()
    generated: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    for step in range(future_actions.shape[1]):
        history[:, -1] = future_actions[:, step]
        noise_batch = 1 if shared_noise_across_batch else batch
        initial_noise = torch.randn(
            (
                noise_batch,
                model.latent_channels,
                context.shape[-2],
                context.shape[-1],
            ),
            generator=generator,
        )
        if shared_noise_across_batch:
            initial_noise = initial_noise.repeat(batch, 1, 1, 1)
        next_latent = model.sample(
            context,
            history,
            steps=sampling_steps,
            initial_noise=initial_noise,
            context_noise_level=0.0 if step == 0 else context_noise_level,
        )
        generated.append(next_latent)
        context = torch.cat((context[:, 1:], next_latent[:, None]), dim=1)
        history = torch.cat((history[:, 1:], torch.zeros_like(history[:, :1])), dim=1)
    return torch.stack(generated, dim=1)


def corrupt_context(
    context: torch.Tensor, levels: torch.Tensor, *, noise: torch.Tensor | None = None
) -> torch.Tensor:
    if levels.shape != (len(context),):
        raise ValueError("context noise levels must be [batch]")
    if torch.any(levels < 0) or torch.any(levels >= 1):
        raise ValueError("context noise levels must be in [0, 1)")
    if noise is None:
        noise = torch.randn_like(context)
    if noise.shape != context.shape:
        raise ValueError("context and context noise must have matching shapes")
    shape = (len(context),) + (1,) * (context.ndim - 1)
    sigma = levels.reshape(shape)
    return (1 - sigma.square()).sqrt() * context + sigma * noise


def draw_context_noise_levels(
    batch_size: int, device: torch.device, maximum: float
) -> torch.Tensor:
    """25% clean, 60% lightly noisy, and 15% more corrupted contexts."""
    if not 0 <= maximum < 1:
        raise ValueError("maximum context noise must be in [0, 1)")
    if maximum == 0:
        return torch.zeros(batch_size, device=device)
    mixture = torch.rand(batch_size, device=device)
    uniform = torch.rand(batch_size, device=device)
    light_max = min(maximum, 0.15)
    levels = torch.where(
        mixture < 0.25,
        torch.zeros_like(uniform),
        torch.where(
            mixture < 0.85,
            uniform * light_max,
            light_max + uniform * (maximum - light_max),
        ),
    )
    return levels


def diffusion_loss(
    model: TemporalActionUNet,
    batch: dict[str, torch.Tensor],
    *,
    maximum_context_noise: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    target = model.normalize_latent(batch["target_latent"])
    context = model.normalize_latent(batch["context_latents"])
    if seed is None:
        timesteps = torch.randint(
            model.schedule.diffusion_steps, (len(target),), device=target.device
        )
        noise = torch.randn_like(target)
        levels = draw_context_noise_levels(len(target), target.device, maximum_context_noise)
        context_noise = torch.randn_like(context)
    else:
        generator = torch.Generator().manual_seed(seed)
        timesteps = torch.randint(
            model.schedule.diffusion_steps, (len(target),), generator=generator
        ).to(target.device)
        noise = torch.randn(target.shape, generator=generator).to(target.device)
        # Fixed evaluation stays clean unless augmentation is explicitly requested.
        levels = torch.zeros(len(target), device=target.device)
        context_noise = torch.randn(context.shape, generator=generator).to(target.device)
    noised = model.schedule.add_noise(target, noise, timesteps)
    context = corrupt_context(context, levels, noise=context_noise)
    expected_velocity = model.schedule.velocity(target, noise, timesteps)
    predicted_velocity = model(noised, context, batch["actions"], timesteps, levels)
    return nn.functional.mse_loss(predicted_velocity, expected_velocity)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def fixed_denoising_loss(
    model: TemporalActionUNet,
    dataset: TemporalLatentDataset,
    device: torch.device,
    *,
    batch_size: int,
    shuffle_actions: bool = False,
    seed: int = 7,
) -> float:
    model.eval()
    total = 0.0
    examples = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        if shuffle_actions:
            batch["actions"] = batch["actions"].roll(1, dims=0)
        loss = diffusion_loss(model, batch, seed=seed + index)
        total += float(loss) * len(batch["actions"])
        examples += len(batch["actions"])
    return total / examples


@dataclass(frozen=True)
class TinyOverfitMetrics:
    sequences: int
    initial_denoising_loss: float
    final_denoising_loss: float
    loss_reduction_percent: float
    shuffled_action_loss: float
    shuffled_action_degradation_percent: float
    shuffled_action_sample_degradation_percent: float
    sample_pixel_mse: float
    sample_pixel_psnr_db: float
    copy_pixel_psnr_db: float
    sample_vs_copy_mse_improvement_percent: float
    oracle_pixel_psnr_db: float
    sample_edge_ratio: float
    oracle_edge_ratio: float
    one_step_action_effect_pixel_l1: float
    five_step_action_effect_pixel_l1: float
    five_step_total_drift_pixel_l1: float
    action_share_of_drift_percent: float
    counterfactual_camera_step: float
    counterfactual_rollout_steps: int
    action_comparison_shared_initial_noise: bool
    sampled_latents_finite: bool
    sampled_latent_min: float
    sampled_latent_max: float


@torch.no_grad()
def evaluate_tiny_overfit(
    model: TemporalActionUNet,
    autoencoder: nn.Module,
    dataset: TemporalLatentDataset,
    device: torch.device,
    *,
    batch_size: int,
    sampling_steps: int,
    seed: int,
    maximum_samples: int = 32,
    rollout_context_noise: float = 0.0,
) -> TinyOverfitMetrics:
    model.eval()
    autoencoder.eval()
    count = min(maximum_samples, len(dataset))
    raw = [dataset[index] for index in range(count)]
    batch = {name: torch.stack([sample[name] for sample in raw]).to(device) for name in raw[0]}
    sampled_latents = model.sample(
        batch["context_latents"], batch["actions"], steps=sampling_steps, seed=seed
    )
    sampled_frames = autoencoder.decode(sampled_latents).clamp(0, 1)
    copy_frames = autoencoder.decode(batch["context_latents"][:, -1]).clamp(0, 1)
    oracle_frames = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
    target_frames = batch["target_frame"]
    sample_mse = nn.functional.mse_loss(sampled_frames, target_frames).item()
    copy_mse = nn.functional.mse_loss(copy_frames, target_frames).item()
    oracle_mse = nn.functional.mse_loss(oracle_frames, target_frames).item()
    target_dx, target_dy = image_gradients(target_frames)
    sample_dx, sample_dy = image_gradients(sampled_frames)
    oracle_dx, oracle_dy = image_gradients(oracle_frames)
    target_edges = target_dx.abs().sum() + target_dy.abs().sum()
    sample_edges = sample_dx.abs().sum() + sample_dy.abs().sum()
    oracle_edges = oracle_dx.abs().sum() + oracle_dy.abs().sum()

    shuffled_latents = model.sample(
        batch["context_latents"],
        batch["actions"].roll(1, dims=0),
        steps=sampling_steps,
        seed=seed,
    )
    shuffled_frames = autoencoder.decode(shuffled_latents).clamp(0, 1)
    shuffled_sample_mse = nn.functional.mse_loss(shuffled_frames, target_frames).item()

    rollout_steps = 5
    seed_index = most_structured_seed(autoencoder, batch["context_latents"])
    variant_context = batch["context_latents"][seed_index : seed_index + 1].repeat(3, 1, 1, 1, 1)
    variant_history = batch["actions"][seed_index : seed_index + 1].repeat(3, 1, 1)
    scripts = counterfactual_action_scripts(rollout_steps, device, dtype=batch["actions"].dtype)
    variant_latents = autoregressive_action_rollout(
        model,
        variant_context,
        variant_history,
        scripts,
        sampling_steps=sampling_steps,
        seed=seed + 10_000,
        shared_noise_across_batch=True,
        context_noise_level=rollout_context_noise,
    )
    decoded_variants = autoencoder.decode(variant_latents.flatten(0, 1)).clamp(0, 1)
    variant_frames = decoded_variants.reshape(3, rollout_steps, *decoded_variants.shape[1:])
    seed_frame = autoencoder.decode(variant_context[:1, -1]).clamp(0, 1)[0]
    total_drift = float((variant_frames[0, -1] - seed_frame).abs().mean())

    def pairwise_effect(frames: torch.Tensor) -> float:
        return float(
            np.mean(
                [
                    nn.functional.l1_loss(frames[first], frames[second]).item()
                    for first, second in ((0, 1), (0, 2), (1, 2))
                ]
            )
        )

    final_loss = fixed_denoising_loss(model, dataset, device, batch_size=batch_size, seed=seed)
    shuffled_loss = fixed_denoising_loss(
        model,
        dataset,
        device,
        batch_size=batch_size,
        shuffle_actions=True,
        seed=seed,
    )
    # Initial loss is filled by the training function after this evaluation.
    return TinyOverfitMetrics(
        sequences=len(dataset),
        initial_denoising_loss=float("nan"),
        final_denoising_loss=final_loss,
        loss_reduction_percent=float("nan"),
        shuffled_action_loss=shuffled_loss,
        shuffled_action_degradation_percent=(shuffled_loss / max(final_loss, 1e-12) - 1) * 100,
        shuffled_action_sample_degradation_percent=(
            shuffled_sample_mse / max(sample_mse, 1e-12) - 1
        )
        * 100,
        sample_pixel_mse=sample_mse,
        sample_pixel_psnr_db=10 * math.log10(1 / max(sample_mse, 1e-12)),
        copy_pixel_psnr_db=10 * math.log10(1 / max(copy_mse, 1e-12)),
        sample_vs_copy_mse_improvement_percent=(1 - sample_mse / max(copy_mse, 1e-12)) * 100,
        oracle_pixel_psnr_db=10 * math.log10(1 / max(oracle_mse, 1e-12)),
        sample_edge_ratio=float(sample_edges / target_edges.clamp_min(1e-12)),
        oracle_edge_ratio=float(oracle_edges / target_edges.clamp_min(1e-12)),
        one_step_action_effect_pixel_l1=pairwise_effect(variant_frames[:, 0]),
        five_step_action_effect_pixel_l1=pairwise_effect(variant_frames[:, -1]),
        five_step_total_drift_pixel_l1=total_drift,
        action_share_of_drift_percent=(
            pairwise_effect(variant_frames[:, -1]) / max(total_drift, 1e-12) * 100
        ),
        counterfactual_camera_step=float(scripts[2, 0, -2]),
        counterfactual_rollout_steps=rollout_steps,
        action_comparison_shared_initial_noise=True,
        sampled_latents_finite=bool(
            torch.isfinite(sampled_latents).all() and torch.isfinite(variant_latents).all()
        ),
        sampled_latent_min=float(sampled_latents.min()),
        sampled_latent_max=float(sampled_latents.max()),
    )


def _save_checkpoint(
    path: Path,
    model: TemporalActionUNet,
    *,
    history: dict[str, list[float]],
    autoencoder_checkpoint: Path,
    manifest_path: Path,
    references: list[TemporalSequenceReference],
    selection_policy: str,
    action_bucket_counts: dict[str, int],
    source_episodes: list[str],
    sampling_steps: int,
    training_steps: int,
    maximum_context_noise: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": "temporal_latent_diffusion_v2",
            "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
            "model_state": model.state_dict(),
            "latent_channels": model.latent_channels,
            "action_dim": model.action_dim,
            "context_frames": model.context_frames,
            "base_channels": model.base_channels,
            "attention_heads": model.attention_heads,
            "diffusion_steps": model.schedule.diffusion_steps,
            "noise_schedule": "linear_beta",
            "beta_start": model.schedule.beta_start,
            "beta_end": model.schedule.beta_end,
            "sampling_steps": sampling_steps,
            "training_steps": training_steps,
            "maximum_context_noise": maximum_context_noise,
            "seed": seed,
            "history": history,
            "fixed_sequence_references": [asdict(reference) for reference in references],
            "selection_policy": selection_policy,
            "action_bucket_counts": action_bucket_counts,
            "source_episodes": source_episodes,
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
        },
        path,
    )


def load_latent_diffusion_v2_checkpoint(
    path: Path, device: torch.device
) -> tuple[TemporalActionUNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "temporal_latent_diffusion_v2":
        raise ValueError("checkpoint is not V2 temporal latent diffusion")
    if checkpoint.get("architecture") != LATENT_DIFFUSION_V2_ARCHITECTURE:
        raise ValueError("V2 checkpoint uses an incompatible or unversioned architecture")
    if checkpoint.get("noise_schedule") != "linear_beta":
        raise ValueError("V2 checkpoint does not use the versioned linear schedule")
    state = checkpoint["model_state"]
    model = TemporalActionUNet(
        latent_channels=int(checkpoint["latent_channels"]),
        action_dim=int(checkpoint["action_dim"]),
        context_frames=int(checkpoint["context_frames"]),
        base_channels=int(checkpoint["base_channels"]),
        attention_heads=int(checkpoint["attention_heads"]),
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        latent_mean=state["latent_mean"],
        latent_std=state["latent_std"],
        action_mean=state["action_mean"],
        action_std=state["action_std"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _save_training_curve(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["step"], history["train"], label="training velocity MSE")
    axis.plot(history["evaluation_step"], history["fixed"], marker="o", label="fixed-noise MSE")
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("velocity prediction MSE")
    axis.set_title("V2 fixed-256 overfit gate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_frame(frame: np.ndarray, tile: int) -> np.ndarray:
    rgb = np.clip(frame * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (tile, tile), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def save_overfit_visuals(
    model: TemporalActionUNet,
    autoencoder: nn.Module,
    dataset: TemporalLatentDataset,
    sample_path: Path,
    action_path: Path,
    device: torch.device,
    *,
    sampling_steps: int,
    seed: int,
    count: int = 6,
    rollout_context_noise: float = 0.0,
) -> None:
    model.eval()
    count = min(count, len(dataset))
    items = [dataset[index] for index in np.linspace(0, len(dataset) - 1, count, dtype=int)]
    batch = {name: torch.stack([item[name] for item in items]).to(device) for name in items[0]}
    sampled = model.sample(
        batch["context_latents"], batch["actions"], steps=sampling_steps, seed=seed
    )
    last_context = autoencoder.decode(batch["context_latents"][:, -1]).clamp(0, 1)
    oracle = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
    generated = autoencoder.decode(sampled).clamp(0, 1)
    columns = (last_context, batch["target_frame"], oracle, generated)
    labels = ("last context", "real target", "decoder oracle", f"V2 sample ({sampling_steps} DDIM)")
    tile, header = 160, 26
    canvas = np.full((count * (tile + header), len(columns) * tile, 3), 24, np.uint8)
    for row in range(count):
        for column, (frames, label) in enumerate(zip(columns, labels, strict=True)):
            x, y = column * tile, row * (tile + header)
            cv2.putText(
                canvas,
                label,
                (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            frame = frames[row].cpu().permute(1, 2, 0).numpy()
            canvas[y + header : y + header + tile, x : x + tile] = _render_frame(frame, tile)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(sample_path), canvas):
        raise ValueError(f"could not write {sample_path}")

    rollout_steps = 5
    seed_index = most_structured_seed(autoencoder, batch["context_latents"])
    history = batch["actions"][seed_index : seed_index + 1].repeat(3, 1, 1)
    context = batch["context_latents"][seed_index : seed_index + 1].repeat(3, 1, 1, 1, 1)
    scripts = counterfactual_action_scripts(rollout_steps, device, dtype=batch["actions"].dtype)
    variants = autoregressive_action_rollout(
        model,
        context,
        history,
        scripts,
        sampling_steps=sampling_steps,
        seed=seed + 10_000,
        shared_noise_across_batch=True,
        context_noise_level=rollout_context_noise,
    )
    decoded = autoencoder.decode(variants.flatten(0, 1)).clamp(0, 1)
    variant_frames = decoded.reshape(3, rollout_steps, *decoded.shape[1:])
    seed_frame = autoencoder.decode(context[:1, -1]).clamp(0, 1)[0]
    row_labels = ("forward + sprint", "look left", "look right")
    action_tile = 128
    columns = rollout_steps + 1
    action_canvas = np.full((3 * (action_tile + header), columns * action_tile, 3), 24, np.uint8)
    for row, row_label in enumerate(row_labels):
        frames = torch.cat((seed_frame[None], variant_frames[row]), dim=0)
        for column, frame_tensor in enumerate(frames):
            x, y = column * action_tile, row * (action_tile + header)
            label = f"{row_label} | seed" if column == 0 else f"t+{column}"
            cv2.putText(
                action_canvas,
                label,
                (x + 5, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            frame = frame_tensor.cpu().permute(1, 2, 0).numpy()
            action_canvas[y + header : y + header + action_tile, x : x + action_tile] = (
                _render_frame(frame, action_tile)
            )
    if not cv2.imwrite(str(action_path), action_canvas):
        raise ValueError(f"could not write {action_path}")


@dataclass(frozen=True)
class TinyOverfitResult:
    checkpoint: Path
    metrics_path: Path
    training_curve: Path
    samples: Path
    action_comparison: Path
    metrics: TinyOverfitMetrics
    parameter_count: int
    device: str


def overfit_latent_diffusion_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    sequences: int = 256,
    training_steps: int = 2_000,
    batch_size: int = 8,
    encode_batch_size: int = 128,
    minimum_episodes: int = 8,
    base_channels: int = 112,
    attention_heads: int = 8,
    diffusion_steps: int = 1_000,
    sampling_steps: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    maximum_context_noise: float = 0.0,
    rollout_context_noise: float = 0.0,
    seed: int = 7,
    requested_device: str = "auto",
) -> TinyOverfitResult:
    if min(sequences, training_steps, batch_size, sampling_steps) < 1:
        raise ValueError("overfit sizes must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    train_paths = DatasetManifest.load(manifest_path).processed_paths(processed_dir, "training")
    print(f"encoding the fixed {sequences}-sequence Stage 2 subset...")
    dataset = TemporalLatentDataset.from_paths(
        train_paths,
        autoencoder,
        device,
        maximum_sequences=sequences,
        encode_batch_size=encode_batch_size,
        selection_policy="action_balanced",
        minimum_episodes=minimum_episodes,
        seed=seed,
    )
    if len(dataset) != sequences:
        raise RuntimeError("the Stage 2 subset is not the requested fixed size")
    print(f"source episodes: {len(dataset.episodes)}")
    print(f"action buckets: {dataset.action_bucket_counts}")
    latent_mean, latent_std, action_mean, action_std = dataset.normalization_statistics()
    model = TemporalActionUNet(
        latent_channels=dataset.latent_shape[0],
        action_dim=9,
        context_frames=dataset.context_frames,
        base_channels=base_channels,
        attention_heads=attention_heads,
        diffusion_steps=diffusion_steps,
        latent_mean=latent_mean,
        latent_std=latent_std,
        action_mean=action_mean,
        action_std=action_std,
    ).to(device)
    print(f"V2 parameters: {model.parameter_count:,}")
    initial_loss = fixed_denoising_loss(model, dataset, device, batch_size=batch_size, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=True,
    )
    iterator = iter(loader)
    history: dict[str, list[float]] = {
        "step": [],
        "train": [],
        "evaluation_step": [0],
        "fixed": [initial_loss],
    }
    started = time.perf_counter()
    model.train()
    for step in range(1, training_steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion_loss(model, batch, maximum_context_noise=maximum_context_noise)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history["step"].append(step)
        history["train"].append(float(loss.detach()))
        if step == 1 or step % 100 == 0 or step == training_steps:
            fixed = fixed_denoising_loss(model, dataset, device, batch_size=batch_size, seed=seed)
            history["evaluation_step"].append(step)
            history["fixed"].append(fixed)
            elapsed = time.perf_counter() - started
            print(
                f"step {step:5d}/{training_steps}: train={loss.item():.6f}  "
                f"fixed={fixed:.6f}  {step / elapsed:.2f} steps/s"
            )
            model.train()

    metrics = evaluate_tiny_overfit(
        model,
        autoencoder,
        dataset,
        device,
        batch_size=batch_size,
        sampling_steps=sampling_steps,
        seed=seed,
        rollout_context_noise=rollout_context_noise,
    )
    metrics = TinyOverfitMetrics(
        **{
            **asdict(metrics),
            "initial_denoising_loss": initial_loss,
            "loss_reduction_percent": (1 - metrics.final_denoising_loss / max(initial_loss, 1e-12))
            * 100,
        }
    )
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    curve = output_dir / "training-curve.png"
    samples = output_dir / "fixed-256-samples.png"
    actions = output_dir / "five-step-action-rollout.png"
    _save_checkpoint(
        checkpoint,
        model,
        history=history,
        autoencoder_checkpoint=autoencoder_checkpoint,
        manifest_path=manifest_path,
        references=dataset.references,
        selection_policy=dataset.selection_policy,
        action_bucket_counts=dataset.action_bucket_counts,
        source_episodes=[episode.episode for episode in dataset.episodes],
        sampling_steps=sampling_steps,
        training_steps=training_steps,
        maximum_context_noise=maximum_context_noise,
        seed=seed,
    )
    _save_training_curve(history, curve)
    save_overfit_visuals(
        model,
        autoencoder,
        dataset,
        samples,
        actions,
        device,
        sampling_steps=sampling_steps,
        seed=seed,
        rollout_context_noise=rollout_context_noise,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "stage": "Stage 2 - fixed 256-sequence overfit",
                "architecture": LATENT_DIFFUSION_V2_ARCHITECTURE,
                "parameters": model.parameter_count,
                "latent_shape": dataset.latent_shape,
                "context_frames": dataset.context_frames,
                "encoded_frames": dataset.encoded_frames,
                "source_episodes": len(dataset.episodes),
                "selection_policy": dataset.selection_policy,
                "action_bucket_counts": dataset.action_bucket_counts,
                "noise_schedule": "linear_beta",
                "beta_start": model.schedule.beta_start,
                "beta_end": model.schedule.beta_end,
                "training_steps": training_steps,
                "sampling_steps": sampling_steps,
                "maximum_context_noise": maximum_context_noise,
                "rollout_context_noise": rollout_context_noise,
                "device": str(device),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
                "dataset_manifest": str(manifest_path),
                "dataset_manifest_sha256": _file_sha256(manifest_path),
                "metrics": asdict(metrics),
                "gate_checks": {
                    "loss_reduction_substantial": metrics.loss_reduction_percent >= 50,
                    "beats_copy_baseline": (metrics.sample_vs_copy_mse_improvement_percent > 0),
                    "actions_change_output": (
                        metrics.shuffled_action_degradation_percent > 0
                        and metrics.shuffled_action_sample_degradation_percent > 0
                        and metrics.five_step_action_effect_pixel_l1 > 0
                    ),
                    "decoder_received_finite_continuous_latents": (metrics.sampled_latents_finite),
                    "recognizability_visual": str(samples),
                    "semantic_five_step_control_visual": str(actions),
                },
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return TinyOverfitResult(
        checkpoint=checkpoint,
        metrics_path=metrics_path,
        training_curve=curve,
        samples=samples,
        action_comparison=actions,
        metrics=metrics,
        parameter_count=model.parameter_count,
        device=str(device),
    )
