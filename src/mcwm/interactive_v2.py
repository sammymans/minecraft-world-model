"""Interactive and scripted rollouts for the V2 latent-diffusion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mcwm.dataset import ProcessedEpisode
from mcwm.dynamics import _file_sha256
from mcwm.interactive import (
    PlaygroundResult,
    parse_action_script,
    run_action_comparison,
    run_scripted_rollout,
)
from mcwm.latent_diffusion_v2 import TemporalActionUNet, load_latent_diffusion_v2_checkpoint
from mcwm.manifest import DatasetManifest
from mcwm.spatial_training import load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device


@dataclass(frozen=True)
class TemporalRolloutSeed:
    episode: str
    current_step: int
    model_fps: float
    context_frames: np.ndarray
    action_history: np.ndarray

    @property
    def previous_frame(self) -> np.ndarray:
        return self.context_frames[-2]

    @property
    def current_frame(self) -> np.ndarray:
        return self.context_frames[-1]


class TemporalRolloutSeedBank:
    """Clean held-out eight-frame contexts for the V2 playground."""

    def __init__(self, episodes: list[ProcessedEpisode], *, context_frames: int = 8):
        if context_frames < 2:
            raise ValueError("V2 rollout seeds need at least two context frames")
        self.episodes = episodes
        self.context_frames = context_frames
        self.index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(episodes):
            for start in range(len(episode.frames) - context_frames):
                if episode.valid[start : start + context_frames].all():
                    self.index.append((episode_index, start))
        if not self.index:
            raise ValueError("validation data has no clean V2 rollout seeds")

    @classmethod
    def load(
        cls, processed_dir: Path, manifest_path: Path, *, context_frames: int = 8
    ) -> TemporalRolloutSeedBank:
        manifest = DatasetManifest.load(manifest_path)
        paths = manifest.processed_paths(processed_dir, "validation")
        return cls(
            [ProcessedEpisode.load(path) for path in paths], context_frames=context_frames
        )

    def __len__(self) -> int:
        return len(self.index)

    def get(self, index: int) -> TemporalRolloutSeed:
        if not 0 <= index < len(self):
            raise ValueError(f"sample_index must be between 0 and {len(self) - 1}")
        episode_index, start = self.index[index]
        episode = self.episodes[episode_index]
        stop = start + self.context_frames
        return TemporalRolloutSeed(
            episode=episode.episode,
            current_step=stop - 1,
            model_fps=episode.model_fps,
            context_frames=episode.frames[start:stop].copy(),
            # Seven transitions connect eight context observations. The live
            # action becomes the eighth, target-driving action at each step.
            action_history=episode.actions[start : stop - 1].astype(
                np.float32, copy=True
            ),
        )


class InteractiveLatentDiffusionEngine:
    """Recursive V2 state with shared sampling seeds across resets/scripts."""

    def __init__(
        self,
        autoencoder: nn.Module,
        model: TemporalActionUNet,
        context_latents: torch.Tensor,
        action_history: torch.Tensor,
        current_frame: np.ndarray,
        device: torch.device,
        *,
        sampling_steps: int = 8,
        sampling_seed: int = 7,
    ) -> None:
        expected_context = (
            1,
            model.context_frames,
            model.latent_channels,
            context_latents.shape[-2],
            context_latents.shape[-1],
        )
        if context_latents.shape != expected_context:
            raise ValueError("V2 seed context does not match the model")
        expected_actions = (1, model.context_frames - 1, model.action_dim)
        if action_history.shape != expected_actions:
            raise ValueError("V2 seed action history does not match the model")
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        self.autoencoder = autoencoder
        self.dynamics = model
        self.device = device
        self.sampling_steps = sampling_steps
        self.sampling_seed = sampling_seed
        self.seed_context = context_latents.detach().clone()
        self.seed_actions = action_history.detach().clone()
        self.seed_frame = current_frame.copy()
        self.context_latents = self.seed_context.clone()
        self.action_history = self.seed_actions.clone()
        self.current_frame = self.seed_frame.copy()
        self.steps = 0

    @classmethod
    @torch.no_grad()
    def from_seed(
        cls,
        autoencoder: nn.Module,
        model: TemporalActionUNet,
        seed: TemporalRolloutSeed,
        device: torch.device,
        *,
        sampling_steps: int = 8,
        sampling_seed: int = 7,
    ) -> InteractiveLatentDiffusionEngine:
        contiguous = np.ascontiguousarray(seed.context_frames.transpose(0, 3, 1, 2))
        frames = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32).div_(255)
        context = autoencoder.encode(frames)[None]
        actions = torch.from_numpy(seed.action_history).to(device=device)[None]
        return cls(
            autoencoder,
            model,
            context,
            actions,
            seed.current_frame,
            device,
            sampling_steps=sampling_steps,
            sampling_seed=sampling_seed,
        )

    @torch.no_grad()
    def step(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.shape == (self.dynamics.action_dim,):
            action_tensor = action_tensor[None]
        if action_tensor.shape != (1, self.dynamics.action_dim):
            raise ValueError("action must contain one nine-value control vector")
        conditioned_actions = torch.cat((self.action_history, action_tensor[:, None]), dim=1)
        predicted = self.dynamics.sample(
            self.context_latents,
            conditioned_actions,
            steps=self.sampling_steps,
            seed=self.sampling_seed + self.steps,
        )
        decoded = self.autoencoder.decode(predicted).clamp(0, 1)[0]
        frame = decoded.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        self.context_latents = torch.cat((self.context_latents[:, 1:], predicted[:, None]), dim=1)
        self.action_history = torch.cat((self.action_history[:, 1:], action_tensor[:, None]), dim=1)
        self.current_frame = frame
        self.steps += 1
        return frame.copy()

    def reset(self) -> np.ndarray:
        self.context_latents = self.seed_context.clone()
        self.action_history = self.seed_actions.clone()
        self.current_frame = self.seed_frame.copy()
        self.steps = 0
        return self.current_frame.copy()

    @torch.no_grad()
    def reseed(self, seed: TemporalRolloutSeed) -> np.ndarray:
        contiguous = np.ascontiguousarray(seed.context_frames.transpose(0, 3, 1, 2))
        frames = torch.from_numpy(contiguous).to(
            device=self.device, dtype=torch.float32
        ).div_(255)
        self.seed_context = self.autoencoder.encode(frames)[None].detach().clone()
        self.seed_actions = (
            torch.from_numpy(seed.action_history).to(device=self.device)[None].detach().clone()
        )
        self.seed_frame = seed.current_frame.copy()
        return self.reset()


def _load_playground_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    model_checkpoint: Path,
    sample_index: int,
    requested_device: str,
    *,
    sampling_steps: int,
    sampling_seed: int,
) -> tuple[
    InteractiveLatentDiffusionEngine,
    TemporalRolloutSeed,
    TemporalRolloutSeedBank,
]:
    device = choose_device(requested_device)
    model, metadata = load_latent_diffusion_v2_checkpoint(model_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("V2 model belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    if autoencoder.latent_channels != model.latent_channels:
        raise ValueError("autoencoder and V2 model latent channels do not match")
    autoencoder.requires_grad_(False)
    seeds = TemporalRolloutSeedBank.load(
        processed_dir, manifest_path, context_frames=model.context_frames
    )
    seed = seeds.get(sample_index)
    engine = InteractiveLatentDiffusionEngine.from_seed(
        autoencoder,
        model,
        seed,
        device,
        sampling_steps=sampling_steps,
        sampling_seed=sampling_seed,
    )
    return engine, seed, seeds


def compare_action_scripts_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    model_checkpoint: Path,
    scripts: list[str],
    *,
    sample_index: int = 0,
    camera_step: float = 30.0,
    sampling_steps: int = 8,
    sampling_seed: int = 7,
    tile: int = 192,
    output_path: Path = Path("artifacts/interactive-rollout-v2/action-comparison.png"),
    requested_device: str = "auto",
) -> tuple[PlaygroundResult, int]:
    engine, seed, seeds = _load_playground_v2(
        processed_dir,
        manifest_path,
        autoencoder_checkpoint,
        model_checkpoint,
        sample_index,
        requested_device,
        sampling_steps=sampling_steps,
        sampling_seed=sampling_seed,
    )
    result = run_action_comparison(
        engine,  # type: ignore[arg-type]
        seed,  # type: ignore[arg-type]
        scripts,
        output_path,
        camera_step=camera_step,
        tile=tile,
    )
    return result, len(seeds)


def launch_playground_v2(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    model_checkpoint: Path,
    *,
    sample_index: int = 0,
    camera_step: float = 30.0,
    sampling_steps: int = 8,
    sampling_seed: int = 7,
    script: str | None = None,
    output_path: Path = Path("artifacts/interactive-rollout-v2/scripted-rollout.png"),
    requested_device: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> tuple[PlaygroundResult, int]:
    engine, seed, seeds = _load_playground_v2(
        processed_dir,
        manifest_path,
        autoencoder_checkpoint,
        model_checkpoint,
        sample_index,
        requested_device,
        sampling_steps=sampling_steps,
        sampling_seed=sampling_seed,
    )
    if script is not None:
        actions = parse_action_script(script, camera_step=camera_step)
        result = run_scripted_rollout(
            engine, seed, actions, output_path  # type: ignore[arg-type]
        )
    else:
        from mcwm.frontend import serve_rollout_frontend

        result = serve_rollout_frontend(
            engine,  # type: ignore[arg-type]
            seed,  # type: ignore[arg-type]
            seed_index=sample_index,
            seed_count=len(seeds),
            seed_loader=seeds.get,  # type: ignore[arg-type]
            camera_step=camera_step,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    return result, len(seeds)
