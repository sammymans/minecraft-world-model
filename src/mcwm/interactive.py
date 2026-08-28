"""A small interactive playground for recursively imagined Minecraft frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from mcwm.dataset import SequenceDataset
from mcwm.dynamics import _file_sha256, _processed_splits
from mcwm.spatial_dynamics import load_spatial_dynamics_checkpoint
from mcwm.spatial_training import load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device

ACTION_NAMES = (
    "w",
    "a",
    "s",
    "d",
    "jump",
    "sprint",
    "sneak",
    "mouse_dx",
    "mouse_dy",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
BINARY_ACTIONS = ACTION_NAMES[:7]


@dataclass(frozen=True)
class RolloutSeed:
    episode: str
    current_step: int
    model_fps: float
    previous_frame: np.ndarray
    current_frame: np.ndarray


@dataclass(frozen=True)
class PlaygroundResult:
    episode: str
    current_step: int
    steps: int
    device: str
    output: Path | None = None


class RolloutSeedBank:
    """Reusable held-out seeds for switching scenes without reloading the model."""

    def __init__(self, dataset: SequenceDataset):
        if not len(dataset):
            raise ValueError("validation data has no clean rollout seeds")
        self.dataset = dataset

    @classmethod
    def load(cls, processed_dir: Path, manifest_path: Path | None) -> RolloutSeedBank:
        _, validation_paths = _processed_splits(processed_dir, manifest_path)
        return cls(SequenceDataset.from_paths(validation_paths, horizon=1))

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, index: int) -> RolloutSeed:
        if not 0 <= index < len(self):
            raise ValueError(f"sample_index must be between 0 and {len(self) - 1}")
        episode_index, current_step = self.dataset.index[index]
        episode = self.dataset.episodes[episode_index]
        return RolloutSeed(
            episode=episode.episode,
            current_step=current_step,
            model_fps=episode.model_fps,
            previous_frame=episode.frames[current_step - 1].copy(),
            current_frame=episode.frames[current_step].copy(),
        )


def make_action(
    controls: Iterable[str] = (),
    *,
    mouse_dx: float = 0.0,
    mouse_dy: float = 0.0,
) -> np.ndarray:
    """Build one raw nine-value action in the training schema."""
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for control in controls:
        normalized = control.lower()
        if normalized not in BINARY_ACTIONS:
            raise ValueError(f"unknown binary control: {control}")
        action[ACTION_INDEX[normalized]] = 1.0
    action[ACTION_INDEX["mouse_dx"]] = mouse_dx
    action[ACTION_INDEX["mouse_dy"]] = mouse_dy
    return action


def _parse_action_spec(spec: str, camera_step: float) -> np.ndarray:
    controls: set[str] = set()
    mouse_dx = 0.0
    mouse_dy = 0.0
    tokens = [token.strip().lower() for token in spec.split("+") if token.strip()]
    if not tokens:
        raise ValueError("an action specification cannot be empty")
    for token in tokens:
        if token in {"idle", "none"}:
            continue
        if token in BINARY_ACTIONS:
            controls.add(token)
        elif token == "look_left":
            mouse_dx -= camera_step
        elif token == "look_right":
            mouse_dx += camera_step
        elif token == "look_up":
            mouse_dy -= camera_step
        elif token == "look_down":
            mouse_dy += camera_step
        else:
            raise ValueError(f"unknown scripted action: {token}")
    return make_action(controls, mouse_dx=mouse_dx, mouse_dy=mouse_dy)


def parse_action_script(script: str, *, camera_step: float = 30.0) -> np.ndarray:
    """Parse ``w+sprint*5, w+look_right*3, idle*2`` into raw actions."""
    if camera_step <= 0:
        raise ValueError("camera_step must be positive")
    actions: list[np.ndarray] = []
    for item in script.split(","):
        item = item.strip()
        if not item:
            continue
        repeat = 1
        if "*" in item:
            item, raw_repeat = item.rsplit("*", 1)
            try:
                repeat = int(raw_repeat)
            except ValueError as error:
                raise ValueError(f"invalid action repetition: {raw_repeat}") from error
            if repeat < 1:
                raise ValueError("action repetitions must be positive")
        action = _parse_action_spec(item.strip(), camera_step)
        actions.extend(action.copy() for _ in range(repeat))
    if not actions:
        raise ValueError("script must contain at least one action")
    return np.stack(actions)


def select_rollout_seed(
    processed_dir: Path,
    manifest_path: Path | None,
    sample_index: int,
) -> tuple[RolloutSeed, int]:
    """Choose two frames from a clean held-out transition."""
    seeds = RolloutSeedBank.load(processed_dir, manifest_path)
    return seeds.get(sample_index), len(seeds)


class InteractiveRolloutEngine:
    """The tested recursive state transition shared by GUI and scripted modes."""

    def __init__(
        self,
        autoencoder: nn.Module,
        dynamics: nn.Module,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        current_frame: np.ndarray,
        device: torch.device,
    ) -> None:
        if previous_latent.shape != current_latent.shape:
            raise ValueError("seed latents must have identical shapes")
        if hasattr(dynamics, "latent_channels"):
            valid_shape = (
                previous_latent.ndim == 4
                and previous_latent.shape[0] == 1
                and previous_latent.shape[1] == dynamics.latent_channels
            )
        else:
            valid_shape = previous_latent.shape == (1, dynamics.latent_dim)
        if not valid_shape:
            raise ValueError("seed latent shape does not match dynamics model")
        if current_frame.ndim != 3 or current_frame.shape[-1] != 3:
            raise ValueError("current_frame must be [height, width, RGB]")
        self.autoencoder = autoencoder
        self.dynamics = dynamics
        self.device = device
        self.seed_previous = previous_latent.detach().clone()
        self.seed_current = current_latent.detach().clone()
        self.seed_frame = current_frame.copy()
        self.previous_latent = self.seed_previous.clone()
        self.current_latent = self.seed_current.clone()
        self.current_frame = self.seed_frame.copy()
        self.steps = 0

    @classmethod
    @torch.no_grad()
    def from_seed(
        cls,
        autoencoder: nn.Module,
        dynamics: nn.Module,
        seed: RolloutSeed,
        device: torch.device,
    ) -> InteractiveRolloutEngine:
        frames = np.stack((seed.previous_frame, seed.current_frame))
        contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        tensor = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32).div_(255)
        latents = autoencoder.encode(tensor)
        return cls(
            autoencoder,
            dynamics,
            latents[0:1],
            latents[1:2],
            seed.current_frame,
            device,
        )

    @torch.no_grad()
    def step(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.shape == (self.dynamics.action_dim,):
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape != (1, self.dynamics.action_dim):
            raise ValueError("action must contain one nine-value control vector")
        predicted = self.dynamics(
            self.previous_latent,
            self.current_latent,
            action_tensor,
        )
        decoded = self.autoencoder.decode(predicted).clamp(0, 1)[0]
        frame = decoded.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        self.previous_latent, self.current_latent = self.current_latent, predicted
        self.current_frame = frame
        self.steps += 1
        return frame.copy()

    def reset(self) -> np.ndarray:
        self.previous_latent = self.seed_previous.clone()
        self.current_latent = self.seed_current.clone()
        self.current_frame = self.seed_frame.copy()
        self.steps = 0
        return self.current_frame.copy()

    @torch.no_grad()
    def reseed(self, seed: RolloutSeed) -> np.ndarray:
        """Replace both real seed frames while keeping the loaded models."""
        frames = np.stack((seed.previous_frame, seed.current_frame))
        contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        tensor = torch.from_numpy(contiguous).to(device=self.device, dtype=torch.float32).div_(255)
        latents = self.autoencoder.encode(tensor)
        self.seed_previous = latents[0:1].detach().clone()
        self.seed_current = latents[1:2].detach().clone()
        self.seed_frame = seed.current_frame.copy()
        return self.reset()


def _load_playground(
    processed_dir: Path,
    manifest_path: Path | None,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    sample_index: int,
    requested_device: str,
) -> tuple[InteractiveRolloutEngine, RolloutSeed, RolloutSeedBank, torch.device]:
    device = choose_device(requested_device)
    dynamics, dynamics_metadata = load_spatial_dynamics_checkpoint(dynamics_checkpoint, device)
    if dynamics_metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial dynamics belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    if autoencoder.latent_channels != dynamics.latent_channels:
        raise ValueError("autoencoder and dynamics latent channels do not match")
    autoencoder.requires_grad_(False)
    seeds = RolloutSeedBank.load(processed_dir, manifest_path)
    seed = seeds.get(sample_index)
    engine = InteractiveRolloutEngine.from_seed(autoencoder, dynamics, seed, device)
    return engine, seed, seeds, device


def _action_label(action: np.ndarray) -> str:
    controls = [
        name.upper() for name, value in zip(BINARY_ACTIONS, action[:7], strict=True) if value
    ]
    keys = "+".join(controls) if controls else "IDLE"
    return f"{keys} mouse=({action[-2]:+.0f},{action[-1]:+.0f})"


def save_scripted_rollout(
    seed: RolloutSeed,
    frames: list[np.ndarray],
    actions: np.ndarray,
    output_path: Path,
) -> None:
    tile_width = 256
    tile_height = 230
    columns = min(6, len(frames))
    rows = (len(frames) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * tile_height
        enlarged = cv2.resize(frame, (208, 176), interpolation=cv2.INTER_NEAREST)
        canvas[y + 30 : y + 206, x + 24 : x + 232] = cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)
        if index == 0:
            label = "real seed t"
        else:
            label = f"imagined t+{index}: {_action_label(actions[index - 1])}"
        cv2.putText(
            canvas,
            label,
            (x + 8, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"could not write scripted rollout: {output_path}")


def save_action_comparison(
    rows: list[tuple[str, list[np.ndarray]]],
    output_path: Path,
    *,
    tile: int = 192,
) -> None:
    """Render one seed imagined forward under several action scripts, one per row.

    Every row starts from the same real frame, so differences down a column are
    caused only by the actions. Frames are enlarged with nearest-neighbour
    sampling: smooth upscaling blurs an already-soft prediction a second time.
    """
    if not rows:
        raise ValueError("action comparison needs at least one script")
    if tile < 64:
        raise ValueError("tile must be at least the 64-pixel frame size")
    columns = max(len(frames) for _, frames in rows)
    label_width = 250
    header = 34
    gap = 6
    canvas = np.full(
        (header + len(rows) * (tile + gap), label_width + columns * (tile + gap), 3),
        20,
        dtype=np.uint8,
    )

    for column in range(columns):
        cv2.putText(
            canvas,
            "real seed t" if column == 0 else f"t+{column}",
            (label_width + column * (tile + gap) + 4, header - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

    for row_index, (label, frames) in enumerate(rows):
        y = header + row_index * (tile + gap)
        cv2.putText(
            canvas,
            label,
            (12, y + tile // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        for column, frame in enumerate(frames):
            x = label_width + column * (tile + gap)
            enlarged = cv2.resize(frame, (tile, tile), interpolation=cv2.INTER_NEAREST)
            canvas[y : y + tile, x : x + tile] = cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"could not write action comparison: {output_path}")


@torch.no_grad()
def run_action_comparison(
    engine: InteractiveRolloutEngine,
    seed: RolloutSeed,
    scripts: list[str],
    output_path: Path,
    *,
    camera_step: float = 30.0,
    tile: int = 192,
) -> PlaygroundResult:
    """Imagine the same seed forward once per script, resetting between runs."""
    if not scripts:
        raise ValueError("action comparison needs at least one script")
    rows: list[tuple[str, list[np.ndarray]]] = []
    longest = 0
    for script in scripts:
        actions = parse_action_script(script, camera_step=camera_step)
        engine.reset()
        frames = [seed.current_frame]
        frames.extend(engine.step(action) for action in actions)
        rows.append((script, frames))
        longest = max(longest, len(actions))
    save_action_comparison(rows, output_path, tile=tile)
    return PlaygroundResult(
        episode=seed.episode,
        current_step=seed.current_step,
        steps=longest,
        device=str(engine.device),
        output=output_path,
    )


def compare_action_scripts(
    processed_dir: Path,
    manifest_path: Path | None,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    scripts: list[str],
    *,
    sample_index: int = 0,
    camera_step: float = 30.0,
    tile: int = 192,
    output_path: Path = Path("artifacts/interactive-rollout/action-comparison.png"),
    requested_device: str = "auto",
) -> tuple[PlaygroundResult, int]:
    """Load the selected checkpoints and compare scripts from one shared seed."""
    engine, seed, seeds, _ = _load_playground(
        processed_dir,
        manifest_path,
        autoencoder_checkpoint,
        dynamics_checkpoint,
        sample_index,
        requested_device,
    )
    result = run_action_comparison(
        engine,
        seed,
        scripts,
        output_path,
        camera_step=camera_step,
        tile=tile,
    )
    return result, len(seeds)


def run_scripted_rollout(
    engine: InteractiveRolloutEngine,
    seed: RolloutSeed,
    actions: np.ndarray,
    output_path: Path,
) -> PlaygroundResult:
    frames = [seed.current_frame]
    frames.extend(engine.step(action) for action in actions)
    save_scripted_rollout(seed, frames, actions, output_path)
    return PlaygroundResult(
        episode=seed.episode,
        current_step=seed.current_step,
        steps=len(actions),
        device=str(engine.device),
        output=output_path,
    )


def make_live_action(
    held: Iterable[str],
    *,
    mouse_dx: float,
    mouse_dy: float,
    camera_step: float,
) -> np.ndarray:
    """Convert held semantic controls into one model action."""
    held = set(held)
    controls = held.intersection(BINARY_ACTIONS)
    mouse_dx += camera_step * (float("look_right" in held) - float("look_left" in held))
    mouse_dy += camera_step * (float("look_down" in held) - float("look_up" in held))
    return make_action(controls, mouse_dx=mouse_dx, mouse_dy=mouse_dy)


def launch_playground(
    processed_dir: Path,
    manifest_path: Path | None,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    *,
    sample_index: int = 0,
    camera_step: float = 30.0,
    script: str | None = None,
    output_path: Path = Path("artifacts/interactive-rollout/scripted-rollout.png"),
    requested_device: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> tuple[PlaygroundResult, int]:
    engine, seed, seeds, _ = _load_playground(
        processed_dir,
        manifest_path,
        autoencoder_checkpoint,
        dynamics_checkpoint,
        sample_index,
        requested_device,
    )
    if script is not None:
        actions = parse_action_script(script, camera_step=camera_step)
        result = run_scripted_rollout(engine, seed, actions, output_path)
    else:
        from mcwm.frontend import serve_rollout_frontend

        result = serve_rollout_frontend(
            engine,
            seed,
            seed_index=sample_index,
            seed_count=len(seeds),
            seed_loader=seeds.get,
            camera_step=camera_step,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    return result, len(seeds)
