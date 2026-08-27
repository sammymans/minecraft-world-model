"""A small interactive playground for recursively imagined Minecraft frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from mcwm.dataset import SequenceDataset
from mcwm.dynamics import (
    _processed_splits,
    _verify_autoencoder,
    load_dynamics_checkpoint,
)
from mcwm.model import LatentDynamics, TinyAutoencoder
from mcwm.training import choose_device, load_autoencoder_checkpoint

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


class LiveActionState:
    """Persistent key toggles plus camera movement consumed once per model step."""

    def __init__(self) -> None:
        self.held: set[str] = set()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0

    def toggle(self, control: str) -> bool:
        if control not in BINARY_ACTIONS:
            raise ValueError(f"unknown binary control: {control}")
        if control in self.held:
            self.held.remove(control)
            return False
        self.held.add(control)
        return True

    def add_camera(self, dx: float, dy: float) -> None:
        self.mouse_dx += dx
        self.mouse_dy += dy

    def clear(self) -> None:
        self.held.clear()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0

    def consume(self) -> np.ndarray:
        action = make_action(
            self.held,
            mouse_dx=self.mouse_dx,
            mouse_dy=self.mouse_dy,
        )
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        return action

    def label(self) -> str:
        ordered = [name.upper() for name in BINARY_ACTIONS if name in self.held]
        keys = "+".join(ordered) if ordered else "IDLE"
        return f"{keys}  mouse pending=({self.mouse_dx:+.0f}, {self.mouse_dy:+.0f})"


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
    _, validation_paths = _processed_splits(processed_dir, manifest_path)
    dataset = SequenceDataset.from_paths(validation_paths, horizon=1)
    if not len(dataset):
        raise ValueError("validation data has no clean rollout seeds")
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"sample_index must be between 0 and {len(dataset) - 1}")
    episode_index, current_step = dataset.index[sample_index]
    episode = dataset.episodes[episode_index]
    return (
        RolloutSeed(
            episode=episode.episode,
            current_step=current_step,
            model_fps=episode.model_fps,
            previous_frame=episode.frames[current_step - 1].copy(),
            current_frame=episode.frames[current_step].copy(),
        ),
        len(dataset),
    )


class InteractiveRolloutEngine:
    """The tested recursive state transition shared by GUI and scripted modes."""

    def __init__(
        self,
        autoencoder: TinyAutoencoder,
        dynamics: LatentDynamics,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        current_frame: np.ndarray,
        device: torch.device,
    ) -> None:
        if previous_latent.shape != current_latent.shape:
            raise ValueError("seed latents must have identical shapes")
        if previous_latent.shape != (1, dynamics.latent_dim):
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
        autoencoder: TinyAutoencoder,
        dynamics: LatentDynamics,
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


def _load_playground(
    processed_dir: Path,
    manifest_path: Path | None,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    sample_index: int,
    requested_device: str,
) -> tuple[InteractiveRolloutEngine, RolloutSeed, int, torch.device]:
    device = choose_device(requested_device)
    dynamics, dynamics_metadata = load_dynamics_checkpoint(dynamics_checkpoint, device)
    _verify_autoencoder(dynamics_metadata, autoencoder_checkpoint)
    autoencoder, autoencoder_metadata = load_autoencoder_checkpoint(
        autoencoder_checkpoint, device
    )
    if int(autoencoder_metadata["latent_dim"]) != dynamics.latent_dim:
        raise ValueError("autoencoder and dynamics latent dimensions do not match")
    autoencoder.requires_grad_(False)
    seed, seed_count = select_rollout_seed(processed_dir, manifest_path, sample_index)
    engine = InteractiveRolloutEngine.from_seed(autoencoder, dynamics, seed, device)
    return engine, seed, seed_count, device


def _action_label(action: np.ndarray) -> str:
    controls = [
        name.upper()
        for name, value in zip(BINARY_ACTIONS, action[:7], strict=True)
        if value
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
        canvas[y + 30 : y + 206, x + 24 : x + 232] = cv2.cvtColor(
            enlarged, cv2.COLOR_RGB2BGR
        )
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


def _enlarge(frame: np.ndarray, size: int = 320) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_NEAREST)


def _viewer_canvas(
    seed: RolloutSeed,
    engine: InteractiveRolloutEngine,
    actions: LiveActionState,
    *,
    running: bool,
) -> np.ndarray:
    panels = [seed.previous_frame, seed.current_frame, engine.current_frame]
    labels = ("real seed t-1", "real seed t", f"imagined t+{engine.steps}")
    canvas = np.full((440, 1000, 3), 22, dtype=np.uint8)
    for index, (frame, label) in enumerate(zip(panels, labels, strict=True)):
        x = 10 + index * 330
        bgr = cv2.cvtColor(_enlarge(frame), cv2.COLOR_RGB2BGR)
        canvas[42:362, x : x + 320] = bgr
        cv2.putText(
            canvas,
            label,
            (x, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    status = "RUNNING" if running else "PAUSED"
    lines = (
        f"{status} | step {engine.steps} ({engine.steps / seed.model_fps:.1f}s) | "
        f"{actions.label()}",
        "Toggle W/A/S/D, E sprint, C sneak, SPACE jump | H/J/K/L look | drag mouse",
        "P run/pause | N single step | X clear controls | R reset | G snapshot | Q quit",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 382 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_interactive_window(
    engine: InteractiveRolloutEngine,
    seed: RolloutSeed,
    *,
    camera_step: float = 30.0,
    snapshot_path: Path = Path("artifacts/interactive-rollout/snapshot.png"),
) -> PlaygroundResult:
    """Open the local recursive playground; movement keys are intentional toggles."""
    if camera_step <= 0:
        raise ValueError("camera_step must be positive")
    window = "Minecraft latent world model"
    actions = LiveActionState()
    running = False
    last_mouse: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        nonlocal last_mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if last_mouse is not None:
                actions.add_camera(x - last_mouse[0], y - last_mouse[1])
            last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            last_mouse = None

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV could not create a GUI window; use --script for headless mode"
        ) from error

    delay_ms = max(1, round(1000 / seed.model_fps))
    try:
        while True:
            cv2.imshow(window, _viewer_canvas(seed, engine, actions, running=running))
            key = cv2.waitKeyEx(delay_ms if running else 40)
            if key >= 0:
                character = chr(key & 0xFF).lower()
                if character == "q":
                    break
                if character in {"w", "a", "s", "d"}:
                    actions.toggle(character)
                elif character == "e":
                    actions.toggle("sprint")
                elif character == "c":
                    actions.toggle("sneak")
                elif character == " ":
                    actions.toggle("jump")
                elif character == "h":
                    actions.add_camera(-camera_step, 0)
                elif character == "l":
                    actions.add_camera(camera_step, 0)
                elif character == "k":
                    actions.add_camera(0, -camera_step)
                elif character == "j":
                    actions.add_camera(0, camera_step)
                elif character == "p":
                    running = not running
                elif character == "n":
                    engine.step(actions.consume())
                elif character == "x":
                    actions.clear()
                elif character == "r":
                    engine.reset()
                    actions.clear()
                    running = False
                elif character == "g":
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(snapshot_path),
                        _viewer_canvas(seed, engine, actions, running=running),
                    )
            if running:
                engine.step(actions.consume())
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    return PlaygroundResult(
        episode=seed.episode,
        current_step=seed.current_step,
        steps=engine.steps,
        device=str(engine.device),
    )


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
) -> tuple[PlaygroundResult, int]:
    engine, seed, seed_count, _ = _load_playground(
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
        result = run_interactive_window(engine, seed, camera_step=camera_step)
    return result, seed_count
