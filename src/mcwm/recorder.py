"""Record synchronized Minecraft pixels and controls in the VPT-compatible schema."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mcwm.data_infrastructure import file_sha256
from mcwm.vpt import load_actions


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def validate(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError("capture left/top must be non-negative")
        if self.width < 64 or self.height < 64:
            raise ValueError("capture width/height must be at least 64 pixels")


@dataclass(frozen=True)
class RecordingResult:
    episode: str
    video_path: Path
    actions_path: Path
    metadata_path: Path
    frames: int
    fps: float
    stopped_early: bool


@dataclass(frozen=True)
class RecordingVerification:
    episode: str
    video_frames: int
    action_records: int
    fps: float
    width: int
    height: int


def default_episode_name() -> str:
    """Create a manifest-compatible local group/date/time episode name."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"local-player-{secrets.token_hex(6)}-{stamp}"


class InputAccumulator:
    """Thread-safe keyboard/button state plus per-frame relative mouse motion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._buttons: set[int] = set()
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._last_mouse: tuple[float, float] | None = None
        self._dx = 0.0
        self._dy = 0.0

    def key_down(self, key: str | None) -> None:
        if key is not None:
            with self._lock:
                self._keys.add(key)

    def key_up(self, key: str | None) -> None:
        if key is not None:
            with self._lock:
                self._keys.discard(key)

    def mouse_move(self, x: float, y: float) -> None:
        with self._lock:
            if self._last_mouse is not None:
                self._dx += x - self._last_mouse[0]
                self._dy += y - self._last_mouse[1]
            self._mouse_x = x
            self._mouse_y = y
            self._last_mouse = (x, y)

    def mouse_button(self, button: int | None, pressed: bool) -> None:
        if button is None:
            return
        with self._lock:
            if pressed:
                self._buttons.add(button)
            else:
                self._buttons.discard(button)

    def snapshot(self, timestamp_ms: int) -> dict[str, Any]:
        """Consume mouse deltas while retaining held keys and buttons."""
        with self._lock:
            record = {
                "mouse": {
                    "x": self._mouse_x,
                    "y": self._mouse_y,
                    "dx": self._dx,
                    "dy": self._dy,
                    "scaledX": 0.0,
                    "scaledY": 0.0,
                    "dwheel": 0.0,
                    "buttons": sorted(self._buttons),
                    "newButtons": [],
                },
                "keyboard": {"keys": sorted(self._keys), "newKeys": [], "chars": ""},
                "isGuiOpen": False,
                "isGuiInventory": False,
                "milli": timestamp_ms,
                "captureSource": "mcwm-local-recorder-v1",
            }
            self._dx = 0.0
            self._dy = 0.0
            return record


def pynput_key_name(key: Any) -> str | None:
    """Map pynput keys to the names understood by the existing VPT parser."""
    character = getattr(key, "char", None)
    if character:
        return str(character).lower()
    name = str(key).lower()
    aliases = {
        "key.space": "space",
        "key.shift": "left.shift",
        "key.shift_l": "left.shift",
        "key.shift_r": "left.shift",
        "key.ctrl": "left.control",
        "key.ctrl_l": "left.control",
        "key.ctrl_r": "left.control",
    }
    return aliases.get(name, name.removeprefix("key."))


def pynput_button_index(button: Any) -> int | None:
    name = str(button).lower()
    return {"button.left": 0, "button.right": 1, "button.middle": 2}.get(name)


def _recording_metadata(
    *,
    episode: str,
    video_path: Path,
    actions_path: Path,
    source_root: Path,
    collector: str | None,
    started_at: str,
    frames: int,
    fps: float,
    region: CaptureRegion,
    stopped_early: bool,
) -> dict[str, Any]:
    def relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(source_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("recording output must be below source root") from error

    return {
        "schema_version": 1,
        "episode": episode,
        "source": "user_recorded",
        "collector": collector,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "frames": frames,
        "fps": fps,
        "capture_region": asdict(region),
        "stopped_early": stopped_early,
        "video_path": relative(video_path),
        "actions_path": relative(actions_path),
        "video_bytes": video_path.stat().st_size,
        "actions_bytes": actions_path.stat().st_size,
        "video_sha256": file_sha256(video_path),
        "actions_sha256": file_sha256(actions_path),
        "action_schema": "vpt-compatible-jsonl-v1",
    }


def record_minecraft_episode(
    output_dir: Path,
    *,
    duration_seconds: float,
    region: CaptureRegion,
    fps: float = 20.0,
    countdown_seconds: float = 3.0,
    episode: str | None = None,
    collector: str | None = None,
    source_root: Path = Path("."),
) -> RecordingResult:
    """Capture a fixed screen region and global input state until F8 or timeout.

    macOS requires Screen Recording permission for the terminal and Accessibility
    permission for global keyboard/mouse listeners. The function imports those
    platform backends lazily so the rest of the project remains headless-safe.
    """
    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    if countdown_seconds < 0:
        raise ValueError("countdown must be non-negative")
    region.validate()
    episode = episode or default_episode_name()
    if Path(episode).name != episode or Path(episode).suffix:
        raise ValueError("episode must be a plain filename stem")
    try:
        import mss
        from pynput import keyboard, mouse
    except ImportError as error:
        raise RuntimeError("recording requires the mss and pynput packages") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{episode}.mp4"
    actions_path = output_dir / f"{episode}.jsonl"
    metadata_path = output_dir / f"{episode}.recording.json"
    for path in (video_path, actions_path, metadata_path):
        if path.exists():
            raise ValueError(f"recording output already exists: {path}")
    temporary_video = output_dir / f"{episode}.part.mp4"
    temporary_actions = output_dir / f"{episode}.jsonl.part"

    print(f"Recording starts in {countdown_seconds:g}s; focus Minecraft. Press F8 to stop.")
    if countdown_seconds:
        time.sleep(countdown_seconds)

    accumulator = InputAccumulator()
    stop_event = threading.Event()

    def on_press(key: Any) -> None:
        if key == keyboard.Key.f8:
            stop_event.set()
            return
        accumulator.key_down(pynput_key_name(key))

    def on_release(key: Any) -> None:
        accumulator.key_up(pynput_key_name(key))

    def on_move(x: float, y: float) -> None:
        accumulator.mouse_move(x, y)

    def on_click(_x: float, _y: float, button: Any, pressed: bool) -> None:
        accumulator.mouse_button(pynput_button_index(button), pressed)

    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (region.width, region.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 recorder")

    started_at = datetime.now(UTC).isoformat()
    frame_count = 0
    start = time.perf_counter()
    frame_interval = 1.0 / fps
    monitor = asdict(region)
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    keyboard_listener.start()
    mouse_listener.start()
    try:
        with mss.mss() as screen, temporary_actions.open("w", encoding="utf-8") as actions:
            while not stop_event.is_set():
                deadline = start + frame_count * frame_interval
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                if time.perf_counter() - start >= duration_seconds:
                    break
                frame = np.asarray(screen.grab(monitor))[:, :, :3]
                writer.write(frame)
                timestamp_ms = int(time.time_ns() // 1_000_000)
                actions.write(json.dumps(accumulator.snapshot(timestamp_ms), separators=(",", ":")))
                actions.write("\n")
                frame_count += 1
    finally:
        keyboard_listener.stop()
        mouse_listener.stop()
        writer.release()

    if frame_count < 2:
        temporary_video.unlink(missing_ok=True)
        temporary_actions.unlink(missing_ok=True)
        raise RuntimeError("recording stopped before two synchronized frames were captured")
    os.replace(temporary_video, video_path)
    os.replace(temporary_actions, actions_path)
    stopped_early = stop_event.is_set()
    metadata = _recording_metadata(
        episode=episode,
        video_path=video_path,
        actions_path=actions_path,
        source_root=source_root,
        collector=collector,
        started_at=started_at,
        frames=frame_count,
        fps=fps,
        region=region,
        stopped_early=stopped_early,
    )
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".part")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return RecordingResult(
        episode=episode,
        video_path=video_path,
        actions_path=actions_path,
        metadata_path=metadata_path,
        frames=frame_count,
        fps=fps,
        stopped_early=stopped_early,
    )


def verify_recording(
    metadata_path: Path, *, source_root: Path = Path(".")
) -> RecordingVerification:
    """Verify checksums plus video/action synchronization for a saved episode."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        episode = str(metadata["episode"])
        video_path = source_root / metadata["video_path"]
        actions_path = source_root / metadata["actions_path"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"invalid recording metadata: {metadata_path}") from error
    if file_sha256(video_path) != metadata.get("video_sha256"):
        raise ValueError("recording video checksum mismatch")
    if file_sha256(actions_path) != metadata.get("actions_sha256"):
        raise ValueError("recording actions checksum mismatch")
    actions = load_actions(actions_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open recording video: {video_path}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if video_frames != len(actions):
        raise ValueError(
            f"recording is not synchronized: {video_frames} video frames, {len(actions)} actions"
        )
    if video_frames != int(metadata.get("frames", -1)):
        raise ValueError("recording frame count does not match metadata")
    return RecordingVerification(episode, video_frames, len(actions), fps, width, height)
