"""Inspect VPT frame/action alignment and create an annotated preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mcwm.vpt import VPTAction, action_counts, load_actions


@dataclass(frozen=True)
class EpisodeInfo:
    fps: float
    video_frames: int
    action_frames: int
    width: int
    height: int

    @property
    def paired_frames(self) -> int:
        return min(self.video_frames, self.action_frames)

    @property
    def duration_seconds(self) -> float:
        return self.paired_frames / self.fps


def inspect_episode(video_path: Path, action_path: Path) -> tuple[EpisodeInfo, dict[str, int]]:
    """Read metadata and basic action counts for a matched episode."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info_without_actions = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "video_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()

    actions = load_actions(action_path)
    info = EpisodeInfo(action_frames=len(actions), **info_without_actions)
    return info, dict(action_counts(actions))


def _draw_overlay(frame: np.ndarray, action: VPTAction, frame_index: int, fps: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (620, 116), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, dst=frame)

    lines = (
        f"source frame {frame_index}  time {frame_index / fps:6.2f}s",
        action.label(),
        "pairing: JSONL line i is shown with video frame i",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (26, 42 + row * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def create_preview(
    video_path: Path,
    action_path: Path,
    output_path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float = 15.0,
    output_fps: float = 10.0,
) -> int:
    """Write a short MP4 with the synchronized action drawn over every frame."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if output_fps <= 0:
        raise ValueError("output_fps must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, round(start_seconds * source_fps))
    stop_frame = start_frame + round(duration_seconds * source_fps)
    stride = max(1, round(source_fps / output_fps))
    actual_output_fps = source_fps / stride
    actions = load_actions(action_path, limit=stop_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create preview: {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    frame_index = start_frame
    while frame_index < min(stop_frame, len(actions)):
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % stride == 0:
            _draw_overlay(frame, actions[frame_index], frame_index, source_fps)
            writer.write(frame)
            written += 1
        frame_index += 1

    capture.release()
    writer.release()
    if written == 0:
        raise ValueError("The requested preview range contained no paired frames")
    return written
