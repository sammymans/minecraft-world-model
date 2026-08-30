from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mcwm.preview import create_preview, inspect_episode


def _write_tiny_episode(directory: Path) -> tuple[Path, Path]:
    video = directory / "tiny.mp4"
    actions = directory / "tiny.jsonl"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 360))
    assert writer.isOpened()
    records = []
    for index in range(10):
        frame = np.full((360, 640, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
        records.append(
            {
                "keyboard": {"keys": ["key.keyboard.w"] if index >= 5 else []},
                "mouse": {"dx": float(index), "dy": 0.0},
                "milli": 1000 + index * 50,
            }
        )
    writer.release()
    actions.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return video, actions


def test_inspect_episode_counts_matching_frames(tmp_path: Path) -> None:
    video, actions = _write_tiny_episode(tmp_path)

    info, counts = inspect_episode(video, actions)

    assert info.fps == 20.0
    assert info.video_frames == 10
    assert info.action_frames == 10
    assert info.paired_frames == 10
    assert counts["W"] == 5


def test_create_preview_writes_downsampled_video(tmp_path: Path) -> None:
    video, actions = _write_tiny_episode(tmp_path)
    output = tmp_path / "preview.mp4"

    written = create_preview(video, actions, output, duration_seconds=0.5, output_fps=10.0)

    assert written == 5
    assert output.stat().st_size > 0
