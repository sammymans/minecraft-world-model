from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mcwm.data_infrastructure import build_recording_catalog, file_sha256
from mcwm.recorder import (
    CaptureRegion,
    InputAccumulator,
    pynput_button_index,
    pynput_key_name,
    verify_recording,
)


class NamedInput:
    def __init__(self, name: str, char: str | None = None):
        self.name = name
        self.char = char

    def __str__(self) -> str:
        return self.name


def test_input_accumulator_consumes_mouse_delta_but_retains_held_state() -> None:
    state = InputAccumulator()
    state.key_down("w")
    state.mouse_button(0, True)
    state.mouse_move(100, 100)
    state.mouse_move(106, 97)

    first = state.snapshot(123)
    second = state.snapshot(124)

    assert first["keyboard"]["keys"] == ["w"]
    assert first["mouse"]["buttons"] == [0]
    assert first["mouse"]["dx"] == 6
    assert first["mouse"]["dy"] == -3
    assert second["mouse"]["dx"] == 0
    assert second["keyboard"]["keys"] == ["w"]


def test_pynput_names_map_to_vpt_controls() -> None:
    assert pynput_key_name(NamedInput("'W'", char="W")) == "w"
    assert pynput_key_name(NamedInput("Key.space")) == "space"
    assert pynput_key_name(NamedInput("Key.shift_r")) == "left.shift"
    assert pynput_button_index(NamedInput("Button.left")) == 0
    assert pynput_button_index(NamedInput("Button.right")) == 1


def test_recording_verification_and_catalog(tmp_path: Path) -> None:
    episode = "local-player-abcdef123456-20260829-120000"
    raw = tmp_path / "data/raw/local"
    raw.mkdir(parents=True)
    video = raw / f"{episode}.mp4"
    actions = raw / f"{episode}.jsonl"
    metadata = raw / f"{episode}.recording.json"

    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 64))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((64, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    records = []
    for index in range(3):
        records.append(
            {
                "mouse": {"dx": 0, "dy": 0, "buttons": []},
                "keyboard": {"keys": []},
                "isGuiOpen": False,
                "milli": 1_000 + index * 50,
            }
        )
    actions.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    value = {
        "schema_version": 1,
        "episode": episode,
        "source": "user_recorded",
        "collector": "test",
        "started_at": "2026-08-29T12:00:00+00:00",
        "frames": 3,
        "fps": 20.0,
        "capture_region": {"left": 0, "top": 0, "width": 64, "height": 64},
        "video_path": video.relative_to(tmp_path).as_posix(),
        "actions_path": actions.relative_to(tmp_path).as_posix(),
        "video_sha256": file_sha256(video),
        "actions_sha256": file_sha256(actions),
    }
    metadata.write_text(json.dumps(value))

    verified = verify_recording(metadata, source_root=tmp_path)
    assert verified.video_frames == 3
    assert verified.action_records == 3

    catalog_path = tmp_path / "artifacts/local.json"
    catalog = build_recording_catalog(metadata, catalog_path, source_root=tmp_path)
    assert catalog.source_type == "local_recording"
    assert [item.role for item in catalog.objects] == [
        "recording_metadata",
        "raw_video",
        "raw_actions",
    ]


def test_capture_region_rejects_invalid_bounds() -> None:
    try:
        CaptureRegion(-1, 0, 64, 64).validate()
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative capture origin should fail")
