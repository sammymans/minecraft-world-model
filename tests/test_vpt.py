import json
from pathlib import Path

import numpy as np

from mcwm.data.vpt import (
    CAMERA_SCALER,
    action_from_row,
    alignment_correlations,
    load_vpt_file,
    resample_episode,
)


def _row(step: int, *, gui: bool = False) -> dict[str, object]:
    return {
        "milli": 1_000 + step * 50,
        "xpos": step * 0.1,
        "ypos": 64.0,
        "zpos": 2.0,
        "yaw": step * 1.5,
        "pitch": step * 0.5,
        "isGuiOpen": gui,
        "keyboard": {"keys": ["key.keyboard.w", "key.keyboard.space"]},
        "mouse": {"dx": 10.0, "dy": -2.0},
    }


def test_action_mapping() -> None:
    action = action_from_row(_row(0))
    assert action[0] == 1.0
    assert action[4] == 1.0
    assert action[7] == 10.0 * CAMERA_SCALER
    assert action[8] == -2.0 * CAMERA_SCALER


def test_vpt_loader_splits_gui_and_derives_velocity(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [_row(index) for index in range(10)]
    rows.extend([_row(10, gui=True)])
    rows.extend([_row(index) for index in range(11, 21)])
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    episodes = load_vpt_file(path, min_segment_states=5)
    assert len(episodes) == 2
    assert sum(episode.transitions for episode in episodes) == 18
    assert episodes[0].states[0, 3] == 0.0
    np.testing.assert_allclose(episodes[0].states[1:, 3], 2.0, atol=1e-4)
    np.testing.assert_allclose(episodes[0].dts, 0.05, atol=1e-5)


def test_resampling_aggregates_held_and_camera_actions(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [_row(index) for index in range(9)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    native = load_vpt_file(path, min_segment_states=5)[0]
    repeated = resample_episode(native, action_repeat=4)
    assert repeated.transitions == 2
    assert repeated.actions[0, 0] == 1.0
    assert repeated.actions[0, 7] == 4 * 10.0 * CAMERA_SCALER
    np.testing.assert_allclose(repeated.dts, 0.2, atol=1e-5)
    assert repeated.states[0, 3] == 0.0
    np.testing.assert_allclose(repeated.states[1:, 3], 2.0, atol=1e-4)


def test_alignment_correlation_peaks_for_synchronized_camera(tmp_path: Path) -> None:
    path = tmp_path / "aligned.jsonl"
    rows = []
    yaw = 0.0
    for index in range(12):
        row = _row(index)
        dx = float(((index * 7) % 11) - 5)
        row["yaw"] = yaw
        row["mouse"] = {"dx": dx, "dy": 0.0}
        rows.append(row)
        yaw += dx * CAMERA_SCALER
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    episode = load_vpt_file(path, min_segment_states=5)[0]
    correlations = alignment_correlations([episode])["yaw"]
    assert correlations[0] > 0.99
    assert correlations[0] > correlations[-1]
    assert correlations[0] > correlations[1]


def test_loader_tolerates_legacy_bytes_in_unused_chars_field(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    lines = [json.dumps(_row(index)).encode() for index in range(8)]
    lines[2] = lines[2].replace(b'"isGuiOpen"', b'"chars":"\x82","isGuiOpen"')
    path.write_bytes(b"\n".join(lines))
    episodes = load_vpt_file(path, min_segment_states=5)
    assert len(episodes) == 1
    assert episodes[0].transitions == 7
