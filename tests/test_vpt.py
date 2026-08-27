from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcwm.download import download_episode
from mcwm.vpt import load_actions, normalize_key, parse_action


def test_normalize_key() -> None:
    assert normalize_key("key.keyboard.w") == "w"
    assert normalize_key("key.keyboard.left.control") == "left.control"
    assert normalize_key("SPACE") == "space"


def test_parse_action_extracts_small_action_space() -> None:
    action = parse_action(
        {
            "keyboard": {"keys": ["key.keyboard.w", "key.keyboard.space"]},
            "mouse": {"dx": -3.5, "dy": 2.0, "buttons": [0]},
            "isGuiOpen": False,
            "milli": 1234,
            "inventory": [{"type": "dirt", "quantity": 64}],
        }
    )

    assert action.label() == "keys=W+JUMP+ATTACK  mouse=(-3.5, +2.0)"
    assert action.movement_vector() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -3.5, 2.0)
    assert action.mouse_buttons == (0,)
    assert action.timestamp_ms == 1234


def test_load_actions_streams_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    records = [
        {"keyboard": {"keys": []}, "mouse": {}},
        {"keyboard": {"keys": ["key.keyboard.d"]}, "mouse": {"dx": 1}},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    actions = load_actions(path)

    assert len(actions) == 2
    assert actions[1].movement_vector()[3] == 1.0
    assert actions[1].mouse_dx == 1.0


def test_load_actions_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"keyboard": {}}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_actions(path)


def test_load_actions_tolerates_legacy_byte_in_unused_chars(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_bytes(
        b'{"keyboard":{"keys":["key.keyboard.escape"],"chars":"\x82"},'
        b'"mouse":{},"isGuiOpen":true}\n'
    )

    actions = load_actions(path)

    assert len(actions) == 1
    assert actions[0].gui_open
    assert actions[0].keys == frozenset({"escape"})


def test_download_rejects_a_path_as_an_episode_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plain filename stem"):
        download_episode(tmp_path, "../not-an-episode")
