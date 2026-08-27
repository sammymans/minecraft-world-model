from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mcwm.cleaning import RejectionReason
from mcwm.dataset import (
    ProcessedEpisode,
    SequenceDataset,
    episode_group,
    preprocess_episode,
    save_sequence_sheet,
    split_episode_paths,
)


def _write_raw_episode(directory: Path, frame_count: int = 10) -> tuple[Path, Path]:
    video = directory / "player-session-20260101-120000.mp4"
    actions = directory / "player-session-20260101-120000.jsonl"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (96, 64))
    assert writer.isOpened()
    records = []
    for index in range(frame_count):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 10
        writer.write(frame)
        records.append(
            {
                "keyboard": {"keys": ["key.keyboard.w"]},
                "mouse": {"dx": 1.0, "dy": -0.5, "buttons": []},
                "milli": 1000 + index * 50,
                "isGuiOpen": False,
            }
        )
    writer.release()
    actions.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return video, actions


def test_preprocess_builds_10hz_canonical_episode(tmp_path: Path) -> None:
    video, actions = _write_raw_episode(tmp_path)
    output = tmp_path / "processed.npz"

    result = preprocess_episode(video, actions, output, target_fps=10, image_size=32, horizon=2)
    episode = ProcessedEpisode.load(output)

    assert result.model_frames == 5
    assert episode.frames.shape == (5, 32, 32, 3)
    assert episode.actions.shape == (4, 9)
    assert episode.source_frame_indices.tolist() == [0, 2, 4, 6, 8]
    assert episode.actions[0, 0] == 1.0
    assert episode.actions[0, -2:].tolist() == [2.0, -1.0]
    assert episode.valid.all()


def test_sequence_dataset_only_indexes_contiguous_valid_windows(tmp_path: Path) -> None:
    frames = np.zeros((8, 16, 16, 3), dtype=np.uint8)
    actions = np.zeros((7, 9), dtype=np.float32)
    reasons = np.zeros(7, dtype=np.int8)
    reasons[3] = RejectionReason.ATTACK
    episode = ProcessedEpisode(
        episode="example",
        frames=frames,
        actions=actions,
        rejection_reasons=reasons,
        source_frame_indices=np.arange(8, dtype=np.int32) * 2,
        model_fps=10.0,
    )

    dataset = SequenceDataset([episode], horizon=2)

    assert len(dataset) == 2
    for item in range(len(dataset)):
        sample = dataset[item]
        assert sample.frames.shape == (4, 16, 16, 3)
        assert sample.actions.shape == (2, 9)
        assert episode.valid[sample.start_step - 1 : sample.start_step + 2].all()

    output = tmp_path / "sample.png"
    save_sequence_sheet(dataset[0], output)
    assert output.stat().st_size > 0


def test_split_keeps_segments_from_one_session_together() -> None:
    paths = [
        Path("player-one-abc123-20260101-120000.npz"),
        Path("player-one-abc123-20260101-120500.npz"),
        Path("player-two-def456-20260102-120000.npz"),
    ]

    train, validation = split_episode_paths(paths)

    assert len(train) == 1
    assert len(validation) == 2
    assert episode_group(validation[0].stem) == episode_group(validation[1].stem)
    assert episode_group(train[0].stem) != episode_group(validation[0].stem)


def test_split_requires_two_groups() -> None:
    with pytest.raises(ValueError, match="at least two"):
        split_episode_paths([Path("player-session-20260101-120000.npz")])
