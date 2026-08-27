from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcwm.manifest import (
    DatasetManifest,
    dataset_status,
    expand_vpt10_manifest,
    split_manifest,
)


def _record(episode: str, split: str) -> dict:
    return {
        "episode": episode,
        "group": episode.rsplit("-", 2)[0],
        "split": split,
        "video_url": f"https://example.test/{episode}.mp4",
        "actions_url": f"https://example.test/{episode}.jsonl",
        "video_bytes": 4,
        "actions_bytes": 3,
    }


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_manifest_preserves_explicit_group_safe_split(tmp_path: Path) -> None:
    training = "player-one-abc123-20260101-120000"
    validation = "player-two-def456-20260102-120000"
    manifest_path = tmp_path / "dataset.jsonl"
    _write_manifest(
        manifest_path,
        [_record(training, "training"), _record(validation, "validation")],
    )

    manifest = DatasetManifest.load(manifest_path)

    assert [entry.episode for entry in manifest.select("training")] == [training]
    assert [entry.episode for entry in manifest.select("validation")] == [validation]


def test_manifest_rejects_group_leakage(tmp_path: Path) -> None:
    first = "player-one-abc123-20260101-120000"
    second = "player-one-abc123-20260101-120500"
    manifest_path = tmp_path / "dataset.jsonl"
    _write_manifest(
        manifest_path,
        [_record(first, "training"), _record(second, "validation")],
    )

    with pytest.raises(ValueError, match="cross dataset splits"):
        DatasetManifest.load(manifest_path)


def test_dataset_status_checks_manifest_file_sizes(tmp_path: Path) -> None:
    training = "player-one-abc123-20260101-120000"
    validation = "player-two-def456-20260102-120000"
    manifest_path = tmp_path / "dataset.jsonl"
    _write_manifest(
        manifest_path,
        [_record(training, "training"), _record(validation, "validation")],
    )
    manifest = DatasetManifest.load(manifest_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / f"{training}.mp4").write_bytes(b"1234")
    (raw_dir / f"{training}.jsonl").write_bytes(b"123")

    status = dataset_status(manifest, raw_dir, tmp_path / "processed")

    assert status.episodes == 2
    assert status.groups == 2
    assert status.raw_complete == 1
    assert status.processed_complete == 0
    assert status.expected_raw_bytes == 14
    assert status.test_groups == 0


def test_expansion_uses_live_matched_pairs_and_preserves_validation(tmp_path: Path) -> None:
    training = "player-one-abc123-20260101-120000"
    validation = "player-two-def456-20260102-120000"
    base_path = tmp_path / "v1.jsonl"
    _write_manifest(
        base_path,
        [_record(training, "training"), _record(validation, "validation")],
    )
    available = {
        "data/10.0/player-three-abc123-20260103-120000.mp4": 10,
        "data/10.0/player-three-abc123-20260103-120000.jsonl": 5,
        "data/10.0/player-four-abc123-20260104-120000.mp4": 20,
        "data/10.0/player-four-abc123-20260104-120000.jsonl": 5,
        "data/10.0/missing-actions-abc123-20260105-120000.mp4": 100,
        "data/10.0/empty-actions-abc123-20260106-120000.mp4": 100,
        "data/10.0/empty-actions-abc123-20260106-120000.jsonl": 0,
    }

    expanded = expand_vpt10_manifest(
        DatasetManifest.load(base_path),
        tmp_path / "v2.jsonl",
        target_bytes=22,
        seed=7,
        available_files=available,
    )

    assert len(expanded.episodes) == 3
    assert expanded.select("validation")[0].episode == validation
    assert expanded.select("training")[-1].group in {
        "player-three-abc123",
        "player-four-abc123",
    }
    assert "missing-actions" not in expanded.select("training")[-1].episode
    assert "empty-actions" not in expanded.select("training")[-1].episode


def test_three_way_split_is_deterministic_group_safe_and_preserves_holdout(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            f"player-{index:02d}-abc123-20260101-{120000 + index:06d}",
            "validation" if index == 0 else "training",
        )
        for index in range(20)
    ]
    source_path = tmp_path / "source.jsonl"
    _write_manifest(source_path, records)
    source = DatasetManifest.load(source_path)

    first = split_manifest(
        source,
        tmp_path / "first.jsonl",
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=9,
    )
    split_manifest(
        source,
        tmp_path / "second.jsonl",
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=9,
    )

    assert (tmp_path / "first.jsonl").read_text() == (tmp_path / "second.jsonl").read_text()
    assert len({entry.group for entry in first.select("training")}) == 12
    assert len({entry.group for entry in first.select("validation")}) == 4
    assert len({entry.group for entry in first.select("test")}) == 4
    assert records[0]["group"] in {entry.group for entry in first.select("validation")}
    assert records[0]["group"] not in {entry.group for entry in first.select("test")}


def test_three_way_split_rejects_impossible_fractions(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_manifest(
        source_path,
        [
            _record("player-one-abc123-20260101-120000", "training"),
            _record("player-two-abc123-20260101-120001", "training"),
            _record("player-three-abc123-20260101-120002", "validation"),
        ],
    )

    with pytest.raises(ValueError, match="sum to less than one"):
        split_manifest(
            DatasetManifest.load(source_path),
            tmp_path / "split.jsonl",
            validation_fraction=0.5,
            test_fraction=0.5,
        )
