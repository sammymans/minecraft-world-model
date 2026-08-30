from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcwm.data_infrastructure import (
    DatasetCatalog,
    build_public_dataset_catalog,
    file_sha256,
    publish_catalog,
)


def _episode_record(episode: str, split: str, video_bytes: int, actions_bytes: int) -> dict:
    base = "https://example.test/vpt"
    return {
        "episode": episode,
        "group": episode.rsplit("-", 2)[0],
        "split": split,
        "video_url": f"{base}/{episode}.mp4",
        "actions_url": f"{base}/{episode}.jsonl",
        "video_bytes": video_bytes,
        "actions_bytes": actions_bytes,
    }


def _write_processed(path: Path, episode: str) -> None:
    frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros((2, 9), dtype=np.float32)
    reasons = np.zeros(2, dtype=np.int8)
    source_indices = np.arange(3, dtype=np.int32)
    metadata = json.dumps({"episode": episode, "model_fps": 10.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=metadata,
        frames=frames,
        actions=actions,
        rejection_reasons=reasons,
        source_frame_indices=source_indices,
    )


def _fixture_tree(root: Path) -> tuple[Path, Path, Path]:
    raw = root / "data/raw"
    processed = root / "data/processed"
    manifest = root / "data/manifests/split.jsonl"
    raw.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    entries = []
    for episode, split in (
        ("player-one-abc-20260101-120000", "training"),
        ("player-two-def-20260101-120001", "validation"),
    ):
        video = raw / f"{episode}.mp4"
        actions = raw / f"{episode}.jsonl"
        video.write_bytes(f"video-{episode}".encode())
        actions.write_text('{"milli":1}\n', encoding="utf-8")
        _write_processed(processed / f"{episode}.npz", episode)
        entries.append(
            _episode_record(episode, split, video.stat().st_size, actions.stat().st_size)
        )
    manifest.write_text("\n".join(json.dumps(item) for item in entries) + "\n", encoding="utf-8")
    return manifest, raw, processed


def test_public_catalog_is_content_addressed_and_round_trips(tmp_path: Path) -> None:
    manifest, raw, processed = _fixture_tree(tmp_path)
    output = tmp_path / "artifacts/catalog.json"
    catalog = build_public_dataset_catalog(
        manifest,
        raw,
        processed,
        output,
        source_root=tmp_path,
        split="training",
    )

    assert catalog.source_type == "public_vpt"
    assert file_sha256(manifest)[:12] in catalog.catalog_id
    assert [item.role for item in catalog.objects] == [
        "dataset_manifest",
        "raw_video",
        "raw_actions",
        "processed_episode",
    ]
    assert all(len(item.sha256) == 64 for item in catalog.objects)
    assert DatasetCatalog.load(output) == catalog


def test_public_catalog_rejects_manifest_size_mismatch(tmp_path: Path) -> None:
    manifest, raw, processed = _fixture_tree(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]["video_bytes"] += 1
    manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n")

    with pytest.raises(ValueError, match="size does not match"):
        build_public_dataset_catalog(
            manifest,
            raw,
            processed,
            tmp_path / "catalog.json",
            source_root=tmp_path,
            split="training",
        )


class _MissingObject(Exception):
    response = {
        "ResponseMetadata": {"HTTPStatusCode": 404},
        "Error": {"Code": "NoSuchKey"},
    }


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.uploads: list[tuple[str, str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            return self.objects[(Bucket, Key)]
        except KeyError as error:
            raise _MissingObject from error

    def upload_file(self, filename: str, bucket: str, key: str, *, ExtraArgs: dict) -> None:
        self.uploads.append((filename, bucket, key))
        self.objects[(bucket, key)] = {
            "ContentLength": Path(filename).stat().st_size,
            "Metadata": ExtraArgs["Metadata"],
        }


def test_publish_is_dry_run_by_default_and_resumable(tmp_path: Path) -> None:
    manifest, raw, processed = _fixture_tree(tmp_path)
    output = tmp_path / "artifacts/catalog.json"
    catalog = build_public_dataset_catalog(
        manifest,
        raw,
        processed,
        output,
        source_root=tmp_path,
        split="training",
    )
    fake = FakeS3()

    dry = publish_catalog(output, "test-bucket", "project", client=fake)
    assert dry.dry_run
    assert not fake.uploads

    first = publish_catalog(output, "test-bucket", "project", execute=True, client=fake)
    assert first.uploaded == len(catalog.objects) + 1
    assert fake.uploads[-1][2].endswith("/_catalog.json")

    second = publish_catalog(output, "test-bucket", "project", execute=True, client=fake)
    assert second.uploaded == 0
    assert second.skipped == len(catalog.objects) + 1


def test_publish_rejects_kms_key_without_kms_encryption(tmp_path: Path) -> None:
    manifest, raw, processed = _fixture_tree(tmp_path)
    output = tmp_path / "catalog.json"
    build_public_dataset_catalog(
        manifest,
        raw,
        processed,
        output,
        source_root=tmp_path,
        split="training",
    )
    with pytest.raises(ValueError, match="requires"):
        publish_catalog(
            output,
            "test-bucket",
            "project",
            execute=True,
            sse="AES256",
            kms_key_id="key-id",
            client=FakeS3(),
        )
