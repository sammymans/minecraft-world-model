"""Versioned manifests for reproducible local Minecraft datasets."""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from mcwm.dataset import ProcessedEpisode, preprocess_episode
from mcwm.download import download_url

DatasetSplit = Literal["training", "validation", "test"]
DATASET_SPLITS: tuple[DatasetSplit, ...] = ("training", "validation", "test")

VPT_CONTAINER_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl"
VPT_10_PREFIX = "data/10.0/"


@dataclass(frozen=True)
class ManifestEpisode:
    """One immutable public video/action pair and its group-safe split."""

    episode: str
    group: str
    split: DatasetSplit
    video_url: str
    actions_url: str
    video_bytes: int
    actions_bytes: int

    @classmethod
    def from_dict(cls, record: dict, *, line_number: int) -> ManifestEpisode:
        required = {
            "episode",
            "group",
            "split",
            "video_url",
            "actions_url",
            "video_bytes",
            "actions_bytes",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"manifest line {line_number} is missing: {sorted(missing)}")
        entry = cls(**{key: record[key] for key in required})
        entry.validate(line_number=line_number)
        return entry

    def validate(self, *, line_number: int | None = None) -> None:
        location = f" on manifest line {line_number}" if line_number else ""
        if not self.episode or Path(self.episode).name != self.episode:
            raise ValueError(f"invalid episode{location}: {self.episode!r}")
        if self.group != self.episode.rsplit("-", 2)[0]:
            raise ValueError(f"group does not match episode{location}: {self.episode}")
        if self.split not in DATASET_SPLITS:
            raise ValueError(f"invalid split{location}: {self.split!r}")
        if not self.video_url.endswith(f"/{self.episode}.mp4"):
            raise ValueError(f"video URL does not match episode{location}")
        if not self.actions_url.endswith(f"/{self.episode}.jsonl"):
            raise ValueError(f"actions URL does not match episode{location}")
        if self.video_bytes < 1 or self.actions_bytes < 1:
            raise ValueError(f"expected file sizes must be positive{location}")

    def raw_paths(self, raw_dir: Path) -> tuple[Path, Path]:
        return raw_dir / f"{self.episode}.mp4", raw_dir / f"{self.episode}.jsonl"

    def processed_path(self, processed_dir: Path) -> Path:
        return processed_dir / f"{self.episode}.npz"


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    episodes: tuple[ManifestEpisode, ...]

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        if not path.exists():
            raise ValueError(f"Dataset manifest does not exist: {path}")
        episodes: list[ManifestEpisode] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            episodes.append(ManifestEpisode.from_dict(record, line_number=line_number))
        manifest = cls(path=path, episodes=tuple(episodes))
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.episodes:
            raise ValueError("dataset manifest is empty")
        names = [entry.episode for entry in self.episodes]
        if len(names) != len(set(names)):
            raise ValueError("dataset manifest contains duplicate episodes")
        group_splits: dict[str, set[str]] = {}
        for entry in self.episodes:
            group_splits.setdefault(entry.group, set()).add(entry.split)
        leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"player/session groups cross dataset splits: {leaked}")
        splits = {entry.split for entry in self.episodes}
        if not {"training", "validation"}.issubset(splits):
            raise ValueError("dataset manifest needs training and validation episodes")

    def select(self, split: DatasetSplit | Literal["all"] = "all") -> tuple[ManifestEpisode, ...]:
        if split == "all":
            return self.episodes
        if split not in DATASET_SPLITS:
            raise ValueError("split must be all, training, validation, or test")
        return tuple(entry for entry in self.episodes if entry.split == split)

    def processed_paths(self, processed_dir: Path, split: DatasetSplit) -> list[Path]:
        paths = [entry.processed_path(processed_dir) for entry in self.select(split)]
        if not paths:
            raise ValueError(f"dataset manifest has no {split} episodes")
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise ValueError(
                f"processed manifest episodes are missing; first missing: {missing[0]}"
            )
        return paths

    def processed_splits(self, processed_dir: Path) -> tuple[list[Path], list[Path]]:
        return (
            self.processed_paths(processed_dir, "training"),
            self.processed_paths(processed_dir, "validation"),
        )


@dataclass(frozen=True)
class DatasetStatus:
    episodes: int
    groups: int
    training_groups: int
    validation_groups: int
    test_groups: int
    raw_complete: int
    processed_complete: int
    expected_raw_bytes: int


def _list_blob_sizes(container_url: str, prefix: str) -> dict[str, int]:
    """Read Azure's live paginated object listing instead of its stale snapshot."""
    blobs: dict[str, int] = {}
    marker = ""
    while True:
        parameters = {
            "restype": "container",
            "comp": "list",
            "prefix": prefix,
            "maxresults": "5000",
        }
        if marker:
            parameters["marker"] = marker
        url = container_url.rstrip("/") + "?" + urllib.parse.urlencode(parameters)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for blob in root.findall("./Blobs/Blob"):
            name = blob.findtext("Name")
            content_length = blob.findtext("./Properties/Content-Length")
            if name is not None and content_length is not None:
                blobs[name] = int(content_length)
        marker = root.findtext("NextMarker") or ""
        if not marker:
            return blobs


def _write_manifest(path: Path, episodes: list[ManifestEpisode]) -> DatasetManifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(entry.__dict__, sort_keys=True) for entry in episodes) + "\n",
        encoding="utf-8",
    )
    return DatasetManifest.load(path)


def expand_vpt10_manifest(
    base_manifest: DatasetManifest,
    output_path: Path,
    *,
    target_bytes: int,
    seed: int = 7,
    container_url: str = VPT_CONTAINER_URL,
    prefix: str = VPT_10_PREFIX,
    available_files: dict[str, int] | None = None,
) -> DatasetManifest:
    """Add diverse VPT 10.x training groups until a raw-byte target is reached."""
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    if available_files is None:
        available_files = _list_blob_sizes(container_url, prefix)

    existing_groups = {entry.group for entry in base_manifest.episodes}
    candidates_by_group: dict[str, tuple[str, str]] = {}
    for video_relpath in sorted(available_files):
        if not video_relpath.endswith(".mp4"):
            continue
        actions_relpath = video_relpath.removesuffix(".mp4") + ".jsonl"
        if actions_relpath not in available_files:
            continue
        if available_files[video_relpath] < 1 or available_files[actions_relpath] < 1:
            continue
        episode = Path(video_relpath).stem
        group = episode.rsplit("-", 2)[0]
        if group not in existing_groups:
            candidates_by_group.setdefault(group, (video_relpath, actions_relpath))

    candidates = list(candidates_by_group.items())
    random.Random(seed).shuffle(candidates)
    selected = list(base_manifest.episodes)
    selected_bytes = sum(
        entry.video_bytes + entry.actions_bytes for entry in base_manifest.episodes
    )

    for group, (video_relpath, actions_relpath) in candidates:
        episode = Path(video_relpath).stem
        entry = ManifestEpisode(
            episode=episode,
            group=group,
            split="training",
            video_url=container_url.rstrip("/") + "/" + video_relpath,
            actions_url=container_url.rstrip("/") + "/" + actions_relpath,
            video_bytes=available_files[video_relpath],
            actions_bytes=available_files[actions_relpath],
        )
        selected.append(entry)
        selected_bytes += entry.video_bytes + entry.actions_bytes
        if selected_bytes >= target_bytes:
            break

    if selected_bytes < target_bytes:
        raise ValueError(
            f"only found {selected_bytes:,} available bytes, below target {target_bytes:,}"
        )

    return _write_manifest(output_path, selected)


def split_manifest(
    source: DatasetManifest,
    output_path: Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 7,
) -> DatasetManifest:
    """Create deterministic group-safe train/validation/test assignments.

    Existing validation and test groups keep their roles so sessions used for
    earlier model selection never become training or final-test evidence.
    Remaining groups are shuffled once and assigned by group count, not by
    individual episode.
    """
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")

    groups = sorted({entry.group for entry in source.episodes})
    if len(groups) < 3:
        raise ValueError("a three-way split needs at least three independent groups")
    test_count = max(1, round(len(groups) * test_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    if test_count + validation_count >= len(groups):
        raise ValueError("split fractions leave no independent training group")

    preserved_validation = {entry.group for entry in source.episodes if entry.split == "validation"}
    preserved_test = {entry.group for entry in source.episodes if entry.split == "test"}
    if len(preserved_validation) > validation_count:
        raise ValueError("existing validation groups exceed the requested validation split")
    if len(preserved_test) > test_count:
        raise ValueError("existing test groups exceed the requested test split")
    remaining = [
        group
        for group in groups
        if group not in preserved_validation and group not in preserved_test
    ]
    random.Random(seed).shuffle(remaining)
    needed_test = test_count - len(preserved_test)
    test_groups = preserved_test | set(remaining[:needed_test])
    needed_validation = validation_count - len(preserved_validation)
    validation_groups = preserved_validation | set(
        remaining[needed_test : needed_test + needed_validation]
    )

    assignments = {
        group: (
            "test"
            if group in test_groups
            else "validation"
            if group in validation_groups
            else "training"
        )
        for group in groups
    }
    episodes = [replace(entry, split=assignments[entry.group]) for entry in source.episodes]
    return _write_manifest(output_path, episodes)


def download_manifest(
    manifest: DatasetManifest,
    raw_dir: Path,
    *,
    split: str = "all",
    force: bool = False,
    workers: int = 3,
) -> None:
    if workers < 1:
        raise ValueError("download workers must be positive")
    selected = manifest.select(split)

    def download_entry(entry: ManifestEpisode) -> None:
        print(f"Fetching: {entry.episode}")
        video_path, action_path = entry.raw_paths(raw_dir)
        download_url(
            entry.video_url,
            video_path,
            force=force,
            expected_bytes=entry.video_bytes,
            show_progress=workers == 1,
        )
        download_url(
            entry.actions_url,
            action_path,
            force=force,
            expected_bytes=entry.actions_bytes,
            show_progress=workers == 1,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(download_entry, selected))


def preprocess_manifest(
    manifest: DatasetManifest,
    raw_dir: Path,
    processed_dir: Path,
    *,
    split: str = "all",
    force: bool = False,
    target_fps: float = 10.0,
    image_size: int = 64,
    horizon: int = 8,
) -> None:
    selected = manifest.select(split)
    for index, entry in enumerate(selected, 1):
        video_path, action_path = entry.raw_paths(raw_dir)
        output_path = entry.processed_path(processed_dir)
        print(f"[{index}/{len(selected)}] {entry.episode}")
        if output_path.exists() and not force:
            print(f"Already processed: {output_path}")
            continue
        if not video_path.exists() or not action_path.exists():
            raise ValueError(f"raw pair is missing for {entry.episode}; run dataset-download")
        result = preprocess_episode(
            video_path,
            action_path,
            output_path,
            target_fps=target_fps,
            image_size=image_size,
            horizon=horizon,
        )
        print(
            f"  frames={result.model_frames:,} valid_transitions={result.valid_transitions:,} "
            f"sequences={result.valid_sequences:,}"
        )


def dataset_status(
    manifest: DatasetManifest,
    raw_dir: Path,
    processed_dir: Path,
    *,
    verify_processed: bool = False,
) -> DatasetStatus:
    raw_complete = 0
    processed_complete = 0
    for entry in manifest.episodes:
        video_path, actions_path = entry.raw_paths(raw_dir)
        raw_ok = (
            video_path.exists()
            and actions_path.exists()
            and video_path.stat().st_size == entry.video_bytes
            and actions_path.stat().st_size == entry.actions_bytes
        )
        raw_complete += int(raw_ok)
        processed_path = entry.processed_path(processed_dir)
        if processed_path.exists():
            if verify_processed:
                episode = ProcessedEpisode.load(processed_path)
                if episode.episode != entry.episode:
                    raise ValueError(f"processed episode identity mismatch: {processed_path}")
                if episode.frames.shape[1:] != (64, 64, 3):
                    raise ValueError(f"processed episode is not 64x64 RGB: {processed_path}")
                if episode.model_fps != 10.0:
                    raise ValueError(f"processed episode is not 10 Hz: {processed_path}")
            processed_complete += 1
    split_groups = {
        split: {entry.group for entry in manifest.select(split)} for split in DATASET_SPLITS
    }
    return DatasetStatus(
        episodes=len(manifest.episodes),
        groups=len(set().union(*split_groups.values())),
        training_groups=len(split_groups["training"]),
        validation_groups=len(split_groups["validation"]),
        test_groups=len(split_groups["test"]),
        raw_complete=raw_complete,
        processed_complete=processed_complete,
        expected_raw_bytes=sum(
            entry.video_bytes + entry.actions_bytes for entry in manifest.episodes
        ),
    )
