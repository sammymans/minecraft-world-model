"""Disk-backed temporal latent data for full-scale V2 training.

The Stage 2 dataset intentionally holds a few encoded episodes in memory.  This
module is the Stage 3 counterpart: encode each frozen observation exactly once,
store contiguous arrays on disk, and expose all clean temporal windows without
keeping the processed RGB dataset in RAM.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from mcwm.dataset import ProcessedEpisode
from mcwm.latent_diffusion_v2 import ACTION_BUCKETS, TemporalSequenceReference

LATENT_CACHE_V2_FORMAT = 1


@dataclass(frozen=True)
class CachedEpisode:
    episode: str
    processed_path: str
    frame_offset: int
    frame_count: int
    action_offset: int
    action_count: int


@dataclass(frozen=True)
class LatentCacheMetadata:
    format_version: int
    split: str
    autoencoder_sha256: str
    manifest_sha256: str
    latent_shape: tuple[int, int, int]
    total_frames: int
    total_actions: int
    latent_mean: tuple[float, ...]
    latent_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    source_signature: tuple[tuple[str, int, int], ...]
    episodes: tuple[CachedEpisode, ...]

    @classmethod
    def from_json(cls, path: Path) -> LatentCacheMetadata:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["latent_shape"] = tuple(record["latent_shape"])
        for name in ("latent_mean", "latent_std", "action_mean", "action_std"):
            record[name] = tuple(record[name])
        record["source_signature"] = tuple(tuple(item) for item in record["source_signature"])
        record["episodes"] = tuple(CachedEpisode(**item) for item in record["episodes"])
        return cls(**record)


def _source_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (path.stem, path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    )


def _cache_path(
    root: Path, autoencoder_sha256: str, manifest_sha256: str, split: str
) -> Path:
    return root / autoencoder_sha256[:12] / manifest_sha256[:12] / split


class LatentEpisodeCache:
    """Memory-mapped latent/action timelines with immutable cache metadata."""

    def __init__(self, path: Path, metadata: LatentCacheMetadata):
        self.path = path
        self.metadata = metadata
        latent_shape = (metadata.total_frames, *metadata.latent_shape)
        self.latents = np.memmap(
            path / "latents.f16", dtype=np.float16, mode="r", shape=latent_shape
        )
        self.actions = np.memmap(
            path / "actions.f32", dtype=np.float32, mode="r", shape=(metadata.total_actions, 9)
        )
        self.valid = np.memmap(
            path / "valid.u8", dtype=np.uint8, mode="r", shape=(metadata.total_actions,)
        )

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        autoencoder_sha256: str,
        manifest_sha256: str,
        split: str,
        paths: list[Path] | None = None,
    ) -> LatentEpisodeCache:
        path = _cache_path(root, autoencoder_sha256, manifest_sha256, split)
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"V2 latent cache is missing: {metadata_path}")
        metadata = LatentCacheMetadata.from_json(metadata_path)
        if metadata.format_version != LATENT_CACHE_V2_FORMAT:
            raise ValueError("V2 latent cache has an incompatible format")
        if metadata.autoencoder_sha256 != autoencoder_sha256:
            raise ValueError("V2 latent cache uses a different autoencoder")
        if metadata.manifest_sha256 != manifest_sha256 or metadata.split != split:
            raise ValueError("V2 latent cache uses a different manifest split")
        if paths is not None and metadata.source_signature != _source_signature(paths):
            raise ValueError("V2 latent cache source episodes changed; rebuild the cache")
        expected_sizes = {
            "latents.f16": metadata.total_frames * int(np.prod(metadata.latent_shape)) * 2,
            "actions.f32": metadata.total_actions * 9 * 4,
            "valid.u8": metadata.total_actions,
        }
        for filename, size in expected_sizes.items():
            candidate = path / filename
            if not candidate.exists() or candidate.stat().st_size != size:
                raise ValueError(f"V2 latent cache file is incomplete: {candidate}")
        return cls(path, metadata)

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        paths: list[Path],
        autoencoder: nn.Module,
        device: torch.device,
        root: Path,
        *,
        autoencoder_sha256: str,
        manifest_sha256: str,
        split: str,
        encode_batch_size: int = 128,
        force: bool = False,
    ) -> LatentEpisodeCache:
        if not paths:
            raise ValueError("cannot build an empty V2 latent cache")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        cache_path = _cache_path(root, autoencoder_sha256, manifest_sha256, split)
        if not force:
            try:
                return cls.open(
                    root,
                    autoencoder_sha256=autoencoder_sha256,
                    manifest_sha256=manifest_sha256,
                    split=split,
                    paths=paths,
                )
            except ValueError:
                pass

        cache_path.mkdir(parents=True, exist_ok=True)
        temporary = {
            name: cache_path / f"{name}.part"
            for name in ("latents.f16", "actions.f32", "valid.u8")
        }
        records: list[CachedEpisode] = []
        frame_offset = 0
        action_offset = 0
        latent_shape: tuple[int, int, int] | None = None
        latent_sum: torch.Tensor | None = None
        latent_squared: torch.Tensor | None = None
        latent_values = 0
        action_sum = torch.zeros(9, dtype=torch.float64)
        action_squared = torch.zeros(9, dtype=torch.float64)
        action_values = 0
        started = time.perf_counter()
        autoencoder.eval()
        with (
            temporary["latents.f16"].open("wb") as latent_file,
            temporary["actions.f32"].open("wb") as action_file,
            temporary["valid.u8"].open("wb") as valid_file,
        ):
            for number, processed_path in enumerate(paths, 1):
                episode = ProcessedEpisode.load(processed_path)
                episode_latents: list[torch.Tensor] = []
                for start in range(0, len(episode.frames), encode_batch_size):
                    frames = episode.frames[start : start + encode_batch_size]
                    contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
                    batch = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32)
                    encoded = autoencoder.encode(batch.div_(255.0)).to(
                        device="cpu", dtype=torch.float16
                    )
                    episode_latents.append(encoded)
                timeline = torch.cat(episode_latents)
                current_shape = tuple(int(value) for value in timeline.shape[1:])
                if latent_shape is None:
                    latent_shape = current_shape
                    latent_sum = torch.zeros(latent_shape[0], dtype=torch.float64)
                    latent_squared = torch.zeros(latent_shape[0], dtype=torch.float64)
                elif current_shape != latent_shape:
                    raise ValueError("cached V2 episodes have inconsistent latent shapes")

                timeline.numpy().tofile(latent_file)
                actions = episode.actions.astype(np.float32, copy=False)
                valid = episode.valid.astype(np.uint8, copy=False)
                actions.tofile(action_file)
                valid.tofile(valid_file)
                records.append(
                    CachedEpisode(
                        episode=episode.episode,
                        processed_path=str(processed_path),
                        frame_offset=frame_offset,
                        frame_count=len(episode.frames),
                        action_offset=action_offset,
                        action_count=len(actions),
                    )
                )
                values = timeline.float()
                assert latent_sum is not None and latent_squared is not None
                latent_sum += values.sum(dim=(0, 2, 3), dtype=torch.float64)
                latent_squared += values.square().sum(dim=(0, 2, 3), dtype=torch.float64)
                latent_values += len(values) * values.shape[2] * values.shape[3]
                valid_actions = torch.from_numpy(actions[valid.astype(bool)]).double()
                action_sum += valid_actions.sum(dim=0)
                action_squared += valid_actions.square().sum(dim=0)
                action_values += len(valid_actions)
                frame_offset += len(episode.frames)
                action_offset += len(actions)
                if number == 1 or number % 25 == 0 or number == len(paths):
                    elapsed = time.perf_counter() - started
                    print(
                        f"cached {split} episode {number:,}/{len(paths):,}: "
                        f"{frame_offset:,} frames ({number / max(elapsed, 1e-9):.2f} episodes/s)"
                    )

        if latent_shape is None or latent_sum is None or latent_squared is None:
            raise RuntimeError("V2 latent cache did not encode any observations")
        if latent_values == 0 or action_values == 0:
            raise RuntimeError("V2 latent cache has no valid statistics")
        latent_mean = latent_sum / latent_values
        latent_variance = latent_squared / latent_values - latent_mean.square()
        action_mean = action_sum / action_values
        action_variance = action_squared / action_values - action_mean.square()
        metadata = LatentCacheMetadata(
            format_version=LATENT_CACHE_V2_FORMAT,
            split=split,
            autoencoder_sha256=autoencoder_sha256,
            manifest_sha256=manifest_sha256,
            latent_shape=latent_shape,
            total_frames=frame_offset,
            total_actions=action_offset,
            latent_mean=tuple(float(value) for value in latent_mean),
            latent_std=tuple(float(value) for value in latent_variance.clamp_min(1e-8).sqrt()),
            action_mean=tuple(float(value) for value in action_mean),
            action_std=tuple(float(value) for value in action_variance.clamp_min(0.05**2).sqrt()),
            source_signature=_source_signature(paths),
            episodes=tuple(records),
        )
        for filename, part in temporary.items():
            part.replace(cache_path / filename)
        metadata_part = cache_path / "metadata.json.part"
        metadata_part.write_text(
            json.dumps(
                {
                    **asdict(metadata),
                    "episodes": [asdict(record) for record in metadata.episodes],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_part.replace(cache_path / "metadata.json")
        return cls.open(
            root,
            autoencoder_sha256=autoencoder_sha256,
            manifest_sha256=manifest_sha256,
            split=split,
            paths=paths,
        )


def _bucket_codes(actions: np.ndarray, camera_threshold: float = 2.0) -> np.ndarray:
    codes = np.full(len(actions), 3, dtype=np.uint8)
    forward = (actions[:, -2] > -camera_threshold) & (
        actions[:, -2] < camera_threshold
    ) & (actions[:, 0] > 0)
    codes[forward] = 0
    codes[actions[:, -2] <= -camera_threshold] = 1
    codes[actions[:, -2] >= camera_threshold] = 2
    return codes


class CachedTemporalLatentDataset(Dataset[dict[str, torch.Tensor]]):
    """All valid temporal windows from one memory-mapped V2 cache."""

    def __init__(self, cache: LatentEpisodeCache, *, context_frames: int = 8):
        if context_frames < 2:
            raise ValueError("cached temporal contexts need at least two frames")
        self.cache = cache
        self.context_frames = context_frames
        episode_indices: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        buckets: list[np.ndarray] = []
        changes: list[np.ndarray] = []
        for episode_index, episode in enumerate(cache.metadata.episodes):
            valid = np.asarray(
                cache.valid[episode.action_offset : episode.action_offset + episode.action_count]
            )
            if len(valid) < context_frames:
                continue
            clean = np.convolve(valid, np.ones(context_frames, dtype=np.int16), mode="valid")
            local_starts = np.flatnonzero(clean == context_frames).astype(np.int32)
            if not len(local_starts):
                continue
            actions = np.asarray(
                cache.actions[
                    episode.action_offset : episode.action_offset + episode.action_count
                ]
            )
            target_codes = _bucket_codes(actions[local_starts + context_frames - 1])
            previous_codes = _bucket_codes(actions[local_starts + context_frames - 2])
            episode_indices.append(np.full(len(local_starts), episode_index, dtype=np.int32))
            starts.append(local_starts)
            buckets.append(target_codes)
            changes.append(target_codes != previous_codes)
        if not starts:
            raise ValueError("V2 latent cache contains no clean temporal windows")
        self.episode_indices = np.concatenate(episode_indices)
        self.starts = np.concatenate(starts)
        self.bucket_codes = np.concatenate(buckets)
        self.action_changes = np.concatenate(changes)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = self.cache.metadata.episodes[int(self.episode_indices[item])]
        start = int(self.starts[item])
        frame_start = episode.frame_offset + start
        action_start = episode.action_offset + start
        stop = frame_start + self.context_frames
        context = np.array(self.cache.latents[frame_start:stop], dtype=np.float32, copy=True)
        target = np.array(self.cache.latents[stop], dtype=np.float32, copy=True)
        actions = np.array(
            self.cache.actions[action_start : action_start + self.context_frames],
            dtype=np.float32,
            copy=True,
        )
        return {
            "context_latents": torch.from_numpy(context),
            "actions": torch.from_numpy(actions),
            "target_latent": torch.from_numpy(target),
            "action_changed": torch.tensor(bool(self.action_changes[item])),
        }

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.cache.metadata.latent_shape

    @property
    def action_bucket_counts(self) -> dict[str, int]:
        return {
            bucket: int((self.bucket_codes == code).sum())
            for code, bucket in enumerate(ACTION_BUCKETS)
        }

    @property
    def action_change_count(self) -> int:
        return int(self.action_changes.sum())

    def normalization_statistics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata = self.cache.metadata
        return tuple(
            torch.tensor(values, dtype=torch.float32)
            for values in (
                metadata.latent_mean,
                metadata.latent_std,
                metadata.action_mean,
                metadata.action_std,
            )
        )  # type: ignore[return-value]

    def reference(self, item: int) -> TemporalSequenceReference:
        episode = self.cache.metadata.episodes[int(self.episode_indices[item])]
        return TemporalSequenceReference(
            episode.episode,
            int(self.starts[item]),
            ACTION_BUCKETS[int(self.bucket_codes[item])],
        )

    def subset_indices(
        self,
        maximum: int,
        *,
        action_changes_only: bool = False,
        seed: int = 7,
    ) -> list[int]:
        if maximum < 1:
            raise ValueError("maximum subset size must be positive")
        candidates = (
            np.flatnonzero(self.action_changes)
            if action_changes_only
            else np.arange(len(self))
        )
        if len(candidates) <= maximum:
            return [int(index) for index in candidates]
        generator = np.random.default_rng(seed)
        return [int(index) for index in generator.choice(candidates, maximum, replace=False)]
