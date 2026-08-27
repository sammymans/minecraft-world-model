"""Canonical processed episodes and contiguous sequence sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mcwm.cleaning import RejectionReason, audit_transitions
from mcwm.vpt import load_actions


@dataclass(frozen=True)
class ProcessedEpisode:
    episode: str
    frames: np.ndarray
    actions: np.ndarray
    rejection_reasons: np.ndarray
    source_frame_indices: np.ndarray
    model_fps: float

    @property
    def valid(self) -> np.ndarray:
        return self.rejection_reasons == RejectionReason.VALID

    @classmethod
    def load(cls, path: Path) -> ProcessedEpisode:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            episode = cls(
                episode=metadata["episode"],
                frames=archive["frames"].copy(),
                actions=archive["actions"].copy(),
                rejection_reasons=archive["rejection_reasons"].copy(),
                source_frame_indices=archive["source_frame_indices"].copy(),
                model_fps=float(metadata["model_fps"]),
            )
        episode.validate()
        return episode

    def validate(self) -> None:
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError("frames must be uint8 [time, height, width, RGB]")
        if self.actions.shape != (len(self.frames) - 1, 9):
            raise ValueError("actions must be [time - 1, 9]")
        if self.rejection_reasons.shape != (len(self.frames) - 1,):
            raise ValueError("rejection_reasons must be [time - 1]")
        if self.source_frame_indices.shape != (len(self.frames),):
            raise ValueError("source_frame_indices must be [time]")


@dataclass(frozen=True)
class SequenceSample:
    episode: str
    start_step: int
    frames: np.ndarray
    actions: np.ndarray
    source_frame_indices: np.ndarray


@dataclass(frozen=True)
class PreprocessResult:
    output_path: Path
    source_frames: int
    model_frames: int
    valid_transitions: int
    valid_sequences: int


class SequenceDataset:
    """Index only valid, contiguous windows; never cross an episode boundary."""

    def __init__(self, episodes: list[ProcessedEpisode], horizon: int = 8):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.episodes = episodes
        self.horizon = horizon
        self.index: list[tuple[int, int]] = []

        for episode_index, episode in enumerate(episodes):
            model_frame_count = len(episode.frames)
            for start in range(1, model_frame_count - horizon):
                if episode.valid[start - 1 : start + horizon].all():
                    self.index.append((episode_index, start))

    @classmethod
    def from_paths(cls, paths: list[Path], horizon: int = 8) -> SequenceDataset:
        return cls([ProcessedEpisode.load(path) for path in paths], horizon=horizon)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> SequenceSample:
        episode_index, start = self.index[item]
        episode = self.episodes[episode_index]
        stop = start + self.horizon
        return SequenceSample(
            episode=episode.episode,
            start_step=start,
            frames=episode.frames[start - 1 : stop + 1],
            actions=episode.actions[start:stop],
            source_frame_indices=episode.source_frame_indices[start - 1 : stop + 1],
        )


def episode_group(stem: str) -> str:
    """Group consecutive VPT segments from the same player/session."""
    parts = stem.rsplit("-", 2)
    return parts[0] if len(parts) == 3 else stem


def split_episode_paths(
    paths: list[Path], validation_group_count: int = 1
) -> tuple[list[Path], list[Path]]:
    """Split whole player/session groups so adjacent segments cannot leak."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(paths):
        groups.setdefault(episode_group(path.stem), []).append(path)
    group_names = sorted(groups)
    if len(group_names) <= validation_group_count:
        raise ValueError("need at least two player/session groups for a held-out split")
    validation_names = set(group_names[:validation_group_count])
    train = [path for name in group_names if name not in validation_names for path in groups[name]]
    validation = [path for name in group_names if name in validation_names for path in groups[name]]
    return train, validation


def _center_crop_resize(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = frame_bgr[top : top + side, left : left + side]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def preprocess_episode(
    video_path: Path,
    action_path: Path,
    output_path: Path,
    *,
    target_fps: float = 10.0,
    image_size: int = 64,
    horizon: int = 8,
) -> PreprocessResult:
    """Decode one immutable raw pair into a small canonical NPZ episode."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = round(source_fps / target_fps)
    if stride < 1 or abs(source_fps / stride - target_fps) > 0.01:
        capture.release()
        raise ValueError(f"target FPS {target_fps} must evenly divide source FPS {source_fps}")

    actions = load_actions(action_path)
    paired_frames = min(video_frames, len(actions))
    report, transitions = audit_transitions(
        actions,
        paired_frames,
        stride=stride,
        horizon=horizon,
    )

    wanted_indices = transitions.source_frame_indices
    decoded: list[np.ndarray] = []
    wanted_position = 0
    for frame_index in range(int(wanted_indices[-1]) + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Video ended unexpectedly at frame {frame_index}")
        if frame_index == int(wanted_indices[wanted_position]):
            decoded.append(_center_crop_resize(frame, image_size))
            wanted_position += 1
            if wanted_position == len(wanted_indices):
                break
    capture.release()

    frames = np.stack(decoded).astype(np.uint8, copy=False)
    metadata = json.dumps(
        {
            "episode": video_path.stem,
            "source_fps": source_fps,
            "model_fps": source_fps / stride,
            "image_size": image_size,
            "action_order": [
                "w",
                "a",
                "s",
                "d",
                "jump",
                "sprint",
                "sneak",
                "mouse_dx",
                "mouse_dy",
            ],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=metadata,
            frames=frames,
            actions=transitions.actions,
            rejection_reasons=transitions.rejection_reasons,
            source_frame_indices=transitions.source_frame_indices,
        )
    temporary.replace(output_path)

    return PreprocessResult(
        output_path=output_path,
        source_frames=report.source_frames,
        model_frames=report.model_frames,
        valid_transitions=report.accepted_transitions,
        valid_sequences=report.valid_sequences,
    )


def _format_action(action: np.ndarray) -> str:
    names = ("W", "A", "S", "D", "JUMP", "SPRINT", "SNEAK")
    keys = [
        name if value == 1 else f"{name}:{value:.1f}"
        for name, value in zip(names, action[:7], strict=True)
        if value
    ]
    key_text = "+".join(keys) if keys else "none"
    return f"{key_text}  mouse=({action[-2]:+.0f},{action[-1]:+.0f})"


def save_sequence_sheet(sample: SequenceSample, output_path: Path, columns: int = 5) -> None:
    """Render every frame and intervening action from one exact training sample."""
    tile_width = 256
    tile_height = 244
    rows = (len(sample.frames) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8)

    for index, frame_rgb in enumerate(sample.frames):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * tile_height
        large = cv2.resize(frame_rgb, (224, 192), interpolation=cv2.INTER_NEAREST)
        canvas[y + 8 : y + 200, x + 16 : x + 240] = cv2.cvtColor(large, cv2.COLOR_RGB2BGR)
        if index == 0:
            role = "context t-1"
        elif index == 1:
            role = "current t"
        else:
            role = f"target t+{index - 1}"
        cv2.putText(
            canvas,
            f"{role} | source {sample.source_frame_indices[index]}",
            (x + 10, y + 218),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if 1 <= index <= len(sample.actions):
            cv2.putText(
                canvas,
                "next: " + _format_action(sample.actions[index - 1]),
                (x + 10, y + 237),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (100, 220, 255),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"Could not write sequence sheet: {output_path}")
