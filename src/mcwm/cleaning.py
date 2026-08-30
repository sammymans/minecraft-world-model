"""Explicit cleaning rules and temporal aggregation for VPT actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mcwm.vpt import MOVEMENT_KEYS, VPTAction

MIN_FRAME_DT_MS = 20
MAX_FRAME_DT_MS = 100
SUPPORTED_KEYS = frozenset(MOVEMENT_KEYS)


class RejectionReason(IntEnum):
    """Exclusive reason assigned using the order in ``transition_reason``."""

    VALID = 0
    GUI_OPEN = 1
    ATTACK = 2
    USE = 3
    PICK_BLOCK = 4
    OTHER_MOUSE_BUTTON = 5
    UNSUPPORTED_KEY = 6
    BAD_TIMESTAMP = 7


@dataclass(frozen=True)
class AggregatedTransitions:
    """A downsampled timeline before frames are decoded."""

    source_frame_indices: np.ndarray
    actions: np.ndarray
    rejection_reasons: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.rejection_reasons == RejectionReason.VALID


@dataclass(frozen=True)
class AuditReport:
    source_frames: int
    model_frames: int
    total_transitions: int
    accepted_transitions: int
    rejection_counts: dict[str, int]
    sequence_horizon: int
    valid_sequences: int

    @property
    def acceptance_rate(self) -> float:
        if self.total_transitions == 0:
            return 0.0
        return self.accepted_transitions / self.total_transitions


def _button_reason(buttons: set[int]) -> RejectionReason | None:
    if 0 in buttons:
        return RejectionReason.ATTACK
    if 1 in buttons:
        return RejectionReason.USE
    if 2 in buttons:
        return RejectionReason.PICK_BLOCK
    if buttons:
        return RejectionReason.OTHER_MOUSE_BUTTON
    return None


def transition_reason(
    actions: list[VPTAction],
    start: int,
    stride: int,
) -> RejectionReason:
    """Return why the model transition from ``start`` to ``start + stride`` is invalid.

    The checks are ordered so every rejected transition has exactly one primary
    reason. GUI state includes the target frame; controls and timing cover the
    source intervals that cause the target frame.
    """
    frame_records = actions[start : start + stride + 1]
    applied_actions = actions[start : start + stride]

    if any(action.gui_open for action in frame_records):
        return RejectionReason.GUI_OPEN

    buttons = {button for action in applied_actions for button in action.mouse_buttons}
    if button_reason := _button_reason(buttons):
        return button_reason

    if any(action.keys - SUPPORTED_KEYS for action in applied_actions):
        return RejectionReason.UNSUPPORTED_KEY

    for index in range(start, start + stride):
        current = actions[index].timestamp_ms
        following = actions[index + 1].timestamp_ms
        if current is None or following is None:
            return RejectionReason.BAD_TIMESTAMP
        delta = following - current
        if not MIN_FRAME_DT_MS <= delta <= MAX_FRAME_DT_MS:
            return RejectionReason.BAD_TIMESTAMP

    return RejectionReason.VALID


def aggregate_actions(actions: list[VPTAction], start: int, stride: int) -> np.ndarray:
    """Combine 20 Hz controls into one lower-frequency model action.

    Key values are the fraction of the interval held. Mouse deltas are summed
    because camera movement accumulates across the interval.
    """
    vectors = np.asarray(
        [action.movement_vector() for action in actions[start : start + stride]],
        dtype=np.float32,
    )
    combined = vectors.mean(axis=0)
    combined[-2:] = vectors[:, -2:].sum(axis=0)
    return combined


def build_aggregated_transitions(
    actions: list[VPTAction],
    paired_frames: int,
    stride: int = 2,
) -> AggregatedTransitions:
    """Create the model-rate timeline shared by auditing and preprocessing."""
    if stride < 1:
        raise ValueError("stride must be positive")
    paired_frames = min(paired_frames, len(actions))
    if paired_frames < stride + 1:
        raise ValueError("episode is too short for the requested stride")

    source_indices = np.arange(0, paired_frames, stride, dtype=np.int32)
    transition_count = len(source_indices) - 1
    model_actions = np.empty((transition_count, 9), dtype=np.float32)
    reasons = np.empty(transition_count, dtype=np.int8)

    for model_index, source_index in enumerate(source_indices[:-1]):
        start = int(source_index)
        model_actions[model_index] = aggregate_actions(actions, start, stride)
        reasons[model_index] = transition_reason(actions, start, stride)

    return AggregatedTransitions(source_indices, model_actions, reasons)


def count_valid_sequences(valid: np.ndarray, horizon: int) -> int:
    """Count windows with one context transition followed by ``horizon`` valid steps."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    model_frame_count = len(valid) + 1
    return sum(
        bool(valid[start - 1 : start + horizon].all())
        for start in range(1, model_frame_count - horizon)
    )


def audit_transitions(
    actions: list[VPTAction],
    paired_frames: int,
    *,
    stride: int = 2,
    horizon: int = 8,
) -> tuple[AuditReport, AggregatedTransitions]:
    transitions = build_aggregated_transitions(actions, paired_frames, stride=stride)
    counts = Counter(
        RejectionReason(int(reason)).name.lower() for reason in transitions.rejection_reasons
    )
    counts.pop(RejectionReason.VALID.name.lower(), None)
    accepted = int(transitions.valid.sum())
    report = AuditReport(
        source_frames=paired_frames,
        model_frames=len(transitions.source_frame_indices),
        total_transitions=len(transitions.actions),
        accepted_transitions=accepted,
        rejection_counts=dict(sorted(counts.items())),
        sequence_horizon=horizon,
        valid_sequences=count_valid_sequences(transitions.valid, horizon),
    )
    return report, transitions
