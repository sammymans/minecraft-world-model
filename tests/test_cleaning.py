from __future__ import annotations

from mcwm.cleaning import (
    RejectionReason,
    audit_transitions,
    build_aggregated_transitions,
)
from mcwm.vpt import VPTAction


def _action(
    *,
    keys: tuple[str, ...] = (),
    dx: float = 0.0,
    buttons: tuple[int, ...] = (),
    gui: bool = False,
    timestamp: int,
) -> VPTAction:
    return VPTAction(
        keys=frozenset(keys),
        mouse_dx=dx,
        mouse_dy=0.0,
        mouse_buttons=buttons,
        gui_open=gui,
        timestamp_ms=timestamp,
    )


def test_two_source_actions_are_aggregated_into_one_model_action() -> None:
    actions = [
        _action(keys=("w",), dx=2, timestamp=0),
        _action(keys=("w",), dx=3, timestamp=50),
        _action(timestamp=100),
    ]

    transitions = build_aggregated_transitions(actions, paired_frames=3, stride=2)

    assert transitions.source_frame_indices.tolist() == [0, 2]
    assert transitions.actions.shape == (1, 9)
    assert transitions.actions[0, 0] == 1.0
    assert transitions.actions[0, -2] == 5.0
    assert transitions.valid.tolist() == [True]


def test_audit_assigns_one_explicit_rejection_reason() -> None:
    actions = [
        _action(buttons=(0,), timestamp=0),
        _action(timestamp=50),
        _action(gui=True, timestamp=100),
        _action(timestamp=150),
        _action(timestamp=200),
    ]

    report, transitions = audit_transitions(actions, paired_frames=5, stride=2, horizon=1)

    assert transitions.rejection_reasons.tolist() == [
        RejectionReason.GUI_OPEN,
        RejectionReason.GUI_OPEN,
    ]
    assert report.accepted_transitions == 0
    assert report.rejection_counts == {"gui_open": 2}


def test_bad_timestamp_is_rejected() -> None:
    actions = [
        _action(timestamp=0),
        _action(timestamp=250),
        _action(timestamp=300),
    ]

    transitions = build_aggregated_transitions(actions, paired_frames=3, stride=2)

    assert transitions.rejection_reasons.tolist() == [RejectionReason.BAD_TIMESTAMP]
