"""Small, explicit helpers for OpenAI's public VPT contractor data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MOVEMENT_KEYS = (
    "w",
    "a",
    "s",
    "d",
    "space",
    "left.shift",
    "left.control",
)

DISPLAY_NAMES = {
    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",
    "space": "JUMP",
    "left.shift": "SNEAK",
    "left.control": "SPRINT",
}

MOUSE_BUTTON_NAMES = {
    0: "ATTACK",
    1: "USE",
    2: "PICK",
}


@dataclass(frozen=True)
class VPTAction:
    """The small action subset used by this project at one VPT frame."""

    keys: frozenset[str]
    mouse_dx: float
    mouse_dy: float
    mouse_buttons: tuple[int, ...]
    gui_open: bool
    timestamp_ms: int | None

    def movement_vector(self) -> tuple[float, ...]:
        """Return [W, A, S, D, jump, sprint, sneak, mouse dx, mouse dy]."""
        return (
            float("w" in self.keys),
            float("a" in self.keys),
            float("s" in self.keys),
            float("d" in self.keys),
            float("space" in self.keys),
            float("left.control" in self.keys),
            float("left.shift" in self.keys),
            self.mouse_dx,
            self.mouse_dy,
        )

    def label(self) -> str:
        """Return a compact human-readable action label."""
        pressed = [DISPLAY_NAMES[key] for key in MOVEMENT_KEYS if key in self.keys]
        pressed.extend(
            MOUSE_BUTTON_NAMES.get(button, f"MOUSE-{button}") for button in self.mouse_buttons
        )
        key_text = "+".join(pressed) if pressed else "none"
        return f"keys={key_text}  mouse=({self.mouse_dx:+.1f}, {self.mouse_dy:+.1f})"


def normalize_key(raw_key: object) -> str:
    """Convert VPT names such as ``key.keyboard.w`` to ``w``."""
    text = str(raw_key).lower()
    prefix = "key.keyboard."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def parse_action(record: dict[str, Any]) -> VPTAction:
    """Extract the controls we need while ignoring unrelated VPT metadata."""
    keyboard = record.get("keyboard") or {}
    mouse = record.get("mouse") or {}
    keys = frozenset(normalize_key(key) for key in (keyboard.get("keys") or []))

    return VPTAction(
        keys=keys,
        mouse_dx=float(mouse.get("dx") or 0.0),
        mouse_dy=float(mouse.get("dy") or 0.0),
        mouse_buttons=tuple(int(button) for button in (mouse.get("buttons") or [])),
        gui_open=bool(record.get("isGuiOpen", False)),
        timestamp_ms=int(record["milli"]) if record.get("milli") is not None else None,
    )


def iter_actions(path: Path) -> Iterable[VPTAction]:
    """Stream a JSONL action file without loading the whole episode into memory."""
    # A small number of official VPT records contain legacy bytes inside the unused
    # keyboard.chars string. Replacing only malformed text keeps JSON and action parsing strict.
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            yield parse_action(record)


def load_actions(path: Path, limit: int | None = None) -> list[VPTAction]:
    """Load at most ``limit`` actions; ``None`` loads the entire episode."""
    actions: list[VPTAction] = []
    for action in iter_actions(path):
        actions.append(action)
        if limit is not None and len(actions) >= limit:
            break
    return actions


def action_counts(actions: Iterable[VPTAction]) -> Counter[str]:
    """Count how many frames contain each movement key."""
    counts: Counter[str] = Counter()
    for action in actions:
        for key in MOVEMENT_KEYS:
            if key in action.keys:
                counts[DISPLAY_NAMES[key]] += 1
    return counts
