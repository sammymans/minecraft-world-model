from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mcwm.interactive import (
    ACTION_INDEX,
    InteractiveRolloutEngine,
    LiveActionState,
    make_action,
    parse_action_script,
)


class _MouseDynamics(nn.Module):
    latent_dim = 1
    action_dim = 9

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        del previous_latent
        return current_latent + action[:, ACTION_INDEX["mouse_dx"] :][:, :1]


class _GrayDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :, None, None].expand(-1, 3, 2, 2)


def test_make_action_uses_canonical_training_order() -> None:
    action = make_action(("w", "jump", "sneak"), mouse_dx=12, mouse_dy=-4)

    assert action.tolist() == [1, 0, 0, 0, 1, 0, 1, 12, -4]


def test_live_action_state_persists_keys_but_consumes_camera_once() -> None:
    state = LiveActionState()
    state.toggle("w")
    state.add_camera(8, -3)

    first = state.consume()
    second = state.consume()

    assert first.tolist() == [1, 0, 0, 0, 0, 0, 0, 8, -3]
    assert second.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert state.toggle("w") is False
    assert state.consume().sum() == 0


def test_script_parses_repetition_and_camera_actions() -> None:
    actions = parse_action_script(
        "w+sprint*2, w+look_right, idle*2", camera_step=30
    )

    assert actions.shape == (5, 9)
    assert actions[:2, ACTION_INDEX["w"]].tolist() == [1, 1]
    assert actions[:2, ACTION_INDEX["sprint"]].tolist() == [1, 1]
    assert actions[2, ACTION_INDEX["mouse_dx"]] == 30
    assert actions[3:].sum() == 0


def test_script_rejects_unknown_controls() -> None:
    with pytest.raises(ValueError, match="unknown scripted action"):
        parse_action_script("attack")


def test_interactive_engine_recursively_feeds_back_predictions_and_resets() -> None:
    engine = InteractiveRolloutEngine(
        _GrayDecoder(),
        _MouseDynamics(),
        torch.tensor([[99.0]]),
        torch.tensor([[0.0]]),
        np.zeros((2, 2, 3), dtype=np.uint8),
        torch.device("cpu"),
    )
    first = make_action(mouse_dx=0.25)
    second = make_action(mouse_dx=0.5)

    engine.step(first)
    frame = engine.step(second)

    assert engine.steps == 2
    assert engine.current_latent.item() == pytest.approx(0.75)
    assert frame[0, 0, 0] == 191
    reset = engine.reset()
    assert engine.steps == 0
    assert engine.current_latent.item() == pytest.approx(0)
    assert np.array_equal(reset, np.zeros((2, 2, 3), dtype=np.uint8))
