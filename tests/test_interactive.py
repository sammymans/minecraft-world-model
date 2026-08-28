from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from mcwm.interactive import (
    ACTION_INDEX,
    InteractiveRolloutEngine,
    RolloutSeed,
    make_action,
    make_live_action,
    parse_action_script,
    run_action_comparison,
    save_action_comparison,
)
from mcwm.model import SpatialLatentDynamics


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
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :, None, None].expand(-1, 3, 2, 2)


def test_make_action_uses_canonical_training_order() -> None:
    action = make_action(("w", "jump", "sneak"), mouse_dx=12, mouse_dy=-4)

    assert action.tolist() == [1, 0, 0, 0, 1, 0, 1, 12, -4]


def test_realtime_action_uses_held_keys_and_relative_camera() -> None:
    action = make_live_action(
        {"w", "sprint", "look_right"},
        mouse_dx=8,
        mouse_dy=-3,
        camera_step=30,
    )

    assert action.tolist() == [1, 0, 0, 0, 0, 1, 0, 38, -3]


def test_script_parses_repetition_and_camera_actions() -> None:
    actions = parse_action_script("w+sprint*2, w+look_right, idle*2", camera_step=30)

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


class _SpatialDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :3]


def test_interactive_engine_accepts_spatial_latents() -> None:
    dynamics = SpatialLatentDynamics(latent_channels=3, hidden_channels=4, blocks=1)
    latent = torch.zeros((1, 3, 2, 2))
    engine = InteractiveRolloutEngine(
        _SpatialDecoder(),
        dynamics,
        latent,
        latent,
        np.zeros((2, 2, 3), dtype=np.uint8),
        torch.device("cpu"),
    )

    frame = engine.step(make_action(("w",)))

    assert frame.shape == (2, 2, 3)
    assert engine.current_latent.shape == (1, 3, 2, 2)


def test_interactive_engine_reseeds_without_reloading_models() -> None:
    engine = InteractiveRolloutEngine(
        _GrayDecoder(),
        _MouseDynamics(),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        np.zeros((2, 2, 3), dtype=np.uint8),
        torch.device("cpu"),
    )
    engine.step(make_action(mouse_dx=0.5))
    frame = np.full((2, 2, 3), 128, dtype=np.uint8)

    result = engine.reseed(RolloutSeed("new", 4, 10, frame, frame))

    assert engine.steps == 0
    assert np.array_equal(result, frame)
    assert engine.current_latent.item() == pytest.approx(128 / 255)


def test_action_comparison_restarts_every_script_from_the_same_seed(tmp_path) -> None:
    engine = InteractiveRolloutEngine(
        _GrayDecoder(),
        _MouseDynamics(),
        torch.tensor([[0.0]]),
        torch.tensor([[0.25]]),
        np.full((2, 2, 3), 64, dtype=np.uint8),
        torch.device("cpu"),
    )
    seed = RolloutSeed(
        "episode", 7, 10.0, np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 64, np.uint8)
    )
    output = tmp_path / "comparison.png"

    result = run_action_comparison(
        engine,
        seed,
        ["look_right*2", "look_left*3"],
        output,
        camera_step=4.0,
        tile=64,
    )

    # The second script must not inherit the first one's drift; both start from
    # the shared seed latent of 0.25.
    assert engine.steps == 3
    assert result.steps == 3
    assert result.episode == "episode"
    assert output.exists()
    written = cv2.imread(str(output))
    # Two rows, and the widest row is the seed frame plus three imagined steps.
    assert written.shape[0] == 34 + 2 * (64 + 6)
    assert written.shape[1] == 250 + 4 * (64 + 6)


def test_action_comparison_rejects_empty_and_undersized_requests(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one script"):
        save_action_comparison([], tmp_path / "empty.png")
    with pytest.raises(ValueError, match="tile must be at least"):
        save_action_comparison(
            [("idle", [np.zeros((2, 2, 3), np.uint8)])], tmp_path / "small.png", tile=16
        )
