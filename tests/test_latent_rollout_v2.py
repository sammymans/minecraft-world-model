from __future__ import annotations

import torch
from torch import nn

from mcwm.latent_diffusion_v2 import TemporalActionUNet
from mcwm.latent_rollout_v2 import (
    CONTEXT_FRAMES,
    MAXIMUM_ROLLOUT_DEPTH,
    evaluate_rollout,
    rollout_diffusion_loss,
    self_generated_context,
)


class _RecordingModel(nn.Module):
    """Marks every generated latent and records the action window it saw."""

    context_frames = CONTEXT_FRAMES
    latent_channels = 2
    action_dim = 9

    def __init__(self) -> None:
        super().__init__()
        self.action_windows: list[torch.Tensor] = []
        self.calls = 0

    def sample(self, context, actions, *, steps, seed, **kwargs):
        del steps, seed, kwargs
        self.action_windows.append(actions.clone())
        self.calls += 1
        # -1000 downward marks generation order and is unmistakable in a window.
        return torch.full_like(context[:, -1], -1000.0 - self.calls)


def _window(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    total = CONTEXT_FRAMES + MAXIMUM_ROLLOUT_DEPTH
    context = (
        torch.arange(total, dtype=torch.float32)[None, :, None, None, None]
        .expand(batch, total, 2, 4, 4)
        .clone()
    )
    actions = (
        torch.arange(total, dtype=torch.float32)[None, :, None].expand(batch, total, 9).clone()
    )
    return context, actions


def test_zero_depth_context_matches_an_ordinary_training_window() -> None:
    model = _RecordingModel()
    context, actions = _window()

    window = self_generated_context(model, context, actions, 0, sampling_steps=4, seed=1)

    assert model.calls == 0
    assert window.shape == (2, CONTEXT_FRAMES, 2, 4, 4)
    # Frames 3..10 are exactly what the one-step trainer would have used.
    assert window[0, :, 0, 0, 0].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]


def test_full_depth_replaces_only_the_newest_frames_and_keeps_alignment() -> None:
    model = _RecordingModel()
    context, actions = _window()

    window = self_generated_context(
        model, context, actions, MAXIMUM_ROLLOUT_DEPTH, sampling_steps=4, seed=1
    )

    assert model.calls == MAXIMUM_ROLLOUT_DEPTH
    values = window[0, :, 0, 0, 0].tolist()
    # Five real frames (3..7) then three generations, oldest generation first.
    assert values[:5] == [3, 4, 5, 6, 7]
    assert values[5:] == [-1001.0, -1002.0, -1003.0]
    # Each generation must be driven by the action for the frame it produces:
    # frame 8 by action 7, frame 9 by action 8, frame 10 by action 9.
    driving = [w[0, -1, 0].item() for w in model.action_windows]
    assert driving == [7, 8, 9]
    # And each generation reads a contiguous eight-action history.
    for produced, seen in enumerate(model.action_windows):
        assert seen[0, :, 0].tolist() == [float(produced + offset) for offset in range(8)]


def test_intermediate_depth_lands_on_the_same_final_frames() -> None:
    model = _RecordingModel()
    context, actions = _window()

    window = self_generated_context(model, context, actions, 1, sampling_steps=4, seed=1)

    assert model.calls == 1
    values = window[0, :, 0, 0, 0].tolist()
    assert values[:7] == [3, 4, 5, 6, 7, 8, 9]
    assert values[7] == -1001.0
    # The single generation stands in for frame 10, so action 9 must drive it.
    assert model.action_windows[0][0, -1, 0].item() == 9


def test_rollout_loss_pairs_the_generated_context_with_its_real_target() -> None:
    torch.manual_seed(4)
    model = TemporalActionUNet(
        latent_channels=2,
        context_frames=CONTEXT_FRAMES,
        base_channels=8,
        attention_heads=2,
        diffusion_steps=20,
    )
    total = CONTEXT_FRAMES + MAXIMUM_ROLLOUT_DEPTH
    batch = {
        "context_latents": torch.randn(2, total, 2, 8, 8),
        "actions": torch.randn(2, total, 9),
        "target_latent": torch.randn(2, 2, 8, 8),
    }

    loss = rollout_diffusion_loss(
        model, batch, depth=2, generation_steps=2, maximum_context_noise=0.2, seed=3
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


class _IdentityDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :3].clamp(0, 1)


def test_rollout_evaluation_reports_one_ratio_per_horizon_step() -> None:
    horizon = 3
    total = CONTEXT_FRAMES + horizon
    dataset = [
        {
            "context_latents": torch.rand(total, 3, 8, 8),
            "actions": torch.randn(total, 9),
        }
        for _ in range(4)
    ]
    model = _RecordingModel()
    model.latent_channels = 3

    metrics = evaluate_rollout(
        model,
        _IdentityDecoder(),
        dataset,
        torch.device("cpu"),
        horizon=horizon,
        sampling_steps=2,
        seed=1,
        examples=4,
    )

    assert model.calls == horizon
    assert len(metrics.edge_ratio_by_step) == horizon
    assert len(metrics.pixel_mse_by_step) == horizon
    assert metrics.edge_ratio_final == metrics.edge_ratio_by_step[-1]
    # Constant generated latents carry no edges, so the ratio must collapse.
    assert metrics.edge_ratio_final == 0.0
