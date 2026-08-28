from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from mcwm.dataset import ProcessedEpisode
from mcwm.model import SpatialLatentDynamics
from mcwm.rollout import (
    EncodedRolloutDataset,
    RolloutHorizonMetrics,
    _metrics_payload,
    evaluate_rollouts,
    recursive_latent_rollout,
)


def _episode() -> ProcessedEpisode:
    frames = np.empty((7, 2, 2, 3), dtype=np.uint8)
    for index in range(7):
        frames[index] = np.array([index * 20, index * 10, index * 5], dtype=np.uint8)
    actions = np.zeros((6, 9), dtype=np.float32)
    actions[:, 0] = np.arange(6)
    return ProcessedEpisode(
        episode="rollout-test",
        frames=frames,
        actions=actions,
        rejection_reasons=np.zeros(6, dtype=np.int8),
        source_frame_indices=np.arange(7, dtype=np.int32) * 2,
        model_fps=10,
    )


def _latents(episode: ProcessedEpisode) -> torch.Tensor:
    frames = np.ascontiguousarray(episode.frames.transpose(0, 3, 1, 2))
    return torch.from_numpy(frames).to(torch.float32).div(255)


class _AddAction(nn.Module):
    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        del previous_latent
        return current_latent + action[..., :1]


class _ColorDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


def test_rollout_dataset_preserves_full_temporal_alignment() -> None:
    episode = _episode()
    latents = _latents(episode)
    dataset = EncodedRolloutDataset([episode], [latents], horizon=3)

    first = dataset[0]

    assert len(dataset) == 3
    assert torch.equal(first["latents"], latents[:5])
    assert torch.equal(first["actions"], torch.from_numpy(episode.actions[1:4]))
    assert first["frames"].shape == (5, 3, 2, 2)


def test_rollout_dataset_promotes_cached_half_latents_for_inference() -> None:
    episode = _episode()
    dataset = EncodedRolloutDataset([episode], [_latents(episode).half()], horizon=2)

    assert dataset[0]["latents"].dtype == torch.float32


def test_rollout_dataset_rejects_empty_sample_budget() -> None:
    episode = _episode()

    with pytest.raises(ValueError, match="maximum_examples must be positive"):
        EncodedRolloutDataset([episode], [_latents(episode)], horizon=2, maximum_examples=0)


def test_recursive_rollout_feeds_predictions_back_without_future_latents() -> None:
    dynamics = _AddAction()
    previous = torch.tensor([[99.0]])
    current = torch.tensor([[0.0]])
    actions = torch.tensor([[[1.0], [2.0], [3.0]]])

    predicted = recursive_latent_rollout(dynamics, previous, current, actions)

    assert predicted.tolist() == [[[1.0], [3.0], [6.0]]]


def test_recursive_rollout_preserves_spatial_latent_shape() -> None:
    class SpatialAdd(nn.Module):
        def forward(self, previous, current, action):
            del previous
            return current + action[:, :1, None, None]

    dynamics = SpatialAdd()
    previous = torch.zeros((2, 3, 4, 4))
    current = torch.zeros((2, 3, 4, 4))
    actions = torch.ones((2, 5, 1))

    predicted = recursive_latent_rollout(dynamics, previous, current, actions)

    assert predicted.shape == (2, 5, 3, 4, 4)


def test_copy_dynamics_matches_recursive_copy_baseline() -> None:
    episode = _episode()
    dataset = EncodedRolloutDataset([episode], [_latents(episode)], horizon=3)
    dynamics = SpatialLatentDynamics(latent_channels=3, hidden_channels=8, blocks=1)

    metrics = evaluate_rollouts(
        dynamics,
        _ColorDecoder(),
        dataset,
        torch.device("cpu"),
        horizons=(1, 2, 3),
        batch_size=2,
    )

    for horizon in metrics:
        assert horizon.recursive_latent_mse == pytest.approx(horizon.copy_latent_mse)
        assert horizon.recursive_pixel_mse == pytest.approx(horizon.copy_pixel_mse)
        assert horizon.oracle_pixel_mse == pytest.approx(0)
        assert horizon.action_effect_latent_mse == pytest.approx(0)
        assert horizon.copy_improvement_percent == pytest.approx(0)
        assert horizon.shuffled_action_pixel_penalty_percent == pytest.approx(0)


def test_metrics_payload_normalizes_numpy_scalars_for_json() -> None:
    metrics = RolloutHorizonMetrics(
        horizon=1,
        seconds=0.1,
        examples=2,
        recursive_latent_mse=0.1,
        recursive_pixel_l1=0.1,
        recursive_pixel_mse=np.float64(0.1),
        recursive_pixel_psnr_db=10.0,
        teacher_forced_latent_mse=0.1,
        teacher_forced_pixel_mse=0.1,
        copy_latent_mse=0.2,
        copy_pixel_mse=np.float64(0.2),
        oracle_pixel_mse=0.01,
        shuffled_action_latent_mse=0.2,
        shuffled_action_pixel_mse=np.float64(0.3),
        action_effect_latent_mse=0.1,
    )

    payload = _metrics_payload(metrics)

    assert payload["beats_copy_pixel"] is True
    assert type(payload["copy_improvement_percent"]) is float
    assert type(payload["shuffled_action_pixel_penalty_percent"]) is float
    json.dumps(payload)
