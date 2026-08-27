from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mcwm.dataset import ProcessedEpisode
from mcwm.dynamics import (
    EncodedDynamicsDataset,
    evaluate_dynamics,
    evaluate_saved_dynamics,
    load_dynamics_checkpoint,
    save_prediction_grid,
    train_dynamics,
)
from mcwm.model import LatentDynamics, TinyAutoencoder


def _episode() -> ProcessedEpisode:
    frames = np.empty((6, 2, 2, 3), dtype=np.uint8)
    for index in range(6):
        frames[index] = np.array([index * 20, index * 10, index * 5], dtype=np.uint8)
    actions = np.zeros((5, 9), dtype=np.float32)
    actions[:, 0] = np.arange(5) % 2
    actions[:, -2] = np.arange(5)
    return ProcessedEpisode(
        episode="tiny",
        frames=frames,
        actions=actions,
        rejection_reasons=np.zeros(5, dtype=np.int8),
        source_frame_indices=np.arange(6, dtype=np.int32) * 2,
        model_fps=10,
    )


def _latents(episode: ProcessedEpisode) -> torch.Tensor:
    return torch.from_numpy(episode.frames[:, 0, 0].copy()).to(torch.float32).div(255)


class _ColorDecoder(nn.Module):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :, None, None].expand(-1, -1, 2, 2)


def test_encoded_dataset_preserves_context_action_target_alignment() -> None:
    episode = _episode()
    latents = _latents(episode)
    dataset = EncodedDynamicsDataset([episode], [latents])

    assert len(dataset) == 4
    first = dataset[0]
    assert torch.equal(first["previous_latent"], latents[0])
    assert torch.equal(first["current_latent"], latents[1])
    assert torch.equal(first["target_latent"], latents[2])
    assert torch.equal(first["action"], torch.from_numpy(episode.actions[1]))


def test_action_statistics_scale_rare_controls_safely() -> None:
    episode = _episode()
    dataset = EncodedDynamicsDataset([episode], [_latents(episode)])

    mean, std = dataset.action_statistics()

    assert mean.shape == (9,)
    assert std.shape == (9,)
    assert torch.all(std >= 0.05)


def test_dynamics_starts_as_copy_latent_baseline() -> None:
    model = LatentDynamics(latent_dim=5, hidden_dim=12, hidden_layers=2)
    previous = torch.randn(4, 5)
    current = torch.randn(4, 5)
    action = torch.randn(4, 9)

    predicted = model(previous, current, action)

    assert torch.equal(predicted, current)
    assert model.parameter_count > 0


def test_dynamics_rejects_misaligned_batches() -> None:
    model = LatentDynamics(latent_dim=5, hidden_dim=8)

    with pytest.raises(ValueError, match="action batch shape"):
        model(torch.zeros(2, 5), torch.zeros(2, 5), torch.zeros(3, 9))


def test_evaluation_reports_copy_and_action_controls(tmp_path: Path) -> None:
    episode = _episode()
    dataset = EncodedDynamicsDataset([episode], [_latents(episode)])
    model = LatentDynamics(latent_dim=3, hidden_dim=8)
    autoencoder = _ColorDecoder()

    metrics = evaluate_dynamics(
        model, autoencoder, dataset, torch.device("cpu"), batch_size=2
    )
    output = tmp_path / "predictions.png"
    save_prediction_grid(model, autoencoder, dataset, output, torch.device("cpu"), count=2)

    assert metrics.examples == len(dataset)
    assert metrics.latent_mse == pytest.approx(metrics.copy_latent_mse)
    assert metrics.pixel_mse == pytest.approx(metrics.decoded_copy_pixel_mse)
    assert metrics.oracle_pixel_mse <= metrics.pixel_mse
    assert metrics.action_effect_latent_mse == 0
    assert metrics.shuffled_action_degradation == pytest.approx(0)
    assert output.stat().st_size > 0


def _save_processed_episode(path: Path, color_offset: int) -> None:
    frames = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    for index in range(6):
        frames[index, :, :, 0] = color_offset + index * 3
        frames[index, :, :, 1] = np.arange(64, dtype=np.uint8)[:, None]
    metadata = json.dumps({"episode": path.stem, "model_fps": 10.0})
    np.savez_compressed(
        path,
        metadata=metadata,
        frames=frames,
        actions=np.zeros((5, 9), dtype=np.float32),
        rejection_reasons=np.zeros(5, dtype=np.int8),
        source_frame_indices=np.arange(6, dtype=np.int32) * 2,
    )


def test_one_epoch_training_and_saved_evaluation_round_trip(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _save_processed_episode(processed / "alpha-session-20260101-120000.npz", 0)
    _save_processed_episode(processed / "beta-session-20260101-120000.npz", 20)
    autoencoder = TinyAutoencoder(latent_dim=4, base_channels=1)
    autoencoder_path = tmp_path / "autoencoder.pt"
    torch.save(
        {
            "model_state": autoencoder.state_dict(),
            "latent_dim": 4,
            "base_channels": 1,
            "image_size": 64,
            "history": {"train_loss": [1.0], "validation_loss": [1.0]},
        },
        autoencoder_path,
    )

    result = train_dynamics(
        processed,
        autoencoder_path,
        tmp_path / "dynamics",
        epochs=1,
        batch_size=4,
        encode_batch_size=6,
        hidden_dim=8,
        hidden_layers=1,
        requested_device="cpu",
    )
    loaded, metadata = load_dynamics_checkpoint(result.checkpoint, torch.device("cpu"))
    evaluated = evaluate_saved_dynamics(
        processed,
        autoencoder_path,
        result.checkpoint,
        tmp_path / "evaluation",
        batch_size=4,
        encode_batch_size=6,
        count=2,
        requested_device="cpu",
    )

    assert loaded.latent_dim == 4
    assert metadata["autoencoder_sha256"]
    assert result.checkpoint.exists()
    assert result.metrics_path.exists()
    assert evaluated.example_count == 4
    assert evaluated.comparison_grid.exists()
