from __future__ import annotations

from pathlib import Path

import numpy as np

from mcwm.cleaning import RejectionReason
from mcwm.dataset import ProcessedEpisode
from mcwm.model import TinyAutoencoder
from mcwm.training import (
    FrameDataset,
    choose_device,
    evaluate_autoencoder,
    save_reconstruction_grid,
)


def _episode() -> ProcessedEpisode:
    frames = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    for index in range(6):
        frames[index, :, :, index % 3] = index * 40
    return ProcessedEpisode(
        episode="tiny",
        frames=frames,
        actions=np.zeros((5, 9), dtype=np.float32),
        rejection_reasons=np.zeros(5, dtype=np.int8),
        source_frame_indices=np.arange(6, dtype=np.int32) * 2,
        model_fps=10,
    )


def test_frame_dataset_deduplicates_overlapping_sequences() -> None:
    dataset = FrameDataset([_episode()], horizon=2)

    assert len(dataset) == 6
    assert dataset[0].shape == (3, 64, 64)
    assert dataset[0].dtype.is_floating_point
    assert 0 <= dataset[-1].max() <= 1


def test_non_gui_frame_policy_keeps_actions_but_excludes_gui_neighbors() -> None:
    episode = _episode()
    reasons = episode.rejection_reasons.copy()
    reasons[1] = RejectionReason.ATTACK
    reasons[3] = RejectionReason.GUI_OPEN
    episode = ProcessedEpisode(
        episode=episode.episode,
        frames=episode.frames,
        actions=episode.actions,
        rejection_reasons=reasons,
        source_frame_indices=episode.source_frame_indices,
        model_fps=episode.model_fps,
    )

    dataset = FrameDataset([episode], horizon=2, policy="non_gui")

    assert [reference.frame_index for reference in dataset.references] == [0, 1, 2, 5]


def test_evaluation_and_reconstruction_grid(tmp_path: Path) -> None:
    dataset = FrameDataset([_episode()], horizon=2)
    model = TinyAutoencoder(latent_dim=8)
    device = choose_device("cpu")
    model.to(device)

    metrics = evaluate_autoencoder(model, dataset, device, batch_size=3)
    output = tmp_path / "grid.png"
    save_reconstruction_grid(model, dataset, output, device, count=3)

    assert metrics.l1 > 0
    assert metrics.mse > 0
    assert metrics.psnr_db > 0
    assert output.stat().st_size > 0
