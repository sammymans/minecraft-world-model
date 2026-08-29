from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mcwm.dataset import ProcessedEpisode
from mcwm.latent_diffusion_v2 import (
    LinearVelocitySchedule,
    TemporalActionUNet,
    TemporalLatentDataset,
    TemporalSequenceReference,
    _save_checkpoint,
    autoregressive_action_rollout,
    corrupt_context,
    diffusion_loss,
    load_latent_diffusion_v2_checkpoint,
)


def _episode(frames: int = 14) -> ProcessedEpisode:
    actions = np.arange((frames - 1) * 9, dtype=np.float32).reshape(frames - 1, 9)
    return ProcessedEpisode(
        episode="fixed-episode",
        frames=np.zeros((frames, 64, 64, 3), dtype=np.uint8),
        actions=actions,
        rejection_reasons=np.zeros(frames - 1, dtype=np.int8),
        source_frame_indices=np.arange(frames, dtype=np.int32) * 2,
        model_fps=10.0,
    )


def _small_model(context_frames: int = 3) -> TemporalActionUNet:
    return TemporalActionUNet(
        latent_channels=4,
        context_frames=context_frames,
        base_channels=8,
        attention_heads=2,
        diffusion_steps=20,
    )


def test_temporal_dataset_aligns_eight_contexts_actions_and_target() -> None:
    episode = _episode()
    timeline = torch.arange(14, dtype=torch.float32)[:, None, None, None].expand(-1, 4, 2, 2)
    dataset = TemporalLatentDataset([episode], [timeline], context_frames=8)
    sample = dataset[0]

    assert sample["context_latents"].shape == (8, 4, 2, 2)
    assert sample["actions"].shape == (8, 9)
    assert sample["target_latent"].shape == (4, 2, 2)
    assert sample["context_latents"][:, 0, 0, 0].tolist() == list(range(8))
    assert sample["target_latent"][0, 0, 0].item() == 8
    assert torch.equal(sample["actions"], torch.from_numpy(episode.actions[:8]))


def test_temporal_dataset_fixed_subset_is_reproducible() -> None:
    episode = _episode(frames=30)
    timeline = torch.randn(30, 4, 2, 2)
    first = TemporalLatentDataset(
        [episode], [timeline], context_frames=8, maximum_sequences=6, seed=17
    )
    second = TemporalLatentDataset(
        [episode], [timeline], context_frames=8, maximum_sequences=6, seed=17
    )

    assert first.index == second.index
    assert len(first) == 6
    assert first.references == second.references


def test_temporal_dataset_action_balances_the_fixed_subset() -> None:
    episode = _episode(frames=80)
    actions = np.zeros_like(episode.actions)
    for index in range(len(actions)):
        bucket = index % 4
        if bucket == 0:
            actions[index, 0] = 1
        elif bucket == 1:
            actions[index, -2] = -5
        elif bucket == 2:
            actions[index, -2] = 5
    balanced_episode = ProcessedEpisode(
        episode=episode.episode,
        frames=episode.frames,
        actions=actions,
        rejection_reasons=episode.rejection_reasons,
        source_frame_indices=episode.source_frame_indices,
        model_fps=episode.model_fps,
    )
    timeline = torch.randn(80, 4, 2, 2)

    dataset = TemporalLatentDataset(
        [balanced_episode],
        [timeline],
        context_frames=8,
        maximum_sequences=32,
        selection_policy="action_balanced",
        seed=19,
    )

    assert dataset.action_bucket_counts == {
        "forward": 8,
        "look_left": 8,
        "look_right": 8,
        "other": 8,
    }
    assert (
        len({(reference.episode, reference.context_start) for reference in dataset.references})
        == 32
    )


def test_velocity_conversion_exactly_reconstructs_clean_target() -> None:
    schedule = LinearVelocitySchedule(diffusion_steps=100)
    clean = torch.randn(5, 4, 3, 3)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 1, 25, 70, 99])
    noised = schedule.add_noise(clean, noise, timesteps)
    perfect_velocity = schedule.velocity(clean, noise, timesteps)

    reconstructed = schedule.clean_from_velocity(noised, perfect_velocity, timesteps)
    reconstructed_noise = schedule.noise_from_velocity(noised, perfect_velocity, timesteps)

    assert torch.allclose(reconstructed, clean, atol=2e-6, rtol=2e-6)
    assert torch.allclose(reconstructed_noise, noise, atol=2e-6, rtol=2e-6)
    assert torch.all(schedule.alpha[1:] < schedule.alpha[:-1])
    assert torch.all(schedule.sigma[1:] > schedule.sigma[:-1])


def test_temporal_unet_shapes_loss_and_default_parameter_budget() -> None:
    model = _small_model()
    batch = {
        "context_latents": torch.randn(2, 3, 4, 16, 16),
        "actions": torch.randn(2, 3, 9),
        "target_latent": torch.randn(2, 4, 16, 16),
    }
    loss = diffusion_loss(model, batch, seed=11)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    default = TemporalActionUNet()
    assert 10_000_000 <= default.parameter_count <= 20_000_000


def test_context_corruption_reports_and_respects_noise_level() -> None:
    context = torch.ones(2, 3, 4, 2, 2)
    noise = torch.zeros_like(context)
    levels = torch.tensor([0.0, 0.5])
    corrupted = corrupt_context(context, levels, noise=noise)

    assert torch.equal(corrupted[0], context[0])
    assert torch.allclose(corrupted[1], torch.full_like(corrupted[1], 0.75**0.5))
    with pytest.raises(ValueError, match=r"in \[0, 1\)"):
        corrupt_context(context, torch.tensor([0.0, 1.0]))


def test_seeded_ddim_sampling_reproduces_across_reset() -> None:
    torch.manual_seed(3)
    model = _small_model()
    context = torch.randn(2, 3, 4, 16, 16)
    actions = torch.randn(2, 3, 9)

    first = model.sample(context, actions, steps=4, seed=29)
    _ = model.sample(context, actions, steps=4, seed=999)
    reset = model.sample(context, actions, steps=4, seed=29)

    assert torch.equal(first, reset)
    assert first.shape == (2, 4, 16, 16)
    assert torch.isfinite(first).all()

    shared = torch.randn(1, 4, 16, 16).repeat(2, 1, 1, 1)
    explicit = model.sample(context, actions, steps=4, initial_noise=shared)
    assert explicit.shape == first.shape
    with pytest.raises(ValueError, match="initial_noise"):
        model.sample(context, actions, steps=4, initial_noise=shared[:1])


def test_autoregressive_rollout_shifts_generated_context_and_shares_noise() -> None:
    torch.manual_seed(5)
    model = _small_model()
    context = torch.randn(1, 3, 4, 16, 16).repeat(2, 1, 1, 1, 1)
    history = torch.randn(1, 3, 9).repeat(2, 1, 1)
    future = torch.zeros(2, 3, 9)

    first = autoregressive_action_rollout(
        model,
        context,
        history,
        future,
        sampling_steps=4,
        seed=31,
        shared_noise_across_batch=True,
    )
    reset = autoregressive_action_rollout(
        model,
        context,
        history,
        future,
        sampling_steps=4,
        seed=31,
        shared_noise_across_batch=True,
    )

    assert first.shape == (2, 3, 4, 16, 16)
    assert torch.equal(first, reset)
    assert torch.equal(first[0], first[1])


def test_v2_checkpoint_round_trip_carries_gate_metadata(tmp_path: Path) -> None:
    model = _small_model()
    autoencoder = tmp_path / "autoencoder.pt"
    autoencoder.write_bytes(b"frozen")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"fixed": True}) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "v2.pt"
    _save_checkpoint(
        checkpoint,
        model,
        history={"train": [1.0]},
        autoencoder_checkpoint=autoencoder,
        manifest_path=manifest,
        references=[TemporalSequenceReference("episode", 12, "forward")],
        selection_policy="action_balanced",
        action_bucket_counts={"forward": 1, "look_left": 0, "look_right": 0, "other": 0},
        source_episodes=["episode"],
        sampling_steps=8,
        training_steps=10,
        maximum_context_noise=0.0,
        seed=7,
    )

    loaded, metadata = load_latent_diffusion_v2_checkpoint(checkpoint, torch.device("cpu"))

    assert metadata["architecture"] == "temporal_multiscale_action_unet_velocity_v2"
    assert metadata["noise_schedule"] == "linear_beta"
    assert metadata["context_frames"] == 3
    assert metadata["fixed_sequence_references"] == [
        {"episode": "episode", "context_start": 12, "action_bucket": "forward"}
    ]
    assert loaded.parameter_count == model.parameter_count
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)
