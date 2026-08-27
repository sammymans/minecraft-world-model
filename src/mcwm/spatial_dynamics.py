"""Bounded training and evaluation for spatial action-conditioned dynamics."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mcwm.dataset import ProcessedEpisode, SequenceDataset
from mcwm.dynamics import (
    DynamicsMetrics,
    _file_sha256,
    _metrics_payload,
    _processed_splits,
    evaluate_dynamics,
    save_prediction_grid,
)
from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class SpatialDynamicsTrainingResult:
    checkpoint: Path
    comparison_grid: Path
    training_curve: Path
    metrics_path: Path
    validation_metrics: DynamicsMetrics
    parameter_count: int
    training_transitions: int
    encoded_frames: int
    latent_shape: tuple[int, int, int]
    device: str


@dataclass(frozen=True)
class SpatialDynamicsEvaluationResult:
    metrics: DynamicsMetrics
    comparison_grid: Path
    metrics_path: Path
    transitions: int
    latent_shape: tuple[int, int, int]
    device: str


def _frame_tensor(frame: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(frame.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).float().div_(255.0)


class SpatialEncodedDynamicsDataset(Dataset[dict[str, torch.Tensor]]):
    """Clean transitions from a bounded set of episodes with float16 latent maps."""

    def __init__(
        self,
        episodes: list[ProcessedEpisode],
        latents: list[torch.Tensor],
        *,
        maximum_transitions: int | None = None,
        horizon: int = 1,
        seed: int = 7,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not episodes or len(episodes) != len(latents):
            raise ValueError("each spatial dynamics episode needs one latent timeline")
        latent_shapes: set[tuple[int, int, int]] = set()
        for episode, timeline in zip(episodes, latents, strict=True):
            if timeline.ndim != 4 or len(timeline) != len(episode.frames):
                raise ValueError("spatial latent timelines must be [time, channels, height, width]")
            latent_shapes.add(tuple(int(value) for value in timeline.shape[1:]))
        if len(latent_shapes) != 1:
            raise ValueError("all episodes must use the same spatial latent shape")
        self.episodes = episodes
        self.latents = latents
        self.latent_shape = latent_shapes.pop()
        self.horizon = horizon
        self.index = SequenceDataset(episodes, horizon=horizon).index
        if maximum_transitions is not None and len(self.index) > maximum_transitions:
            generator = torch.Generator().manual_seed(seed)
            selected = torch.randperm(len(self.index), generator=generator)[:maximum_transitions]
            self.index = [self.index[int(item)] for item in selected]

    @classmethod
    @torch.no_grad()
    def from_paths(
        cls,
        paths: list[Path],
        autoencoder: SpatialAutoencoder,
        device: torch.device,
        *,
        maximum_transitions: int | None = None,
        horizon: int = 1,
        encode_batch_size: int = 128,
        seed: int = 7,
    ) -> SpatialEncodedDynamicsDataset:
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        if maximum_transitions is not None and maximum_transitions < 1:
            raise ValueError("maximum_transitions must be positive when supplied")
        ordered_paths = list(paths)
        if maximum_transitions is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(ordered_paths)
        autoencoder.eval()
        episodes: list[ProcessedEpisode] = []
        timelines: list[torch.Tensor] = []
        available_transitions = 0
        for path in ordered_paths:
            episode = ProcessedEpisode.load(path)
            valid_transitions = len(SequenceDataset([episode], horizon=horizon))
            if valid_transitions == 0:
                continue
            chunks: list[torch.Tensor] = []
            for start in range(0, len(episode.frames), encode_batch_size):
                frames = episode.frames[start : start + encode_batch_size]
                contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
                batch = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32)
                batch.div_(255.0)
                chunks.append(autoencoder.encode(batch).to(device="cpu", dtype=torch.float16))
            episodes.append(episode)
            timelines.append(torch.cat(chunks))
            available_transitions += valid_transitions
            if maximum_transitions is not None and available_transitions >= maximum_transitions:
                break
        if not episodes:
            raise ValueError("no clean spatial dynamics transitions were found")
        return cls(
            episodes,
            timelines,
            maximum_transitions=maximum_transitions,
            horizon=horizon,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        episode = self.episodes[episode_index]
        timeline = self.latents[episode_index]
        sample = {
            "previous_latent": timeline[current_index - 1].float(),
            "current_latent": timeline[current_index].float(),
            "target_latent": timeline[current_index + 1].float(),
            "action": torch.from_numpy(episode.actions[current_index]).float(),
            "current_frame": _frame_tensor(episode.frames[current_index]),
            "target_frame": _frame_tensor(episode.frames[current_index + 1]),
        }
        if self.horizon > 1:
            stop = current_index + self.horizon
            # [horizon + 2] latents seed the rollout and supply one target per step.
            sample["latent_sequence"] = timeline[current_index - 1 : stop + 1].float()
            sample["action_sequence"] = torch.from_numpy(
                episode.actions[current_index:stop]
            ).float()
        return sample

    @property
    def encoded_frames(self) -> int:
        return sum(len(timeline) for timeline in self.latents)

    def normalization_statistics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate per-channel latent/motion statistics and per-action statistics."""
        channels = self.latent_shape[0]
        latent_sum = torch.zeros(channels, dtype=torch.float64)
        latent_squared = torch.zeros(channels, dtype=torch.float64)
        motion_sum = torch.zeros(channels, dtype=torch.float64)
        motion_squared = torch.zeros(channels, dtype=torch.float64)
        latent_values = 0
        motion_values = 0
        actions: list[np.ndarray] = []
        by_episode: dict[int, list[int]] = {}
        for episode_index, current_index in self.index:
            by_episode.setdefault(episode_index, []).append(current_index)
            actions.append(self.episodes[episode_index].actions[current_index])
        for episode_index, indices in by_episode.items():
            selected = torch.tensor(indices)
            current = self.latents[episode_index][selected].float()
            previous = self.latents[episode_index][selected - 1].float()
            motion = current - previous
            latent_sum += current.sum(dim=(0, 2, 3), dtype=torch.float64)
            latent_squared += current.square().sum(dim=(0, 2, 3), dtype=torch.float64)
            motion_sum += motion.sum(dim=(0, 2, 3), dtype=torch.float64)
            motion_squared += motion.square().sum(dim=(0, 2, 3), dtype=torch.float64)
            latent_values += len(current) * current.shape[2] * current.shape[3]
            motion_values += len(motion) * motion.shape[2] * motion.shape[3]
        latent_mean = latent_sum / latent_values
        latent_variance = latent_squared / latent_values - latent_mean.square()
        motion_mean = motion_sum / motion_values
        motion_variance = motion_squared / motion_values - motion_mean.square()
        action_array = np.stack(actions).astype(np.float32, copy=False)
        action_mean = torch.from_numpy(action_array.mean(axis=0))
        action_std = torch.from_numpy(action_array.std(axis=0)).clamp_min(0.05)
        return (
            action_mean,
            action_std,
            latent_mean.float(),
            latent_variance.clamp_min(1e-8).sqrt().float(),
            motion_mean.float(),
            motion_variance.clamp_min(1e-8).sqrt().float(),
        )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _step_losses(
    predicted: torch.Tensor,
    target_latent: torch.Tensor,
    autoencoder: SpatialAutoencoder,
    latent_std: torch.Tensor,
    *,
    pixel_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one predicted latent against the decoder oracle, not the raw frame.

    Comparing against `decode(target_latent)` keeps the dynamics model from being
    charged for reconstruction error it cannot fix.
    """
    latent_loss = ((predicted - target_latent) / latent_std).square().mean()
    if not (pixel_weight or edge_weight):
        zero = latent_loss.new_zeros(())
        return latent_loss, zero, zero
    predicted_frame = autoencoder.decode(predicted)
    with torch.no_grad():
        oracle_frame = autoencoder.decode(target_latent)
    pixel_loss = nn.functional.mse_loss(predicted_frame, oracle_frame)
    if edge_weight:
        predicted_dx, predicted_dy = image_gradients(predicted_frame)
        oracle_dx, oracle_dy = image_gradients(oracle_frame)
        edge_loss = 0.5 * (
            nn.functional.l1_loss(predicted_dx, oracle_dx)
            + nn.functional.l1_loss(predicted_dy, oracle_dy)
        )
    else:
        edge_loss = latent_loss.new_zeros(())
    return latent_loss, pixel_loss, edge_loss


def _prediction_loss(
    dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float,
    pixel_weight: float,
    edge_weight: float = 0.0,
    rollout_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-step objective, or an unrolled one when `rollout_steps` exceeds one.

    `edge_weight` penalizes the difference in image gradients, the same term the
    spatial autoencoder uses. Squared error alone is minimized by the blurry
    average of every plausible next frame, and two structurally different
    dynamics models both land at 59% of the real frame's edge energy, so
    sharpness is set by this objective rather than by the architecture. Note the
    scale: the normalized latent term is O(0.4) while the gradient term is
    O(0.005), so a useful `edge_weight` is tens, not fractions.

    `rollout_steps` feeds each prediction back in as the next input, so blur that
    compounds across steps is charged to the model during training rather than
    discovered later during rollout.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if rollout_steps == 1:
        predicted = dynamics(
            batch["previous_latent"], batch["current_latent"], batch["action"]
        )
        latent_loss, pixel_loss, edge_loss = _step_losses(
            predicted,
            batch["target_latent"],
            autoencoder,
            dynamics.latent_std,
            pixel_weight=pixel_weight,
            edge_weight=edge_weight,
        )
    else:
        if "latent_sequence" not in batch:
            raise ValueError("multi-step training needs a dataset built with a horizon")
        latents = batch["latent_sequence"]
        actions = batch["action_sequence"]
        if latents.shape[1] < rollout_steps + 2:
            raise ValueError("the latent sequence is shorter than the requested rollout")
        previous, current = latents[:, 0], latents[:, 1]
        latent_terms: list[torch.Tensor] = []
        pixel_terms: list[torch.Tensor] = []
        edge_terms: list[torch.Tensor] = []
        for step in range(rollout_steps):
            predicted = dynamics(previous, current, actions[:, step])
            step_latent, step_pixel, step_edge = _step_losses(
                predicted,
                latents[:, step + 2],
                autoencoder,
                dynamics.latent_std,
                pixel_weight=pixel_weight,
                edge_weight=edge_weight,
            )
            latent_terms.append(step_latent)
            pixel_terms.append(step_pixel)
            edge_terms.append(step_edge)
            previous, current = current, predicted
        latent_loss = torch.stack(latent_terms).mean()
        pixel_loss = torch.stack(pixel_terms).mean()
        edge_loss = torch.stack(edge_terms).mean()
    total = latent_weight * latent_loss + pixel_weight * pixel_loss + edge_weight * edge_loss
    return total, latent_loss, pixel_loss


@torch.no_grad()
def _validation_objective(
    dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedDynamicsDataset,
    device: torch.device,
    *,
    batch_size: int,
    latent_weight: float,
    pixel_weight: float,
    edge_weight: float,
    rollout_steps: int = 1,
) -> tuple[float, float, float]:
    dynamics.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = np.zeros(3, dtype=np.float64)
    examples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        losses = _prediction_loss(
            dynamics,
            autoencoder,
            batch,
            latent_weight=latent_weight,
            pixel_weight=pixel_weight,
            edge_weight=edge_weight,
            rollout_steps=rollout_steps,
        )
        count = len(batch["action"])
        totals += np.array([float(loss) for loss in losses]) * count
        examples += count
    return tuple(float(value / examples) for value in totals)  # type: ignore[return-value]


def _save_checkpoint(
    path: Path,
    dynamics: SpatialLatentDynamics,
    *,
    history: dict[str, list[float]],
    autoencoder_checkpoint: Path,
    autoencoder_sha256: str,
    manifest_path: Path,
    latent_weight: float,
    pixel_weight: float,
    edge_weight: float = 0.0,
    rollout_steps: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "spatial_latent_dynamics",
            "model_state": dynamics.state_dict(),
            "latent_channels": dynamics.latent_channels,
            "action_dim": dynamics.action_dim,
            "hidden_channels": dynamics.hidden_channels,
            "blocks": dynamics.blocks,
            "history": history,
            "latent_weight": latent_weight,
            "pixel_weight": pixel_weight,
            "edge_weight": edge_weight,
            "rollout_steps": rollout_steps,
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": autoencoder_sha256,
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
        },
        path,
    )


def load_spatial_dynamics_checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialLatentDynamics, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "spatial_latent_dynamics":
        raise ValueError("checkpoint is not spatial latent dynamics")
    state = checkpoint["model_state"]
    dynamics = SpatialLatentDynamics(
        latent_channels=int(checkpoint["latent_channels"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        blocks=int(checkpoint["blocks"]),
        action_mean=state["action_mean"],
        action_std=state["action_std"],
        latent_mean=state["latent_mean"].flatten(),
        latent_std=state["latent_std"].flatten(),
        motion_mean=state["motion_mean"].flatten(),
        motion_std=state["motion_std"].flatten(),
    ).to(device)
    dynamics.load_state_dict(state)
    dynamics.eval()
    return dynamics, checkpoint


def _save_curve(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train"], label="training objective")
    axis.plot(history["validation"], label="validation objective")
    axis.plot(history["validation_latent"], label="normalized latent MSE")
    axis.plot(history["validation_pixel"], label="decoded pixel MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("Spatial action-conditioned dynamics")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_spatial_dynamics(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    encode_batch_size: int = 128,
    maximum_transitions: int = 30_000,
    hidden_channels: int = 64,
    blocks: int = 3,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    latent_weight: float = 1.0,
    pixel_weight: float = 1.0,
    edge_weight: float = 0.0,
    rollout_steps: int = 1,
    patience: int = 5,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialDynamicsTrainingResult:
    if min(epochs, batch_size, encode_batch_size, maximum_transitions, patience) < 1:
        raise ValueError("training sizes and patience must be positive")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if min(latent_weight, pixel_weight, edge_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if latent_weight + pixel_weight + edge_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha256 = _file_sha256(autoencoder_checkpoint)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    train_paths, validation_paths = _processed_splits(processed_dir, manifest_path)
    print(f"encoding up to {maximum_transitions:,} diverse training transitions...")
    training = SpatialEncodedDynamicsDataset.from_paths(
        train_paths,
        autoencoder,
        device,
        maximum_transitions=maximum_transitions,
        horizon=rollout_steps,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    print("encoding frozen validation transitions...")
    validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths,
        autoencoder,
        device,
        horizon=rollout_steps,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    statistics = training.normalization_statistics()
    dynamics = SpatialLatentDynamics(
        latent_channels=training.latent_shape[0],
        action_dim=9,
        hidden_channels=hidden_channels,
        blocks=blocks,
        action_mean=statistics[0],
        action_std=statistics[1],
        latent_mean=statistics[2],
        latent_std=statistics[3],
        motion_mean=statistics[4],
        motion_std=statistics[5],
    ).to(device)
    optimizer = torch.optim.AdamW(
        dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: dict[str, list[float]] = {
        "train": [],
        "validation": [],
        "validation_latent": [],
        "validation_pixel": [],
    }
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
        dynamics.train()
        total = 0.0
        examples = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _prediction_loss(
                dynamics,
                autoencoder,
                batch,
                latent_weight=latent_weight,
                pixel_weight=pixel_weight,
                edge_weight=edge_weight,
                rollout_steps=rollout_steps,
            )
            loss.backward()
            optimizer.step()
            count = len(batch["action"])
            total += float(loss.detach()) * count
            examples += count
        train_loss = total / examples
        validation_loss, validation_latent, validation_pixel = _validation_objective(
            dynamics,
            autoencoder,
            validation,
            device,
            batch_size=batch_size,
            latent_weight=latent_weight,
            pixel_weight=pixel_weight,
            edge_weight=edge_weight,
            rollout_steps=rollout_steps,
        )
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        history["validation_latent"].append(validation_latent)
        history["validation_pixel"].append(validation_pixel)
        print(
            f"epoch {epoch + 1:3d}/{epochs}: train={train_loss:.6f}  "
            f"validation={validation_loss:.6f}  latent={validation_latent:.6f}  "
            f"pixel={validation_pixel:.6f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(dynamics.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {epoch + 1} epochs")
                break
    if best_state is None:
        raise RuntimeError("spatial dynamics training produced no checkpoint")
    dynamics.load_state_dict(best_state)
    if _file_sha256(autoencoder_checkpoint) != autoencoder_sha256:
        raise ValueError("the spatial autoencoder checkpoint changed during training")
    checkpoint = output_dir / "best.pt"
    _save_checkpoint(
        checkpoint,
        dynamics,
        history=history,
        autoencoder_checkpoint=autoencoder_checkpoint,
        autoencoder_sha256=autoencoder_sha256,
        manifest_path=manifest_path,
        latent_weight=latent_weight,
        pixel_weight=pixel_weight,
        edge_weight=edge_weight,
        rollout_steps=rollout_steps,
    )
    metrics = evaluate_dynamics(
        dynamics, autoencoder, validation, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / "one-step-predictions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    save_prediction_grid(dynamics, autoencoder, validation, grid, device)
    _save_curve(history, curve)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "spatial latent dynamics",
                "rollout_steps": rollout_steps,
                "edge_weight": edge_weight,
                "training_transitions": len(training),
                "encoded_training_frames": training.encoded_frames,
                "validation_transitions": len(validation),
                "latent_shape": training.latent_shape,
                "parameters": dynamics.parameter_count,
                "device": str(device),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "autoencoder_sha256": autoencoder_sha256,
                "dataset_manifest": str(manifest_path),
                "validation": _metrics_payload(metrics),
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SpatialDynamicsTrainingResult(
        checkpoint=checkpoint,
        comparison_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        validation_metrics=metrics,
        parameter_count=dynamics.parameter_count,
        training_transitions=len(training),
        encoded_frames=training.encoded_frames,
        latent_shape=training.latent_shape,
        device=str(device),
    )


def evaluate_saved_spatial_dynamics(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    output_dir: Path,
    *,
    batch_size: int = 32,
    encode_batch_size: int = 128,
    count: int = 6,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialDynamicsEvaluationResult:
    device = choose_device(requested_device)
    dynamics, metadata = load_spatial_dynamics_checkpoint(dynamics_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial dynamics belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    _, validation_paths = _processed_splits(processed_dir, manifest_path)
    validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths,
        autoencoder,
        device,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    metrics = evaluate_dynamics(
        dynamics, autoencoder, validation, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / "validation-one-step-predictions.png"
    metrics_path = output_dir / "validation-metrics.json"
    save_prediction_grid(dynamics, autoencoder, validation, grid, device, count=count)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "saved spatial dynamics evaluation",
                "transitions": len(validation),
                "latent_shape": validation.latent_shape,
                "metrics": asdict(metrics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SpatialDynamicsEvaluationResult(
        metrics=metrics,
        comparison_grid=grid,
        metrics_path=metrics_path,
        transitions=len(validation),
        latent_shape=validation.latent_shape,
        device=str(device),
    )


@dataclass(frozen=True)
class SpatialRolloutHorizon:
    horizon: int
    latent_mse: float
    frozen_latent_mse: float
    pixel_l1: float
    frozen_pixel_l1: float
    sharpness: float
    real_sharpness: float

    @property
    def sharpness_ratio(self) -> float:
        return self.sharpness / max(self.real_sharpness, 1e-12)

    @property
    def beats_frozen(self) -> bool:
        return self.latent_mse < self.frozen_latent_mse


def _image_sharpness(images: torch.Tensor) -> float:
    """Mean absolute image gradient: how much edge energy survived."""
    horizontal, vertical = image_gradients(images)
    return float(0.5 * (horizontal.abs().mean() + vertical.abs().mean()))


@torch.no_grad()
def evaluate_spatial_rollouts(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    output_dir: Path,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20),
    starts: int = 120,
    encode_batch_size: int = 128,
    seed: int = 7,
    requested_device: str = "auto",
) -> list[SpatialRolloutHorizon]:
    """Recursively imagine forward with true actions and measure the decay.

    One-step accuracy does not imply multi-step stability, and the interactive
    playground is a rollout, not a single step. This reports error against a
    frozen-frame baseline and, separately, how much edge energy survives - the
    two decay differently, and only the second one tracks how blurry it looks.
    """
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be positive")
    if starts < 1:
        raise ValueError("starts must be positive")
    device = choose_device(requested_device)
    dynamics, metadata = load_spatial_dynamics_checkpoint(dynamics_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial dynamics belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    _, validation_paths = _processed_splits(processed_dir, manifest_path)

    longest = max(horizons)
    episodes = [ProcessedEpisode.load(path) for path in validation_paths]
    index = SequenceDataset(episodes, horizon=longest).index
    if not index:
        raise ValueError(f"no held-out episode supports a {longest}-step rollout")
    generator = np.random.default_rng(seed)
    picks = generator.choice(len(index), size=min(starts, len(index)), replace=False)

    totals = {
        horizon: dict.fromkeys(
            ("latent", "frozen_latent", "pixel", "frozen_pixel", "sharp", "real_sharp"), 0.0
        )
        for horizon in horizons
    }
    wanted = set(horizons)
    for pick in picks:
        episode_index, start = index[int(pick)]
        episode = episodes[episode_index]
        window = episode.frames[start - 1 : start + longest + 1]
        frames = torch.from_numpy(
            np.ascontiguousarray(window.transpose(0, 3, 1, 2))
        ).to(device=device, dtype=torch.float32).div_(255.0)
        true_latents = autoencoder.encode(frames)
        seed_latent = true_latents[1:2]
        previous, current = true_latents[0:1], seed_latent.clone()
        frozen_frame = autoencoder.decode(seed_latent).clamp(0, 1)
        for step in range(1, longest + 1):
            action = torch.from_numpy(episode.actions[start + step - 1])
            action = action.to(device=device, dtype=torch.float32)[None]
            previous, current = current, dynamics(previous, current, action)
            if step not in wanted:
                continue
            target = true_latents[step + 1 : step + 2]
            real_frame = frames[step + 1 : step + 2]
            imagined = autoencoder.decode(current).clamp(0, 1)
            bucket = totals[step]
            bucket["latent"] += float((current - target).square().mean())
            bucket["frozen_latent"] += float((seed_latent - target).square().mean())
            bucket["pixel"] += float((imagined - real_frame).abs().mean())
            bucket["frozen_pixel"] += float((frozen_frame - real_frame).abs().mean())
            bucket["sharp"] += _image_sharpness(imagined)
            bucket["real_sharp"] += _image_sharpness(real_frame)

    count = len(picks)
    results = [
        SpatialRolloutHorizon(
            horizon=horizon,
            latent_mse=totals[horizon]["latent"] / count,
            frozen_latent_mse=totals[horizon]["frozen_latent"] / count,
            pixel_l1=totals[horizon]["pixel"] / count,
            frozen_pixel_l1=totals[horizon]["frozen_pixel"] / count,
            sharpness=totals[horizon]["sharp"] / count,
            real_sharpness=totals[horizon]["real_sharp"] / count,
        )
        for horizon in sorted(horizons)
    ]
    metrics_path = output_dir / "rollout-metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "spatial recursive rollout",
                "starts": count,
                "dynamics_checkpoint": str(dynamics_checkpoint),
                "rollout_steps_trained": int(metadata.get("rollout_steps", 1)),
                "edge_weight_trained": float(metadata.get("edge_weight", 0.0)),
                "horizons": [
                    asdict(item) | {"sharpness_ratio": item.sharpness_ratio}
                    for item in results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results
