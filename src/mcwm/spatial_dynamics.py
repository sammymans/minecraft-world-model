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
    evaluate_dynamics,
    save_prediction_grid,
)
from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics
from mcwm.spatial_training import load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SPATIAL_DYNAMICS_ARCHITECTURE = "additive_residual_v1"


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
    rollout_steps: int = 1


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
        seed: int = 7,
    ):
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
        self.index = SequenceDataset(episodes, horizon=1).index
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
        count_horizon: int = 1,
        encode_batch_size: int = 128,
        seed: int = 7,
    ) -> SpatialEncodedDynamicsDataset:
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        if maximum_transitions is not None and maximum_transitions < 1:
            raise ValueError("maximum_transitions must be positive when supplied")
        if count_horizon < 1:
            raise ValueError("count_horizon must be positive")
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
            valid_transitions = len(SequenceDataset([episode], horizon=count_horizon))
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
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        episode = self.episodes[episode_index]
        timeline = self.latents[episode_index]
        return {
            "previous_latent": timeline[current_index - 1].float(),
            "current_latent": timeline[current_index].float(),
            "target_latent": timeline[current_index + 1].float(),
            "action": torch.from_numpy(episode.actions[current_index]).float(),
            "current_frame": _frame_tensor(episode.frames[current_index]),
            "target_frame": _frame_tensor(episode.frames[current_index + 1]),
        }

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


class SpatialEncodedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-horizon latent windows used for differentiable recursive training."""

    def __init__(
        self,
        encoded: SpatialEncodedDynamicsDataset,
        *,
        horizon: int,
        maximum_sequences: int | None = None,
        seed: int = 7,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if maximum_sequences is not None and maximum_sequences < 1:
            raise ValueError("maximum_sequences must be positive when supplied")
        self.episodes = encoded.episodes
        self.latents = encoded.latents
        self.latent_shape = encoded.latent_shape
        self.horizon = horizon
        self.index = SequenceDataset(self.episodes, horizon=horizon).index
        if maximum_sequences is not None and len(self.index) > maximum_sequences:
            selected = torch.randperm(
                len(self.index), generator=torch.Generator().manual_seed(seed)
            )[:maximum_sequences]
            self.index = [self.index[int(item)] for item in selected]
        if not self.index:
            raise ValueError("no clean spatial sequences were found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        stop = current_index + self.horizon
        episode = self.episodes[episode_index]
        return {
            "latents": self.latents[episode_index][current_index - 1 : stop + 1].float(),
            "actions": torch.from_numpy(
                episode.actions[current_index:stop].astype(np.float32, copy=False)
            ),
        }

    @property
    def encoded_frames(self) -> int:
        return sum(len(timeline) for timeline in self.latents)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _prediction_loss(
    dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float,
    pixel_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = dynamics(
        batch["previous_latent"], batch["current_latent"], batch["action"]
    )
    normalized_error = (predicted - batch["target_latent"]) / dynamics.latent_std
    latent_loss = normalized_error.square().mean()
    if pixel_weight:
        predicted_frame = autoencoder.decode(predicted)
        with torch.no_grad():
            oracle_frame = autoencoder.decode(batch["target_latent"])
        pixel_loss = nn.functional.mse_loss(predicted_frame, oracle_frame)
    else:
        pixel_loss = latent_loss.new_zeros(())
    total = latent_weight * latent_loss + pixel_weight * pixel_loss
    return total, latent_loss, pixel_loss


def _recursive_prediction_loss(
    dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float,
    pixel_weight: float,
    horizon_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backpropagate through a rollout that consumes its own predicted latents."""
    latents = batch["latents"]
    actions = batch["actions"]
    if latents.ndim != 5 or actions.ndim != 3:
        raise ValueError("recursive batches need latent and action time axes")
    if len(latents) != len(actions) or latents.shape[1] != actions.shape[1] + 2:
        raise ValueError("recursive latent and action timelines are misaligned")
    if not 0 < horizon_decay <= 1:
        raise ValueError("horizon_decay must be in (0, 1]")

    previous = latents[:, 0]
    current = latents[:, 1]
    latent_total = current.new_zeros(())
    pixel_total = current.new_zeros(())
    weight_total = 0.0
    for step in range(actions.shape[1]):
        predicted = dynamics(previous, current, actions[:, step])
        target = latents[:, step + 2]
        weight = horizon_decay**step
        normalized_error = (predicted - target) / dynamics.latent_std
        latent_total = latent_total + weight * normalized_error.square().mean()
        if pixel_weight:
            predicted_frame = autoencoder.decode(predicted)
            with torch.no_grad():
                target_frame = autoencoder.decode(target)
            pixel_total = pixel_total + weight * nn.functional.mse_loss(
                predicted_frame, target_frame
            )
        weight_total += weight
        previous, current = current, predicted

    latent_loss = latent_total / weight_total
    pixel_loss = pixel_total / weight_total
    total = latent_weight * latent_loss + pixel_weight * pixel_loss
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
        )
        count = len(batch["action"])
        totals += np.array([float(loss) for loss in losses]) * count
        examples += count
    return tuple(float(value / examples) for value in totals)  # type: ignore[return-value]


@torch.no_grad()
def _recursive_validation_objective(
    dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedSequenceDataset,
    device: torch.device,
    *,
    batch_size: int,
    latent_weight: float,
    pixel_weight: float,
    horizon_decay: float,
) -> tuple[float, float, float]:
    dynamics.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = np.zeros(3, dtype=np.float64)
    examples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        losses = _recursive_prediction_loss(
            dynamics,
            autoencoder,
            batch,
            latent_weight=latent_weight,
            pixel_weight=pixel_weight,
            horizon_decay=horizon_decay,
        )
        count = len(batch["actions"])
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
    rollout_steps: int = 1,
    horizon_decay: float = 1.0,
    initial_checkpoint: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_type": "spatial_latent_dynamics",
            "architecture": SPATIAL_DYNAMICS_ARCHITECTURE,
            "model_state": dynamics.state_dict(),
            "latent_channels": dynamics.latent_channels,
            "action_dim": dynamics.action_dim,
            "hidden_channels": dynamics.hidden_channels,
            "blocks": dynamics.blocks,
            "history": history,
            "latent_weight": latent_weight,
            "pixel_weight": pixel_weight,
            "rollout_steps": rollout_steps,
            "horizon_decay": horizon_decay,
            "initial_checkpoint": (
                None if initial_checkpoint is None else str(initial_checkpoint)
            ),
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
    if checkpoint.get("architecture") != SPATIAL_DYNAMICS_ARCHITECTURE:
        raise ValueError(
            "spatial dynamics checkpoint uses an incompatible or unversioned architecture; "
            "retrain it with the current code"
        )
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


def _save_curve(history: dict[str, list[float]], path: Path, *, rollout_steps: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train"], label="training objective")
    axis.plot(history["validation"], label="validation objective")
    axis.plot(history["validation_latent"], label="normalized latent MSE")
    axis.plot(history["validation_pixel"], label="decoded pixel MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    title = "Spatial action-conditioned dynamics"
    if rollout_steps > 1:
        title = f"{title} ({rollout_steps}-step recursive objective)"
    axis.set_title(title)
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
    patience: int = 5,
    rollout_steps: int = 1,
    horizon_decay: float = 0.8,
    gradient_clip: float = 0.0,
    maximum_validation_sequences: int = 5_000,
    initial_checkpoint: Path | None = None,
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialDynamicsTrainingResult:
    if min(epochs, batch_size, encode_batch_size, maximum_transitions, patience) < 1:
        raise ValueError("training sizes and patience must be positive")
    if latent_weight < 0 or pixel_weight < 0 or latent_weight + pixel_weight == 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if not 0 < horizon_decay <= 1:
        raise ValueError("horizon_decay must be in (0, 1]")
    if gradient_clip < 0:
        raise ValueError("gradient_clip must be non-negative")
    if maximum_validation_sequences < 1:
        raise ValueError("maximum_validation_sequences must be positive")
    recursive = rollout_steps > 1
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha256 = _file_sha256(autoencoder_checkpoint)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    manifest = DatasetManifest.load(manifest_path)
    train_paths, validation_paths = manifest.processed_splits(processed_dir)
    window = "windows" if recursive else "transitions"
    print(f"encoding up to {maximum_transitions:,} diverse training {window}...")
    training = SpatialEncodedDynamicsDataset.from_paths(
        train_paths,
        autoencoder,
        device,
        maximum_transitions=maximum_transitions,
        count_horizon=rollout_steps,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    print(f"encoding frozen validation {window}...")
    validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths,
        autoencoder,
        device,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    if initial_checkpoint is not None:
        dynamics, initial_metadata = load_spatial_dynamics_checkpoint(
            initial_checkpoint, device
        )
        if initial_metadata["autoencoder_sha256"] != autoencoder_sha256:
            raise ValueError("the initial dynamics checkpoint uses a different autoencoder")
        if dynamics.latent_channels != training.latent_shape[0]:
            raise ValueError("the initial dynamics checkpoint uses a different latent shape")
        print(
            f"fine-tuning {initial_checkpoint} "
            f"({dynamics.hidden_channels} channels, {dynamics.blocks} blocks); "
            "its frozen normalization statistics are reused"
        )
    else:
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
    if recursive:
        training_windows = SpatialEncodedSequenceDataset(
            training,
            horizon=rollout_steps,
            maximum_sequences=maximum_transitions,
            seed=seed,
        )
        validation_windows = SpatialEncodedSequenceDataset(
            validation,
            horizon=rollout_steps,
            maximum_sequences=maximum_validation_sequences,
            seed=seed,
        )
    else:
        training_windows = training
        validation_windows = validation
    loader = DataLoader(
        training_windows,
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
            if recursive:
                loss, _, _ = _recursive_prediction_loss(
                    dynamics,
                    autoencoder,
                    batch,
                    latent_weight=latent_weight,
                    pixel_weight=pixel_weight,
                    horizon_decay=horizon_decay,
                )
                count = len(batch["actions"])
            else:
                loss, _, _ = _prediction_loss(
                    dynamics,
                    autoencoder,
                    batch,
                    latent_weight=latent_weight,
                    pixel_weight=pixel_weight,
                )
                count = len(batch["action"])
            loss.backward()
            if gradient_clip:
                nn.utils.clip_grad_norm_(dynamics.parameters(), gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * count
            examples += count
        train_loss = total / examples
        if recursive:
            validation_loss, validation_latent, validation_pixel = (
                _recursive_validation_objective(
                    dynamics,
                    autoencoder,
                    validation_windows,
                    device,
                    batch_size=batch_size,
                    latent_weight=latent_weight,
                    pixel_weight=pixel_weight,
                    horizon_decay=horizon_decay,
                )
            )
        else:
            validation_loss, validation_latent, validation_pixel = _validation_objective(
                dynamics,
                autoencoder,
                validation_windows,
                device,
                batch_size=batch_size,
                latent_weight=latent_weight,
                pixel_weight=pixel_weight,
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
        rollout_steps=rollout_steps,
        horizon_decay=horizon_decay,
        initial_checkpoint=initial_checkpoint,
    )
    metrics = evaluate_dynamics(
        dynamics, autoencoder, validation, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / "one-step-predictions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    save_prediction_grid(dynamics, autoencoder, validation, grid, device)
    _save_curve(history, curve, rollout_steps=rollout_steps)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": (
                    f"spatial {rollout_steps}-step recursive latent dynamics"
                    if recursive
                    else "spatial one-step latent dynamics"
                ),
                "rollout_steps": rollout_steps,
                "horizon_decay": horizon_decay,
                "gradient_clip": gradient_clip,
                "initial_checkpoint": (
                    None if initial_checkpoint is None else str(initial_checkpoint)
                ),
                "training_transitions": len(training_windows),
                "encoded_training_frames": training.encoded_frames,
                "validation_transitions": len(validation_windows),
                "one_step_validation_transitions": len(validation),
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
        training_transitions=len(training_windows),
        encoded_frames=training.encoded_frames,
        latent_shape=training.latent_shape,
        device=str(device),
        rollout_steps=rollout_steps,
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
    split: DatasetSplit = "validation",
    seed: int = 7,
    requested_device: str = "auto",
) -> SpatialDynamicsEvaluationResult:
    if split not in {"validation", "test"}:
        raise ValueError("spatial dynamics evaluation split must be validation or test")
    device = choose_device(requested_device)
    dynamics, metadata = load_spatial_dynamics_checkpoint(dynamics_checkpoint, device)
    if metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("spatial dynamics belongs to a different autoencoder checkpoint")
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    selected_paths = DatasetManifest.load(manifest_path).processed_paths(processed_dir, split)
    dataset = SpatialEncodedDynamicsDataset.from_paths(
        selected_paths,
        autoencoder,
        device,
        encode_batch_size=encode_batch_size,
        seed=seed,
    )
    metrics = evaluate_dynamics(
        dynamics, autoencoder, dataset, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / f"{split}-one-step-predictions.png"
    metrics_path = output_dir / f"{split}-metrics.json"
    save_prediction_grid(dynamics, autoencoder, dataset, grid, device, count=count)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "saved spatial dynamics evaluation",
                "split": split,
                "transitions": len(dataset),
                "latent_shape": dataset.latent_shape,
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
        transitions=len(dataset),
        latent_shape=dataset.latent_shape,
        device=str(device),
    )
