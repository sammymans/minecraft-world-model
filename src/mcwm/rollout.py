"""Recursive held-out evaluation for the latent dynamics model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mcwm.dataset import ProcessedEpisode, SequenceDataset
from mcwm.dynamics import (
    EncodedDynamicsDataset,
    _file_sha256,
    _processed_splits,
    _verify_autoencoder,
    load_dynamics_checkpoint,
)
from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import (
    LatentDynamics,
    SpatialAutoencoder,
    SpatialLatentDynamics,
    TinyAutoencoder,
)
from mcwm.spatial_dynamics import (
    SpatialEncodedDynamicsDataset,
    load_spatial_dynamics_checkpoint,
)
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, load_autoencoder_checkpoint

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

Autoencoder = TinyAutoencoder | SpatialAutoencoder
Dynamics = LatentDynamics | SpatialLatentDynamics


@dataclass(frozen=True)
class RolloutHorizonMetrics:
    horizon: int
    seconds: float
    examples: int
    recursive_latent_mse: float
    recursive_pixel_l1: float
    recursive_pixel_mse: float
    recursive_pixel_psnr_db: float
    teacher_forced_latent_mse: float
    teacher_forced_pixel_mse: float
    copy_latent_mse: float
    copy_pixel_mse: float
    oracle_pixel_mse: float
    shuffled_action_latent_mse: float
    shuffled_action_pixel_mse: float
    action_effect_latent_mse: float
    recursive_edge_ratio: float
    oracle_edge_ratio: float

    @property
    def beats_copy_pixel(self) -> bool:
        return self.recursive_pixel_mse < self.copy_pixel_mse

    @property
    def copy_improvement_percent(self) -> float:
        return 100.0 * (1.0 - self.recursive_pixel_mse / self.copy_pixel_mse)

    @property
    def shuffled_action_pixel_penalty_percent(self) -> float:
        return 100.0 * (
            self.shuffled_action_pixel_mse / self.recursive_pixel_mse - 1.0
        )


@dataclass(frozen=True)
class RolloutEvaluationResult:
    horizons: tuple[RolloutHorizonMetrics, ...]
    metrics_path: Path
    error_curve: Path
    filmstrips: Path
    example_count: int
    max_horizon: int
    device: str


def _frame_timeline(frames: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    return torch.from_numpy(contiguous).to(torch.float32).div_(255.0)


class EncodedRolloutDataset(Dataset[dict[str, torch.Tensor]]):
    """Clean fixed-horizon windows with frozen latent timelines."""

    def __init__(
        self,
        episodes: list[ProcessedEpisode],
        latents: list[torch.Tensor],
        *,
        horizon: int,
        maximum_examples: int | None = None,
        seed: int = 7,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if maximum_examples is not None and maximum_examples < 1:
            raise ValueError("maximum_examples must be positive when supplied")
        if len(episodes) != len(latents):
            raise ValueError("each episode needs one latent timeline")
        for episode, episode_latents in zip(episodes, latents, strict=True):
            if episode_latents.ndim not in {2, 4} or len(episode_latents) != len(
                episode.frames
            ):
                raise ValueError("latent timelines must be flat or spatial with a time axis")
        latent_shapes = {tuple(values.shape[1:]) for values in latents}
        if len(latent_shapes) != 1:
            raise ValueError("all latent timelines must use the same shape")
        self.episodes = episodes
        self.latents = latents
        self.horizon = horizon
        self.index = SequenceDataset(episodes, horizon=horizon).index
        if maximum_examples is not None and len(self.index) > maximum_examples:
            selected = torch.randperm(
                len(self.index), generator=torch.Generator().manual_seed(seed)
            )[:maximum_examples]
            self.index = [self.index[int(item)] for item in selected]
        self.latent_shape = latent_shapes.pop()

    @classmethod
    @torch.no_grad()
    def from_paths(
        cls,
        paths: list[Path],
        autoencoder: TinyAutoencoder,
        device: torch.device,
        *,
        horizon: int,
        encode_batch_size: int = 128,
        maximum_examples: int | None = None,
        seed: int = 7,
    ) -> EncodedRolloutDataset:
        encoded = EncodedDynamicsDataset.from_paths(
            paths,
            autoencoder,
            device,
            encode_batch_size=encode_batch_size,
        )
        return cls(
            encoded.episodes,
            encoded.latents,
            horizon=horizon,
            maximum_examples=maximum_examples,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        episode = self.episodes[episode_index]
        stop = current_index + self.horizon
        return {
            "latents": self.latents[episode_index][current_index - 1 : stop + 1].float(),
            "frames": _frame_timeline(
                episode.frames[current_index - 1 : stop + 1]
            ),
            "actions": torch.from_numpy(
                episode.actions[current_index:stop].astype(np.float32, copy=False)
            ),
        }


@torch.no_grad()
def recursive_latent_rollout(
    dynamics: Dynamics,
    previous_latent: torch.Tensor,
    current_latent: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Predict a latent timeline without reading any real future latent."""
    if previous_latent.shape != current_latent.shape:
        raise ValueError("seed latents must have the same shape")
    if actions.ndim != 3 or actions.shape[0] != current_latent.shape[0]:
        raise ValueError("action batch shape does not match seed latent batch shape")
    predictions: list[torch.Tensor] = []
    previous = previous_latent
    current = current_latent
    for step in range(actions.shape[-2]):
        predicted = dynamics(previous, current, actions[..., step, :])
        predictions.append(predicted)
        previous, current = current, predicted
    return torch.stack(predictions, dim=1)


def _psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _validate_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and increasing")
    return horizons


def _metrics_payload(metrics: RolloutHorizonMetrics) -> dict[str, float | int | bool]:
    return {
        **asdict(metrics),
        # Metric accumulation can produce NumPy scalar types. Normalize derived
        # values here so the public metrics artifact is always valid JSON.
        "beats_copy_pixel": bool(metrics.beats_copy_pixel),
        "copy_improvement_percent": float(metrics.copy_improvement_percent),
        "shuffled_action_pixel_penalty_percent": (
            float(metrics.shuffled_action_pixel_penalty_percent)
        ),
    }


@torch.no_grad()
def evaluate_rollouts(
    dynamics: Dynamics,
    autoencoder: Autoencoder,
    dataset: EncodedRolloutDataset,
    device: torch.device,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20),
    batch_size: int = 64,
    seed: int = 7,
) -> tuple[RolloutHorizonMetrics, ...]:
    """Measure recursive error growth against controls on one fixed window set."""
    horizons = _validate_horizons(horizons)
    if dataset.horizon < horizons[-1]:
        raise ValueError("dataset horizon is shorter than requested evaluation horizon")
    if not len(dataset):
        raise ValueError("cannot evaluate an empty rollout dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dynamics.eval()
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_actions = torch.stack([dataset[index]["actions"] for index in range(len(dataset))])
    if len(all_actions) > 1:
        shift = 1 + seed % (len(all_actions) - 1)
        mismatched_actions = torch.roll(all_actions, shifts=shift, dims=0)
    else:
        mismatched_actions = torch.flip(all_actions, dims=(1,))

    names = (
        "recursive_latent_squared",
        "recursive_pixel_absolute",
        "recursive_pixel_squared",
        "teacher_latent_squared",
        "teacher_pixel_squared",
        "copy_latent_squared",
        "copy_pixel_squared",
        "oracle_pixel_squared",
        "shuffled_latent_squared",
        "shuffled_pixel_squared",
        "action_effect_squared",
        "recursive_edge_energy",
        "oracle_edge_energy",
        "target_edge_energy",
    )
    sums = {name: np.zeros(dataset.horizon, dtype=np.float64) for name in names}
    examples = 0
    latent_values = 0
    pixel_values = 0
    offset = 0

    for raw_batch in loader:
        latents = raw_batch["latents"].to(device)
        frames = raw_batch["frames"].to(device)
        actions = raw_batch["actions"].to(device)
        count = len(actions)
        shuffled = mismatched_actions[offset : offset + count].to(device)
        offset += count

        predicted = recursive_latent_rollout(
            dynamics, latents[:, 0], latents[:, 1], actions
        )
        shuffled_predicted = recursive_latent_rollout(
            dynamics, latents[:, 0], latents[:, 1], shuffled
        )
        decoded_copy = autoencoder.decode(latents[:, 1]).clamp(0, 1)

        for step in range(dataset.horizon):
            target_latent = latents[:, step + 2]
            target_frame = frames[:, step + 2]
            teacher_latent = dynamics(
                latents[:, step], latents[:, step + 1], actions[:, step]
            )
            predicted_frame = autoencoder.decode(predicted[:, step]).clamp(0, 1)
            teacher_frame = autoencoder.decode(teacher_latent).clamp(0, 1)
            oracle_frame = autoencoder.decode(target_latent).clamp(0, 1)
            shuffled_frame = autoencoder.decode(shuffled_predicted[:, step]).clamp(0, 1)

            sums["recursive_latent_squared"][step] += torch.square(
                predicted[:, step] - target_latent
            ).sum().item()
            sums["recursive_pixel_absolute"][step] += torch.abs(
                predicted_frame - target_frame
            ).sum().item()
            sums["recursive_pixel_squared"][step] += torch.square(
                predicted_frame - target_frame
            ).sum().item()
            sums["teacher_latent_squared"][step] += torch.square(
                teacher_latent - target_latent
            ).sum().item()
            sums["teacher_pixel_squared"][step] += torch.square(
                teacher_frame - target_frame
            ).sum().item()
            sums["copy_latent_squared"][step] += torch.square(
                latents[:, 1] - target_latent
            ).sum().item()
            sums["copy_pixel_squared"][step] += torch.square(
                decoded_copy - target_frame
            ).sum().item()
            sums["oracle_pixel_squared"][step] += torch.square(
                oracle_frame - target_frame
            ).sum().item()
            sums["shuffled_latent_squared"][step] += torch.square(
                shuffled_predicted[:, step] - target_latent
            ).sum().item()
            sums["shuffled_pixel_squared"][step] += torch.square(
                shuffled_frame - target_frame
            ).sum().item()
            sums["action_effect_squared"][step] += torch.square(
                predicted[:, step] - shuffled_predicted[:, step]
            ).sum().item()
            # Squared error rewards blur, so track how much edge energy each
            # prediction actually carries relative to the real frame.
            for name, image in (
                ("recursive_edge_energy", predicted_frame),
                ("oracle_edge_energy", oracle_frame),
                ("target_edge_energy", target_frame),
            ):
                horizontal, vertical = image_gradients(image)
                sums[name][step] += (
                    horizontal.abs().sum().item() + vertical.abs().sum().item()
                )

        examples += count
        latent_values += predicted[:, 0].numel()
        pixel_values += frames[:, 0].numel()

    results: list[RolloutHorizonMetrics] = []
    model_fps = dataset.episodes[0].model_fps
    for horizon in horizons:
        step = horizon - 1
        recursive_pixel_mse = sums["recursive_pixel_squared"][step] / pixel_values
        results.append(
            RolloutHorizonMetrics(
                horizon=horizon,
                seconds=horizon / model_fps,
                examples=examples,
                recursive_latent_mse=sums["recursive_latent_squared"][step]
                / latent_values,
                recursive_pixel_l1=sums["recursive_pixel_absolute"][step]
                / pixel_values,
                recursive_pixel_mse=recursive_pixel_mse,
                recursive_pixel_psnr_db=_psnr(recursive_pixel_mse),
                teacher_forced_latent_mse=sums["teacher_latent_squared"][step]
                / latent_values,
                teacher_forced_pixel_mse=sums["teacher_pixel_squared"][step]
                / pixel_values,
                copy_latent_mse=sums["copy_latent_squared"][step] / latent_values,
                copy_pixel_mse=sums["copy_pixel_squared"][step] / pixel_values,
                oracle_pixel_mse=sums["oracle_pixel_squared"][step] / pixel_values,
                shuffled_action_latent_mse=sums["shuffled_latent_squared"][step]
                / latent_values,
                shuffled_action_pixel_mse=sums["shuffled_pixel_squared"][step]
                / pixel_values,
                action_effect_latent_mse=sums["action_effect_squared"][step]
                / latent_values,
                recursive_edge_ratio=sums["recursive_edge_energy"][step]
                / max(sums["target_edge_energy"][step], 1e-12),
                oracle_edge_ratio=sums["oracle_edge_energy"][step]
                / max(sums["target_edge_energy"][step], 1e-12),
            )
        )
    return tuple(results)


def save_error_curve(metrics: tuple[RolloutHorizonMetrics, ...], path: Path) -> None:
    horizons = [item.horizon for item in metrics]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        horizons,
        [item.recursive_pixel_mse for item in metrics],
        marker="o",
        label="recursive model",
    )
    axis.plot(
        horizons,
        [item.teacher_forced_pixel_mse for item in metrics],
        marker="o",
        label="teacher-forced one-step",
    )
    axis.plot(
        horizons,
        [item.copy_pixel_mse for item in metrics],
        marker="o",
        label="frozen decoded copy",
    )
    axis.plot(
        horizons,
        [item.shuffled_action_pixel_mse for item in metrics],
        marker="o",
        label="mismatched actions",
    )
    axis.plot(
        horizons,
        [item.oracle_pixel_mse for item in metrics],
        linestyle="--",
        label="decoder oracle",
    )
    axis.set_xticks(horizons)
    axis.set_xlabel("recursive rollout horizon (10 Hz steps)")
    axis.set_ylabel("held-out pixel MSE")
    axis.set_title("Minecraft latent world model: recursive error growth")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


@torch.no_grad()
def save_rollout_filmstrips(
    dynamics: Dynamics,
    autoencoder: Autoencoder,
    dataset: EncodedRolloutDataset,
    path: Path,
    device: torch.device,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20),
    count: int = 3,
) -> None:
    horizons = _validate_horizons(horizons)
    if not len(dataset):
        raise ValueError("cannot visualize an empty rollout dataset")
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    columns = (0, *horizons)
    row_labels = (
        "real",
        "recursive prediction",
        "mismatched actions",
        "frozen copy",
        "error x4",
    )
    tile = 128
    header = 28
    label_width = 210
    sample_height = len(row_labels) * (tile + header)
    canvas = np.full(
        (count * sample_height, label_width + len(columns) * tile, 3),
        24,
        dtype=np.uint8,
    )

    dynamics.eval()
    autoencoder.eval()
    for sample_number, dataset_index in enumerate(indices):
        sample = dataset[int(dataset_index)]
        latents = sample["latents"].unsqueeze(0).to(device)
        actions = sample["actions"].unsqueeze(0).to(device)
        predicted = recursive_latent_rollout(
            dynamics, latents[:, 0], latents[:, 1], actions
        )
        mismatch_index = (int(dataset_index) + max(1, len(dataset) // 2)) % len(dataset)
        mismatched_actions = dataset[mismatch_index]["actions"].unsqueeze(0).to(device)
        mismatched = recursive_latent_rollout(
            dynamics, latents[:, 0], latents[:, 1], mismatched_actions
        )
        predicted_frames = autoencoder.decode(predicted[0]).clamp(0, 1).cpu().numpy()
        predicted_frames = predicted_frames.transpose(0, 2, 3, 1)
        mismatched_frames = autoencoder.decode(mismatched[0]).clamp(0, 1).cpu().numpy()
        mismatched_frames = mismatched_frames.transpose(0, 2, 3, 1)
        copy_frame = autoencoder.decode(latents[:, 1]).clamp(0, 1)[0]
        copy_frame = copy_frame.cpu().permute(1, 2, 0).numpy()
        real_frames = sample["frames"].permute(0, 2, 3, 1).numpy()
        sample_y = sample_number * sample_height

        for row, row_label in enumerate(row_labels):
            y = sample_y + row * (tile + header)
            cv2.putText(
                canvas,
                f"{row_label} | sample {int(dataset_index)}",
                (8, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            for column, horizon in enumerate(columns):
                x = label_width + column * tile
                cv2.putText(
                    canvas,
                    "seed t" if horizon == 0 else f"t+{horizon}",
                    (x + 7, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                real = real_frames[1 if horizon == 0 else horizon + 1]
                predicted_frame = real_frames[1] if horizon == 0 else predicted_frames[horizon - 1]
                mismatched_frame = (
                    real_frames[1] if horizon == 0 else mismatched_frames[horizon - 1]
                )
                if row == 0:
                    image = _to_bgr(real)
                elif row == 1:
                    image = _to_bgr(predicted_frame)
                elif row == 2:
                    image = _to_bgr(mismatched_frame)
                elif row == 3:
                    image = _to_bgr(real_frames[1] if horizon == 0 else copy_frame)
                else:
                    error = np.abs(predicted_frame - real).mean(axis=2)
                    image = cv2.applyColorMap(
                        np.clip(error * 4 * 255, 0, 255).astype(np.uint8),
                        cv2.COLORMAP_INFERNO,
                    )
                canvas[y + header : y + header + tile, x : x + tile] = cv2.resize(
                    image, (tile, tile), interpolation=cv2.INTER_NEAREST
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write rollout filmstrips: {path}")


def evaluate_saved_rollouts(
    processed_dir: Path,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20),
    batch_size: int = 64,
    encode_batch_size: int = 128,
    count: int = 3,
    maximum_examples: int = 5_000,
    split: DatasetSplit = "validation",
    seed: int = 7,
    requested_device: str = "auto",
) -> RolloutEvaluationResult:
    horizons = _validate_horizons(horizons)
    if split not in {"validation", "test"}:
        raise ValueError("rollout evaluation split must be validation or test")
    device = choose_device(requested_device)
    checkpoint = torch.load(dynamics_checkpoint, map_location="cpu", weights_only=True)
    spatial = checkpoint.get("model_type") == "spatial_latent_dynamics"
    if spatial:
        dynamics, dynamics_metadata = load_spatial_dynamics_checkpoint(
            dynamics_checkpoint, device
        )
        if dynamics_metadata["autoencoder_sha256"] != _file_sha256(
            autoencoder_checkpoint
        ):
            raise ValueError("spatial dynamics belongs to a different autoencoder checkpoint")
        autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
        if autoencoder.latent_channels != dynamics.latent_channels:
            raise ValueError("autoencoder and dynamics latent channels do not match")
    else:
        dynamics, dynamics_metadata = load_dynamics_checkpoint(dynamics_checkpoint, device)
        _verify_autoencoder(dynamics_metadata, autoencoder_checkpoint)
        autoencoder, autoencoder_metadata = load_autoencoder_checkpoint(
            autoencoder_checkpoint, device
        )
        if int(autoencoder_metadata["latent_dim"]) != dynamics.latent_dim:
            raise ValueError("autoencoder and dynamics latent dimensions do not match")
    autoencoder.requires_grad_(False)
    if manifest_path is not None:
        paths = DatasetManifest.load(manifest_path).processed_paths(processed_dir, split)
    else:
        if split == "test":
            raise ValueError("test evaluation requires an explicit manifest")
        _, paths = _processed_splits(processed_dir, None)
    if spatial:
        encoded = SpatialEncodedDynamicsDataset.from_paths(
            paths, autoencoder, device, encode_batch_size=encode_batch_size
        )
        dataset = EncodedRolloutDataset(
            encoded.episodes,
            encoded.latents,
            horizon=horizons[-1],
            maximum_examples=maximum_examples,
            seed=seed,
        )
    else:
        dataset = EncodedRolloutDataset.from_paths(
            paths,
            autoencoder,
            device,
            horizon=horizons[-1],
            encode_batch_size=encode_batch_size,
            maximum_examples=maximum_examples,
            seed=seed,
        )
    metrics = evaluate_rollouts(
        dynamics,
        autoencoder,
        dataset,
        device,
        horizons=horizons,
        batch_size=batch_size,
        seed=seed,
    )
    metrics_path = output_dir / "metrics.json"
    curve_path = output_dir / "error-vs-horizon.png"
    filmstrip_path = output_dir / "rollout-filmstrips.png"
    save_error_curve(metrics, curve_path)
    save_rollout_filmstrips(
        dynamics,
        autoencoder,
        dataset,
        filmstrip_path,
        device,
        horizons=horizons,
        count=count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": "recursive latent rollout evaluation",
                "split": split,
                "examples": len(dataset),
                "horizons": [_metrics_payload(item) for item in metrics],
                "dynamics_checkpoint": str(dynamics_checkpoint),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
                "dataset_manifest": str(manifest_path) if manifest_path else None,
                "dataset_manifest_sha256": (
                    _file_sha256(manifest_path) if manifest_path else None
                ),
                "seed": seed,
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return RolloutEvaluationResult(
        horizons=metrics,
        metrics_path=metrics_path,
        error_curve=curve_path,
        filmstrips=filmstrip_path,
        example_count=len(dataset),
        max_horizon=horizons[-1],
        device=str(device),
    )
