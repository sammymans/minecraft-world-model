"""Training and one-step evaluation for action-conditioned latent dynamics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mcwm.dataset import ProcessedEpisode, SequenceDataset, split_episode_paths
from mcwm.manifest import DatasetManifest, DatasetSplit
from mcwm.model import LatentDynamics, TinyAutoencoder
from mcwm.training import choose_device, load_autoencoder_checkpoint, seed_everything

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class DynamicsMetrics:
    examples: int
    latent_mse: float
    pixel_l1: float
    pixel_mse: float
    pixel_psnr_db: float
    copy_latent_mse: float
    copy_pixel_l1: float
    copy_pixel_mse: float
    copy_pixel_psnr_db: float
    decoded_copy_pixel_l1: float
    decoded_copy_pixel_mse: float
    decoded_copy_pixel_psnr_db: float
    oracle_pixel_l1: float
    oracle_pixel_mse: float
    oracle_pixel_psnr_db: float
    shuffled_action_latent_mse: float
    shuffled_action_pixel_mse: float
    action_effect_latent_mse: float

    @property
    def shuffled_action_degradation(self) -> float:
        return self.shuffled_action_latent_mse - self.latent_mse


@dataclass(frozen=True)
class DynamicsTrainingResult:
    checkpoint: Path
    comparison_grid: Path
    training_curve: Path
    metrics_path: Path
    train_metrics: DynamicsMetrics
    validation_metrics: DynamicsMetrics
    parameter_count: int
    latent_dim: int
    device: str


@dataclass(frozen=True)
class DynamicsEvaluationResult:
    metrics: DynamicsMetrics
    comparison_grid: Path
    training_curve: Path
    example_count: int
    latent_dim: int
    device: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_splits(
    processed_dir: Path,
    manifest_path: Path | None,
) -> tuple[list[Path], list[Path]]:
    if manifest_path is not None:
        return DatasetManifest.load(manifest_path).processed_splits(processed_dir)
    paths = sorted(processed_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No processed episodes in {processed_dir}")
    return split_episode_paths(paths)


def _frame_tensor(frame: np.ndarray) -> torch.Tensor:
    contiguous = np.ascontiguousarray(frame.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).to(torch.float32).div_(255.0)


class EncodedDynamicsDataset(Dataset[dict[str, torch.Tensor]]):
    """Clean one-step transitions with frozen visual latents cached in memory."""

    def __init__(self, episodes: list[ProcessedEpisode], latents: list[torch.Tensor]):
        if len(episodes) != len(latents):
            raise ValueError("each episode needs one latent timeline")
        for episode, episode_latents in zip(episodes, latents, strict=True):
            if episode_latents.ndim != 2 or len(episode_latents) != len(episode.frames):
                raise ValueError("latent timelines must be [time, latent_dim]")
        latent_dims = {int(values.shape[1]) for values in latents}
        if len(latent_dims) != 1:
            raise ValueError("all latent timelines must use the same latent_dim")
        self.episodes = episodes
        self.latents = latents
        self.index = SequenceDataset(episodes, horizon=1).index
        self.latent_dim = latent_dims.pop()

    @classmethod
    @torch.no_grad()
    def from_paths(
        cls,
        paths: list[Path],
        autoencoder: TinyAutoencoder,
        device: torch.device,
        *,
        encode_batch_size: int = 128,
    ) -> EncodedDynamicsDataset:
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be positive")
        episodes = [ProcessedEpisode.load(path) for path in paths]
        autoencoder.eval()
        encoded_episodes: list[torch.Tensor] = []
        for episode in episodes:
            chunks: list[torch.Tensor] = []
            for start in range(0, len(episode.frames), encode_batch_size):
                frames = episode.frames[start : start + encode_batch_size]
                contiguous = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
                batch = torch.from_numpy(contiguous).to(device=device, dtype=torch.float32)
                batch.div_(255.0)
                chunks.append(autoencoder.encode(batch).cpu())
            encoded_episodes.append(torch.cat(chunks))
        return cls(episodes, encoded_episodes)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, current_index = self.index[item]
        episode = self.episodes[episode_index]
        episode_latents = self.latents[episode_index]
        return {
            "previous_latent": episode_latents[current_index - 1],
            "current_latent": episode_latents[current_index],
            "target_latent": episode_latents[current_index + 1],
            "action": torch.from_numpy(episode.actions[current_index]).to(torch.float32),
            "current_frame": _frame_tensor(episode.frames[current_index]),
            "target_frame": _frame_tensor(episode.frames[current_index + 1]),
        }

    def action_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.index:
            raise ValueError("cannot fit action statistics on an empty dataset")
        actions = np.stack(
            [self.episodes[episode].actions[current] for episode, current in self.index]
        ).astype(np.float32, copy=False)
        mean = torch.from_numpy(actions.mean(axis=0))
        # A floor avoids exploding rare binary controls while still scaling mouse movement.
        std = torch.from_numpy(actions.std(axis=0)).clamp_min(0.05)
        return mean, std


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _prediction_loss(
    dynamics: LatentDynamics,
    autoencoder: TinyAutoencoder,
    batch: dict[str, torch.Tensor],
    *,
    latent_weight: float,
    pixel_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_latent = dynamics(
        batch["previous_latent"], batch["current_latent"], batch["action"]
    )
    predicted_frame = autoencoder.decode(predicted_latent)
    latent_loss = nn.functional.mse_loss(predicted_latent, batch["target_latent"])
    pixel_loss = nn.functional.mse_loss(predicted_frame, batch["target_frame"])
    total = latent_weight * latent_loss + pixel_weight * pixel_loss
    return total, latent_loss, pixel_loss


@torch.no_grad()
def _mean_objective(
    dynamics: LatentDynamics,
    autoencoder: TinyAutoencoder,
    dataset: EncodedDynamicsDataset,
    device: torch.device,
    *,
    batch_size: int,
    latent_weight: float,
    pixel_weight: float,
) -> tuple[float, float, float]:
    dynamics.eval()
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_sum = 0.0
    latent_sum = 0.0
    pixel_sum = 0.0
    examples = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        total, latent, pixel = _prediction_loss(
            dynamics,
            autoencoder,
            batch,
            latent_weight=latent_weight,
            pixel_weight=pixel_weight,
        )
        count = len(batch["action"])
        total_sum += float(total) * count
        latent_sum += float(latent) * count
        pixel_sum += float(pixel) * count
        examples += count
    if not examples:
        raise ValueError("cannot evaluate an empty dynamics dataset")
    return total_sum / examples, latent_sum / examples, pixel_sum / examples


def _psnr(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def evaluate_dynamics(
    dynamics: LatentDynamics,
    autoencoder: TinyAutoencoder,
    dataset: EncodedDynamicsDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    seed: int = 7,
) -> DynamicsMetrics:
    """Measure prediction, copy-frame baseline, and shuffled-action behavior."""
    if not len(dataset):
        raise ValueError("cannot evaluate an empty dynamics dataset")
    dynamics.eval()
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_actions = torch.stack([dataset[index]["action"] for index in range(len(dataset))])
    generator = torch.Generator().manual_seed(seed)
    shuffled_actions = all_actions[torch.randperm(len(dataset), generator=generator)]

    latent_squared = 0.0
    pixel_absolute = 0.0
    pixel_squared = 0.0
    copy_latent_squared = 0.0
    copy_pixel_absolute = 0.0
    copy_pixel_squared = 0.0
    decoded_copy_pixel_absolute = 0.0
    decoded_copy_pixel_squared = 0.0
    oracle_pixel_absolute = 0.0
    oracle_pixel_squared = 0.0
    shuffled_latent_squared = 0.0
    shuffled_pixel_squared = 0.0
    action_effect_squared = 0.0
    latent_values = 0
    pixel_values = 0
    examples = 0

    offset = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        count = len(batch["action"])
        shuffled = shuffled_actions[offset : offset + count].to(device)
        offset += count

        predicted_latent = dynamics(
            batch["previous_latent"], batch["current_latent"], batch["action"]
        )
        predicted_frame = autoencoder.decode(predicted_latent).clamp(0, 1)
        decoded_copy_frame = autoencoder.decode(batch["current_latent"]).clamp(0, 1)
        oracle_frame = autoencoder.decode(batch["target_latent"]).clamp(0, 1)
        shuffled_latent = dynamics(
            batch["previous_latent"], batch["current_latent"], shuffled
        )
        shuffled_frame = autoencoder.decode(shuffled_latent).clamp(0, 1)

        latent_squared += torch.square(predicted_latent - batch["target_latent"]).sum().item()
        pixel_absolute += torch.abs(predicted_frame - batch["target_frame"]).sum().item()
        pixel_squared += torch.square(predicted_frame - batch["target_frame"]).sum().item()
        copy_latent_squared += torch.square(
            batch["current_latent"] - batch["target_latent"]
        ).sum().item()
        copy_pixel_absolute += torch.abs(
            batch["current_frame"] - batch["target_frame"]
        ).sum().item()
        copy_pixel_squared += torch.square(
            batch["current_frame"] - batch["target_frame"]
        ).sum().item()
        decoded_copy_pixel_absolute += torch.abs(
            decoded_copy_frame - batch["target_frame"]
        ).sum().item()
        decoded_copy_pixel_squared += torch.square(
            decoded_copy_frame - batch["target_frame"]
        ).sum().item()
        oracle_pixel_absolute += torch.abs(
            oracle_frame - batch["target_frame"]
        ).sum().item()
        oracle_pixel_squared += torch.square(
            oracle_frame - batch["target_frame"]
        ).sum().item()
        shuffled_latent_squared += torch.square(
            shuffled_latent - batch["target_latent"]
        ).sum().item()
        shuffled_pixel_squared += torch.square(
            shuffled_frame - batch["target_frame"]
        ).sum().item()
        action_effect_squared += torch.square(predicted_latent - shuffled_latent).sum().item()
        latent_values += predicted_latent.numel()
        pixel_values += predicted_frame.numel()
        examples += count

    pixel_mse = pixel_squared / pixel_values
    copy_pixel_mse = copy_pixel_squared / pixel_values
    decoded_copy_pixel_mse = decoded_copy_pixel_squared / pixel_values
    oracle_pixel_mse = oracle_pixel_squared / pixel_values
    return DynamicsMetrics(
        examples=examples,
        latent_mse=latent_squared / latent_values,
        pixel_l1=pixel_absolute / pixel_values,
        pixel_mse=pixel_mse,
        pixel_psnr_db=_psnr(pixel_mse),
        copy_latent_mse=copy_latent_squared / latent_values,
        copy_pixel_l1=copy_pixel_absolute / pixel_values,
        copy_pixel_mse=copy_pixel_mse,
        copy_pixel_psnr_db=_psnr(copy_pixel_mse),
        decoded_copy_pixel_l1=decoded_copy_pixel_absolute / pixel_values,
        decoded_copy_pixel_mse=decoded_copy_pixel_mse,
        decoded_copy_pixel_psnr_db=_psnr(decoded_copy_pixel_mse),
        oracle_pixel_l1=oracle_pixel_absolute / pixel_values,
        oracle_pixel_mse=oracle_pixel_mse,
        oracle_pixel_psnr_db=_psnr(oracle_pixel_mse),
        shuffled_action_latent_mse=shuffled_latent_squared / latent_values,
        shuffled_action_pixel_mse=shuffled_pixel_squared / pixel_values,
        action_effect_latent_mse=action_effect_squared / latent_values,
    )


def _save_dynamics_checkpoint(
    path: Path,
    dynamics: LatentDynamics,
    *,
    history: dict[str, list[float]],
    autoencoder_checkpoint: Path,
    autoencoder_sha256: str,
    train_paths: list[Path],
    validation_paths: list[Path],
    manifest_path: Path | None,
    latent_weight: float,
    pixel_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _file_sha256(manifest_path) if manifest_path is not None else None
    torch.save(
        {
            "format_version": 1,
            "model_state": dynamics.state_dict(),
            "latent_dim": dynamics.latent_dim,
            "action_dim": dynamics.action_dim,
            "hidden_dim": dynamics.hidden_dim,
            "hidden_layers": dynamics.hidden_layers,
            "action_mean": dynamics.action_mean.detach().cpu(),
            "action_std": dynamics.action_std.detach().cpu(),
            "history": history,
            "latent_weight": latent_weight,
            "pixel_weight": pixel_weight,
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": autoencoder_sha256,
            "train_episodes": [item.stem for item in train_paths],
            "validation_episodes": [item.stem for item in validation_paths],
            "dataset_manifest": str(manifest_path) if manifest_path is not None else None,
            "dataset_manifest_sha256": manifest_hash,
        },
        path,
    )


def load_dynamics_checkpoint(
    path: Path, device: torch.device
) -> tuple[LatentDynamics, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    dynamics = LatentDynamics(
        latent_dim=int(checkpoint["latent_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
        action_mean=checkpoint["action_mean"],
        action_std=checkpoint["action_std"],
    ).to(device)
    dynamics.load_state_dict(checkpoint["model_state"])
    dynamics.eval()
    return dynamics, checkpoint


def _verify_autoencoder(checkpoint: dict, autoencoder_path: Path) -> None:
    actual_hash = _file_sha256(autoencoder_path)
    if checkpoint["autoencoder_sha256"] != actual_hash:
        raise ValueError(
            "dynamics checkpoint belongs to a different autoencoder checkpoint; "
            "latent coordinate systems cannot be mixed"
        )


def _save_training_curve(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="training objective")
    axis.plot(history["validation_loss"], label="validation objective")
    axis.plot(history["validation_latent_mse"], label="validation latent MSE", alpha=0.8)
    axis.plot(history["validation_pixel_mse"], label="validation pixel MSE", alpha=0.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean loss")
    axis.set_title("Action-conditioned latent dynamics")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_prediction_grid(
    dynamics: LatentDynamics,
    autoencoder: TinyAutoencoder,
    dataset: EncodedDynamicsDataset,
    path: Path,
    device: torch.device,
    *,
    count: int = 6,
) -> None:
    if not len(dataset):
        raise ValueError("cannot visualize an empty dynamics dataset")
    dynamics.eval()
    autoencoder.eval()
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    samples = [dataset[int(index)] for index in indices]
    previous = torch.stack([sample["previous_latent"] for sample in samples]).to(device)
    current = torch.stack([sample["current_latent"] for sample in samples]).to(device)
    actions = torch.stack([sample["action"] for sample in samples]).to(device)
    targets = torch.stack([sample["target_latent"] for sample in samples]).to(device)
    predicted = autoencoder.decode(dynamics(previous, current, actions)).clamp(0, 1)
    decoded_copy = autoencoder.decode(current).clamp(0, 1)
    oracle = autoencoder.decode(targets).clamp(0, 1)

    tile_size = 144
    header = 28
    labels = (
        "previous",
        "current",
        "real next",
        "decoded copy",
        "predicted next",
        "decoder oracle",
        "error x4",
    )
    canvas = np.full(
        (count * (tile_size + header), len(labels) * tile_size, 3), 24, dtype=np.uint8
    )
    rows = zip(samples, decoded_copy.cpu(), predicted.cpu(), oracle.cpu(), strict=True)
    for row, (sample, decoded_copy_frame, predicted_frame, oracle_frame) in enumerate(rows):
        previous_frame = dataset.episodes[dataset.index[int(indices[row])][0]].frames[
            dataset.index[int(indices[row])][1] - 1
        ]
        current_frame = sample["current_frame"].permute(1, 2, 0).numpy()
        target_frame = sample["target_frame"].permute(1, 2, 0).numpy()
        decoded_copy_np = decoded_copy_frame.permute(1, 2, 0).numpy()
        predicted_np = predicted_frame.permute(1, 2, 0).numpy()
        oracle_np = oracle_frame.permute(1, 2, 0).numpy()
        error = np.abs(predicted_np - target_frame).mean(axis=2)
        images = (
            cv2.cvtColor(previous_frame, cv2.COLOR_RGB2BGR),
            cv2.cvtColor((current_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((target_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((decoded_copy_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((predicted_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((oracle_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.applyColorMap(
                np.clip(error * 4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            ),
        )
        y = row * (tile_size + header)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = column * tile_size
            cv2.putText(
                canvas,
                label,
                (x + 7, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            canvas[y + header : y + header + tile_size, x : x + tile_size] = cv2.resize(
                image, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write prediction grid: {path}")


def _metrics_payload(metrics: DynamicsMetrics) -> dict[str, float | int | bool]:
    payload: dict[str, float | int | bool] = asdict(metrics)
    payload.update(
        {
            "beats_copy_latent": metrics.latent_mse < metrics.copy_latent_mse,
            "beats_raw_copy_pixel": metrics.pixel_mse < metrics.copy_pixel_mse,
            "beats_decoded_copy_pixel": (
                metrics.pixel_mse < metrics.decoded_copy_pixel_mse
            ),
            "pixel_mse_above_decoder_oracle": metrics.pixel_mse - metrics.oracle_pixel_mse,
            "shuffled_action_degradation": metrics.shuffled_action_degradation,
            "shuffling_actions_is_worse": metrics.shuffled_action_degradation > 0,
        }
    )
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_dynamics(
    processed_dir: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    batch_size: int = 128,
    encode_batch_size: int = 128,
    hidden_dim: int = 512,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    latent_weight: float = 1.0,
    pixel_weight: float = 1.0,
    manifest_path: Path | None = None,
    patience: int = 8,
    seed: int = 7,
    requested_device: str = "auto",
) -> DynamicsTrainingResult:
    """Train dynamics while keeping the checkpointed encoder and decoder frozen."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if latent_weight < 0 or pixel_weight < 0 or latent_weight + pixel_weight == 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder_sha256 = _file_sha256(autoencoder_checkpoint)
    autoencoder, autoencoder_metadata = load_autoencoder_checkpoint(
        autoencoder_checkpoint, device
    )
    if _file_sha256(autoencoder_checkpoint) != autoencoder_sha256:
        raise ValueError(
            "autoencoder checkpoint changed while it was being loaded; "
            "train dynamics from a stable checkpoint path"
        )
    autoencoder.requires_grad_(False)
    train_paths, validation_paths = _processed_splits(processed_dir, manifest_path)
    print("encoding frozen training latents once...")
    training = EncodedDynamicsDataset.from_paths(
        train_paths, autoencoder, device, encode_batch_size=encode_batch_size
    )
    print("encoding frozen validation latents once...")
    validation = EncodedDynamicsDataset.from_paths(
        validation_paths, autoencoder, device, encode_batch_size=encode_batch_size
    )
    if not len(training) or not len(validation):
        raise ValueError("training and validation each need clean one-step transitions")
    if training.latent_dim != int(autoencoder_metadata["latent_dim"]):
        raise ValueError("encoded latent size does not match the autoencoder checkpoint")

    action_mean, action_std = training.action_statistics()
    dynamics = LatentDynamics(
        latent_dim=training.latent_dim,
        action_dim=9,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        action_mean=action_mean,
        action_std=action_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        dynamics.parameters(), learning_rate, weight_decay=weight_decay
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
        "train_loss": [],
        "validation_loss": [],
        "validation_latent_mse": [],
        "validation_pixel_mse": [],
    }
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(epochs):
        dynamics.train()
        total_loss = 0.0
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
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch["action"])
            examples += len(batch["action"])
        train_loss = total_loss / examples
        validation_loss, validation_latent, validation_pixel = _mean_objective(
            dynamics,
            autoencoder,
            validation,
            device,
            batch_size=batch_size,
            latent_weight=latent_weight,
            pixel_weight=pixel_weight,
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_latent_mse"].append(validation_latent)
        history["validation_pixel_mse"].append(validation_pixel)
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
        raise RuntimeError("dynamics training did not produce a checkpoint")
    dynamics.load_state_dict(best_state)
    if _file_sha256(autoencoder_checkpoint) != autoencoder_sha256:
        raise ValueError(
            "autoencoder checkpoint changed during dynamics training; "
            "use a distinct, stable checkpoint path"
        )
    checkpoint_path = output_dir / "best.pt"
    _save_dynamics_checkpoint(
        checkpoint_path,
        dynamics,
        history=history,
        autoencoder_checkpoint=autoencoder_checkpoint,
        autoencoder_sha256=autoencoder_sha256,
        train_paths=train_paths,
        validation_paths=validation_paths,
        manifest_path=manifest_path,
        latent_weight=latent_weight,
        pixel_weight=pixel_weight,
    )

    train_metrics = evaluate_dynamics(
        dynamics, autoencoder, training, device, batch_size=batch_size, seed=seed
    )
    validation_metrics = evaluate_dynamics(
        dynamics, autoencoder, validation, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / "one-step-predictions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / "metrics.json"
    save_prediction_grid(dynamics, autoencoder, validation, grid, device)
    _save_training_curve(history, curve)
    _write_json(
        metrics_path,
        {
            "mode": "one-step latent dynamics",
            "latent_dim": dynamics.latent_dim,
            "action_dim": dynamics.action_dim,
            "hidden_dim": dynamics.hidden_dim,
            "hidden_layers": dynamics.hidden_layers,
            "parameters": dynamics.parameter_count,
            "device": str(device),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": autoencoder_sha256,
            "training_episodes": [item.stem for item in train_paths],
            "validation_episodes": [item.stem for item in validation_paths],
            "training": _metrics_payload(train_metrics),
            "validation": _metrics_payload(validation_metrics),
            "history": history,
        },
    )
    return DynamicsTrainingResult(
        checkpoint=checkpoint_path,
        comparison_grid=grid,
        training_curve=curve,
        metrics_path=metrics_path,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        parameter_count=dynamics.parameter_count,
        latent_dim=dynamics.latent_dim,
        device=str(device),
    )


def evaluate_saved_dynamics(
    processed_dir: Path,
    autoencoder_checkpoint: Path,
    dynamics_checkpoint: Path,
    output_dir: Path,
    *,
    split: DatasetSplit = "validation",
    manifest_path: Path | None = None,
    batch_size: int = 128,
    encode_batch_size: int = 128,
    count: int = 6,
    seed: int = 7,
    requested_device: str = "auto",
) -> DynamicsEvaluationResult:
    device = choose_device(requested_device)
    dynamics, dynamics_metadata = load_dynamics_checkpoint(dynamics_checkpoint, device)
    _verify_autoencoder(dynamics_metadata, autoencoder_checkpoint)
    autoencoder, autoencoder_metadata = load_autoencoder_checkpoint(
        autoencoder_checkpoint, device
    )
    if int(autoencoder_metadata["latent_dim"]) != dynamics.latent_dim:
        raise ValueError("autoencoder and dynamics latent dimensions do not match")
    autoencoder.requires_grad_(False)
    if manifest_path is not None:
        selected_paths = DatasetManifest.load(manifest_path).processed_paths(
            processed_dir, split
        )
    else:
        if split == "test":
            raise ValueError("test evaluation requires an explicit manifest")
        training_paths, validation_paths = _processed_splits(processed_dir, None)
        selected_paths = training_paths if split == "training" else validation_paths
    dataset = EncodedDynamicsDataset.from_paths(
        selected_paths, autoencoder, device, encode_batch_size=encode_batch_size
    )
    metrics = evaluate_dynamics(
        dynamics, autoencoder, dataset, device, batch_size=batch_size, seed=seed
    )
    grid = output_dir / f"{split}-one-step-predictions.png"
    curve = output_dir / "training-curve.png"
    metrics_path = output_dir / f"{split}-metrics.json"
    save_prediction_grid(dynamics, autoencoder, dataset, grid, device, count=count)
    _save_training_curve(dynamics_metadata["history"], curve)
    _write_json(
        metrics_path,
        {
            "mode": "saved one-step latent dynamics evaluation",
            "split": split,
            "latent_dim": dynamics.latent_dim,
            "dynamics_checkpoint": str(dynamics_checkpoint),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "metrics": _metrics_payload(metrics),
        },
    )
    return DynamicsEvaluationResult(
        metrics=metrics,
        comparison_grid=grid,
        training_curve=curve,
        example_count=len(dataset),
        latent_dim=dynamics.latent_dim,
        device=str(device),
    )
