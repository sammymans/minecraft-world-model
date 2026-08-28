"""Short-horizon action-conditioned latent video flow matching."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mcwm.dynamics import _file_sha256
from mcwm.manifest import DatasetManifest
from mcwm.model import SpatialAutoencoder, SpatialLatentDynamics, SpatialLatentVideoFlow
from mcwm.spatial_dynamics import (
    SpatialEncodedDynamicsDataset,
    SpatialEncodedSequenceDataset,
    load_spatial_dynamics_checkpoint,
)
from mcwm.spatial_training import image_gradients, load_spatial_autoencoder_checkpoint
from mcwm.training import choose_device, seed_everything

FLOW_ARCHITECTURE = "v1_refinement_rectified_flow_v3"


@dataclass(frozen=True)
class FlowMetrics:
    flow_mse: float
    sampled_latent_mse: float
    sampled_pixel_mse: float
    copy_pixel_mse: float
    edge_energy_ratio: float
    gradient_alignment: float
    shuffled_action_pixel_mse: float
    action_penalty_percent: float
    milliseconds_per_clip: float


@dataclass(frozen=True)
class FlowTrainingResult:
    checkpoint: Path
    metrics_path: Path
    filmstrip: Path
    metrics: FlowMetrics
    parameter_count: int
    training_sequences: int
    device: str


def flow_matching_loss(
    model: SpatialLatentVideoFlow,
    batch: dict[str, torch.Tensor],
    base_future: torch.Tensor,
    *,
    action_dropout: float = 0.15,
    noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress the straight optimal-transport velocity from noise to a future clip."""
    latents = batch["latents"]
    actions = batch["actions"]
    context, future = latents[:, :2], latents[:, 2:]
    target = (future - base_future) / model.latent_std
    if noise is None:
        noise = torch.randn_like(target)
    if flow_time is None:
        flow_time = torch.rand(len(target), device=target.device, dtype=target.dtype)
    if noise.shape != target.shape or flow_time.shape != (len(target),):
        raise ValueError("fixed flow noise or time has the wrong shape")
    mask = (torch.rand(len(target), device=target.device) >= action_dropout).to(target.dtype)
    interpolation = flow_time[:, None, None, None, None]
    noisy_future = (1 - interpolation) * noise + interpolation * target
    target_velocity = target - noise
    predicted_velocity = model(noisy_future, context, base_future, actions, flow_time, mask)
    return nn.functional.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def base_rollout(
    dynamics: SpatialLatentDynamics, context: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Roll the proven deterministic V1 model over one action plan."""
    previous, current = context[:, 0], context[:, 1]
    predictions: list[torch.Tensor] = []
    for step in range(actions.shape[1]):
        predicted = dynamics(previous, current, actions[:, step])
        predictions.append(predicted)
        previous, current = current, predicted
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def evaluate_flow(
    model: SpatialLatentVideoFlow,
    base_dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedSequenceDataset,
    device: torch.device,
    *,
    batch_size: int = 16,
    maximum_examples: int = 256,
    sampling_steps: int = 8,
    guidance_scale: float = 2.0,
    refinement_strength: float = 0.2,
    seed: int = 7,
) -> FlowMetrics:
    """Score held-out clips with common noise for correct and shuffled actions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cpu_generator = torch.Generator().manual_seed(seed)
    totals = np.zeros(9, dtype=np.float64)
    examples = 0
    elapsed = 0.0
    for raw in loader:
        if examples >= maximum_examples:
            break
        keep = min(len(raw["actions"]), maximum_examples - examples)
        batch = {name: value[:keep].to(device) for name, value in raw.items()}
        context, target = batch["latents"][:, :2], batch["latents"][:, 2:]
        actions = batch["actions"]
        base_future = base_rollout(base_dynamics, context, actions)
        normalized_target = (target - base_future) / model.latent_std
        fixed_noise = torch.randn(
            normalized_target.shape, generator=cpu_generator, dtype=normalized_target.dtype
        ).to(device)
        fixed_time = torch.full((keep,), 0.5, device=device)
        midpoint = 0.5 * fixed_noise + 0.5 * normalized_target
        velocity = model(midpoint, context, base_future, actions, fixed_time)
        flow_mse = nn.functional.mse_loss(
            velocity, normalized_target - fixed_noise, reduction="none"
        ).flatten(1).mean(1)

        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        predicted = model.sample(
            context,
            base_future,
            actions,
            steps=sampling_steps,
            guidance_scale=guidance_scale,
            refinement_strength=refinement_strength,
            initial_noise=fixed_noise,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed += time.perf_counter() - started

        shuffled_actions = actions.roll(1, dims=0) if keep > 1 else actions.flip(1)
        shuffled_base = base_rollout(base_dynamics, context, shuffled_actions)
        shuffled = model.sample(
            context,
            shuffled_base,
            shuffled_actions,
            steps=sampling_steps,
            guidance_scale=guidance_scale,
            refinement_strength=refinement_strength,
            initial_noise=fixed_noise,
        )
        current = context[:, 1:2].expand_as(target)
        latent_mse = (predicted - target).square().flatten(1).mean(1)

        flat_predicted = predicted.flatten(0, 1)
        flat_target = target.flatten(0, 1)
        flat_copy = current.flatten(0, 1)
        flat_shuffled = shuffled.flatten(0, 1)
        decoded_predicted = autoencoder.decode(flat_predicted).clamp(0, 1)
        decoded_target = autoencoder.decode(flat_target).clamp(0, 1)
        decoded_copy = autoencoder.decode(flat_copy).clamp(0, 1)
        decoded_shuffled = autoencoder.decode(flat_shuffled).clamp(0, 1)
        pixel_mse = (decoded_predicted - decoded_target).square().flatten(1).mean(1)
        copy_mse = (decoded_copy - decoded_target).square().flatten(1).mean(1)
        shuffled_mse = (decoded_shuffled - decoded_target).square().flatten(1).mean(1)
        predicted_parts = image_gradients(decoded_predicted)
        target_parts = image_gradients(decoded_target)
        predicted_gradients = torch.cat([part.flatten(1) for part in predicted_parts], dim=1)
        target_gradients = torch.cat([part.flatten(1) for part in target_parts], dim=1)
        edge_energy = predicted_gradients.abs().sum(1)
        target_energy = target_gradients.abs().sum(1).clamp_min(1e-8)
        alignment = nn.functional.cosine_similarity(
            predicted_gradients, target_gradients, dim=1
        )
        clip_pixel = pixel_mse.reshape(keep, model.horizon).mean(1)
        clip_copy = copy_mse.reshape(keep, model.horizon).mean(1)
        clip_shuffled = shuffled_mse.reshape(keep, model.horizon).mean(1)
        clip_edge = (edge_energy / target_energy).reshape(keep, model.horizon).mean(1)
        clip_alignment = alignment.reshape(keep, model.horizon).mean(1)
        totals += np.array(
            [
                flow_mse.sum().item(),
                latent_mse.sum().item(),
                clip_pixel.sum().item(),
                clip_copy.sum().item(),
                clip_edge.sum().item(),
                clip_alignment.sum().item(),
                clip_shuffled.sum().item(),
                0.0,
                0.0,
            ]
        )
        examples += keep
    if not examples:
        raise ValueError("flow evaluation dataset is empty")
    means = totals / examples
    action_penalty = 100 * (means[6] - means[2]) / max(means[2], 1e-12)
    return FlowMetrics(
        flow_mse=float(means[0]),
        sampled_latent_mse=float(means[1]),
        sampled_pixel_mse=float(means[2]),
        copy_pixel_mse=float(means[3]),
        edge_energy_ratio=float(means[4]),
        gradient_alignment=float(means[5]),
        shuffled_action_pixel_mse=float(means[6]),
        action_penalty_percent=float(action_penalty),
        milliseconds_per_clip=1000 * elapsed / examples,
    )


def _save_checkpoint(
    path: Path,
    model: SpatialLatentVideoFlow,
    *,
    autoencoder_checkpoint: Path,
    base_dynamics_checkpoint: Path,
    manifest_path: Path,
    sampling_steps: int,
    guidance_scale: float,
    refinement_strength: float,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "spatial_latent_video_flow",
            "architecture": FLOW_ARCHITECTURE,
            "latent_channels": model.latent_channels,
            "action_dim": model.action_dim,
            "horizon": model.horizon,
            "hidden_channels": model.hidden_channels,
            "condition_dim": model.condition_dim,
            "sampling_steps": sampling_steps,
            "guidance_scale": guidance_scale,
            "refinement_strength": refinement_strength,
            "model_state": model.state_dict(),
            "autoencoder_checkpoint": str(autoencoder_checkpoint),
            "autoencoder_sha256": _file_sha256(autoencoder_checkpoint),
            "base_dynamics_checkpoint": str(base_dynamics_checkpoint),
            "base_dynamics_sha256": _file_sha256(base_dynamics_checkpoint),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
            "history": history,
        },
        path,
    )


def load_spatial_flow_checkpoint(
    path: Path, device: torch.device
) -> tuple[SpatialLatentVideoFlow, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != "spatial_latent_video_flow":
        raise ValueError("checkpoint is not spatial latent video flow")
    if checkpoint.get("architecture") != FLOW_ARCHITECTURE:
        raise ValueError("flow checkpoint uses an incompatible architecture")
    state = checkpoint["model_state"]
    model = SpatialLatentVideoFlow(
        latent_channels=int(checkpoint["latent_channels"]),
        action_dim=int(checkpoint["action_dim"]),
        horizon=int(checkpoint["horizon"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        condition_dim=int(checkpoint["condition_dim"]),
        action_mean=state["action_mean"],
        action_std=state["action_std"],
        latent_mean=state["latent_mean"].flatten(),
        latent_std=state["latent_std"].flatten(),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _save_filmstrip(
    model: SpatialLatentVideoFlow,
    base_dynamics: SpatialLatentDynamics,
    autoencoder: SpatialAutoencoder,
    dataset: SpatialEncodedSequenceDataset,
    path: Path,
    device: torch.device,
    *,
    sampling_steps: int,
    guidance_scale: float,
    refinement_strength: float,
    seed: int,
) -> None:
    raw = dataset[0]
    latents = raw["latents"].unsqueeze(0).to(device)
    actions = raw["actions"].unsqueeze(0).to(device)
    base_future = base_rollout(base_dynamics, latents[:, :2], actions)
    noise = torch.randn(
        (1, model.horizon, model.latent_channels, *latents.shape[-2:]),
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    with torch.no_grad():
        predicted = model.sample(
            latents[:, :2],
            base_future,
            actions,
            steps=sampling_steps,
            guidance_scale=guidance_scale,
            refinement_strength=refinement_strength,
            initial_noise=noise,
        )
        decoded = autoencoder.decode(predicted.flatten(0, 1)).clamp(0, 1)
        decoded_base = autoencoder.decode(base_future.flatten(0, 1)).clamp(0, 1)
        target = autoencoder.decode(latents[:, 2:].flatten(0, 1)).clamp(0, 1)
    predicted_frames = decoded.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    base_frames = decoded_base.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    target_frames = target.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    tile = 160
    label = 30
    canvas = np.full((3 * (tile + label), model.horizon * tile, 3), 20, np.uint8)
    for index in range(model.horizon):
        for row, frames in enumerate((base_frames, predicted_frames, target_frames)):
            frame = cv2.resize(frames[index], (tile, tile), interpolation=cv2.INTER_NEAREST)
            y = row * (tile + label) + label
            canvas[y : y + tile, index * tile : (index + 1) * tile] = cv2.cvtColor(
                frame, cv2.COLOR_RGB2BGR
            )
        cv2.putText(canvas, f"t+{index + 1}", (index * tile + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "V1 base", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "flow", (5, tile + label + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 190, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "target", (5, 2 * (tile + label) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 230, 230), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"could not write flow filmstrip: {path}")


def train_spatial_flow(
    processed_dir: Path,
    manifest_path: Path,
    autoencoder_checkpoint: Path,
    output_dir: Path,
    *,
    base_dynamics_checkpoint: Path = Path(
        "artifacts/spatial-dynamics-v4-multistep/best.pt"
    ),
    epochs: int = 5,
    batch_size: int = 16,
    encode_batch_size: int = 128,
    maximum_sequences: int = 4_000,
    maximum_validation_sequences: int = 256,
    horizon: int = 8,
    hidden_channels: int = 128,
    condition_dim: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    action_dropout: float = 0.15,
    sampling_steps: int = 8,
    guidance_scale: float = 2.0,
    refinement_strength: float = 0.2,
    initial_checkpoint: Path | None = None,
    seed: int = 7,
    requested_device: str = "auto",
) -> FlowTrainingResult:
    if min(epochs, batch_size, encode_batch_size, maximum_sequences, horizon) < 1:
        raise ValueError("training sizes must be positive")
    if not 0 <= action_dropout < 1:
        raise ValueError("action_dropout must be in [0, 1)")
    seed_everything(seed)
    device = choose_device(requested_device)
    autoencoder, _ = load_spatial_autoencoder_checkpoint(autoencoder_checkpoint, device)
    autoencoder.requires_grad_(False)
    base_dynamics, base_metadata = load_spatial_dynamics_checkpoint(
        base_dynamics_checkpoint, device
    )
    if base_metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
        raise ValueError("base dynamics uses a different autoencoder")
    base_dynamics.requires_grad_(False)
    train_paths, validation_paths = DatasetManifest.load(manifest_path).processed_splits(
        processed_dir
    )
    print(f"encoding up to {maximum_sequences:,} training clips...")
    encoded_train = SpatialEncodedDynamicsDataset.from_paths(
        train_paths, autoencoder, device, maximum_transitions=maximum_sequences,
        count_horizon=horizon, encode_batch_size=encode_batch_size, seed=seed
    )
    print("encoding held-out validation clips...")
    encoded_validation = SpatialEncodedDynamicsDataset.from_paths(
        validation_paths, autoencoder, device,
        maximum_transitions=maximum_validation_sequences, count_horizon=horizon,
        encode_batch_size=encode_batch_size, seed=seed
    )
    training = SpatialEncodedSequenceDataset(
        encoded_train, horizon=horizon, maximum_sequences=maximum_sequences, seed=seed
    )
    validation = SpatialEncodedSequenceDataset(
        encoded_validation, horizon=horizon,
        maximum_sequences=maximum_validation_sequences, seed=seed
    )
    if initial_checkpoint is None:
        statistics = encoded_train.normalization_statistics()
        model = SpatialLatentVideoFlow(
            latent_channels=training.latent_shape[0], action_dim=9, horizon=horizon,
            hidden_channels=hidden_channels, condition_dim=condition_dim,
            action_mean=statistics[0], action_std=statistics[1],
            latent_mean=statistics[2], latent_std=statistics[3]
        ).to(device)
    else:
        model, initial_metadata = load_spatial_flow_checkpoint(initial_checkpoint, device)
        if initial_metadata["autoencoder_sha256"] != _file_sha256(autoencoder_checkpoint):
            raise ValueError("initial flow checkpoint uses a different autoencoder")
        if initial_metadata["base_dynamics_sha256"] != _file_sha256(
            base_dynamics_checkpoint
        ):
            raise ValueError("initial flow checkpoint uses different base dynamics")
        if model.horizon != horizon or model.latent_channels != training.latent_shape[0]:
            raise ValueError("initial flow checkpoint uses an incompatible horizon or latent shape")
        print(f"continuing from {initial_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(
        training, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0
    )
    history: list[dict[str, float]] = []
    best_flow = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for raw in loader:
            batch = {name: value.to(device) for name, value in raw.items()}
            with torch.no_grad():
                base_future = base_rollout(
                    base_dynamics, batch["latents"][:, :2], batch["actions"]
                )
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(
                model, batch, base_future, action_dropout=action_dropout
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch["actions"])
            examples += len(batch["actions"])
        metrics = evaluate_flow(
            model, base_dynamics, autoencoder, validation, device, batch_size=batch_size,
            maximum_examples=maximum_validation_sequences,
            sampling_steps=sampling_steps, guidance_scale=guidance_scale,
            refinement_strength=refinement_strength, seed=seed
        )
        row = {"train_flow_mse": total / examples, **asdict(metrics)}
        history.append(row)
        print(
            f"epoch {epoch + 1}/{epochs}: train={row['train_flow_mse']:.5f}  "
            f"val={metrics.flow_mse:.5f}  pixel={metrics.sampled_pixel_mse:.5f}  "
            f"edge={metrics.edge_energy_ratio:.3f}  action={metrics.action_penalty_percent:+.1f}%"
        )
        if metrics.flow_mse < best_flow:
            best_flow = metrics.flow_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("flow training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output_dir / "best.pt"
    _save_checkpoint(
        checkpoint, model, autoencoder_checkpoint=autoencoder_checkpoint,
        base_dynamics_checkpoint=base_dynamics_checkpoint,
        manifest_path=manifest_path, sampling_steps=sampling_steps,
        guidance_scale=guidance_scale, refinement_strength=refinement_strength,
        history=history
    )
    metrics = evaluate_flow(
        model, base_dynamics, autoencoder, validation, device, batch_size=batch_size,
        maximum_examples=maximum_validation_sequences,
        sampling_steps=sampling_steps, guidance_scale=guidance_scale,
        refinement_strength=refinement_strength, seed=seed
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2) + "\n")
    filmstrip = output_dir / "validation-filmstrip.png"
    _save_filmstrip(
        model, base_dynamics, autoencoder, validation, filmstrip, device,
        sampling_steps=sampling_steps, guidance_scale=guidance_scale,
        refinement_strength=refinement_strength, seed=seed
    )
    return FlowTrainingResult(
        checkpoint=checkpoint, metrics_path=metrics_path, filmstrip=filmstrip,
        metrics=metrics, parameter_count=model.parameter_count,
        training_sequences=len(training), device=str(device)
    )


class FlowStepAdapter(nn.Module):
    """Expose the first frame of a joint clip as a live one-step dynamics model."""

    def __init__(
        self,
        flow: SpatialLatentVideoFlow,
        base_dynamics: SpatialLatentDynamics,
        *,
        steps: int,
        guidance_scale: float,
        refinement_strength: float = 0.2,
        seed: int = 7,
    ):
        super().__init__()
        self.flow = flow
        self.base_dynamics = base_dynamics
        self.latent_channels = flow.latent_channels
        self.action_dim = flow.action_dim
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.refinement_strength = refinement_strength
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.cached_action: torch.Tensor | None = None
        self.cached_future: torch.Tensor | None = None
        self.cache_index = 0

    def reset_sampling(self) -> None:
        self.generator = torch.Generator().manual_seed(self.seed)
        self.cached_action = None
        self.cached_future = None
        self.cache_index = 0

    def forward(
        self, previous_latent: torch.Tensor, current_latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if (
            self.cached_future is not None
            and self.cached_action is not None
            and self.cache_index < self.flow.horizon
            and torch.equal(action, self.cached_action)
        ):
            predicted = self.cached_future[:, self.cache_index]
            self.cache_index += 1
            return predicted
        context = torch.stack((previous_latent, current_latent), dim=1)
        plan = action[:, None].expand(-1, self.flow.horizon, -1)
        base_future = base_rollout(self.base_dynamics, context, plan)
        self.cached_future = self.flow.sample(
            context, base_future, plan, steps=self.steps, guidance_scale=self.guidance_scale,
            refinement_strength=self.refinement_strength,
            generator=self.generator
        ).detach()
        self.cached_action = action.detach().clone()
        self.cache_index = 1
        return self.cached_future[:, 0]
