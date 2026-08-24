"""Supervised V0 training and checkpoint handling."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mcwm.data.features import Standardizer, flatten_episodes, model_inputs
from mcwm.data.schema import LEARNED_TARGET_NAMES, MODEL_INPUT_NAMES, Episode
from mcwm.models.baselines import constant_velocity
from mcwm.models.dynamics import DynamicsMLP


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    hidden_layers: int = 2
    seed: int = 7
    device: str = "cpu"


@dataclass
class TrainedDynamics:
    model: DynamicsMLP
    input_standardizer: Standardizer
    target_standardizer: Standardizer
    config: TrainConfig
    history: list[dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        dts: np.ndarray,
    ) -> np.ndarray:
        raw_inputs = model_inputs(states, actions, dts)
        baseline = constant_velocity(states, actions, dts)
        inputs = torch.from_numpy(self.input_standardizer.transform(raw_inputs))
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            normalized = self.model(inputs.to(device)).cpu().numpy()
        learned_residual = self.target_standardizer.inverse(normalized)
        residual = np.zeros_like(baseline)
        residual[:, : len(LEARNED_TARGET_NAMES)] = learned_residual
        return baseline + residual


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_dynamics(
    train_episodes: list[Episode],
    validation_episodes: list[Episode],
    config: TrainConfig,
) -> TrainedDynamics:
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    train = flatten_episodes(train_episodes)
    validation = flatten_episodes(validation_episodes)
    train_inputs = model_inputs(train.states, train.actions, train.dts)
    validation_inputs = model_inputs(
        validation.states, validation.actions, validation.dts
    )
    input_standardizer = Standardizer.fit(train_inputs)
    train_baseline = constant_velocity(train.states, train.actions, train.dts)
    validation_baseline = constant_velocity(
        validation.states, validation.actions, validation.dts
    )
    learned_columns = slice(0, len(LEARNED_TARGET_NAMES))
    train_residuals = (train.targets - train_baseline)[:, learned_columns]
    validation_residuals = (validation.targets - validation_baseline)[
        :, learned_columns
    ]
    target_standardizer = Standardizer.fit(train_residuals)

    x_train = torch.from_numpy(input_standardizer.transform(train_inputs))
    y_train = torch.from_numpy(target_standardizer.transform(train_residuals))
    x_validation = torch.from_numpy(input_standardizer.transform(validation_inputs))
    y_validation = torch.from_numpy(
        target_standardizer.transform(validation_residuals)
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = DynamicsMLP(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_state: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(inputs)
            total_examples += len(inputs)

        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                loss_function(
                    model(x_validation.to(device)), y_validation.to(device)
                ).cpu()
            )
        train_loss = total_loss / total_examples
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"epoch={epoch:03d} train={train_loss:.6f} "
                f"validation={validation_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainedDynamics(
        model=model,
        input_standardizer=input_standardizer,
        target_standardizer=target_standardizer,
        config=config,
        history=history,
    )


def save_checkpoint(
    trained: TrainedDynamics,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": trained.model.state_dict(),
            "input_standardizer": trained.input_standardizer.to_dict(),
            "target_standardizer": trained.target_standardizer.to_dict(),
            "config": asdict(trained.config),
            "history": trained.history,
            "input_names": MODEL_INPUT_NAMES,
            "target_names": LEARNED_TARGET_NAMES,
            "prediction_mode": "constant_velocity_residual",
            "metadata": trained.metadata if metadata is None else metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> TrainedDynamics:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = TrainConfig(**checkpoint["config"])
    model = DynamicsMLP(
        input_dim=len(checkpoint["input_names"]),
        output_dim=len(checkpoint["target_names"]),
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return TrainedDynamics(
        model=model,
        input_standardizer=Standardizer.from_dict(
            checkpoint["input_standardizer"]
        ),
        target_standardizer=Standardizer.from_dict(
            checkpoint["target_standardizer"]
        ),
        config=config,
        history=checkpoint["history"],
        metadata=checkpoint.get("metadata", {}),
    )
