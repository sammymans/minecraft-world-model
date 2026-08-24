"""Prediction metrics, action ablations, and rollout visualization."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcwm-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mcwm.data.features import flatten_episodes
from mcwm.data.schema import TARGET_NAMES, Episode
from mcwm.math import integrate_delta
from mcwm.models.baselines import constant_velocity, persistence
from mcwm.training import TrainedDynamics


def _prediction_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    error = prediction - target
    group_columns = {
        "position_blocks": slice(0, 3),
        "velocity_blocks_per_second": slice(3, 6),
        "orientation_degrees": slice(6, 8),
    }
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rmse_by_group": {
            name: float(np.sqrt(np.mean(error[:, columns] ** 2)))
            for name, columns in group_columns.items()
        },
        "mae_by_target": {
            name: float(value)
            for name, value in zip(
                TARGET_NAMES, np.mean(np.abs(error), axis=0), strict=True
            )
        },
    }


def evaluate_one_step(
    trained: TrainedDynamics,
    episodes: list[Episode],
    *,
    seed: int = 0,
) -> dict[str, object]:
    data = flatten_episodes(episodes)
    learned = trained.predict(data.states, data.actions, data.dts)
    zero = persistence(data.states)
    kinematic = constant_velocity(data.states, data.actions, data.dts)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(data.actions))
    shuffled_actions = data.actions[permutation]
    shuffled = trained.predict(data.states, shuffled_actions, data.dts)
    shuffled_movement_actions = data.actions.copy()
    shuffled_movement_actions[:, :7] = data.actions[permutation, :7]
    shuffled_movement = trained.predict(
        data.states, shuffled_movement_actions, data.dts
    )
    learned_metrics = _prediction_metrics(data.targets, learned)
    shuffled_metrics = _prediction_metrics(data.targets, shuffled)
    shuffled_movement_metrics = _prediction_metrics(
        data.targets, shuffled_movement
    )
    learned_groups = learned_metrics["rmse_by_group"]
    shuffled_movement_groups = shuffled_movement_metrics["rmse_by_group"]
    return {
        "transitions": len(data),
        "learned": learned_metrics,
        "persistence": _prediction_metrics(data.targets, zero),
        "constant_velocity": _prediction_metrics(data.targets, kinematic),
        "shuffled_actions": shuffled_metrics,
        "shuffled_movement_actions": shuffled_movement_metrics,
        "action_conditioning_gain": float(
            shuffled_metrics["rmse"] - learned_metrics["rmse"]
        ),
        "movement_action_conditioning_gain_by_group": {
            name: float(shuffled_movement_groups[name] - learned_groups[name])
            for name in ("position_blocks", "velocity_blocks_per_second")
        },
    }


def rollout_episode(
    episode: Episode,
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    start: int = 0,
    horizon: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    stop = min(start + horizon, episode.transitions)
    predicted = [episode.states[start].copy()]
    state = episode.states[start].copy()
    for index in range(start, stop):
        delta = predictor(
            state[None, :],
            episode.actions[index : index + 1],
            episode.dts[index : index + 1],
        )[0]
        state = integrate_delta(state, delta)
        predicted.append(state.copy())
    actual = episode.states[start : stop + 1]
    return actual, np.stack(predicted)


def rollout_position_errors(
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    episodes: list[Episode],
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    errors: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    maximum = max(horizons)
    for episode in episodes:
        if episode.transitions < maximum:
            continue
        for start in range(0, episode.transitions - maximum + 1, maximum):
            actual, predicted = rollout_episode(
                episode, predictor, start=start, horizon=maximum
            )
            for horizon in horizons:
                errors[horizon].append(
                    float(np.linalg.norm(predicted[horizon, :3] - actual[horizon, :3]))
                )
    return {
        str(horizon): float(np.mean(values)) if values else float("nan")
        for horizon, values in errors.items()
    }


def save_rollout_plot(
    trained: TrainedDynamics,
    episode: Episode,
    path: str | Path,
    *,
    horizon: int = 200,
) -> None:
    actual, predicted = rollout_episode(episode, trained.predict, horizon=horizon)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(actual[:, 0], actual[:, 2], label="real Minecraft", linewidth=2)
    axis.plot(predicted[:, 0], predicted[:, 2], label="world model", linewidth=2)
    axis.scatter(actual[0, 0], actual[0, 2], label="start", marker="o")
    axis.set_xlabel("world x")
    axis.set_ylabel("world z")
    axis.set_title("Open-loop V0 trajectory")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
