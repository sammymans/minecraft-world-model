from pathlib import Path

import numpy as np

from mcwm.data.schema import ACTION_NAMES, Episode
from mcwm.evaluation import (
    rollout_episode,
    rollout_position_errors,
    save_rollout_plot,
)


def _linear_episode(steps: int = 8) -> Episode:
    states = np.zeros((steps + 1, 8), dtype=np.float32)
    states[:, 0] = np.arange(steps + 1)
    states[:, 3] = 1.0
    return Episode(
        states=states,
        actions=np.zeros((steps, len(ACTION_NAMES)), dtype=np.float32),
        dts=np.ones(steps, dtype=np.float32),
        source="linear",
    )


def _one_block_predictor(
    states: np.ndarray, actions: np.ndarray, dts: np.ndarray
) -> np.ndarray:
    prediction = np.zeros((len(states), 8), dtype=np.float32)
    prediction[:, 0] = 1.0
    return prediction


def test_recursive_rollout_and_horizon_metrics() -> None:
    episode = _linear_episode()
    actual, predicted = rollout_episode(
        episode, _one_block_predictor, horizon=8
    )
    np.testing.assert_allclose(predicted, actual)
    errors = rollout_position_errors(
        _one_block_predictor, [episode], horizons=(1, 2, 4)
    )
    assert errors == {"1": 0.0, "2": 0.0, "4": 0.0}


def test_rollout_plot_is_written(tmp_path: Path) -> None:
    class Predictor:
        predict = staticmethod(_one_block_predictor)

    output = tmp_path / "rollout.png"
    save_rollout_plot(Predictor(), _linear_episode(), output, horizon=8)  # type: ignore[arg-type]
    assert output.stat().st_size > 0
