"""Simple predictions that the learned model must beat."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def persistence(states: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.zeros((len(states), 8), dtype=np.float32)


def constant_velocity(
    states: NDArray[np.float32],
    actions: NDArray[np.float32],
    dts: NDArray[np.float32],
) -> NDArray[np.float32]:
    prediction = np.zeros((len(states), 8), dtype=np.float32)
    prediction[:, :3] = states[:, 3:6] * dts[:, None]
    prediction[:, 6] = actions[:, 7]
    prediction[:, 7] = actions[:, 8]
    return prediction

