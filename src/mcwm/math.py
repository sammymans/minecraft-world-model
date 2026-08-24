"""Coordinate and state-integration utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wrap_degrees(angle: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Wrap angles to the half-open interval [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


def state_delta(
    state: NDArray[np.float32], next_state: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return next_state - state, using circular subtraction for yaw."""

    delta = np.asarray(next_state - state, dtype=np.float32)
    delta[..., 6] = wrap_degrees(delta[..., 6])
    return delta


def integrate_delta(
    state: NDArray[np.float32], delta: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Apply a predicted state delta and wrap the resulting yaw."""

    result = np.asarray(state + delta, dtype=np.float32)
    result[..., 6] = wrap_degrees(result[..., 6])
    result[..., 7] = np.clip(result[..., 7], -90.0, 90.0)
    return result


def rotate_xz(
    vectors: NDArray[np.float32], yaw_degrees: float | NDArray[np.float32]
) -> NDArray[np.float32]:
    """Rotate x/z vectors counterclockwise by yaw_degrees."""

    vectors = np.asarray(vectors, dtype=np.float32)
    yaw = np.deg2rad(yaw_degrees)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x = vectors[..., 0]
    z = vectors[..., 1]
    return np.stack(
        (cos_yaw * x - sin_yaw * z, sin_yaw * x + cos_yaw * z), axis=-1
    ).astype(np.float32)

