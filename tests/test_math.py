import numpy as np

from mcwm.math import integrate_delta, rotate_xz, state_delta, wrap_degrees


def test_wrap_degrees_handles_boundary() -> None:
    values = np.asarray([179.0, 181.0, -181.0, 540.0], dtype=np.float32)
    np.testing.assert_allclose(
        wrap_degrees(values),
        np.asarray([179.0, -179.0, 179.0, -180.0], dtype=np.float32),
    )


def test_state_delta_and_integration_round_trip() -> None:
    state = np.asarray([1, 2, 3, 4, 5, 6, 179, 10], dtype=np.float32)
    next_state = np.asarray([2, 4, 6, 3, 2, 1, -179, 12], dtype=np.float32)
    delta = state_delta(state, next_state)
    assert delta[6] == 2.0
    np.testing.assert_allclose(integrate_delta(state, delta), next_state)


def test_rotate_xz_quarter_turn() -> None:
    vector = np.asarray([[1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(
        rotate_xz(vector, 90.0),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )

