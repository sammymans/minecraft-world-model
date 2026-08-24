import numpy as np
import pytest

from mcwm.data.features import Standardizer, flatten_episodes, model_inputs
from mcwm.data.schema import ACTION_NAMES, MODEL_INPUT_NAMES
from mcwm.data.synthetic import generate_synthetic_episode


def test_synthetic_episode_contract_and_features() -> None:
    episode = generate_synthetic_episode(steps=32, seed=1)
    assert episode.states.shape == (33, 8)
    assert episode.actions.shape == (32, len(ACTION_NAMES))
    flattened = flatten_episodes([episode])
    features = model_inputs(flattened.states, flattened.actions, flattened.dts)
    assert features.shape == (32, len(MODEL_INPUT_NAMES))
    assert flattened.targets.shape == (32, 8)


def test_standardizer_round_trip() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    standardizer = Standardizer.fit(values)
    restored = standardizer.inverse(standardizer.transform(values))
    np.testing.assert_allclose(restored, values)
    assert standardizer.std[1] == 1.0


def test_episode_rejects_wrong_action_count() -> None:
    episode = generate_synthetic_episode(steps=4)
    with pytest.raises(ValueError, match="one row per state transition"):
        type(episode)(
            states=episode.states,
            actions=episode.actions[:-1],
            dts=episode.dts[:-1],
            source="bad",
        )

