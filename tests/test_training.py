from pathlib import Path

from mcwm.data.synthetic import generate_synthetic_episode
from mcwm.evaluation import evaluate_one_step
from mcwm.training import (
    TrainConfig,
    fit_dynamics,
    load_checkpoint,
    save_checkpoint,
)


def test_model_learns_synthetic_dynamics_and_round_trips_checkpoint(
    tmp_path: Path,
) -> None:
    episodes = [
        generate_synthetic_episode(steps=160, seed=index) for index in range(5)
    ]
    trained = fit_dynamics(
        episodes[:4],
        episodes[4:],
        TrainConfig(
            epochs=15,
            batch_size=128,
            learning_rate=1e-3,
            hidden_dim=64,
            seed=3,
        ),
    )
    metrics = evaluate_one_step(trained, episodes[4:])
    assert metrics["learned"]["rmse"] < metrics["persistence"]["rmse"]
    assert "position_blocks" in metrics["movement_action_conditioning_gain_by_group"]
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(trained, checkpoint, metadata={"action_repeat": 4})
    restored = load_checkpoint(checkpoint)
    assert restored.metadata == {"action_repeat": 4}
    restored_metrics = evaluate_one_step(restored, episodes[4:])
    assert restored_metrics["learned"]["rmse"] == metrics["learned"]["rmse"]
