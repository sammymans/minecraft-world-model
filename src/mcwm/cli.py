"""Command line interface for the V0 learning workflow."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from mcwm.data.schema import ACTION_NAMES
from mcwm.data.synthetic import generate_synthetic_episode
from mcwm.data.vpt import alignment_correlations, load_vpt_file, load_vpt_files
from mcwm.download import DEFAULT_INDEX_URL, download_vpt_actions
from mcwm.training import TrainConfig, fit_dynamics, save_checkpoint


def _json_dump(value: object, path: Path | None = None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True)
    print(rendered)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def _jsonl_files(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no JSONL files found in {directory}")
    return files


def command_download(args: argparse.Namespace) -> None:
    paths = download_vpt_actions(
        args.output,
        index_source=args.index,
        limit=args.limit,
        start=args.start,
    )
    print(f"ready: {len(paths)} recordings in {args.output}")


def command_inspect(args: argparse.Namespace) -> None:
    episodes = load_vpt_file(args.path, action_repeat=args.action_repeat)
    report = {
        "file": str(args.path),
        "segments": len(episodes),
        "transitions": sum(episode.transitions for episode in episodes),
        "duration_seconds": sum(float(episode.dts.sum()) for episode in episodes),
        "camera_alignment_correlation": alignment_correlations(episodes),
    }
    _json_dump(report)


def _recording_group(path: Path) -> str:
    match = re.search(r"-([0-9a-f]+)-\d{8}-\d{6}$", path.stem)
    return match.group(1) if match else path.stem


def _split_files(
    files: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(_recording_group(path), []).append(path)
    if len(groups) < 3:
        raise SystemExit(
            "training requires recordings from at least 3 players/sessions"
        )
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    validation_count = max(1, round(len(group_names) * 0.15))
    test_count = max(1, round(len(group_names) * 0.15))
    train_count = len(group_names) - validation_count - test_count

    def files_for(names: list[str]) -> list[Path]:
        return sorted(path for name in names for path in groups[name])

    return (
        files_for(group_names[:train_count]),
        files_for(group_names[train_count : train_count + validation_count]),
        files_for(group_names[train_count + validation_count :]),
    )


def command_audit(args: argparse.Namespace) -> None:
    files = _jsonl_files(args.data)
    non_utf8_recordings: list[str] = []
    for path in files:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            non_utf8_recordings.append(path.name)
    episodes = load_vpt_files(files, action_repeat=args.action_repeat)
    actions = np.concatenate([episode.actions for episode in episodes])
    dts = np.concatenate([episode.dts for episode in episodes])
    train, validation, test = _split_files(files, args.seed)
    report = {
        "recordings": len(files),
        "recordings_with_replaced_legacy_text_bytes": non_utf8_recordings,
        "player_or_session_groups": len({_recording_group(path) for path in files}),
        "segments": len(episodes),
        "transitions": int(sum(episode.transitions for episode in episodes)),
        "duration_seconds": float(dts.sum()),
        "action_repeat": args.action_repeat,
        "dt_seconds": {
            "median": float(np.median(dts)),
            "p05": float(np.quantile(dts, 0.05)),
            "p95": float(np.quantile(dts, 0.95)),
        },
        "held_action_fraction": {
            name: float(actions[:, index].mean())
            for index, name in enumerate(ACTION_NAMES[:7])
        },
        "mean_absolute_camera_degrees_per_step": {
            "yaw": float(np.abs(actions[:, 7]).mean()),
            "pitch": float(np.abs(actions[:, 8]).mean()),
        },
        "camera_alignment_correlation": alignment_correlations(episodes),
        "recording_split": {
            "train": [path.name for path in train],
            "validation": [path.name for path in validation],
            "test": [path.name for path in test],
        },
    }
    _json_dump(report, Path(args.output) if args.output else None)


def _train_and_report(
    train_episodes,
    validation_episodes,
    test_episodes,
    args: argparse.Namespace,
    checkpoint_metadata: dict[str, object] | None = None,
) -> None:
    from mcwm.evaluation import (
        evaluate_one_step,
        rollout_position_errors,
        save_rollout_plot,
    )
    from mcwm.models.baselines import constant_velocity

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
    )
    trained = fit_dynamics(train_episodes, validation_episodes, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        trained, output / "model.pt", metadata=checkpoint_metadata
    )
    metrics = {
        "split_transitions": {
            "train": sum(episode.transitions for episode in train_episodes),
            "validation": sum(episode.transitions for episode in validation_episodes),
            "test": sum(episode.transitions for episode in test_episodes),
        },
        "one_step": evaluate_one_step(trained, test_episodes, seed=args.seed),
        "rollout_position_error_blocks": {
            "learned": rollout_position_errors(trained.predict, test_episodes),
            "constant_velocity": rollout_position_errors(
                constant_velocity, test_episodes
            ),
        },
        "best_validation_loss": min(
            row["validation_loss"] for row in trained.history
        ),
        "checkpoint_metadata": checkpoint_metadata or {},
    }
    _json_dump(metrics, output / "metrics.json")
    longest = max(test_episodes, key=lambda episode: episode.transitions)
    save_rollout_plot(trained, longest, output / "rollout.png")
    (output / "history.json").write_text(
        json.dumps(trained.history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"checkpoint: {output / 'model.pt'}")
    print(f"rollout plot: {output / 'rollout.png'}")


def command_train(args: argparse.Namespace) -> None:
    files = _jsonl_files(args.data)
    train_files, validation_files, test_files = _split_files(files, args.seed)
    print(
        f"recordings: train={len(train_files)} validation={len(validation_files)} "
        f"test={len(test_files)}"
    )
    _train_and_report(
        load_vpt_files(train_files, action_repeat=args.action_repeat),
        load_vpt_files(validation_files, action_repeat=args.action_repeat),
        load_vpt_files(test_files, action_repeat=args.action_repeat),
        args,
        checkpoint_metadata={
            "dataset": "openai_vpt",
            "action_repeat": args.action_repeat,
            "train_recordings": [path.name for path in train_files],
            "validation_recordings": [path.name for path in validation_files],
            "test_recordings": [path.name for path in test_files],
        },
    )


def command_evaluate(args: argparse.Namespace) -> None:
    from mcwm.evaluation import (
        evaluate_one_step,
        rollout_position_errors,
        save_rollout_plot,
    )
    from mcwm.models.baselines import constant_velocity
    from mcwm.training import load_checkpoint, resolve_device

    trained = load_checkpoint(args.model, device=resolve_device(args.device))
    recorded_repeat = trained.metadata.get("action_repeat")
    action_repeat = args.action_repeat
    if action_repeat is None:
        action_repeat = int(recorded_repeat) if recorded_repeat is not None else 4
    elif recorded_repeat is not None and action_repeat != int(recorded_repeat):
        raise SystemExit(
            f"checkpoint was trained with action_repeat={recorded_repeat}; "
            f"received {action_repeat}"
        )
    recorded_test_files = trained.metadata.get("test_recordings")
    if recorded_test_files:
        test_files = [Path(args.data) / str(name) for name in recorded_test_files]
        missing = [path for path in test_files if not path.exists()]
        if missing:
            raise SystemExit(f"checkpoint test recording is missing: {missing[0]}")
    else:
        files = _jsonl_files(args.data)
        _, _, test_files = _split_files(files, trained.config.seed)
    test_episodes = load_vpt_files(
        test_files, action_repeat=action_repeat
    )
    report = {
        "model": str(args.model),
        "seed": trained.config.seed,
        "action_repeat": action_repeat,
        "test_recordings": [path.name for path in test_files],
        "test_transitions": sum(
            episode.transitions for episode in test_episodes
        ),
        "one_step": evaluate_one_step(
            trained, test_episodes, seed=trained.config.seed
        ),
        "rollout_position_error_blocks": {
            "learned": rollout_position_errors(
                trained.predict, test_episodes
            ),
            "constant_velocity": rollout_position_errors(
                constant_velocity, test_episodes
            ),
        },
    }
    output = Path(args.output)
    _json_dump(report, output / "evaluation.json")
    longest = max(test_episodes, key=lambda episode: episode.transitions)
    save_rollout_plot(trained, longest, output / "rollout.png")
    print(f"evaluation: {output / 'evaluation.json'}")
    print(f"rollout plot: {output / 'rollout.png'}")


def command_synthetic(args: argparse.Namespace) -> None:
    episodes = [
        generate_synthetic_episode(steps=args.steps, seed=args.seed + index)
        for index in range(6)
    ]
    _train_and_report(episodes[:4], episodes[4:5], episodes[5:], args)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="artifacts/v0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcwm", description="Small Minecraft world-model learning project"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-vpt", help="download official VPT action/state recordings"
    )
    download.add_argument("--output", default="data/raw/vpt/episodes")
    download.add_argument("--index", default=DEFAULT_INDEX_URL)
    download.add_argument("--limit", type=int, default=24)
    download.add_argument("--start", type=int, default=0)
    download.set_defaults(function=command_download)

    inspect = subparsers.add_parser(
        "inspect-vpt", help="inspect transitions and camera/action alignment"
    )
    inspect.add_argument("path")
    inspect.add_argument("--action-repeat", type=int, default=1)
    inspect.set_defaults(function=command_inspect)

    audit = subparsers.add_parser(
        "audit-vpt", help="audit a public VPT dataset before training"
    )
    audit.add_argument("--data", default="data/raw/vpt/episodes")
    audit.add_argument("--action-repeat", type=int, default=4)
    audit.add_argument("--seed", type=int, default=7)
    audit.add_argument("--output", default="artifacts/v0/data-audit.json")
    audit.set_defaults(function=command_audit)

    train = subparsers.add_parser(
        "train-v0", help="train V0 on downloaded VPT recordings"
    )
    train.add_argument("--data", default="data/raw/vpt/episodes")
    train.add_argument("--action-repeat", type=int, default=4)
    _add_training_arguments(train)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser(
        "evaluate-v0", help="evaluate a saved V0 checkpoint on held-out recordings"
    )
    evaluate.add_argument("--model", default="artifacts/v0/model.pt")
    evaluate.add_argument("--data", default="data/raw/vpt/episodes")
    evaluate.add_argument("--action-repeat", type=int)
    evaluate.add_argument("--output", default="artifacts/v0/reloaded")
    evaluate.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    evaluate.set_defaults(function=command_evaluate)

    synthetic = subparsers.add_parser(
        "synthetic-v0", help="run the same pipeline on known synthetic dynamics"
    )
    synthetic.add_argument("--steps", type=int, default=512)
    _add_training_arguments(synthetic)
    synthetic.set_defaults(function=command_synthetic)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
