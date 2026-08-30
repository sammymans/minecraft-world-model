"""Command-line entry point for the learning project."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcwm.cleaning import audit_transitions
from mcwm.data_infrastructure import (
    build_public_dataset_catalog,
    build_recording_catalog,
    publish_catalog,
)
from mcwm.dataset import (
    ProcessedEpisode,
    SequenceDataset,
    preprocess_episode,
    save_sequence_sheet,
)
from mcwm.download import DEMO_STEM, download_episode
from mcwm.interactive import compare_action_scripts, launch_playground
from mcwm.interactive_v2 import compare_action_scripts_v2, launch_playground_v2
from mcwm.latent_diffusion_v2 import overfit_latent_diffusion_v2
from mcwm.latent_rollout_v2 import finetune_rollout_v2
from mcwm.latent_training_v2 import train_latent_diffusion_v2
from mcwm.manifest import (
    DATASET_SPLITS,
    DatasetManifest,
    dataset_status,
    download_manifest,
    expand_vpt10_manifest,
    preprocess_manifest,
    split_manifest,
)
from mcwm.preview import create_preview, inspect_episode
from mcwm.recorder import CaptureRegion, record_minecraft_episode, verify_recording
from mcwm.rollout import evaluate_saved_rollouts
from mcwm.spatial_dynamics import (
    evaluate_saved_spatial_dynamics,
    train_spatial_dynamics,
)
from mcwm.spatial_training import (
    evaluate_saved_spatial_autoencoder,
    sanity_overfit_spatial_autoencoder,
    train_spatial_autoencoder,
)
from mcwm.vpt import load_actions

DEFAULT_DATA_DIR = Path("data/raw/vpt")
DEFAULT_PROCESSED_DIR = Path("data/processed/vpt_v1")
DEFAULT_MANIFEST = Path("data/manifests/vpt_v1.jsonl")


def _episode_paths(data_dir: Path, episode: str) -> tuple[Path, Path]:
    return data_dir / f"{episode}.mp4", data_dir / f"{episode}.jsonl"


def _add_episode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episode", default=DEMO_STEM, help="VPT episode filename stem")


def _add_spatial_dynamics_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/vpt_v4"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl"))
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcwm", description="Tiny Minecraft world model")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download-demo", help="download one official VPT pair")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    download.add_argument("--force", action="store_true")
    _add_episode_argument(download)

    dataset_download = commands.add_parser(
        "dataset-download", help="download every raw pair selected by a manifest"
    )
    dataset_download.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    dataset_download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    dataset_download.add_argument("--split", choices=("all", *DATASET_SPLITS), default="all")
    dataset_download.add_argument("--workers", type=int, default=3)
    dataset_download.add_argument("--force", action="store_true")

    expand_manifest = commands.add_parser(
        "dataset-expand-manifest",
        help="expand a manifest with diverse available VPT 10.x training groups",
    )
    expand_manifest.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST)
    expand_manifest.add_argument("--output", type=Path, default=Path("data/manifests/vpt_v2.jsonl"))
    expand_manifest.add_argument("--target-gib", type=float, default=10.0)
    expand_manifest.add_argument("--seed", type=int, default=7)

    split_manifest_parser = commands.add_parser(
        "dataset-split-manifest",
        help="assign complete session groups to train, validation, and test",
    )
    split_manifest_parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/manifests/vpt_v4.jsonl")
    )
    split_manifest_parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    split_manifest_parser.add_argument("--validation-fraction", type=float, default=0.1)
    split_manifest_parser.add_argument("--test-fraction", type=float, default=0.1)
    split_manifest_parser.add_argument("--seed", type=int, default=7)

    dataset_preprocess = commands.add_parser(
        "dataset-preprocess", help="preprocess every raw pair selected by a manifest"
    )
    dataset_preprocess.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    dataset_preprocess.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    dataset_preprocess.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    dataset_preprocess.add_argument("--split", choices=("all", *DATASET_SPLITS), default="all")
    dataset_preprocess.add_argument("--target-fps", type=float, default=10.0)
    dataset_preprocess.add_argument("--size", type=int, default=64)
    dataset_preprocess.add_argument("--horizon", type=int, default=8)
    dataset_preprocess.add_argument("--force", action="store_true")

    dataset_verify = commands.add_parser(
        "dataset-verify", help="verify manifest, raw files, and processed episodes"
    )
    dataset_verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    dataset_verify.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    dataset_verify.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)

    dataset_catalog = commands.add_parser(
        "dataset-catalog",
        help="hash and catalog a local public-data snapshot before object-storage upload",
    )
    dataset_catalog.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    dataset_catalog.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    dataset_catalog.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    dataset_catalog.add_argument(
        "--output", type=Path, default=Path("artifacts/dataset-catalogs/vpt-v4.json")
    )
    dataset_catalog.add_argument("--source-root", type=Path, default=Path("."))
    dataset_catalog.add_argument("--split", choices=("all", *DATASET_SPLITS), default="all")
    dataset_catalog.add_argument("--without-raw", action="store_true")
    dataset_catalog.add_argument("--without-processed", action="store_true")
    dataset_catalog.add_argument("--target-fps", type=float, default=10.0)
    dataset_catalog.add_argument("--size", type=int, default=64)
    dataset_catalog.add_argument("--horizon", type=int, default=8)

    record_minecraft = commands.add_parser(
        "record-minecraft",
        help="record a screen region and global controls as a synchronized local episode",
    )
    record_minecraft.add_argument("--output-dir", type=Path, default=Path("data/raw/local"))
    record_minecraft.add_argument("--duration", type=float, default=300.0)
    record_minecraft.add_argument("--fps", type=float, default=20.0)
    record_minecraft.add_argument("--countdown", type=float, default=3.0)
    record_minecraft.add_argument("--left", type=int, default=0)
    record_minecraft.add_argument("--top", type=int, default=0)
    record_minecraft.add_argument("--width", type=int, default=1280)
    record_minecraft.add_argument("--height", type=int, default=720)
    record_minecraft.add_argument("--episode")
    record_minecraft.add_argument("--collector")
    record_minecraft.add_argument("--source-root", type=Path, default=Path("."))

    verify_recording_parser = commands.add_parser(
        "recording-verify", help="verify one local recording's checksums and synchronization"
    )
    verify_recording_parser.add_argument("metadata", type=Path)
    verify_recording_parser.add_argument("--source-root", type=Path, default=Path("."))

    recording_catalog = commands.add_parser(
        "recording-catalog",
        help="catalog one real local recording and its optional processed episode",
    )
    recording_catalog.add_argument("metadata", type=Path)
    recording_catalog.add_argument("--processed", type=Path)
    recording_catalog.add_argument(
        "--output", type=Path, default=Path("artifacts/dataset-catalogs/local-recording.json")
    )
    recording_catalog.add_argument("--source-root", type=Path, default=Path("."))

    publish_s3 = commands.add_parser(
        "dataset-publish-s3",
        help="publish a verified catalog to S3 or an S3-compatible object store",
    )
    publish_s3.add_argument("catalog", type=Path)
    publish_s3.add_argument("--bucket", required=True)
    publish_s3.add_argument("--prefix", default="minecraft-world-model")
    publish_s3.add_argument("--source-root", type=Path)
    publish_s3.add_argument("--workers", type=int, default=4)
    publish_s3.add_argument("--endpoint-url")
    publish_s3.add_argument("--region")
    publish_s3.add_argument("--storage-class")
    publish_s3.add_argument("--sse", choices=("AES256", "aws:kms"))
    publish_s3.add_argument("--kms-key-id")
    publish_s3.add_argument(
        "--execute",
        action="store_true",
        help="perform uploads; without this flag the command only prints the plan",
    )

    inspect = commands.add_parser("inspect-demo", help="inspect video/action alignment")
    inspect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    _add_episode_argument(inspect)

    action = commands.add_parser("show-action", help="print one parsed action")
    action.add_argument("index", type=int)
    action.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    _add_episode_argument(action)

    preview = commands.add_parser("make-preview", help="write a video with action overlays")
    preview.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    preview.add_argument("--output", type=Path, default=Path("artifacts/vpt-preview.mp4"))
    preview.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    preview.add_argument("--duration", type=float, default=15.0, help="duration in seconds")
    preview.add_argument("--fps", type=float, default=10.0, help="preview frame rate")
    _add_episode_argument(preview)

    audit = commands.add_parser("audit-data", help="count accepted and rejected transitions")
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    audit.add_argument("--target-fps", type=float, default=10.0)
    audit.add_argument("--horizon", type=int, default=8)
    _add_episode_argument(audit)

    preprocess = commands.add_parser("preprocess-data", help="build one canonical NPZ episode")
    preprocess.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    preprocess.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    preprocess.add_argument("--target-fps", type=float, default=10.0)
    preprocess.add_argument("--size", type=int, default=64)
    preprocess.add_argument("--horizon", type=int, default=8)
    _add_episode_argument(preprocess)

    summary = commands.add_parser("dataset-summary", help="show group split and sequence counts")
    summary.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    summary.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    summary.add_argument("--horizon", type=int, default=8)

    sequence = commands.add_parser("show-sequence", help="render one exact training sequence")
    sequence.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    sequence.add_argument("--output", type=Path, default=Path("artifacts/sequence-sample.png"))
    sequence.add_argument("--horizon", type=int, default=8)
    sequence.add_argument("--index", type=int, default=0)
    _add_episode_argument(sequence)

    spatial_sanity = commands.add_parser(
        "sanity-spatial-autoencoder",
        help="memorize a tiny frame set with the spatial autoencoder",
    )
    spatial_sanity.add_argument("--processed-dir", type=Path, default=Path("data/processed/vpt_v3"))
    spatial_sanity.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v3.jsonl")
    )
    spatial_sanity.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/spatial-autoencoder-sanity")
    )
    spatial_sanity.add_argument("--frames", type=int, default=32)
    spatial_sanity.add_argument("--steps", type=int, default=500)
    spatial_sanity.add_argument("--latent-channels", type=int, default=16)
    spatial_sanity.add_argument("--base-channels", type=int, default=32)
    spatial_sanity.add_argument("--edge-weight", type=float, default=0.25)
    spatial_sanity.add_argument("--learning-rate", type=float, default=1e-3)
    spatial_sanity.add_argument("--seed", type=int, default=7)
    spatial_sanity.add_argument("--device", default="auto")

    spatial_train = commands.add_parser(
        "train-spatial-autoencoder",
        help="train the edge-preserving spatial representation",
    )
    spatial_train.add_argument("--processed-dir", type=Path, default=Path("data/processed/vpt_v3"))
    spatial_train.add_argument("--manifest", type=Path, default=Path("data/manifests/vpt_v3.jsonl"))
    spatial_train.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/spatial-autoencoder-v3")
    )
    spatial_train.add_argument("--epochs", type=int, default=30)
    spatial_train.add_argument("--batch-size", type=int, default=64)
    spatial_train.add_argument("--latent-channels", type=int, default=16)
    spatial_train.add_argument("--base-channels", type=int, default=32)
    spatial_train.add_argument("--edge-weight", type=float, default=0.25)
    spatial_train.add_argument("--learning-rate", type=float, default=1e-3)
    spatial_train.add_argument("--patience", type=int, default=6)
    spatial_train.add_argument(
        "--max-training-frames",
        type=int,
        default=0,
        help="deterministic frame subset; 0 uses all available training frames",
    )
    spatial_train.add_argument("--seed", type=int, default=7)
    spatial_train.add_argument("--device", default="auto")

    spatial_evaluate = commands.add_parser(
        "evaluate-spatial-autoencoder",
        help="evaluate a saved spatial autoencoder checkpoint",
    )
    spatial_evaluate.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v3")
    )
    spatial_evaluate.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v3.jsonl")
    )
    spatial_evaluate.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/spatial-autoencoder-v3/best.pt")
    )
    spatial_evaluate.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/spatial-autoencoder-v3-eval")
    )
    spatial_evaluate.add_argument("--split", choices=DATASET_SPLITS, default="validation")
    spatial_evaluate.add_argument("--batch-size", type=int, default=64)
    spatial_evaluate.add_argument("--count", type=int, default=8)
    spatial_evaluate.add_argument("--device", default="auto")

    train_spatial_dynamics_parser = commands.add_parser(
        "train-spatial-dynamics",
        help="train one-step action-conditioned dynamics on spatial latent maps",
    )
    _add_spatial_dynamics_data_arguments(train_spatial_dynamics_parser)
    train_spatial_dynamics_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/spatial-dynamics-v4")
    )
    train_spatial_dynamics_parser.add_argument("--epochs", type=int, default=20)
    train_spatial_dynamics_parser.add_argument(
        "--maximum-transitions",
        type=int,
        default=30_000,
        help="bounded in-memory latent cache; keeps the pilot within a few GiB of RAM",
    )
    train_spatial_dynamics_parser.add_argument("--hidden-channels", type=int, default=64)
    train_spatial_dynamics_parser.add_argument("--blocks", type=int, default=3)
    train_spatial_dynamics_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_spatial_dynamics_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_spatial_dynamics_parser.add_argument("--latent-weight", type=float, default=1.0)
    train_spatial_dynamics_parser.add_argument("--pixel-weight", type=float, default=1.0)
    train_spatial_dynamics_parser.add_argument("--patience", type=int, default=5)
    train_spatial_dynamics_parser.add_argument(
        "--selection-policy",
        choices=("random", "nested_prefix"),
        default="random",
        help=(
            "transition subset policy; nested_prefix makes smaller seed-matched "
            "data ablations strict subsets of larger ones"
        ),
    )
    train_spatial_dynamics_parser.add_argument(
        "--validation-every",
        type=int,
        default=1,
        help="evaluate every N epochs and always on the final epoch",
    )
    train_spatial_dynamics_parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="recursive steps unrolled per training window; 1 keeps one-step training",
    )
    train_spatial_dynamics_parser.add_argument(
        "--horizon-decay",
        type=float,
        default=0.8,
        help="weight applied to each further recursive step",
    )
    train_spatial_dynamics_parser.add_argument(
        "--gradient-clip",
        type=float,
        default=0.0,
        help="global gradient-norm clip; 0 disables it",
    )
    train_spatial_dynamics_parser.add_argument(
        "--maximum-validation-sequences",
        type=int,
        default=5_000,
        help="bounded recursive validation windows scored after every epoch",
    )
    train_spatial_dynamics_parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="fine-tune this saved spatial dynamics checkpoint instead of starting fresh",
    )

    evaluate_spatial_dynamics_parser = commands.add_parser(
        "evaluate-spatial-dynamics",
        help="evaluate a saved spatial dynamics checkpoint against its baselines",
    )
    _add_spatial_dynamics_data_arguments(evaluate_spatial_dynamics_parser)
    evaluate_spatial_dynamics_parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-dynamics-v4-multistep/best.pt"),
    )
    evaluate_spatial_dynamics_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial-dynamics-v4-multistep-eval"),
    )
    evaluate_spatial_dynamics_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate_spatial_dynamics_parser.add_argument("--count", type=int, default=6)

    evaluate_rollout_parser = commands.add_parser(
        "evaluate-rollout", help="measure recursive latent prediction over several horizons"
    )
    _add_spatial_dynamics_data_arguments(evaluate_rollout_parser)
    evaluate_rollout_parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-dynamics-v4-multistep/best.pt"),
    )
    evaluate_rollout_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/spatial-rollout-v4-multistep")
    )
    evaluate_rollout_parser.add_argument(
        "--horizons", type=int, nargs="+", default=(1, 2, 5, 10, 20)
    )
    evaluate_rollout_parser.add_argument("--count", type=int, default=3)
    evaluate_rollout_parser.add_argument("--maximum-examples", type=int, default=5_000)
    evaluate_rollout_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )

    play_rollout_parser = commands.add_parser(
        "play-rollout", help="control a recursively imagined Minecraft latent state"
    )
    play_rollout_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    play_rollout_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    play_rollout_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    play_rollout_parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-dynamics-v4-multistep/best.pt"),
    )
    play_rollout_parser.add_argument(
        "--sample-index",
        type=int,
        default=20_000,
        help="clean held-out transition used for the two real seed frames",
    )
    play_rollout_parser.add_argument(
        "--camera-step",
        type=float,
        default=12.0,
        help=(
            "raw mouse delta generated by each scripted or arrow-key camera input; "
            "rollouts hold detail near 10 and dissolve by 30"
        ),
    )
    play_rollout_parser.add_argument(
        "--script",
        help="run headlessly, e.g. 'w+sprint*5,w+look_right*3,idle*2'",
    )
    play_rollout_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interactive-rollout/scripted-rollout.png"),
    )
    play_rollout_parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    play_rollout_parser.add_argument("--host", default="127.0.0.1")
    play_rollout_parser.add_argument("--port", type=int, default=8765)
    play_rollout_parser.add_argument(
        "--no-open", action="store_true", help="do not open the frontend in a browser"
    )

    compare_actions_parser = commands.add_parser(
        "compare-actions",
        help="imagine one seed forward under several action scripts, side by side",
    )
    compare_actions_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    compare_actions_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    compare_actions_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    compare_actions_parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-dynamics-v4-multistep/best.pt"),
    )
    compare_actions_parser.add_argument(
        "--scripts",
        nargs="+",
        default=("w+sprint*6", "look_left*6", "look_right*6", "idle*6"),
        help="one action script per row, e.g. 'w+sprint*6' 'look_left*6'",
    )
    compare_actions_parser.add_argument(
        "--sample-index",
        type=int,
        default=20_000,
        help="clean held-out transition used for the shared real seed frames",
    )
    compare_actions_parser.add_argument("--camera-step", type=float, default=30.0)
    compare_actions_parser.add_argument(
        "--tile", type=int, default=192, help="rendered size of each imagined frame"
    )
    compare_actions_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interactive-rollout/action-comparison.png"),
    )
    compare_actions_parser.add_argument("--device", default="auto")

    overfit_v2_parser = commands.add_parser(
        "overfit-latent-diffusion-v2",
        help="run the fixed-256 overfit gate for V2 temporal latent diffusion",
    )
    overfit_v2_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    overfit_v2_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    overfit_v2_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    overfit_v2_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/fixed-256-action-balanced-v2"),
    )
    overfit_v2_parser.add_argument("--sequences", type=int, default=256)
    overfit_v2_parser.add_argument("--steps", type=int, default=2_000)
    overfit_v2_parser.add_argument("--batch-size", type=int, default=8)
    overfit_v2_parser.add_argument("--encode-batch-size", type=int, default=128)
    overfit_v2_parser.add_argument("--minimum-episodes", type=int, default=8)
    overfit_v2_parser.add_argument("--base-channels", type=int, default=112)
    overfit_v2_parser.add_argument("--attention-heads", type=int, default=8)
    overfit_v2_parser.add_argument("--diffusion-steps", type=int, default=1_000)
    overfit_v2_parser.add_argument("--sampling-steps", type=int, default=8)
    overfit_v2_parser.add_argument("--learning-rate", type=float, default=2e-4)
    overfit_v2_parser.add_argument("--weight-decay", type=float, default=1e-5)
    overfit_v2_parser.add_argument(
        "--maximum-context-noise",
        type=float,
        default=0.0,
        help="Stage 2 defaults to clean context; Stage 3 will enable augmentation",
    )
    overfit_v2_parser.add_argument(
        "--rollout-context-noise",
        type=float,
        default=0.0,
        help="context noise level reported to the model during counterfactual rollout",
    )
    overfit_v2_parser.add_argument("--seed", type=int, default=7)
    overfit_v2_parser.add_argument("--device", default="auto")

    train_v2_parser = commands.add_parser(
        "train-latent-diffusion-v2",
        help="train V2 latent diffusion with fixed held-out evaluation",
    )
    train_v2_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    train_v2_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    train_v2_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    train_v2_parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/latent-cache-v2")
    )
    train_v2_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/full-v4"),
    )
    train_v2_parser.add_argument("--steps", type=int, default=60_000)
    train_v2_parser.add_argument("--batch-size", type=int, default=8)
    train_v2_parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help=(
            "encoded frames conditioning each prediction; 8 is 0.8s of history, "
            "and a longer window keeps real frames in context deeper into a rollout"
        ),
    )
    train_v2_parser.add_argument("--encode-batch-size", type=int, default=128)
    train_v2_parser.add_argument("--base-channels", type=int, default=112)
    train_v2_parser.add_argument("--attention-heads", type=int, default=8)
    train_v2_parser.add_argument("--diffusion-steps", type=int, default=1_000)
    train_v2_parser.add_argument("--sampling-steps", type=int, default=8)
    train_v2_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_v2_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_v2_parser.add_argument("--maximum-context-noise", type=float, default=0.2)
    train_v2_parser.add_argument(
        "--action-change-fraction",
        type=float,
        default=0.35,
        help="target share after adding switch-point examples; every natural window remains",
    )
    train_v2_parser.add_argument("--evaluation-every", type=int, default=2_000)
    train_v2_parser.add_argument(
        "--maximum-training-windows",
        type=int,
        help=(
            "use a deterministic nested subset of this many training windows; "
            "omit to use every available window"
        ),
    )
    train_v2_parser.add_argument("--maximum-validation-sequences", type=int, default=512)
    train_v2_parser.add_argument("--sample-count", type=int, default=16)
    train_v2_parser.add_argument("--resume", type=Path)
    train_v2_parser.add_argument(
        "--force-cache", action="store_true", help="re-encode and replace this exact V2 cache"
    )
    train_v2_parser.add_argument("--seed", type=int, default=7)
    train_v2_parser.add_argument("--device", default="auto")

    finetune_v2_parser = commands.add_parser(
        "finetune-rollout-v2",
        help="fine-tune V2 on its own generated context to resist rollout collapse",
    )
    finetune_v2_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    finetune_v2_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    finetune_v2_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    finetune_v2_parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/latent-cache-v2")
    )
    finetune_v2_parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/full-v4/best.pt"),
    )
    finetune_v2_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/rollout-finetune"),
    )
    finetune_v2_parser.add_argument("--steps", type=int, default=6_000)
    finetune_v2_parser.add_argument("--batch-size", type=int, default=8)
    finetune_v2_parser.add_argument("--encode-batch-size", type=int, default=128)
    finetune_v2_parser.add_argument("--learning-rate", type=float, default=5e-5)
    finetune_v2_parser.add_argument("--weight-decay", type=float, default=1e-5)
    finetune_v2_parser.add_argument(
        "--generation-steps",
        type=int,
        default=8,
        help="DDIM steps used to build the self-generated context during training",
    )
    finetune_v2_parser.add_argument(
        "--sampling-steps", type=int, default=32, help="DDIM steps used for evaluation"
    )
    finetune_v2_parser.add_argument("--rollout-horizon", type=int, default=8)
    finetune_v2_parser.add_argument("--maximum-context-noise", type=float, default=0.2)
    finetune_v2_parser.add_argument("--evaluation-every", type=int, default=1_000)
    finetune_v2_parser.add_argument("--maximum-validation-sequences", type=int, default=256)
    finetune_v2_parser.add_argument("--rollout-examples", type=int, default=24)
    finetune_v2_parser.add_argument("--seed", type=int, default=7)
    finetune_v2_parser.add_argument("--device", default="auto")

    play_v2_parser = commands.add_parser(
        "play-v2", help="control a recursively sampled V2 latent-diffusion world"
    )
    play_v2_parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/vpt_v4"))
    play_v2_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    play_v2_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    play_v2_parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/full-v4/best.pt"),
    )
    play_v2_parser.add_argument(
        "--sample-index",
        type=int,
        default=42_750,
        help=(
            "clean held-out eight-frame context used as the seed; the default is "
            "a bright outdoor scene, because dark caves hide all camera motion"
        ),
    )
    play_v2_parser.add_argument(
        "--camera-step",
        type=float,
        default=12.0,
        help=(
            "raw mouse delta per camera input; rollouts hold detail near 10-12 "
            "and dissolve above 30, since blur tracks new content per step"
        ),
    )
    play_v2_parser.add_argument(
        "--sampling-steps",
        type=int,
        default=32,
        help=(
            "DDIM steps per imagined frame; 8 loses roughly one fifth of edge energy per "
            "step and washes out by t+3, 32 holds structure, 64 saturates"
        ),
    )
    play_v2_parser.add_argument("--sampling-seed", type=int, default=7)
    play_v2_parser.add_argument("--script")
    play_v2_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interactive-rollout-v2/scripted-rollout.png"),
    )
    play_v2_parser.add_argument("--device", default="auto")
    play_v2_parser.add_argument("--host", default="127.0.0.1")
    play_v2_parser.add_argument("--port", type=int, default=8765)
    play_v2_parser.add_argument("--no-open", action="store_true")

    compare_v2_parser = commands.add_parser(
        "compare-actions-v2",
        help="compare V2 action scripts from one shared held-out context",
    )
    compare_v2_parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vpt_v4")
    )
    compare_v2_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/vpt_v4_split.jsonl")
    )
    compare_v2_parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-autoencoder-v3/best.pt"),
    )
    compare_v2_parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=Path("artifacts/spatial-latent-diffusion-v2/full-v4/best.pt"),
    )
    compare_v2_parser.add_argument(
        "--scripts",
        nargs="+",
        default=("w+sprint*6", "look_left*6", "look_right*6", "idle*6"),
    )
    compare_v2_parser.add_argument("--sample-index", type=int, default=42_750)
    compare_v2_parser.add_argument("--camera-step", type=float, default=12.0)
    compare_v2_parser.add_argument(
        "--sampling-steps",
        type=int,
        default=32,
        help=(
            "DDIM steps per imagined frame; 8 loses roughly one fifth of edge energy per "
            "step and washes out by t+3, 32 holds structure, 64 saturates"
        ),
    )
    compare_v2_parser.add_argument("--sampling-seed", type=int, default=7)
    compare_v2_parser.add_argument("--tile", type=int, default=192)
    compare_v2_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interactive-rollout-v2/action-comparison.png"),
    )
    compare_v2_parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download-demo":
        video, actions = download_episode(args.data_dir, args.episode, force=args.force)
        print(f"Video:   {video}")
        print(f"Actions: {actions}")
        return 0

    if args.command == "dataset-download":
        manifest = DatasetManifest.load(args.manifest)
        download_manifest(
            manifest,
            args.data_dir,
            split=args.split,
            force=args.force,
            workers=args.workers,
        )
        return 0

    if args.command == "dataset-expand-manifest":
        manifest = expand_vpt10_manifest(
            DatasetManifest.load(args.base_manifest),
            args.output,
            target_bytes=int(args.target_gib * 1024**3),
            seed=args.seed,
        )
        expected = sum(entry.video_bytes + entry.actions_bytes for entry in manifest.episodes)
        print(f"Wrote: {args.output}")
        print(f"Episodes: {len(manifest.episodes)}")
        print(f"Independent groups: {len({entry.group for entry in manifest.episodes})}")
        print(f"Expected raw data: {expected / 1024**3:.2f} GiB")
        return 0

    if args.command == "dataset-split-manifest":
        manifest = split_manifest(
            DatasetManifest.load(args.source_manifest),
            args.output,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        print(f"Wrote: {args.output}")
        for split in DATASET_SPLITS:
            episodes = manifest.select(split)
            groups = {entry.group for entry in episodes}
            raw_gib = sum(entry.video_bytes + entry.actions_bytes for entry in episodes) / 2**30
            print(
                f"{split:10s}: {len(groups):3d} groups, {len(episodes):3d} episodes, "
                f"{raw_gib:5.2f} GiB"
            )
        return 0

    if args.command == "dataset-preprocess":
        manifest = DatasetManifest.load(args.manifest)
        preprocess_manifest(
            manifest,
            args.data_dir,
            args.output_dir,
            split=args.split,
            force=args.force,
            target_fps=args.target_fps,
            image_size=args.size,
            horizon=args.horizon,
        )
        return 0

    if args.command == "dataset-verify":
        manifest = DatasetManifest.load(args.manifest)
        status = dataset_status(
            manifest,
            args.data_dir,
            args.processed_dir,
            verify_processed=True,
        )
        print(f"manifest episodes:  {status.episodes}")
        print(f"independent groups: {status.groups}")
        print(f"training groups:    {status.training_groups}")
        print(f"validation groups:  {status.validation_groups}")
        print(f"test groups:        {status.test_groups}")
        print(f"raw pairs complete: {status.raw_complete}/{status.episodes}")
        print(f"processed complete: {status.processed_complete}/{status.episodes}")
        print(f"expected raw size:  {status.expected_raw_bytes / 2**30:.2f} GiB")
        if status.raw_complete != status.episodes:
            raise SystemExit("Raw dataset is incomplete. Run: uv run mcwm dataset-download")
        if status.processed_complete != status.episodes:
            raise SystemExit("Processed dataset is incomplete. Run: uv run mcwm dataset-preprocess")
        print("dataset verification passed")
        return 0

    if args.command == "dataset-catalog":
        catalog = build_public_dataset_catalog(
            args.manifest,
            args.data_dir,
            args.processed_dir,
            args.output,
            source_root=args.source_root,
            split=args.split,
            include_raw=not args.without_raw,
            include_processed=not args.without_processed,
            target_fps=args.target_fps,
            image_size=args.size,
            horizon=args.horizon,
        )
        print(f"catalog:       {args.output}")
        print(f"catalog id:    {catalog.catalog_id}")
        print(f"objects:       {len(catalog.objects):,}")
        print(f"catalog bytes: {sum(item.bytes for item in catalog.objects) / 2**30:.2f} GiB")
        return 0

    if args.command == "record-minecraft":
        result = record_minecraft_episode(
            args.output_dir,
            duration_seconds=args.duration,
            region=CaptureRegion(args.left, args.top, args.width, args.height),
            fps=args.fps,
            countdown_seconds=args.countdown,
            episode=args.episode,
            collector=args.collector,
            source_root=args.source_root,
        )
        print(f"episode:  {result.episode}")
        print(f"frames:   {result.frames:,} at {result.fps:g} Hz")
        print(f"video:    {result.video_path}")
        print(f"actions:  {result.actions_path}")
        print(f"metadata: {result.metadata_path}")
        return 0

    if args.command == "recording-verify":
        result = verify_recording(args.metadata, source_root=args.source_root)
        print(f"episode:        {result.episode}")
        print(f"video frames:   {result.video_frames:,}")
        print(f"action records: {result.action_records:,}")
        print(f"video:          {result.width}x{result.height} at {result.fps:g} Hz")
        print("recording verification passed")
        return 0

    if args.command == "recording-catalog":
        catalog = build_recording_catalog(
            args.metadata,
            args.output,
            source_root=args.source_root,
            processed_path=args.processed,
        )
        print(f"catalog:    {args.output}")
        print(f"catalog id: {catalog.catalog_id}")
        print(f"objects:    {len(catalog.objects):,}")
        return 0

    if args.command == "dataset-publish-s3":
        result = publish_catalog(
            args.catalog,
            args.bucket,
            args.prefix,
            source_root=args.source_root,
            execute=args.execute,
            workers=args.workers,
            endpoint_url=args.endpoint_url,
            region=args.region,
            storage_class=args.storage_class,
            sse=args.sse,
            kms_key_id=args.kms_key_id,
        )
        print(f"destination: {result.destination}")
        print(f"objects:     {result.planned:,}")
        if result.dry_run:
            print("dry run only; pass --execute to upload (remote objects are never deleted)")
        else:
            print(f"uploaded:    {result.uploaded:,}")
            print(f"skipped:     {result.skipped:,}")
            print(f"bytes:       {result.bytes_uploaded / 2**30:.2f} GiB")
        return 0

    if args.command == "dataset-summary":
        try:
            manifest = DatasetManifest.load(args.manifest)
            split_paths = {
                split: manifest.processed_paths(args.processed_dir, split)
                for split in DATASET_SPLITS
                if manifest.select(split)
            }
        except ValueError as error:
            raise SystemExit(str(error)) from error
        for split, paths in split_paths.items():
            sequences = sum(
                len(SequenceDataset.from_paths([path], horizon=args.horizon)) for path in paths
            )
            groups = {entry.group for entry in manifest.select(split)}
            print(f"{split} groups:    {len(groups):,}")
            print(f"{split} episodes:  {len(paths):,}")
            print(f"{split} sequences: {sequences:,}")
        return 0

    if args.command == "show-sequence":
        processed_path = args.processed_dir / f"{args.episode}.npz"
        if not processed_path.exists():
            raise SystemExit(f"Processed episode is missing: {processed_path}")
        dataset = SequenceDataset([ProcessedEpisode.load(processed_path)], horizon=args.horizon)
        if not 0 <= args.index < len(dataset):
            raise SystemExit(f"index must be between 0 and {len(dataset) - 1}")
        sample = dataset[args.index]
        save_sequence_sheet(sample, args.output)
        print(f"episode: {sample.episode}")
        print(f"model frames: {sample.frames.shape}")
        print(f"model actions: {sample.actions.shape}")
        print(f"source frames: {sample.source_frame_indices.tolist()}")
        print(f"wrote: {args.output}")
        return 0

    if args.command == "sanity-spatial-autoencoder":
        result = sanity_overfit_spatial_autoencoder(
            args.processed_dir,
            args.manifest,
            args.output_dir,
            frame_count=args.frames,
            steps=args.steps,
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            edge_weight=args.edge_weight,
            learning_rate=args.learning_rate,
            seed=args.seed,
            requested_device=args.device,
        )
        print(f"device:       {result.device}")
        print(f"latent shape: {result.latent_shape}")
        print(f"parameters:   {result.parameter_count:,}")
        print(f"memorized L1: {result.train_metrics.pixel_l1:.6f}")
        print(f"edge ratio:   {result.train_metrics.gradient_energy_ratio:.3f}")
        print(f"checkpoint:   {result.checkpoint}")
        print(f"visual:       {result.reconstruction_grid}")
        return 0

    if args.command == "train-spatial-autoencoder":
        result = train_spatial_autoencoder(
            args.processed_dir,
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            edge_weight=args.edge_weight,
            learning_rate=args.learning_rate,
            patience=args.patience,
            max_training_frames=args.max_training_frames or None,
            seed=args.seed,
            requested_device=args.device,
        )
        validation = result.validation_metrics
        if validation is None:
            raise RuntimeError("spatial training did not evaluate validation data")
        print(f"device:          {result.device}")
        print(f"latent shape:    {result.latent_shape}")
        print(f"parameters:      {result.parameter_count:,}")
        print(f"validation L1:   {validation.pixel_l1:.6f}")
        print(f"validation PSNR: {validation.psnr_db:.2f} dB")
        print(f"edge ratio:      {validation.gradient_energy_ratio:.3f}")
        print(f"checkpoint:      {result.checkpoint}")
        print(f"visual:          {result.reconstruction_grid}")
        return 0

    if args.command == "evaluate-spatial-autoencoder":
        result = evaluate_saved_spatial_autoencoder(
            args.processed_dir,
            args.manifest,
            args.checkpoint,
            args.output_dir,
            split=args.split,
            batch_size=args.batch_size,
            count=args.count,
            requested_device=args.device,
        )
        print(f"device:       {result.device}")
        print(f"frames:       {result.frame_count:,}")
        print(f"latent shape: {result.latent_shape}")
        print(f"pixel L1:     {result.metrics.pixel_l1:.6f}")
        print(f"pixel MSE:    {result.metrics.pixel_mse:.6f}")
        print(f"PSNR:         {result.metrics.psnr_db:.2f} dB")
        print(f"edge ratio:   {result.metrics.gradient_energy_ratio:.3f}")
        print(f"visual:       {result.reconstruction_grid}")
        return 0

    if args.command == "train-spatial-dynamics":
        result = train_spatial_dynamics(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            maximum_transitions=args.maximum_transitions,
            hidden_channels=args.hidden_channels,
            blocks=args.blocks,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            latent_weight=args.latent_weight,
            pixel_weight=args.pixel_weight,
            patience=args.patience,
            rollout_steps=args.rollout_steps,
            horizon_decay=args.horizon_decay,
            gradient_clip=args.gradient_clip,
            maximum_validation_sequences=args.maximum_validation_sequences,
            initial_checkpoint=args.initial_checkpoint,
            selection_policy=args.selection_policy,
            validation_every=args.validation_every,
            seed=args.seed,
            requested_device=args.device,
        )
        validation = result.validation_metrics
        print(f"device:                      {result.device}")
        print(f"latent shape:                {result.latent_shape}")
        print(f"dynamics parameters:         {result.parameter_count:,}")
        print(f"recursive training steps:    {result.rollout_steps}")
        print(f"training windows:            {result.training_transitions:,}")
        print(f"encoded training frames:     {result.encoded_frames:,}")
        print(f"validation latent MSE:       {validation.latent_mse:.6f}")
        print(f"copy baseline latent MSE:    {validation.copy_latent_mse:.6f}")
        print(f"validation pixel L1:         {validation.pixel_l1:.6f}")
        print(f"decoded-copy pixel L1:       {validation.decoded_copy_pixel_l1:.6f}")
        print(f"decoder-oracle pixel L1:     {validation.oracle_pixel_l1:.6f}")
        print(f"shuffled-action degradation: {validation.shuffled_action_degradation:+.6f}")
        print(f"checkpoint:                  {result.checkpoint}")
        print(f"visual:                      {result.comparison_grid}")
        return 0

    if args.command == "evaluate-spatial-dynamics":
        result = evaluate_saved_spatial_dynamics(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.dynamics_checkpoint,
            args.output_dir,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            count=args.count,
            split=args.split,
            seed=args.seed,
            requested_device=args.device,
        )
        metrics = result.metrics
        print(f"device:                      {result.device}")
        print(f"transitions:                 {result.transitions:,}")
        print(f"latent shape:                {result.latent_shape}")
        print(f"latent MSE:                  {metrics.latent_mse:.6f}")
        print(f"copy baseline latent MSE:    {metrics.copy_latent_mse:.6f}")
        print(f"pixel L1:                    {metrics.pixel_l1:.6f}")
        print(f"decoded-copy pixel L1:       {metrics.decoded_copy_pixel_l1:.6f}")
        print(f"decoder-oracle pixel L1:     {metrics.oracle_pixel_l1:.6f}")
        print(f"pixel PSNR:                  {metrics.pixel_psnr_db:.2f} dB")
        print(f"shuffled-action degradation: {metrics.shuffled_action_degradation:+.6f}")
        print(f"visual:                      {result.comparison_grid}")
        return 0

    if args.command == "evaluate-rollout":
        result = evaluate_saved_rollouts(
            args.processed_dir,
            args.autoencoder_checkpoint,
            args.dynamics_checkpoint,
            args.output_dir,
            manifest_path=args.manifest,
            horizons=tuple(args.horizons),
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            count=args.count,
            maximum_examples=args.maximum_examples,
            split=args.split,
            seed=args.seed,
            requested_device=args.device,
        )
        print(f"device:        {result.device}")
        print(f"examples:      {result.example_count:,}")
        print(f"max horizon:   {result.max_horizon} steps")
        print("horizon  recursive MSE  copy gain  action penalty  beats copy")
        for metrics in result.horizons:
            print(
                f"{metrics.horizon:7d}  {metrics.recursive_pixel_mse:13.6f}  "
                f"{metrics.copy_improvement_percent:8.1f}%  "
                f"{metrics.shuffled_action_pixel_penalty_percent:12.1f}%  "
                f"{str(metrics.beats_copy_pixel):>10s}"
            )
        print(f"curve:         {result.error_curve}")
        print(f"filmstrips:    {result.filmstrips}")
        print(f"metrics:       {result.metrics_path}")
        return 0

    if args.command == "play-rollout":
        result, seed_count = launch_playground(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.dynamics_checkpoint,
            sample_index=args.sample_index,
            camera_step=args.camera_step,
            script=args.script,
            output_path=args.output,
            requested_device=args.device,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        print(f"device:         {result.device}")
        print(f"held-out seeds: {seed_count:,}")
        print(f"episode:        {result.episode}")
        print(f"seed step:      {result.current_step}")
        print(f"imagined steps: {result.steps}")
        if result.output is not None:
            print(f"visual:         {result.output}")
        return 0

    if args.command == "compare-actions":
        result, seed_count = compare_action_scripts(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.dynamics_checkpoint,
            list(args.scripts),
            sample_index=args.sample_index,
            camera_step=args.camera_step,
            tile=args.tile,
            output_path=args.output,
            requested_device=args.device,
        )
        print(f"device:         {result.device}")
        print(f"held-out seeds: {seed_count:,}")
        print(f"episode:        {result.episode}")
        print(f"seed step:      {result.current_step}")
        print(f"scripts:        {len(args.scripts)}")
        print(f"imagined steps: {result.steps}")
        print(f"visual:         {result.output}")
        return 0

    if args.command == "overfit-latent-diffusion-v2":
        result = overfit_latent_diffusion_v2(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.output_dir,
            sequences=args.sequences,
            training_steps=args.steps,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            encode_batch_size=args.encode_batch_size,
            base_channels=args.base_channels,
            attention_heads=args.attention_heads,
            diffusion_steps=args.diffusion_steps,
            sampling_steps=args.sampling_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            maximum_context_noise=args.maximum_context_noise,
            rollout_context_noise=args.rollout_context_noise,
            seed=args.seed,
            requested_device=args.device,
        )
        metrics = result.metrics
        print(f"device:                       {result.device}")
        print(f"fixed sequences:              {metrics.sequences}")
        print(f"V2 parameters:                {result.parameter_count:,}")
        print(f"initial denoising loss:        {metrics.initial_denoising_loss:.6f}")
        print(f"final denoising loss:          {metrics.final_denoising_loss:.6f}")
        print(f"loss reduction:                {metrics.loss_reduction_percent:.1f}%")
        print(f"shuffled-action degradation:  {metrics.shuffled_action_degradation_percent:+.1f}%")
        print(
            "shuffled sample degradation:  "
            f"{metrics.shuffled_action_sample_degradation_percent:+.1f}%"
        )
        print(f"sample PSNR:                   {metrics.sample_pixel_psnr_db:.2f} dB")
        print(f"copy baseline PSNR:            {metrics.copy_pixel_psnr_db:.2f} dB")
        print(f"sample edge ratio:             {metrics.sample_edge_ratio:.3f}")
        print(f"five-step action effect L1:    {metrics.five_step_action_effect_pixel_l1:.6f}")
        print(f"five-step total drift L1:      {metrics.five_step_total_drift_pixel_l1:.6f}")
        print(f"action share of drift:         {metrics.action_share_of_drift_percent:.1f}%")
        print(f"sampled latents finite:        {metrics.sampled_latents_finite}")
        print(f"checkpoint:                    {result.checkpoint}")
        print(f"samples:                       {result.samples}")
        print(f"five-step action rollout:      {result.action_comparison}")
        return 0

    if args.command == "train-latent-diffusion-v2":
        result = train_latent_diffusion_v2(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.cache_dir,
            args.output_dir,
            training_steps=args.steps,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            encode_batch_size=args.encode_batch_size,
            base_channels=args.base_channels,
            attention_heads=args.attention_heads,
            diffusion_steps=args.diffusion_steps,
            sampling_steps=args.sampling_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            maximum_context_noise=args.maximum_context_noise,
            target_action_change_fraction=args.action_change_fraction,
            evaluation_every=args.evaluation_every,
            maximum_training_windows=args.maximum_training_windows,
            maximum_validation_sequences=args.maximum_validation_sequences,
            sample_count=args.sample_count,
            resume_checkpoint=args.resume,
            force_cache=args.force_cache,
            seed=args.seed,
            requested_device=args.device,
        )
        print(f"device:                    {result.device}")
        print(f"training windows:          {result.training_windows:,}")
        print(f"validation windows:        {result.validation_windows:,}")
        print(f"training action changes:   {result.action_change_windows:,}")
        print(f"V2 parameters:             {result.parameter_count:,}")
        print(f"completed steps:           {result.completed_steps:,}")
        print(f"best checkpoint:           {result.checkpoint}")
        print(f"latest checkpoint:         {result.latest_checkpoint}")
        print(f"metrics:                   {result.metrics_path}")
        print(f"validation samples:        {result.samples}")
        print(f"validation action rollout: {result.action_comparison}")
        return 0

    if args.command == "finetune-rollout-v2":
        result = finetune_rollout_v2(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.cache_dir,
            args.initial_checkpoint,
            args.output_dir,
            training_steps=args.steps,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            generation_steps=args.generation_steps,
            sampling_steps=args.sampling_steps,
            rollout_horizon=args.rollout_horizon,
            maximum_context_noise=args.maximum_context_noise,
            evaluation_every=args.evaluation_every,
            maximum_validation_sequences=args.maximum_validation_sequences,
            rollout_examples=args.rollout_examples,
            seed=args.seed,
            requested_device=args.device,
        )
        print(f"device:              {result.device}")
        print(f"training windows:    {result.training_windows:,}")
        print(f"V2 parameters:       {result.parameter_count:,}")
        print(f"completed steps:     {result.completed_steps:,}")
        print(f"best checkpoint:     {result.checkpoint}")
        print(f"latest checkpoint:   {result.latest_checkpoint}")
        print(f"metrics:             {result.metrics_path}")
        print(f"validation samples:  {result.samples}")
        print(f"action rollout:      {result.action_comparison}")
        return 0

    if args.command == "play-v2":
        result, seed_count = launch_playground_v2(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.model_checkpoint,
            sample_index=args.sample_index,
            camera_step=args.camera_step,
            sampling_steps=args.sampling_steps,
            sampling_seed=args.sampling_seed,
            script=args.script,
            output_path=args.output,
            requested_device=args.device,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        print(f"device:         {result.device}")
        print(f"held-out seeds: {seed_count:,}")
        print(f"episode:        {result.episode}")
        print(f"seed step:      {result.current_step}")
        print(f"imagined steps: {result.steps}")
        if result.output is not None:
            print(f"visual:         {result.output}")
        return 0

    if args.command == "compare-actions-v2":
        result, seed_count = compare_action_scripts_v2(
            args.processed_dir,
            args.manifest,
            args.autoencoder_checkpoint,
            args.model_checkpoint,
            list(args.scripts),
            sample_index=args.sample_index,
            camera_step=args.camera_step,
            sampling_steps=args.sampling_steps,
            sampling_seed=args.sampling_seed,
            tile=args.tile,
            output_path=args.output,
            requested_device=args.device,
        )
        print(f"device:         {result.device}")
        print(f"held-out seeds: {seed_count:,}")
        print(f"episode:        {result.episode}")
        print(f"seed step:      {result.current_step}")
        print(f"scripts:        {len(args.scripts)}")
        print(f"imagined steps: {result.steps}")
        print(f"visual:         {result.output}")
        return 0

    video, actions = _episode_paths(args.data_dir, args.episode)
    if not video.exists() or not actions.exists():
        raise SystemExit("Demo files are missing. Run: uv run mcwm download-demo")

    if args.command == "inspect-demo":
        info, counts = inspect_episode(video, actions)
        print(f"video:         {info.width}x{info.height} at {info.fps:.2f} fps")
        print(f"video frames:  {info.video_frames:,}")
        print(f"action lines:  {info.action_frames:,}")
        print(f"paired frames: {info.paired_frames:,} ({info.duration_seconds:.1f} seconds)")
        print("movement-key frames:")
        for key, count in sorted(counts.items()):
            print(f"  {key:7s} {count:7,d}")
        return 0

    if args.command == "show-action":
        if args.index < 0:
            raise SystemExit("index must be non-negative")
        parsed = load_actions(actions, limit=args.index + 1)
        if args.index >= len(parsed):
            raise SystemExit(f"episode contains only {len(parsed)} actions")
        selected = parsed[args.index]
        print(selected.label())
        print(f"model vector: {selected.movement_vector()}")
        return 0

    if args.command == "make-preview":
        written = create_preview(
            video,
            actions,
            args.output,
            start_seconds=args.start,
            duration_seconds=args.duration,
            output_fps=args.fps,
        )
        print(f"Wrote {written} annotated frames to {args.output}")
        return 0

    if args.command == "audit-data":
        info, _ = inspect_episode(video, actions)
        parsed = load_actions(actions)
        stride = round(info.fps / args.target_fps)
        if stride < 1 or abs(info.fps / stride - args.target_fps) > 0.01:
            raise SystemExit("target FPS must evenly divide the source FPS")
        report, _ = audit_transitions(
            parsed,
            min(info.video_frames, info.action_frames),
            stride=stride,
            horizon=args.horizon,
        )
        print(f"source frames:        {report.source_frames:,}")
        print(f"10 Hz model frames:   {report.model_frames:,}")
        print(f"model transitions:    {report.total_transitions:,}")
        print(
            f"accepted transitions: {report.accepted_transitions:,} ({report.acceptance_rate:.1%})"
        )
        print("exclusive rejection reasons:")
        for reason, count in report.rejection_counts.items():
            print(f"  {reason:20s} {count:7,d}")
        print(f"valid {report.sequence_horizon}-step sequences: {report.valid_sequences:,}")
        return 0

    if args.command == "preprocess-data":
        output = args.output_dir / f"{args.episode}.npz"
        result = preprocess_episode(
            video,
            actions,
            output,
            target_fps=args.target_fps,
            image_size=args.size,
            horizon=args.horizon,
        )
        print(f"source frames:      {result.source_frames:,}")
        print(f"model frames:       {result.model_frames:,}")
        print(f"valid transitions:  {result.valid_transitions:,}")
        print(f"valid sequences:    {result.valid_sequences:,}")
        print(f"wrote: {result.output_path}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
