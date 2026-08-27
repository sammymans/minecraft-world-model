# Tiny Minecraft Latent World Model

This is a deliberately small learning project about the core idea behind world
models:

> Compress an observation into a latent state, predict how an action changes
> that state, and decode the predicted state so we can see the imagined future.

The complete project has four parts:

1. a Minecraft data recorder;
2. a sequence dataset pipeline;
3. a tiny latent world model; and
4. an interactive imagined rollout.

We will first prove that the pipeline works using a small subset of the public
OpenAI VPT Minecraft dataset. Once the model can learn from those synchronized
videos and actions, we will build our own recorder and feed its output through
the same pipeline.

There is no agent, planning system, MPC, robotics layer, or high-resolution
video generator in scope.

The canonical project plan is [docs/PROJECT.md](docs/PROJECT.md). The hands-on
lessons are:

1. [Public frames and actions](docs/LEARNING_01_PUBLIC_DATA.md)
2. [Cleaning and sequence datasets](docs/LEARNING_02_DATASET.md)
3. [Visual autoencoder](docs/LEARNING_03_AUTOENCODER.md)
4. [Local dataset pipeline](docs/LEARNING_04_LOCAL_DATA_PIPELINE.md)
5. [Action-conditioned latent dynamics](docs/LEARNING_05_LATENT_DYNAMICS.md)
6. [Multi-step recursive evaluation](docs/LEARNING_06_MULTI_STEP_EVALUATION.md)
7. [Interactive latent rollout](docs/LEARNING_07_INTERACTIVE_ROLLOUT.md)

Measured scaling results are tracked in
[docs/RESULTS_DATA_SCALING.md](docs/RESULTS_DATA_SCALING.md).

## Play the world model

Open the interactive rollout viewer using the selected V2 checkpoints:

```bash
uv run mcwm play-rollout
```

The viewer begins with two real held-out seed frames. After that, every image is
recursively imagined from your controls; press `N` for one step or `P` to run.
See [Learning 07](docs/LEARNING_07_INTERACTIVE_ROLLOUT.md) for the controls and
scripted mode.

## Reproduce the local dataset

The committed manifests define exact public episode pairs and group-safe
splits; large data files remain ignored locally. The current larger experiment
uses `vpt_v2`.

```bash
uv run mcwm dataset-download --manifest data/manifests/vpt_v2.jsonl
uv run mcwm dataset-preprocess \
  --manifest data/manifests/vpt_v2.jsonl \
  --output-dir data/processed/vpt_v2
uv run mcwm dataset-verify \
  --manifest data/manifests/vpt_v2.jsonl \
  --processed-dir data/processed/vpt_v2
uv run mcwm dataset-summary \
  --manifest data/manifests/vpt_v2.jsonl \
  --processed-dir data/processed/vpt_v2
```

## Try the first data pipeline

```bash
uv sync
uv run mcwm download-demo
uv run mcwm inspect-demo
uv run mcwm show-action 100
uv run mcwm make-preview --start 3 --duration 15
open artifacts/vpt-preview.mp4
```

The preview is real Minecraft footage with the synchronized keyboard and mouse
action drawn on each frame. It is not a model prediction yet.

## Current status

The verified `vpt_v2` pipeline has 173 episodes, 12.81 training hours, and
148,069 clean one-step examples. The selected 1.4-million-parameter autoencoder
compresses each frame into 256 values and reaches 28.27 dB on the frozen
held-out session. Its paired action-conditioned dynamics model beats decoded
copy by 17.0%; shuffling actions worsens pixel MSE by 26.9%. Recursive
predictions beat frozen copy through 20 held-out steps, although they become
blurred and action influence weakens by two seconds. The next milestone is the
V3 scaling comparison and optional short-horizon rollout training. The first
interactive viewer is implemented and uses the same recursive loop as the
offline evaluator.
