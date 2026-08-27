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
8. [Spatial representation redesign](docs/LEARNING_08_SPATIAL_AUTOENCODER.md)

Measured scaling results are tracked in
[docs/RESULTS_DATA_SCALING.md](docs/RESULTS_DATA_SCALING.md).

## Play the world model

Open the interactive rollout viewer using the reproducible V1 checkpoints:

```bash
uv run mcwm play-rollout
```

The model initializes from two held-out seed frames, but the browser frontend
shows only one large current viewport. Every later image is recursively imagined
from your controls. Hold a movement key and the model automatically advances at
10 Hz; no separate step or play command is required.
See [Learning 07](docs/LEARNING_07_INTERACTIVE_ROLLOUT.md) for the controls and
scripted mode.

## Reproduce the local dataset

The committed manifests define exact public episode pairs and group-safe
splits; large data files remain ignored locally. The current larger experiment
uses `vpt_v3`.

```bash
uv run mcwm dataset-download --manifest data/manifests/vpt_v3.jsonl
uv run mcwm dataset-preprocess \
  --manifest data/manifests/vpt_v3.jsonl \
  --output-dir data/processed/vpt_v3
uv run mcwm dataset-verify \
  --manifest data/manifests/vpt_v3.jsonl \
  --processed-dir data/processed/vpt_v3
uv run mcwm dataset-summary \
  --manifest data/manifests/vpt_v3.jsonl \
  --processed-dir data/processed/vpt_v3
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

The verified `vpt_v3` pipeline has 345 episodes, 343 training episodes, 245,087
clean eight-step training sequences, and 781,732 usable representation frames.
The first flat-latent world model remains reproducible, but its interactive
rollout exposed immediate decoder blur and weak action influence. We therefore
stopped scaling that model and replaced its 256-value flat representation with
a 253,395-parameter spatial autoencoder whose latent shape is
$16\times16\times16$. The selected 100,000-frame checkpoint reaches 37.46 dB,
L1 0.00674, and a 0.974 edge-energy ratio on the frozen held-out episodes,
versus 28.27 dB and L1 0.02128 for the old model. The next step is to train and
evaluate spatial action-conditioned dynamics, then reconnect the browser
frontend. The recorder remains last.
