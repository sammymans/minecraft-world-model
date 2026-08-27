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
9. [Spatial action-conditioned dynamics](docs/LEARNING_09_SPATIAL_DYNAMICS.md)

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
The 253,395-parameter spatial autoencoder reaches 37.46 dB, L1 0.00674, and a
0.974 edge-energy ratio on the frozen held-out episodes, versus 28.27 dB and
L1 0.02128 for the old flat-latent model.

Spatial action-conditioned dynamics now exist and pass their acceptance gate. A
260,390-parameter model trained on a bounded 30,000-transition pilot reaches
0.003902 held-out latent MSE against a 0.007160 copy baseline, and 0.028623
pixel L1 against 0.033883 for decoded copy. Shuffling the actions raises the
error by 43% of the model's own error, so the controls are load-bearing rather
than decorative.

Two honest caveats. An ablation at matched data scale shows the architecture
changes are worth only 1-3%: **data scale, not architecture, drove the
improvement**, and the pilot used 30,000 of 245,087 available transitions. And
the prediction still carries only 59% of the real frame's edge energy, so blur
is reduced rather than solved. See
[Learning 09](docs/LEARNING_09_SPATIAL_DYNAMICS.md).

The next steps are an on-disk latent cache to train past the in-memory bound,
then the `--edge-weight` sharpness term, then recursive rollout, and only then
reconnecting the browser frontend. The recorder remains last.
