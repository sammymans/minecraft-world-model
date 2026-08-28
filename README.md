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

The canonical V1 project plan is [docs/PROJECT.md](docs/PROJECT.md). The approved
[V2 latent-diffusion plan](docs/V2_ACTION_CONDITIONED_LATENT_DIFFUSION.md)
targets a more recognizable visual demo while preserving V1. The hands-on
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
10. [Multi-step dynamics training](docs/LEARNING_10_MULTI_STEP_TRAINING.md)
11. [Rejected EDM diffusion pilot](docs/LEARNING_11_EDM_DIFFUSION.md)
12. [Rejected discrete-tokenizer pilot](docs/LEARNING_12_DISCRETE_TOKENIZER.md)
13. [Latent-video flow refinement](docs/LEARNING_13_LATENT_VIDEO_FLOW.md)

Measured scaling results are tracked in
[docs/RESULTS_DATA_SCALING.md](docs/RESULTS_DATA_SCALING.md).

## Play the world model

Open the interactive rollout viewer using the selected spatial checkpoints:

```bash
uv run mcwm play-rollout
```

The model initializes from two held-out seed frames, but the browser frontend
shows only one large current viewport. Every later image is recursively imagined
from your controls. Playback defaults to one step per second, and movement keys
advance once while paused; the speed control can restore 10 Hz. Use the seed
buttons to jump to another held-out Minecraft scene without restarting the model.
See [Learning 07](docs/LEARNING_07_INTERACTIVE_ROLLOUT.md) for the controls and
scripted mode.

## Reproduce the local dataset

The committed manifests define exact public episode pairs and group-safe
splits; large data files remain ignored locally. `vpt_v4.jsonl` defines the
downloaded 100 GiB inventory, while `vpt_v4_split.jsonl` assigns its complete
sessions to training, validation, and test.

```bash
uv run mcwm dataset-download --manifest data/manifests/vpt_v4.jsonl
uv run mcwm dataset-preprocess \
  --manifest data/manifests/vpt_v4.jsonl \
  --output-dir data/processed/vpt_v4
uv run mcwm dataset-verify \
  --manifest data/manifests/vpt_v4_split.jsonl \
  --processed-dir data/processed/vpt_v4
uv run mcwm dataset-summary \
  --manifest data/manifests/vpt_v4_split.jsonl \
  --processed-dir data/processed/vpt_v4
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

The verified `vpt_v4` pipeline contains 707 episodes from 705 independent
player/session groups and 100.03 GiB of synchronized raw data. Its deterministic
group-safe split is:

| split | groups | episodes | raw size | clean eight-step sequences |
|---|---:|---:|---:|---:|
| training | 565 | 566 | 80.35 GiB | 398,354 |
| validation | 70 | 71 | 9.85 GiB | 52,331 |
| test | 70 | 70 | 9.82 GiB | 49,906 |

Validation is used for training decisions; test remains untouched until a model
is selected. The original narrow held-out session remains in validation because
it has already been used during model development; the test sessions are new.
The 253,395-parameter spatial autoencoder reaches 37.46 dB, L1 0.00674, and a
0.974 edge-energy ratio on the frozen held-out episodes, versus 28.27 dB and
L1 0.02128 for the old flat-latent model. A final evaluation over 61,831 frames
from the new 70-session test split remains strong at 36.82 dB, L1 0.00782, and
a 0.969 edge-energy ratio.

The selected 255,376-parameter spatial dynamics checkpoint trained on 100,000
V4 transitions. On 62,646 transitions from the final 70-group test split, it
reaches latent MSE 0.003874 against 0.007246 for copy and pixel L1 0.03069
against 0.03749 for decoded copy. Shuffling actions raises latent MSE to
0.005700, so controls are load-bearing.

On 5,000 recursive test windows, the model beats frozen copy through every
measured horizon from 1 to 20 steps. At 20 steps (2 seconds), pixel MSE is
0.02364 versus 0.02985 for copy, and mismatched actions are 22.9% worse. Images
still become smooth after several recursive steps; this is a small deterministic
world model, not a crisp video generator. The spatial checkpoint now powers the
local browser rollout. See [Learning 09](docs/LEARNING_09_SPATIAL_DYNAMICS.md).

Try it with `uv run mcwm play-rollout`. The recorder remains last.
