# Data-Scaling Experiment Log

## Why this file exists

Every training command writes an exact `metrics.json` beside its checkpoint,
but `artifacts/` is intentionally ignored by Git. This committed log preserves
the selected results and the experimental protocol as the local dataset grows.

The question is not merely whether training loss falls. We want to measure:

$$
N_{data}\uparrow
\quad\Longrightarrow\quad
\text{held-out prediction improves and actions matter more}
$$

## Rules for a fair scaling experiment

1. Keep the validation session unchanged across dataset sizes.
2. Change only one major factor in the controlled comparison.
3. Never pair dynamics with a different autoencoder checkpoint.
4. Evaluate copy and shuffled-action controls, not only learned error.
5. Preserve every run in a distinct artifact directory.
6. Record failed and neutral results as well as improvements.

The frozen validation set currently contains:

```text
1 independent player/session group
2 consecutive episodes
978 autoencoder validation frames
1,034 one-step dynamics examples
618 clean eight-step sequences
```

This fixed set makes the existing rows directly comparable. It is sufficient
for the present learning experiment, although a future generalization study
should add several new held-out groups as a second test set.

## Dataset scale

| dataset | raw size | training hours | training frames | one-step examples | eight-step sequences |
|---|---:|---:|---:|---:|---:|
| `vpt_v1` | 2.16 GiB | 0.89 | 31,936 | 7,314 | 5,605 |
| `vpt_v2` | 25.14 GiB | 12.81 | 461,191 | 148,069 | 117,348 |

`vpt_v2` therefore supplies approximately 20.2 times as many clean one-step
examples as `vpt_v1`.

## Autoencoder results

Both rows use the same architecture: 256 latent values and 1,396,835 trainable
parameters.

| run | training frames | validation L1 | validation MSE | validation PSNR |
|---|---:|---:|---:|---:|
| `artifacts/autoencoder` | 27,157 | 0.025175 | 0.002126 | 26.72 dB |
| `artifacts/autoencoder-v2` | 392,924 | **0.021275** | **0.001491** | **28.27 dB** |

The larger visual dataset lowers reconstruction MSE by approximately 30%
without increasing the latent dimension.

## One-step dynamics results

The fair decoded-copy improvement is:

$$
100\left(1-
\frac{\operatorname{MSE}_{model}}
{\operatorname{MSE}_{decoded\ copy}}
\right)
$$

Action sensitivity is measured by replacing every action with an action from a
different validation example:

$$
100\left(
\frac{\operatorname{MSE}_{shuffled}}
{\operatorname{MSE}_{correct}}-1
\right)
$$

| run | learned pixel MSE | copy improvement | shuffled-action latent penalty | shuffled-action pixel penalty |
|---|---:|---:|---:|---:|
| v1 data + old encoder | 0.006077 | 2.6% | 0.2% | 0.4% |
| v2 data + old encoder | **0.005044** | **19.2%** | **24.0%** | **33.5%** |
| v2 data + new encoder | 0.005070 | 17.0% | 13.1% | 26.9% |

The most important controlled comparison is the first two rows because only the
dynamics training data changed. More data turns a nearly action-insensitive
model into one whose error rises substantially when the controls are wrong.

The new encoder has a better decoder-oracle MSE, $0.001417$ instead of
$0.002027$, and its dynamics prediction has lower pixel L1. Its one-step pixel
MSE is within 0.5% of the old-encoder dynamics model. We select the new pair for
recursive evaluation while retaining the old pair as evidence about data
scaling.

## Exact checkpoint pairings

```text
v1 baseline:
  artifacts/autoencoder/best.pt
  artifacts/dynamics/best.pt

controlled data-only result:
  artifacts/autoencoder/best.pt
  artifacts/dynamics-v2-old-ae/best.pt

selected recursive-rollout foundation:
  artifacts/autoencoder-v2/best.pt
  artifacts/dynamics-v2-new-ae/best.pt
```

Each dynamics checkpoint stores the SHA-256 fingerprint of its autoencoder.
Evaluation refuses a mismatched pair.

## What to record for the next scale

When expanding beyond 25 GiB, create a superset of the existing manifest:

```bash
uv run mcwm dataset-expand-manifest \
  --base-manifest data/manifests/vpt_v2.jsonl \
  --output data/manifests/vpt_v3.jsonl \
  --target-gib 50 \
  --seed 7
```

Then run two experiments:

1. freeze `artifacts/autoencoder-v2/best.pt` and train dynamics on `vpt_v3`;
2. train a new autoencoder on `vpt_v3`, then train its paired dynamics model.

The first isolates dynamics-data scaling. The second measures the complete
pipeline at the new scale. Add both rows here, along with:

- manifest SHA-256;
- raw size, training hours, frames, and clean examples;
- autoencoder L1, MSE, and PSNR;
- learned, decoded-copy, and decoder-oracle pixel error;
- correct-versus-shuffled action error; and
- multi-step errors at horizons 1, 2, 5, 10, and 20 once implemented.

## Current conclusion

The first scaling step is a clear positive result. The main improvement came
from increasing action-labelled dynamics examples, not increasing network size:

$$
7{,}314\longrightarrow148{,}069
$$

The next scientific question is whether this one-step improvement survives
recursive use. That protocol is defined in
`LEARNING_06_MULTI_STEP_EVALUATION.md`.
