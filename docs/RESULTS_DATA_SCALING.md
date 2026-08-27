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
| `vpt_v3` | 50.05 GiB | — | 781,732 | — | 245,087 |

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

### Spatial representation redesign

The V1 interactive rollout showed that the flat 256-value representation was
too destructive: a decoder oracle supplied with the real next latent was
already blurry. We therefore changed both architecture and data scale. These
rows are not a controlled data-only comparison; they answer whether the new
representation is a better foundation for future dynamics.

| run | representation | sampled training frames | validation L1 | validation MSE | validation PSNR | edge-energy ratio |
|---|---|---:|---:|---:|---:|---:|
| flat V2 checkpoint | 256 values | 392,924 | 0.021275 | 0.001491 | 28.27 dB | not recorded |
| spatial V3 pilot | $16\times16\times16$ | 20,000 | **0.016133** | **0.000934** | **30.30 dB** | 0.711 |
| spatial V3 selected | $16\times16\times16$ | 100,000 | **0.006743** | **0.000180** | **37.46 dB** | **0.974** |

The pilot also passed a 32-frame memorization gate with L1 $0.006326$ and an
edge-energy ratio of $0.896$. The selected result trained for 12 epochs; its
training L1 is $0.006390$, close to held-out L1. See
`LEARNING_08_SPATIAL_AUTOENCODER.md` for the architecture, loss, failed first
attempt, and interpretation.

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

caffeinate -i uv run mcwm dataset-download \
  --manifest data/manifests/vpt_v3.jsonl \
  --workers 3
```

The target is 50 GiB in total, so this adds approximately 25 GiB to the existing
V2 download. Existing complete files are detected and skipped.

The V3 dataset is now downloaded and verified. The flat-latent scaling plan was
superseded after decoder-oracle and interactive tests identified the
representation itself as a bottleneck. The active experiment trains the spatial
autoencoder on V3, followed by its paired spatial dynamics model. Record:

- manifest SHA-256;
- raw size, training hours, frames, and clean examples;
- autoencoder L1, MSE, and PSNR;
- learned, decoded-copy, and decoder-oracle pixel error;
- correct-versus-shuffled action error; and
- multi-step errors at horizons 1, 2, 5, 10, and 20.

## Current conclusion

The first scaling step was a clear positive result. The main V1 dynamics
improvement came from increasing action-labelled examples, not network size:

$$
7{,}314\longrightarrow148{,}069
$$

That improvement survived recursive evaluation, but interactive use exposed
visual blur and weak control more clearly than aggregate error did. The spatial
autoencoder pilot improves held-out reconstruction substantially; spatial
dynamics has not yet been trained, so we do not yet claim the redesigned world
model works. The evaluation protocol remains the one in
`LEARNING_06_MULTI_STEP_EVALUATION.md`.

## Multi-step baseline for future data scales

The selected `vpt_v2` pair was recursively evaluated on 288 held-out starts
that each have a clean 20-step future:

| horizon | recursive pixel MSE | copy improvement | mismatched-action penalty |
|---:|---:|---:|---:|
| 1 | 0.009323 | 20.4% | 33.7% |
| 2 | 0.013902 | 24.0% | 38.5% |
| 5 | 0.023579 | 23.7% | 33.8% |
| 10 | 0.033395 | 25.7% | 30.7% |
| 20 | 0.044008 | 21.6% | 10.0% |

These values are the baseline for `vpt_v3`. Future rollout comparisons must
retain the same frozen validation group, maximum horizon, and 288-window index.
The horizon-1 value is not directly comparable to the earlier one-step result:
this table deliberately uses only starting points that also possess a clean
20-step future, keeping the sample set identical across every horizon.
