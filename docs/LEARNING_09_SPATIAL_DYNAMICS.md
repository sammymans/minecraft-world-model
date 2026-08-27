# Learning 09: spatial action-conditioned dynamics

Learning 08 fixed the *representation*. This lesson builds the piece that
consumes it:

$$
\hat e_{t+1}=F_\phi(e_{t-1},e_t,a_t),
\qquad e\in\mathbb{R}^{16\times16\times16},
\qquad a\in\mathbb{R}^{9}.
$$

The 255,376-parameter network broadcasts the action across the grid as nine
extra channels and predicts a residual change, initialized to zero so training
starts exactly at the copy baseline and has to earn every change.

## The pilot result

30,000 bounded transitions from `vpt_v3`, 87,375 encoded frames, 20 epochs,
evaluated on the frozen held-out transitions:

| measurement | prediction | decoded copy | decoder oracle |
|---|---:|---:|---:|
| latent MSE | **0.003941** | 0.007160 | 0 |
| pixel L1 | **0.029529** | 0.033883 | 0.006611 |

Shuffling the actions raises the latent error by 37.6% of the model's own
error, so the controls are load-bearing rather than decorative.

## What we learned trying to improve it

Three things worth recording, because each cost a real experiment.

**Data beats architecture, by a lot.** The same model trained on 256
transitions loses to decoded copy in pixel space and has a 2.7% action effect.
At 30,000 transitions it wins and reaches 37.6%. Adding FiLM conditioning and
an explicit action-conditioned warp to the architecture moved one-step error by
only 1-3% at matched data scale. The 256-transition failure was evidence about
data, not about architecture.

**An overfit gate only proves something if the target count is below the
parameter count.** Each transition carries $16\cdot16\cdot16=4{,}096$ target
values, so 256 transitions is over a million numbers against 255,376
parameters. A plateau there is arithmetic, not a bug. At 16 transitions the
model drives training loss from 0.754 to 0.0037.

**Sharpness is set by the objective, not the architecture.** The prediction
carries about 59% of the real frame's edge energy, and the decoder is not at
fault - handing it a real latent gives 97%. Two structurally different dynamics
models both land at 59%, because squared error is minimized by the average over
plausible futures and that average is smooth. An image-gradient penalty does
not fix this: it is still a regression loss, minimized by the mean of the
gradient field, and measurably made rollout sharpness worse. The literature
agrees - [MineWorld](https://arxiv.org/abs/2504.08388) uses discrete tokens
with next-token prediction and [Oasis](https://oasis-model.github.io/) uses
diffusion, both modeling a distribution rather than a point estimate.

## The measurement problem

Every number above rests on two held-out episodes from a single player,
recorded five minutes apart. `sprint` is *constant* across the entire held-out
set, and an inverse-dynamics probe scored 0.92 on the training split against
-0.32 held out. The split is group-safe and leak-free, but it is one draw and
too narrow to support fine-grained comparisons. Widening it is a prerequisite
for trusting any further tuning.

That prerequisite is now resolved by `vpt_v4_split.jsonl`: 70 independent
groups are used for validation and a different 70 groups are reserved for the
final test. The original held-out session remains in validation because earlier
experiments already used it for model development. Historical V3 metrics above
remain useful evidence, but they are not
directly comparable to the new multi-session validation numbers.

## The selected V4 result

The versioned additive model trained for 20 epochs on 100,000 transitions
sampled across the 565 training groups. No test session was used during
training or checkpoint selection.

| split | latent MSE | copy latent MSE | pixel L1 | decoded-copy L1 | shuffled latent MSE |
|---|---:|---:|---:|---:|---:|
| validation | **0.004292** | 0.008068 | **0.032997** | 0.040351 | 0.006401 |
| final test | **0.003874** | 0.007246 | **0.030689** | 0.037488 | 0.005700 |

The final test result confirms that the model beats copy and depends on the
action outside the data used to select it.

Recursive evaluation uses the model's own prediction as its next input. On
5,000 final-test windows:

| horizon | seconds | recursive pixel MSE | copy improvement | mismatched-action penalty |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 0.003608 | 49.0% | 108.3% |
| 2 | 0.2 | 0.005655 | 48.9% | 103.7% |
| 5 | 0.5 | 0.009933 | 45.4% | 79.3% |
| 10 | 1.0 | 0.014907 | 38.3% | 50.3% |
| 20 | 2.0 | 0.023640 | 20.8% | 22.9% |

The model passes the quantitative rollout gate at every measured horizon.
Visually, scene layout remains interpretable for short horizons but becomes
smooth after repeated feedback. That is the current model limitation, not a UI
problem.

Checkpoints now store the exact spatial-dynamics architecture identifier. This
prevents an experimental checkpoint from being silently paired with reverted
model code. The old V3 artifact predates that identifier and cannot be loaded;
its replacement is the selected V4 checkpoint reported above.

## Running it

```bash
uv run mcwm train-spatial-dynamics \
  --processed-dir data/processed/vpt_v4 \
  --manifest data/manifests/vpt_v4_split.jsonl \
  --autoencoder-checkpoint artifacts/spatial-autoencoder-v3/best.pt \
  --output-dir artifacts/spatial-dynamics-v4 \
  --maximum-transitions 100000 --epochs 20
```

```bash
uv run mcwm evaluate-spatial-dynamics \
  --dynamics-checkpoint artifacts/spatial-dynamics-v4/best.pt \
  --output-dir artifacts/spatial-dynamics-v4/validation \
  --split validation
```

Measure recursive validation rollouts with:

```bash
uv run mcwm evaluate-rollout --split validation
```

Launch the selected checkpoint in the browser with:

```bash
uv run mcwm play-rollout --sample-index 20000
```

`--maximum-transitions` bounds the in-memory training latent cache. The next
infrastructure refinement should be a simple on-disk latent cache before trying
to consume all 398,354 eight-step V4 training sequences.

## What comes next

The V0 world-model loop is complete: representation, action-conditioned
dynamics, recursive evaluation, and interactive rollout all share the selected
spatial checkpoint. The next model improvement would address deterministic
blur, likely with a discrete or probabilistic latent objective. Per the project
scope, the next pipeline milestone is the Minecraft recorder.
