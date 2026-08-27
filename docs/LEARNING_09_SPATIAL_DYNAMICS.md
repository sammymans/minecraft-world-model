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
final test. The original held-out session is part of test, so it never enters
training. Historical V3 metrics above remain useful evidence, but they are not
directly comparable to the new multi-session validation numbers.

Checkpoints now store the exact spatial-dynamics architecture identifier. This
prevents an experimental checkpoint from being silently paired with reverted
model code. The old V3 artifact predates that identifier and must be retrained;
its recorded metrics and images remain experiment evidence, not the selected
runnable model.

## Running it

```bash
uv run mcwm train-spatial-dynamics \
  --processed-dir data/processed/vpt_v4 \
  --manifest data/manifests/vpt_v4_split.jsonl \
  --autoencoder-checkpoint artifacts/spatial-autoencoder-v3/best.pt \
  --output-dir artifacts/spatial-dynamics-v4 \
  --maximum-transitions 30000 --epochs 20
```

```bash
uv run mcwm evaluate-spatial-dynamics \
  --dynamics-checkpoint artifacts/spatial-dynamics-v4/best.pt \
  --output-dir artifacts/spatial-dynamics-v4/validation \
  --split validation
```

After architecture and training decisions are frozen, run the same command with
`--split test` and a separate output directory exactly once.

`--maximum-transitions` bounds the in-memory training latent cache. The next
infrastructure refinement should be a simple on-disk latent cache before trying
to consume all 398,354 eight-step V4 training sequences.

## What comes next

Retrain the versioned baseline on the new split, use validation for model
selection, then restore recursive spatial evaluation. Only a selected model is
evaluated on test or connected to the frontend.
