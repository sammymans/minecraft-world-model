# Learning 10: training on the model's own predictions

The selected V0 dynamics model is useful for one or two imagined steps, but its
frames become smooth after roughly two to five recursive steps. This lesson
targets that exact failure without changing the encoder or dynamics
architecture.

## Why one-step training and interactive use differ

The current model was trained on real latent pairs:

$$
\hat z_{t+1}=F_\phi(z_{t-1},z_t,a_t).
$$

At the next training example it receives real context again. The interactive
rollout cannot do that. It feeds the prediction back into the model:

$$
\hat z_{t+2}=F_\phi(z_t,\hat z_{t+1},a_{t+1}),
$$

$$
\hat z_{t+3}=F_\phi(\hat z_{t+1},\hat z_{t+2},a_{t+2}).
$$

This mismatch is often called **exposure bias**. A small error in
$\hat z_{t+1}$ creates a context the model did not see during one-step
training. The next prediction adds another error, and the latent trajectory
gradually leaves the data distribution.

The final-test measurements match the browser experience:

| recursive horizon | pixel PSNR | pixel MSE |
|---:|---:|---:|
| 1 step | 24.43 dB | 0.003608 |
| 5 steps | 20.03 dB | 0.009933 |
| 20 steps | 16.26 dB | 0.023640 |

## The smallest useful intervention

Keep the frozen spatial autoencoder and the same 255,376-parameter dynamics
network. Start from the selected V0 checkpoint, recursively unroll it for five
steps, and backpropagate through its own predicted states.

For step $k$:

$$
\hat z_{t+k}=F_\phi(\hat z_{t+k-2},\hat z_{t+k-1},a_{t+k-1}),
$$

where the first two inputs are the real seed latents $z_{t-1}$ and $z_t$.
The normalized latent error is:

$$
\mathcal L^{latent}_k=
\left\|
\frac{\hat z_{t+k}-z_{t+k}}{\sigma_z}
\right\|_2^2.
$$

The pixel-space term compares decoded predicted and target latents:

$$
\mathcal L^{pixel}_k=
\left\|D(\hat z_{t+k})-D(z_{t+k})\right\|_2^2.
$$

We combine all five steps with a decay $\lambda$ and normalize the weights:

$$
\mathcal L_{multi}=
\frac{1}{\sum_{k=1}^{5}\lambda^{k-1}}
\sum_{k=1}^{5}\lambda^{k-1}
\left(
\mathcal L^{latent}_k+\mathcal L^{pixel}_k
\right).
$$

With $\lambda=0.8$, early accuracy remains important while later predictions
still contribute substantial gradient.

## Why fine-tune instead of starting over

V0 already learned Minecraft one-step motion and meaningful action
conditioning from 100,000 transitions. Fine-tuning asks it to become robust to
its own errors. Starting from scratch would mix two questions—whether it can
learn one-step physics and whether recursive supervision helps—and would cost
more compute.

The first bounded experiment uses:

- five recursive steps;
- 30,000 training windows, or 150,000 predicted training steps per epoch;
- 5,000 broad validation windows;
- ten fine-tuning epochs with early stopping; and
- the same frozen spatial autoencoder and additive dynamics architecture.

Two settings the plan above did not fix, chosen when the run was launched and
recorded here so the result stays attributable:

- **learning rate 1e-4**, a third of V0's 3e-4, because this is a fine-tune of
  an already-trained model rather than a fresh run; and
- **gradient-norm clipping at 1.0**, because the loss now backpropagates
  through five chained applications of the same network.

## Honest comparison

The V0 checkpoint remains frozen. V1 is selected only with the validation
split and is compared on the identical validation windows at horizons
$1,2,5,10,20$.

V1 passes the experiment gate if:

1. five-step validation pixel MSE improves over V0's $0.011852$;
2. one-step validation error does not regress by more than 10%;
3. mismatched actions remain worse than correct actions; and
4. filmstrips remain recognizable for longer than V0.

Only after those decisions are frozen do we run V1 on the final test split.

## What we measured

Training selected epoch 7 of 10. The recursive validation objective moved very
little across the run—0.465758 at epoch 1 to 0.465089 at epoch 7, changes in
the fourth decimal—so almost all of the adaptation happened in the first epoch.
A flat curve on the training objective did not mean a flat rollout, which is
why the gate is measured with the rollout evaluator rather than the loss.

Both models were scored by `evaluate-rollout` on the identical 5,000 windows
(same manifest, same seed, same horizon). The copy and decoder-oracle baselines
match to six decimals across the two runs, which is what confirms the window
sets are the same.

| horizon | V0 validation | V1 validation | change | V0 test | V1 test | change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.004174 | 0.004179 | +0.1% | 0.003608 | 0.003616 | +0.2% |
| 2 | 0.006617 | 0.006444 | -2.6% | 0.005655 | 0.005500 | -2.7% |
| 5 | 0.011852 | **0.010948** | **-7.6%** | 0.009933 | **0.009273** | **-6.6%** |
| 10 | 0.017567 | 0.015623 | -11.1% | 0.014907 | 0.013302 | -10.8% |
| 20 | 0.026763 | **0.021114** | **-21.1%** | 0.023640 | **0.018877** | **-20.1%** |

The gain scales with horizon and is near zero at one step. That is the exact
signature exposure bias predicts: the further a rollout runs on its own
predictions, the more the correction is worth. Test and validation agree to
within a percentage point at every horizon.

The four gates:

1. **Pass.** Five-step validation MSE fell from 0.011852 to 0.010948.
2. **Pass.** One-step validation latent MSE rose 0.004292 to 0.004301, a 0.22%
   regression against a 10% budget; one-step pixel MSE actually improved 0.10%.
3. **Pass.** Mismatched actions stay worse at every horizon, and the gap grew.
   The mismatched-action penalty at 20 steps went from 19.5% to 34.6% on
   validation, and the one-step action effect rose 14.7%. A model trained to
   survive its own errors leans *harder* on the controls.
4. **Marginal.** Across the three filmstrip samples, two hold structure
   modestly longer and one is not clearly better. Real, but small.

## What this does not fix

Blur is still the dominant failure. V1 degrades more slowly; it does not stay
sharp. At 20 steps it reaches 17.24 dB on test, up a little under 1 dB from
16.26 dB, and both are far short of a recognizable frame. The interactive
rollout still becomes indistinct after several imagined steps.

So the honest reading is that exposure bias was *a* real contributor and not
the main one. Removing it bought a consistent, horizon-scaling improvement for
one fine-tuning run and no architecture change, which is a good trade. It did
not make the model a convincing interactive simulator.

That is the evidence the experiment was built to produce. The remaining blur is
attributable to deterministic continuous-latent regression—squared error is
minimized by the average over plausible futures, and that average is smooth—so
the next redesign is a discrete or probabilistic latent objective, as
[MineWorld](https://arxiv.org/abs/2504.08388) and
[Oasis](https://oasis-model.github.io/) both do.

## Getting the most out of the model we have

While measuring V1 we found that recursive coherence is governed by **how much
new content enters the frame per step**, not by how many steps have elapsed.
Three consequences, all measured on the selected checkpoint:

- **Camera speed dominates.** At a camera delta of 10 the rollout still shows
  the shoreline and structures at t+6. At 30 the same seed and script is
  indistinct by t+3. The browser previously sent raw pointer deltas, which at
  10 Hz reach 50-300 per step and are clamped at 500, so ordinary mouse
  movement was driving the model far outside the range where it stays coherent.
- **Standing still stays sharp.** An `idle` rollout holds crisp block edges for
  the whole horizon, because the prediction is close to the identity and there
  is nothing uncertain to average over. Blur is a function of motion.
- **Scene matters.** High-contrast outdoor scenes with large flat regions
  (water, grass, sky) survive noticeably longer than dark caves or fine detail,
  where low contrast makes any error read as mush.

The frontend now scales and caps pointer deltas the way an in-game sensitivity
slider would, and renders the 64x64 frame with nearest-neighbour upscaling
instead of the browser's default smoothing, which was blurring an already-soft
prediction a second time.

`compare-actions` renders one seed forward under several scripts, one per row,
which is the clearest single view of action conditioning:

```bash
uv run mcwm compare-actions --sample-index 20000 --camera-step 10 \
  --scripts 'look_left*6' 'look_right*6' 'w+sprint*6' 'idle*6'
```

None of this changes the model. It changes how the model is driven and shown,
and it is the difference between a rollout that reads as Minecraft and one that
reads as fog.

## Running it

The recursive objective is the same `train-spatial-dynamics` command with
`--rollout-steps` above 1. At `--rollout-steps 1` it is bit-for-bit the
one-step objective, so V0 stays reproducible from the same entry point.

```bash
uv run mcwm train-spatial-dynamics \
  --output-dir artifacts/spatial-dynamics-v4-multistep \
  --initial-checkpoint artifacts/spatial-dynamics-v4/best.pt \
  --rollout-steps 5 --horizon-decay 0.8 --gradient-clip 1.0 \
  --maximum-transitions 30000 --maximum-validation-sequences 5000 \
  --epochs 10 --batch-size 32 --learning-rate 1e-4
```

```bash
uv run mcwm evaluate-rollout --split validation
uv run mcwm evaluate-rollout --split test \
  --output-dir artifacts/spatial-rollout-v4-multistep/test
```

The selected V1 checkpoint is now the default for `play-rollout` and both
evaluators. V0 remains on disk and is still measurable by passing
`--dynamics-checkpoint artifacts/spatial-dynamics-v4/best.pt` explicitly.

Checkpoints record `rollout_steps`, `horizon_decay`, and `initial_checkpoint`,
so a recursively fine-tuned artifact cannot be mistaken for a one-step one.

## What this experiment does not add

It does not add a new encoder, larger network, diffusion model, service, or
experiment framework. It reuses the current data, checkpoint format, evaluator,
and browser. That keeps the result attributable and the code small.

## The training/inference horizon mismatch (2026-08-29)

The V1 data ablation measured a decomposition that reframes the blur problem.
On 5,000 held-out windows, the selected 100K checkpoint scores:

| horizon | recursive | teacher-forced |
|---:|---:|---:|
| 1 | 0.004165 | 0.004165 |
| 2 | 0.006466 | 0.004143 |
| 5 | 0.010986 | 0.004072 |
| 10 | 0.015631 | 0.003912 |
| 20 | 0.020934 | 0.003835 |

Teacher-forced error is flat, and slightly *decreasing*, across every horizon.
Given a real frame, the one-step model predicts as well at step 20 as at step 1.
All of the observed degradation is compounding feedback: each prediction is
marginally smoother than the real latent, that output becomes the next input,
and repeated application acts as a low-pass filter.

This bounds what more data can buy. At fixed compute, 50K to 100K transitions
improved one-step latent MSE by only 2.0% and left the wrong-action penalty
roughly flat, because more data sharpens each individual step rather than
changing how errors accumulate across twenty of them.

The immediate mismatch is that this model was fine-tuned to unroll **five**
steps while the interactive rollout and the 20-step evaluator run far longer.
The model never practises the regime it is asked to perform in. Extending the
recursive training horizon targets the measured failure directly and needs no
new architecture, data, or code.

### Planned rollout-horizon ablation

Three points, all fine-tuned from the same one-step checkpoint
(`artifacts/spatial-dynamics-v4/best.pt`) with identical 30,000 windows, ten
epochs, learning rate 1e-4, gradient clipping 1.0, and horizon decay 0.8:

| rollout steps | status | approximate cost |
|---:|---|---|
| 5 | already trained as `spatial-dynamics-v4-multistep` | ~8 minutes |
| 10 | to train | ~16 minutes |
| 20 | to train | ~32 minutes |

Unlike the data ablation, this study holds **windows and epochs** fixed rather
than optimizer compute. Horizon is a change to the training objective, not a
resource being traded, and per-window cost necessarily scales with the number of
chained applications. Holding compute fixed instead would have starved the long
horizons of data and reintroduced the confound the data ablation just removed.
This design assumes 30,000 windows is past the steep part of the data curve;
that is consistent with the measured 50K-to-100K saturation but is not proven
for the recursive objective specifically.

Window availability does not bound the study. Clean training windows number
435,478 at horizon 5, 377,245 at horizon 10, and 299,520 at horizon 20, so the
sampled 30,000 is far below supply at every point. The exact 30,000 differ
across horizons because the underlying valid-window sets differ; only the count,
seed, and schedule are matched.

Evaluation is the unchanged `evaluate-rollout` sweep at horizons 1, 2, 5, 10,
and 20 over 5,000 broad validation windows, so copy and decoder-oracle baselines
stay identical and the three points remain directly comparable.

Interpret the outcome as follows:

- 20-step error falls materially and one-step error does not regress: the
  mismatch was real and the demo improves with no architecture change;
- 20-step error falls but one-step error regresses: the objective trades
  near-term sharpness for long-horizon stability, and the correct horizon is a
  demo decision rather than a strict win;
- error is flat across horizons: recursive supervision is exhausted at this
  model size, and the remaining lever is structural. The next candidate is
  predicting a motion field and warping the previous latent instead of adding a
  regenerated residual, because warping relocates existing detail rather than
  re-synthesising it and therefore cannot smooth under repeated application.

### Rollout-horizon ablation result (2026-08-29)

Three checkpoints, each fine-tuned from `artifacts/spatial-dynamics-v4/best.pt`
with identical 30,000 windows, ten epochs, learning rate 1e-4, gradient clip
1.0, and horizon decay 0.8. Only `--rollout-steps` differed. All three were
scored by one `evaluate-rollout` sweep over 5,000 broad validation windows;
copy and decoder-oracle baselines match to six decimals, confirming a single
shared window set.

Recursive pixel MSE:

| horizon | train@5 | train@10 | train@20 | 10 vs 5 | 20 vs 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.004179 | 0.004204 | 0.004254 | +0.6% | +1.8% |
| 2 | 0.006444 | 0.006451 | 0.006564 | +0.1% | +1.9% |
| 5 | 0.010948 | 0.010866 | 0.011075 | -0.7% | +1.2% |
| 10 | 0.015623 | 0.015232 | 0.015569 | -2.5% | -0.3% |
| 20 | 0.021114 | **0.019788** | 0.020399 | **-6.3%** | -3.4% |

Mismatched-action pixel penalty, where higher means the controls carry more of
the prediction:

| horizon | train@5 | train@10 | train@20 |
|---:|---:|---:|---:|
| 1 | 109.4% | 107.3% | 103.1% |
| 5 | 87.9% | 87.6% | 85.8% |
| 10 | 60.3% | 61.7% | 62.6% |
| 20 | 34.6% | **39.0%** | 40.4% |

**The training horizon has an optimum near ten steps, and the relationship is
not monotonic.** Training at ten steps reduces 20-step error 6.3% for a
negligible 0.6% one-step cost, and raises the 20-step action penalty from 34.6%
to 39.0%. Training at twenty steps is *worse* than ten on error: it pays 1.8%
at one step and returns only 3.4% at twenty. Longer-horizon supervision keeps
improving action reliance monotonically, but its accuracy benefit peaks and
then reverses, so "train on the horizon you will deploy at" is not the right
rule. This matches the second interpretation branch stated above.

**The gain is real but not visible.** The winning ten-step filmstrip is
indistinguishable from the five-step one by eye: the desert seed still washes
to a flat gradient by `t+20` and the underwater seed still dissolves. A 6%
error reduction does not change what a person sees.

That is the important conclusion, because it can now be triangulated:

| lever | best gain at 20 steps | visible |
|---|---:|---|
| training transitions, 10K to 50K | -8.8% | no |
| training horizon, 5 to 10 steps | -6.3% | no |

Two independent levers, each moved across a wide range under matched
conditions, both return single-digit improvements and neither changes the
rendered result. Neither data volume nor the recursive objective is the
binding constraint. What remains is the formulation: the model regenerates the
whole next latent as an additive residual, so each pass is slightly smoother
than the truth and twenty passes compound into a low-pass filter. Predicting a
motion field and warping the previous latent would relocate existing detail
rather than re-synthesising it, and therefore cannot smooth under repeated
application.

Two axes remain untested and are cheaper than the rebuild: model **capacity**
(the dynamics network is 255,376 parameters, smaller than the autoencoder that
feeds it) and the training **distribution** (the cleaning filter rejects every
transition where attack is held, which is the majority of contractor gameplay
and biases the surviving data toward uneventful frames).

Artifacts are under `artifacts/v1-horizon-ablation/`. The ten-step checkpoint
is better than the deployed five-step one at every horizon at or above five,
with stronger action reliance and no meaningful one-step regression, so it is
the better demo default irrespective of what follows.
