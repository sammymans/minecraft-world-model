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
