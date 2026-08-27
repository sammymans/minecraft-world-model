# Learning 05: Action-Conditioned Latent Dynamics

## Where this fits

This lesson implements canonical Milestone 4. Learning 04 added the reproducible
local data pipeline, so the lesson number and milestone number are now offset by
one.

The autoencoder answers a static question:

$$
o_t \longrightarrow e_t \longrightarrow \hat o_t
$$

The dynamics model adds time and control:

$$
(e_{t-1}, e_t, a_t) \longrightarrow \hat e_{t+1}
$$

It is the first component that tries to predict what happens after the player
takes an action.

## Why the autoencoder is frozen

The meaning of every latent coordinate is defined by one exact autoencoder
checkpoint. Retraining that autoencoder can rotate, rescale, add, or remove
latent features. A dynamics model trained against one checkpoint therefore
cannot safely be paired with another checkpoint, even when both use the same
latent dimension.

Dynamics checkpoints record the SHA-256 fingerprint of their autoencoder. The
evaluation command refuses to combine checkpoints whose fingerprints differ.
This turns a subtle experimental mistake into an explicit error.

The encoder and decoder parameters remain frozen during dynamics training. The
encoder is run once over each episode and its latents are cached in memory. The
decoder still participates in the computation graph so pixel loss can teach the
dynamics network, but the decoder weights themselves do not change.

## Exact temporal alignment

Each one-step example comes from three consecutive clean frames and the action
between the last two:

```text
o_(t-1) ---- a_(t-1) ----> o_t ---- a_t ----> o_(t+1)
   |                         |                   |
   v                         v                   v
 e_(t-1)                    e_t               e_(t+1)
                              \                 ^
                               \---- a_t -------/
```

Both transitions must pass the cleaning rules. The first transition provides
motion context; the second is the event being predicted. No example crosses an
episode boundary.

## Model structure

A single frame cannot reveal velocity. The network therefore receives the
current latent, the observed latent change, and the normalized raw action:

$$
x_t = [\operatorname{LN}(e_t),
       \operatorname{LN}(e_t-e_{t-1}),
       (a_t-\mu_a)/\sigma_a]
$$

A small MLP predicts a residual rather than a completely new state:

$$
\hat e_{t+1}=e_t+\operatorname{MLP}(x_t)
$$

The final layer begins at zero, so the untrained network exactly implements the
copy-latent baseline. Training must improve on that honest starting point.

The seven key controls are already bounded, but mouse deltas can be hundreds of
units. Action mean and standard deviation are fitted only on the training split
and stored inside the model. A small standard-deviation floor prevents rare
binary controls from being amplified excessively.

## Training objective

The first implementation combines latent and decoded-image losses:

$$
\mathcal L =
\lambda_{latent}\lVert\hat e_{t+1}-e_{t+1}\rVert_2^2
+
\lambda_{pixel}\lVert D(\hat e_{t+1})-o_{t+1}\rVert_2^2
$$

Latent loss asks the prediction to land near the encoder's target. Pixel loss
asks that prediction to decode into the real next image. Both weights default to
one and are recorded with the checkpoint.

## Run with the selected visual checkpoint

Use a distinct autoencoder artifact directory so a running experiment never
overwrites the checkpoint that defines a dynamics run:

```bash
uv run mcwm train-dynamics \
  --processed-dir data/processed/vpt_v2 \
  --manifest data/manifests/vpt_v2.jsonl \
  --autoencoder-checkpoint artifacts/autoencoder-v2/best.pt \
  --output-dir artifacts/dynamics-v2-new-ae
```

The latent dimension is read from the checkpoint; there is no separate
`--latent-dim` option that can accidentally disagree.

Recreate held-out measurements and visuals independently:

```bash
uv run mcwm evaluate-dynamics \
  --processed-dir data/processed/vpt_v2 \
  --manifest data/manifests/vpt_v2.jsonl \
  --autoencoder-checkpoint artifacts/autoencoder-v2/best.pt \
  --dynamics-checkpoint artifacts/dynamics-v2-new-ae/best.pt \
  --output-dir artifacts/dynamics-v2-new-ae-eval
```

Training writes:

```text
artifacts/dynamics-v2-new-ae/best.pt
artifacts/dynamics-v2-new-ae/metrics.json
artifacts/dynamics-v2-new-ae/training-curve.png
artifacts/dynamics-v2-new-ae/one-step-predictions.png
```

## What counts as evidence

The evaluator reports three controlled comparisons on held-out episodes:

1. learned prediction versus the real next latent and frame;
2. copying the current latent/frame versus the same target; and
3. using the correct action versus actions shuffled across examples.

The important gates are not training loss alone:

- learned held-out error should beat the copy baseline;
- shuffled actions should increase error;
- predictions under different actions should measurably differ; and
- real-versus-predicted images should show coherent next-frame changes.

Passing those gates completes Milestone 4. Recursive use of the model belongs to
Milestone 5, where errors are measured over progressively longer open-loop
rollouts.

## Controlled data-scaling results

All rows use the exact same 1,034 transitions from the frozen held-out session.
`Decoded copy` means decoding $e_t$ and treating it as the next frame; this is
the fair baseline because it passes through the same decoder as the learned
prediction.

| experiment | learned pixel MSE | decoded-copy MSE | improvement | shuffled-action penalty |
|---|---:|---:|---:|---:|
| v1 data, old encoder | 0.006077 | 0.006239 | 2.6% | 0.2% latent |
| v2 data, old encoder | **0.005044** | 0.006239 | **19.2%** | **24.0% latent / 33.5% pixel** |
| v2 data, new encoder | 0.005070 | 0.006108 | 17.0% | 13.1% latent / 26.9% pixel |

The old-encoder comparison isolates the effect of data. Increasing clean
one-step examples from 7,314 to 148,069 turns the shuffled-action test from an
almost-zero 0.2% penalty into a 24.0% penalty. The learned model also beats
copying by about 19%. This is direct held-out evidence that the network now uses
the action instead of merely extrapolating visual similarity.

The new encoder lowers its decoder-oracle MSE from 0.002027 to 0.001417 and
produces visibly clearer reconstructions. Its paired dynamics model has nearly
the same one-step pixel MSE, lower pixel L1, and still a strong 26.9% pixel
penalty when actions are shuffled. We select this pair for the interactive path
because it retains action control while providing the stronger visual state.

The checkpoint pairs must not be crossed:

```text
controlled data-only comparison:
  artifacts/autoencoder/best.pt
  artifacts/dynamics-v2-old-ae/best.pt

selected interactive foundation:
  artifacts/autoencoder-v2/best.pt
  artifacts/dynamics-v2-new-ae/best.pt
```

The next milestone is recursive evaluation. One-step success does not establish
that predictions remain coherent when fed back into the model:

$$
\hat e_{t+k+1}=F(\hat e_{t+k-1},\hat e_{t+k},a_{t+k})
$$

We will measure horizons of 1, 2, 5, 10, and 20 steps before building the
keyboard-controlled viewer.
