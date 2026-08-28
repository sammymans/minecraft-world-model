# Learning 13: latent-video flow refinement

## Question

Can a small flow-matching model make V1's eight-frame forecasts look sharper
without replacing its learned action-conditioned dynamics?

The experiment follows the rectified-flow form of conditional Flow Matching:
interpolate between Gaussian noise and a target latent correction, regress the
constant conditional velocity, and integrate the learned field at inference.
Unlike the earlier standalone diffusion pilot, this model conditions on the
complete V1 forecast and generates only a correction to it.

## Bounded implementation

- frozen 64x64 spatial autoencoder and frozen V1 multi-step dynamics;
- two context latents, eight V1 future latents, and eight actions;
- a 4.8M-parameter convolutional flow network;
- 4,000 training clips, 256 held-out validation clips, and five epochs on MPS;
- eight Heun sampling steps; and
- classifier-free action dropout with guidance at sampling time.

The live adapter plays the jointly generated clip while an action remains held
and replans when the action changes. V1 remains the default browser checkpoint.

## Verification

The flow equations have an exact perfect-velocity regression test, the training
objective memorizes a fixed clip, checkpoint loading and deterministic reset are
tested, and the complete suite has 71 passing tests. Held-out evaluation uses
identical initial noise for correct and shuffled actions.

At full correction strength, the model over-sharpened V1 into noisy texture. A
held-out strength sweep selected 0.2:

| metric | V1 base | calibrated flow |
| --- | ---: | ---: |
| decoded pixel MSE | 0.006147 | 0.006316 |
| edge-energy ratio | 0.543 | 0.855 |
| gradient alignment | 0.764 | 0.727 |
| wrong-action penalty | 39.6% | 39.4% |

The trade is explicit: edge energy improves by 57%, pixel error regresses by
2.7%, and action sensitivity is preserved. This is a visual refinement layer,
not a replacement for the world model or decoder.

## Human gate

Metrics cannot decide whether the browser looks better. Run:

```bash
uv run mcwm play-rollout \
  --dynamics-checkpoint artifacts/spatial-flow-v4-refiner/best.pt
```

The original V1 remains available by omitting `--dynamics-checkpoint`. The flow
checkpoint should not become the default unless the interactive comparison is
visibly better to the user.

Reference: Lipman et al., [Flow Matching for Generative
Modeling](https://arxiv.org/abs/2210.02747).
