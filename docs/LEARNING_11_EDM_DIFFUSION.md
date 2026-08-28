# Learning 11: EDM diffusion pilot

## Question

Can diffusion remove the deterministic dynamics model's immediate blur without
destroying its action conditioning or latent-state accuracy?

The first diffusion attempt did not answer this successfully. It used a
DDPM-style noise-prediction objective over the next latent residual. Its samples
had high edge energy, but the edges were unstructured noise; it lost to the copy
baseline and effectively ignored actions.

## Literature-driven redesign

The redesign uses the parts of published interactive world models that address
our measured failure:

- [DIAMOND](https://arxiv.org/abs/2405.12399) reports substantially more stable
  autoregressive world modeling with EDM preconditioning and clean-sample
  prediction than with DDPM noise prediction. It uses a U-Net conditioned on
  four frames and actions, with adaptive normalization for action and diffusion
  time conditioning.
- [GameNGen](https://arxiv.org/abs/2408.14837) conditions on frame/action
  history and corrupts conditioning frames during training. Its ablations show
  this context-noise augmentation prevents rapid autoregressive drift.
- [Oasis](https://oasis-model.github.io/) combines latent diffusion with
  Diffusion Forcing and dynamic noising for long-context stability.
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392) assigns independent
  noise levels to sequence tokens so a model learns to denoise imperfect past
  and future states under multiple causal sampling schedules.

We implemented the less expensive one-step prerequisites before attempting
Diffusion Forcing:

1. EDM input/output/skip preconditioning and a Heun sampler;
2. a two-level spatial U-Net with 1.72M parameters;
3. four latent context frames and four aligned action vectors;
4. FiLM-style adaptive normalization from action and noise embeddings in every
   residual block;
5. Gaussian context corruption during training; and
6. an optional frozen V1 anchor, with EDM modeling only the normalized
   correction from V1's forecast to the real next latent.

The existing autoencoder and V1 checkpoint remain frozen. No checkpoint or CLI
default was replaced.

## Evaluation repairs

Stochastic correct-action and shuffled-action rollouts now reuse identical
noise. Their difference therefore isolates action conditioning rather than
sampling variance. Resetting or reseeding the interactive engine also resets
the sampler's random stream.

Edge-energy magnitude is retained only as a diagnostic. We added gradient
cosine alignment, which measures whether generated and target gradients point
in the same places and directions. Random noise may have high edge energy but
poor gradient alignment.

## Bounded experiments

All values below use 512 frozen validation examples. These pilot comparisons
were used to decide whether a 100K transition run was justified.

| model | sampling | latent MSE | reference MSE | gradient cosine | edge ratio | shuffled-action penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| anchored EDM | sigma max 0.25 | 0.002412 | V1 0.002207 | 0.713 | 0.832 | +43.0% |
| anchored EDM | sigma max 5.0 | 0.004039 | V1 0.002207 | 0.581 | 1.380 | +25.3% |
| direct EDM | sigma max 5.0 | 0.004685 | copy 0.004017 | 0.554 | 1.258 | +2.5% |

The low-noise anchored model is stable and remains action-controlled, but it is
9.3% worse than V1 in latent MSE. Its extra edges appear as mild grain rather
than correctly reconstructed block boundaries. Full-noise anchored sampling is
approximately as inaccurate as copying the current latent and visibly noisy.
Direct EDM is 16.6% worse than copy and nearly action-insensitive.

Explicitly feeding V1's predicted anchor into the correction network did not
materially change the result. The denoising objective continued to improve, but
the low-noise sample metrics were nearly flat across twelve epochs.

## Decision

Neither variant passes the one-step gate, so neither is scaled to 100K
transitions, trained with multi-frame Diffusion Forcing, or integrated into the
browser. The deterministic V1 model remains the default.

This is not evidence that diffusion cannot work. It shows that, with the
current deterministic autoencoder latent space and this bounded dataset/model,
diffusion can trade blur for grain but has not recovered coherent Minecraft
structure. Scaling before resolving that one-step failure would be an
uncontrolled and expensive experiment.

## Reproduction

Anchored pilot:

```bash
uv run mcwm train-spatial-edm \
  --output-dir artifacts/spatial-edm-v4-anchor-aware-pilot \
  --maximum-transitions 20000 \
  --evaluation-examples 512 \
  --epochs 12 \
  --hidden-channels 64 \
  --blocks-per-level 2 \
  --sampling-steps 8 \
  --sigma-max 0.25 \
  --context-noise 0.1
```

Direct comparison:

```bash
uv run mcwm train-spatial-edm \
  --direct \
  --output-dir artifacts/spatial-edm-v4-direct-pilot \
  --maximum-transitions 20000 \
  --evaluation-examples 512 \
  --epochs 12 \
  --hidden-channels 64 \
  --blocks-per-level 2 \
  --sampling-steps 8 \
  --sigma-max 5.0 \
  --context-noise 0.1
```

The artifact directories contain the checkpoint, training curve, metrics JSON,
and fixed one-step comparison grid.
