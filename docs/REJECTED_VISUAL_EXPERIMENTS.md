# Rejected visual experiments

V1's continuous spatial representation and deterministic dynamics are the
supported baseline. Three bounded attempts to improve rollout sharpness were
measured and rejected; their runtime code has been removed so they cannot be
mistaken for the selected model.

## Direct latent diffusion

A small conditional denoiser replaced one-step regression. It produced more
edge energy, but either became grainy or ignored actions and failed to beat the
copy baseline. Low-noise anchored sampling retained some action sensitivity but
was less accurate than V1. This pilot was too small and too close to a one-step
residual refiner to test the sequence-model design used by successful game
world models.

## Discrete tokenizer

A warm-started VQ tokenizer used its codebook (perplexity 183.9) but degraded
the representation before token dynamics were trained:

| representation | validation PSNR | edge-energy ratio |
|---|---:|---:|
| continuous V1 | 37.46 dB | 0.974 |
| VQ pilot | 28.75 dB | 0.671 |

It failed the representation gate, so the discrete dynamics stage was stopped.
This rejects that small tokenizer configuration, not discrete world models in
general.

## Latent-video flow refiner

A 4.8M-parameter flow model refined eight-frame V1 clips with classifier-free
action guidance. Its calibrated validation metrics were:

| metric | V1 | flow-refined |
|---|---:|---:|
| decoded pixel MSE | 0.006147 | 0.006316 |
| edge-energy ratio | 0.543 | 0.855 |
| wrong-action penalty | 39.6% | 39.4% |

The numerical edge gain did not survive the human gate: the browser still
looked effectively as blurry and degraded just as quickly. The refiner and its
interactive adapter were therefore removed.

## What these failures changed

V2 does not stack another cosmetic refiner on V1. It keeps the proven frozen
spatial autoencoder, then trains a larger action-conditioned diffusion sequence
model to predict the next latent directly from a longer history. The complete
design and gates are in
[V2_ACTION_CONDITIONED_LATENT_DIFFUSION.md](V2_ACTION_CONDITIONED_LATENT_DIFFUSION.md).
