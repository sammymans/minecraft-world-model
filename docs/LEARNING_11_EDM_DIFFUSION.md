# Learning 11: rejected diffusion pilots

## Question

Could diffusion remove the deterministic dynamics model's blur while retaining
its latent accuracy and action conditioning?

## What we tested

Two bounded latent-diffusion approaches were evaluated against the frozen V1
dynamics model:

- DDPM-style noise prediction over the next latent residual; and
- EDM clean-sample prediction, both directly and as a correction around V1.

The EDM pilot used four latent/action context steps, a small U-Net, adaptive
action/noise conditioning, context corruption, and common random noise for fair
correct-versus-shuffled-action comparisons. These choices came from DIAMOND,
GameNGen, Oasis, and Diffusion Forcing.

## Result

| model | latent MSE | reference MSE | gradient cosine | edge ratio | shuffled-action penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchored EDM, low noise | 0.002412 | V1 0.002207 | 0.713 | 0.832 | +43.0% |
| anchored EDM, full noise | 0.004039 | V1 0.002207 | 0.581 | 1.380 | +25.3% |
| direct EDM, full noise | 0.004685 | copy 0.004017 | 0.554 | 1.258 | +2.5% |

Low-noise anchored sampling remained action-sensitive but was less accurate
than V1 and added grain rather than reconstructing Minecraft block boundaries.
Full-noise sampling was noisier, while direct diffusion was nearly
action-insensitive and worse than copying the current latent.

## Decision

None of the pilots passed the one-step quality gate. They were never made the
interactive default or scaled to a full training run. Their implementation was
removed from the main code after the experiment so the working project remains
focused on the deterministic spatial autoencoder and V1 dynamics model.

The git history and ignored artifact directories retain the experimental work
if it is ever needed for reference. The important result is preserved here so
the same unsuccessful approach is not repeated accidentally.

This does not prove that diffusion cannot work. It shows that sampling over the
current deterministic autoencoder latents traded blur for unstructured noise
without improving the usable world model.
