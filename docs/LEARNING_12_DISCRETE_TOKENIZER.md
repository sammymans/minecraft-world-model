# Learning 12: rejected discrete-tokenizer pilot

## Question

Can a discrete visual representation prevent the continuous dynamics model
from averaging future latents into blurry frames?

Discrete tokens are a legitimate world-model design. IRIS, Genie, and
MineWorld use learned visual tokens and categorical next-token prediction.
They do not establish that a small tokenizer trained on our data will preserve
enough Minecraft detail, so reconstruction was tested before adding token
dynamics.

## Bounded implementation

The pilot kept V1 and its checkpoints untouched. It:

- copied the trained spatial autoencoder's encoder and decoder;
- inserted a 512-entry codebook into the existing 16x16x16 latent grid;
- initialized the codebook with k-means over pretrained continuous latents;
- fine-tuned with reconstruction, codebook, and commitment losses;
- used 20,000 training frames and all 978 eligible V3 validation frames; and
- trained for 10 epochs on MPS.

The gate required the quantized representation to stay within 3 dB of the
continuous reconstruction, retain at least 90% of its edge-energy ratio, and
use the codebook meaningfully.

## Result

| metric | continuous autoencoder | 512-code tokenizer |
| --- | ---: | ---: |
| validation PSNR | 37.46 dB | 28.75 dB |
| edge-energy ratio | 0.974 | 0.671 |
| pixel L1 | 0.00674 | 0.02031 |
| codes used | n/a | 509 / 512 |
| code perplexity | n/a | 183.9 |

The codebook did not collapse: almost every entry appeared on validation data.
The failure was the information bottleneck itself. Quantization removed fine
structure and produced visibly softer reconstructions before any dynamics model
was involved.

## Decision

The tokenizer failed its gate, so action-conditioned token dynamics was not
built. Starting dynamics from a representation already 8.7 dB worse than the
current decoder oracle would work against the demo goal.

The experimental implementation was removed from the working code after the
measurement. The ignored artifacts remain in
`artifacts/spatial-vq-v3-pilot/`, including the checkpoint, metrics, curve,
and reconstruction grid.

This result does not show that discrete world models are ineffective. Larger
systems use substantially larger codebooks, pretrained perceptual tokenizers,
larger dynamics networks, and far more data. It shows that the smallest
warm-started VQ conversion is not a free sharpness fix for this project.
