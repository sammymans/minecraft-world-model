# V2: action-conditioned latent diffusion world model

## 1. Goal

V1 proved the original learning objective: a small model can encode Minecraft,
use player actions to predict held-out latent transitions, and recursively
imagine a controllable future. Its failure is visual: motion predictions quickly
average into smooth color fields.

V2 has one additional completion requirement:

> From one real seed, forward, left, and right controls should produce visibly
> different, recognizable 64x64 futures for roughly five to ten steps.

Oasis-level resolution, long-term memory, and photorealism are not required.
Minecraft block boundaries and the main scene layout should remain readable.

## 2. Why this remains a world model

“World model” describes the role of the system: predict the next environment
state from previous states and an action. “Latent diffusion” describes the
probabilistic transition model used to perform that prediction.

```text
past frames ──encoder──> past latent states ─┐
                                             ├─> diffusion dynamics ─> next latent
recent actions ──────────────────────────────┘                         │
                                                                       v
                                                                    decoder
                                                                       │
                                                                       v
                                                               predicted frame
```

The decoder does not become a diffusion model. The existing encoder and decoder
remain the observation interface; diffusion replaces V1's deterministic latent
dynamics network.

## 3. What changes from V1

| | V1 | V2 |
|---|---|---|
| latent representation | continuous 16x16x16 | unchanged |
| context | two latent frames | eight latent frames |
| controls | current 9-value action | recent action sequence |
| prediction | one deterministic latent | one sampled next latent |
| objective | squared latent/pixel error | diffusion velocity prediction |
| training context | clean real latents | clean and artificially corrupted latents |
| dynamics size | 255K parameters | approximately 10–20M parameters |
| decoder | frozen spatial decoder | initially unchanged and frozen |

Squared-error regression is rewarded for predicting the conditional average.
When several sharp futures are possible, that average can be blurry. Diffusion
instead learns a conditional distribution and samples one plausible next state.

## 4. Proposed model

### 4.1 Existing representation

Keep `artifacts/spatial-autoencoder-v3/best.pt` frozen. Its held-out decoder
oracle reaches about 36.8 dB and preserves approximately 97% of real edge
energy, so it can already display clear 64x64 Minecraft frames. Replacing it is
not part of the initial V2 experiment.

### 4.2 Inputs

For target frame `t+1`, the diffusion U-Net receives:

- the noised target latent;
- the previous eight encoded latent frames;
- the corresponding action history, including the action for `t+1`;
- the target diffusion timestep; and
- the noise level applied to the context frames.

The latent history is concatenated spatially in the input channels. Each action
is embedded as a token. A small cross-attention block at the U-Net bottleneck
lets spatial features read the action history without broadcasting nine raw
numbers over the whole image.

### 4.3 Denoising objective

During training, sample noise `epsilon` and a noise level. Corrupt the real next
latent according to the diffusion schedule:

```text
noisy_next = alpha * real_next + sigma * epsilon
```

The U-Net predicts the velocity parameterization used by GameNGen. It is
conditioned on past latents and actions, so the learned distribution is:

```text
p(next latent | latent history, action history)
```

At inference, begin with Gaussian noise and use approximately eight DDIM steps
to obtain one next latent. The existing decoder renders that latent.

### 4.4 Context noise augmentation

Autoregressive inference conditions on generated frames, but ordinary training
conditions on perfect real frames. GameNGen addresses this mismatch by adding
noise to conditioning frames during training and telling the model the context
noise level.

V2 will do the same in latent space. This is the main stability mechanism:

```text
clean context       some batches
lightly noisy       most augmented batches
more corrupted      a smaller fraction of batches
```

The model must learn to recover a sharp next state even when its recent history
contains the kind of small errors it will encounter during rollout.

## 5. Data and compute

No new download is needed for the first V2 attempt. The existing V4 pipeline
contains:

- 707 processed episodes;
- 398,354 clean eight-step training sequences;
- 52,331 validation sequences; and
- 49,906 untouched test sequences.

The old flow pilot used only 4,000 clips and roughly 1,250 optimizer updates.
That was enough to test plumbing, not enough evidence about a generative video
model.

The V2 pilot will use 50,000 diverse training windows and train by optimizer
step rather than by a misleadingly small epoch count. After the first 200
steps, training throughput will be measured on MPS and the run length reported
before continuing. The full 398K-window dataset is used only if the pilot
produces recognizable held-out frames.

## 6. Implementation stages and gates

### Stage 1 — model and mathematical tests

Implement one temporal latent U-Net, one diffusion schedule, and DDIM sampling.

Gate:

- tensor shapes and checkpoint metadata are tested;
- a perfect denoiser exactly reconstructs its target in a controlled test;
- seeded sampling is reproducible across reset; and
- the model stays within the interactive memory budget.

### Stage 2 — tiny fixed-set overfit

Train on a fixed set of 256 short sequences.

Gate:

- denoising loss falls substantially;
- generated next frames become recognizable rather than noise;
- changing actions changes the output; and
- the decoder still receives valid continuous latent maps.

If the model cannot pass this gate, the implementation or architecture is
wrong. More data is not the response.

### Stage 3 — held-out pilot

Train on 50,000 V4 sequences with context noise augmentation. Evaluate on fixed
validation sequences and render the same seeds and actions used for V1.

Gate:

- held-out frames are recognizable through approximately `t+5`;
- edges represent scene structure rather than random grain;
- left, right, and forward visibly diverge from the same seed;
- shuffled actions remain worse than correct actions; and
- sampling is fast enough for the one-step-per-second demo mode.

The user makes the visual pass/fail decision. Edge metrics, PSNR, and latent
error are diagnostics, not substitutes for that decision.

### Stage 4 — scale only after a visual pass

If Stage 3 works, train longer and use more of the existing 398K training
windows. Only then consider reducing sampling steps or restoring 10 Hz playback.

## 7. What is deliberately excluded

The first V2 implementation will not include:

- the V1 flow refiner;
- a discrete tokenizer;
- a transformer or full Diffusion Forcing implementation;
- a new autoencoder or higher resolution;
- decoder fine-tuning;
- rewards, planning, or an agent;
- additional data downloads; or
- multiple competing diffusion architectures.

Those are follow-ups only if the direct model identifies a specific need. This
keeps V2 to one architecture and one visual hypothesis.

## 8. Repository separation

V1 stays reproducible and remains the default until V2 passes the visual gate.
V2 receives separate code paths, commands, checkpoints, artifacts, and
documentation. A failed V2 experiment cannot overwrite the selected V1 model.

Planned artifact directory:

```text
artifacts/spatial-latent-diffusion-v2/
```

The browser will load V2 only when its checkpoint is explicitly selected. It
becomes the default only after the user accepts the interactive comparison.

## 9. Reading guide

### Read first

1. **GameNGen — Diffusion Models Are Real-Time Game Engines**  
   [Paper](https://arxiv.org/abs/2408.14837)  
   Read Sections 3.2, 3.2.1, and 3.3. This is the closest implementation
   reference: next-frame latent diffusion conditioned on previous observations
   and actions, with context noise augmentation for autoregressive stability.

2. **High-Resolution Image Synthesis with Latent Diffusion Models**  
   [Paper](https://arxiv.org/abs/2112.10752)  
   Read the latent-space formulation and conditioning sections. This explains
   why a frozen autoencoder can make diffusion substantially cheaper while
   retaining visual detail.

### Read for context

3. **DIAMOND — Diffusion for World Modeling: Visual Details Matter in Atari**  
   [Paper](https://arxiv.org/abs/2405.12399)  
   Useful for the world-model motivation and the interactive CS:GO result.
   DIAMOND operates differently from this V2, so it is supporting evidence, not
   the implementation template.

4. **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**  
   [Paper](https://arxiv.org/abs/2407.01392)  
   Read after the simpler model is clear. Independent per-token noise levels
   are a possible V2.1 improvement if ordinary context augmentation still
   accumulates errors.

5. **Oasis: A Universe in a Transformer**  
   [Project and architecture overview](https://oasis-model.github.io/)  
   This shows the scaled Minecraft version: a transformer autoencoder, a 500M
   parameter latent diffusion transformer, Diffusion Forcing, and dynamic
   noising. It is the visual inspiration, not the size target.

## 10. Definition of done

V2 is complete when the same held-out seed can be reset and controlled with
forward, left, and right actions, and a person can recognize the Minecraft scene
and its motion for the next five to ten frames. The result may remain 64x64,
soft, imperfect, and short. It must not immediately collapse into a smooth
color field or unrelated high-frequency noise.
