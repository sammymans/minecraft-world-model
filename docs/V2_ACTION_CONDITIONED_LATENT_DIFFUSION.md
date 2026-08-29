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

## 11. Implementation status (2026-08-28)

V2 now has an isolated implementation in `mcwm.latent_diffusion_v2`; V1's model,
checkpoint loading, rollout commands, and default artifacts are unchanged. The
new path contains:

- a fixed temporal latent dataset with eight context maps, eight exactly aligned
  transition actions, and one target map;
- an action-balanced fixed-set policy that draws equally from forward,
  look-left, look-right, and other target actions across at least eight episodes;
- a 1,000-level linear beta diffusion schedule with velocity targets and
  deterministic eight-step DDIM sampling, matching GameNGen's reported schedule
  family;
- a 19,326,960-parameter latent U-Net with timestep/context-noise conditioning
  and action-history cross-attention at 16x16, 8x8, and 4x4 resolutions;
- a shared-noise five-step autoregressive comparison in which generated latents
  are shifted back into context under forward, left, and right action scripts;
- clean/light/heavier context corruption support, left disabled for the clean
  Stage 2 memorization gate; and
- versioned V2-only checkpoints that record the frozen autoencoder hash, manifest
  hash, architecture, schedule, action buckets, source episodes, and every
  fixed-sequence reference.

Stage 1 passed. The mathematical tests reconstruct the exact clean target from a
perfect velocity prediction, checkpoint metadata round-trips, seeded DDIM output
is identical after a reset, and the five-step shared-noise rollout is reproducible.
A real V4 readiness check selected exactly 64 examples from each action bucket
across eight episodes. A full-size MPS forward/backward and five-step rollout
used approximately 292 MB of MPS driver memory; all rollout latents were finite.
The complete repository suite passes with 65 tests.

The first Stage 2 run, retained under `fixed-256-overfit`, proved that the model
could memorize sharp next frames: velocity MSE fell 76.8%, sampled frames reached
27.50 dB, and samples beat the copy-last-frame baseline. Its same-noise one-step
action differences were too faint and texture-like to count as convincing
semantic control. That checkpoint was trained under the first V2 architecture
(`temporal_action_unet_velocity_v1`, cosine schedule, bottleneck-only action
attention). The current loader deliberately rejects it, so its PNGs and metrics
remain readable as diagnostic evidence while the checkpoint itself cannot be
mistaken for the candidate to scale.

The revised action-balanced Stage 2 implementation is ready to train with:

```text
uv run mcwm overfit-latent-diffusion-v2 \
  --sequences 256 \
  --minimum-episodes 8 \
  --steps 2000 \
  --batch-size 8 \
  --device mps
```

It writes new artifacts without overwriting the original diagnostic run:

```text
artifacts/spatial-latent-diffusion-v2/fixed-256-action-balanced-v2/
```

The 50K pilot means a later optimizer-step run over a deterministic 50,000-window
slice of the existing V4 training split, evaluated against untouched validation
episodes with context-noise augmentation enabled. It is deliberately between the
256-window memorization gate and all 398K available training windows. It is not a
download and it has not been implemented or started because the revised Stage 2
visual gate must pass first. No full-dataset training or default-model switch has
occurred.

## 12. Revised Stage 2 result (2026-08-28)

The action-balanced run under `fixed-256-action-balanced-v2/` converged roughly
twice as fast as the first attempt: fixed-noise velocity MSE reached 0.245 by
step 900, a level the original needed about 1,900 steps to hit, and finished at
0.111560 (an 88.95% reduction) after 2,000 steps.

Four of the five Stage 2 checks pass, decisively:

| check | result |
|---|---|
| denoising loss falls substantially | 88.95% reduction |
| generated frames are recognizable | 30.33 dB, versus 22.39 dB for copy-last-frame |
| beats the copy baseline | 83.9% lower pixel MSE |
| decoder receives valid continuous latents | finite, range -1.11 to 1.36 |

Recursive stability is also better than V1 ever managed. Frame contrast is held
across all five rollout steps from dark, median, and bright seeds (for example
0.2253 at the seed against 0.2629 at `t+5`), so the V1 failure of dissolving into
a smooth colour field does not occur.

**The action-control check still fails.** Holding context and initial noise fixed
and varying only the camera script, the left-versus-right difference is real,
monotonically growing, and close to linear in camera magnitude — but far too
small to see:

| camera step | left/right L1 at `t+5` | share of total drift |
|---|---|---|
| 12 | 0.00109 | 0.8% |
| 30 | 0.00270 | 1.9% |
| 60 | 0.00521 | 3.6% |
| 120 | 0.00922 | 6.5% |

So the conditioning pathway demonstrably works — the model reads the action and
responds in proportion — but roughly 93 to 99% of what changes between frames is
action-independent drift. Shuffling action histories costs 9.0% denoising loss
and 12.0% sample MSE, confirming the actions carry signal without carrying
control.

Three harness defects were found and fixed while diagnosing this; none of them
created the failure, but all three obscured it:

- the counterfactual seeded from `dataset[0]`, which happened to be a near-black
  cave frame (brightness 0.044) in which no camera motion could be visible. Seeds
  are now chosen by gradient energy;
- `counterfactual_action_scripts` defaulted to a camera step of 12, below the
  median real turn of 16 in this subset (p90 is 121). The default is now 30; and
- `autoregressive_action_rollout` could not report a context noise level to the
  model, so a model trained with augmentation could not be evaluated correctly.
  It is now plumbed through to a `--rollout-context-noise` flag.

Metrics now also record `five_step_total_drift_pixel_l1` and
`action_share_of_drift_percent`, since a raw difference is not interpretable
without the drift it is measured against.

The leading hypothesis for the weak control is that context noise augmentation
is disabled. Given eight clean context frames, the next latent is nearly
determined by extrapolating observed motion, which makes the action close to
redundant. Corrupting the context is precisely the mechanism GameNGen uses to
force the model to depend on the action instead. Stage 2 deliberately ran clean,
so this is the next thing to test rather than a defect in the architecture.

## 13. Context noise result: hypothesis refuted (2026-08-28)

`fixed-256-ctxnoise/` trained 8,000 steps with `maximum_context_noise 0.2`. As a
*predictor* it is clearly better: 33.64 dB against 30.33 dB, 92.5% better than
copy-last-frame against 83.9%, and the grain problem eased (edge ratio 1.053
against 1.130). Rollouts stay stable and recognizable for five steps.

As a *controller* it is worse. Measured through one identical harness — same
seed frame, same camera scripts, same RNG — action share of total drift:

| run | camera 30 | camera 120 | shuffled-action loss penalty |
|---|---|---|---|
| 2,000 steps, clean context | 5.9% | 20.5% | 9.0% |
| 8,000 steps, context noise 0.2 | 3.3% | 12.0% | 3.2% |

The context-noise model relies on actions roughly 1.7 times *less*. The
loss-based and pixel-based measures agree, so this is not a measurement artefact.
The hypothesis in section 12 — that clean context makes the action redundant and
that corrupting it would force action reliance — is refuted.

**The comparison is confounded.** That run changed two variables at once: context
noise 0.0 to 0.2 *and* steps 2,000 to 8,000. Either could explain the regression.

The second possibility is the more troubling one, and it is a property of the
gate rather than of the model. On a fixed 256-sequence set each context has
exactly one true continuation, so a model that memorizes the context-to-target
mapping does not need the action at all. Training longer should therefore be
expected to *reduce* measured action reliance, because memorization and action
conditioning are substitutes for this objective. If that is what is happening,
Stage 2 cannot demonstrate semantic control no matter how it is tuned, and doing
better on its other four checks actively makes the fifth look worse.

The disentangling run is 8,000 steps with clean context. If action share falls to
roughly 3% it is memorization and the Stage 2 control check should be retired in
favour of a held-out measurement. If it stays near 6% then context noise is
genuinely harmful here and should not carry into Stage 3.

One earlier figure needs correcting: section 12 quoted 1.9% action share for the
clean run. That measurement used a different seed frame. Under the controlled
harness above the same checkpoint measures 5.9%. Action share is strongly
scene-dependent, so only same-harness comparisons are meaningful.

## 14. The Stage 2 control check is unsound (2026-08-28)

The disentangling run isolates the variable. All three checkpoints measured
through one identical harness — same seed frame, same scripts, same RNG:

| run | PSNR | shuffled-action penalty | action share @30 | @120 |
|---|---|---|---|---|
| 2,000 steps, clean | 30.33 dB | 9.00% | 5.9% | 20.5% |
| 8,000 steps, clean | 33.57 dB | 4.61% | 2.6% | 9.3% |
| 8,000 steps, noise 0.2 | 33.64 dB | 3.25% | 3.3% | 12.0% |

**Training length, not context noise, causes the regression.** Holding context
clean and going from 2,000 to 8,000 steps more than halves action share, 5.9% to
2.6%. At matched 8,000 steps, context noise is roughly neutral and if anything
slightly better on action share, 3.3% against 2.6%. Section 13 attributed the
regression to context noise; that attribution was wrong.

This is the memorization mechanism anticipated in section 13. On a fixed
256-sequence set each context has exactly one true continuation, so memorizing
the context-to-target mapping makes the action redundant. Prediction quality and
measured action reliance therefore move in opposite directions: the run with the
best PSNR has the worst control, and the run with the worst PSNR has the best.

The Stage 2 control check is therefore structurally incapable of demonstrating
semantic control, and it anti-correlates with the gate's other four checks. It
should not be tuned against and should not gate progress. Sections 12 and 13 both
treated it as a real failure of the model; on this evidence it is a failure of
the test.

The other four Stage 2 checks pass convincingly and stand: recognizable frames at
33.6 dB, 92.5% better than copy-last-frame, valid continuous latents, and stable
five-step rollouts with no colour-field collapse — the V1 failure mode is gone.

Consequences for Stage 3:

- action control must be measured on held-out sequences, where memorization is
  not available, and this is the first point at which the V2 completion criterion
  can be honestly tested;
- context noise augmentation should be carried forward, per GameNGen, since the
  evidence against it has been withdrawn and it is roughly neutral here; and
- longer training is not intrinsically harmful. Its apparent harm here was an
  artefact of a 256-window training set.

## 15. Stage 3 full-V4 implementation (2026-08-28)

Stage 3 now trains the same 19,326,960-parameter latent-diffusion U-Net from
Stage 2; it does not introduce another architecture. The new
`train-latent-diffusion-v2` command adds the production data and evaluation
path that the fixed-set harness deliberately lacked:

- the frozen autoencoder encodes each unique observation once into a contiguous
  float16 disk cache, keyed by both autoencoder and manifest hashes;
- training and validation latents/actions are memory mapped rather than keeping
  the 8.5 GiB processed RGB dataset in RAM;
- every natural training window appears once per sampling epoch, with additional
  action-change windows added to reach a 35% switch-point share;
- validation comes only from the group-separated V4 validation split;
- natural and action-change validation separately compare the correct final
  action against repeating the previous action, shuffling the final action, and
  zeroing the final action while holding diffusion noise fixed;
- checkpoints contain optimizer and data-sampler state and resume without
  changing the model; and
- fixed held-out samples and shared-noise five-step action rollouts are rendered
  at every evaluation.

The cache contains 1,483,566 unique training frames and exposes 409,910 clean
V2 windows. Of those, 109,443 (26.7%) change action bucket on the final
transition. The corresponding validation cache contains 53,734 windows. This
V2 count is slightly larger than the historical 398,354 eight-step count in
Section 5 because V2 needs exactly eight clean context-to-target transitions;
the older generic `SequenceDataset` also required a preceding clean transition
for its two-frame V1 seed.

The full-data run is:

```text
uv run mcwm train-latent-diffusion-v2 \
  --steps 60000 \
  --evaluation-every 2000 \
  --maximum-validation-sequences 512 \
  --sample-count 16 \
  --output-dir artifacts/spatial-latent-diffusion-v2/full-v4 \
  --device mps
```

Resume the same run with:

```text
uv run mcwm train-latent-diffusion-v2 \
  --steps 60000 \
  --evaluation-every 2000 \
  --maximum-validation-sequences 512 \
  --sample-count 16 \
  --output-dir artifacts/spatial-latent-diffusion-v2/full-v4 \
  --resume artifacts/spatial-latent-diffusion-v2/full-v4/latest.pt \
  --device mps
```

A two-step end-to-end smoke gate passed, including cache reuse, training,
natural/action-change validation, DDIM sampling, checkpoint save/load, and both
visual renderers. The first real checkpoint at step 200 reduced training
velocity MSE from 0.9405 to 0.6507 and reached 18.08 dB against the held-out
decoder oracle. Action penalties remained effectively zero at this deliberately
early checkpoint. Warm MPS throughput was approximately 12.8 optimizer steps
per second at batch size 8.

The Stage 3 decision is made from held-out results, not training loss. A useful
candidate must beat decoded copy, produce recognizable five-step futures, and
show a positive, visually meaningful correct-action advantage—especially on
the action-change subset. The final test split remains untouched until a
checkpoint passes those validation gates.

The selected/best Stage 3 checkpoint can be exercised without changing the V1
playground. A reproducible comparison image uses:

```text
uv run mcwm compare-actions-v2 \
  --scripts 'w+sprint*6' 'look_left*6' 'look_right*6' 'idle*6'
```

An interactive browser session uses:

```text
uv run mcwm play-v2
```

Both commands seed V2 from eight real held-out frames and their seven connecting
actions. Each live control becomes the eighth, target-driving action; the
sampled latent and action are then shifted back into their respective histories.
Resetting or comparing scripts restores both histories and the diffusion RNG so
differences between rows are attributable to controls rather than initial noise.
