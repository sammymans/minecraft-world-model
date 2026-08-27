# Tiny Minecraft Latent World Model

## 1. The project in one sentence

We will train a small neural network to observe Minecraft video, compress what
it sees, predict how keyboard and mouse actions change that compressed state,
and display the predicted future as an interactive rollout.

The whole project is:

```text
Minecraft frames + player actions
                |
                v
         sequence dataset
                |
                v
     tiny latent world model
                |
                v
   interactive imagined rollout
```

This is intentionally not a model of all of Minecraft. It is a small experiment
that demonstrates the central world-model loop from pixels to imagined pixels.

## 2. What a world model means here

At time $t$, the model receives an observation $o_t$ and the action $a_t$ taken
by the player. The observation is a Minecraft image. The action contains a
small set of keyboard and mouse controls.

The model first compresses the observation into a latent state:

$$
z_t = E_\theta(o_t)
$$

The latent state $z_t$ is a small vector. It is the model's internal numerical
description of what it currently sees.

The dynamics model predicts how an action changes that state:

$$
\hat z_{t+1} = F_\phi(z_t, a_t)
$$

The decoder turns the predicted latent state back into an image:

$$
\hat o_{t+1} = D_\psi(\hat z_{t+1})
$$

We obtain an imagined trajectory by feeding each prediction back into the
dynamics model:

$$
\hat z_{t+2} = F_\phi(\hat z_{t+1}, a_{t+1})
$$

This recursive prediction is the defining demonstration. The final interface
will let a person provide the actions while watching the model imagine what
happens next.

## 3. Exactly what is in scope

The project contains four components.

### 3.1 Data recorder

The recorder will eventually capture our own Minecraft gameplay at a fixed
rate. Each timestep will contain:

$$
(o_t, a_t, \tau_t)
$$

where $o_t$ is a frame, $a_t$ is the action applied after that frame, and
$\tau_t$ is a timestamp.

The initial action space will contain only:

$$
a_t = [W,A,S,D,\text{jump},\text{sprint},\text{sneak},
\Delta x_{mouse},\Delta y_{mouse}]
$$

We do not need inventory, crafting, chat, attacks, or every Minecraft key for
the first model.

The recorder comes later. We will first use public data so that recorder bugs
and model bugs cannot be confused with each other.

### 3.2 Dataset pipeline

The dataset pipeline will load synchronized frames and actions and return short,
contiguous sequences:

$$
(o_t,a_t,o_{t+1},a_{t+1},\ldots,o_{t+H})
$$

It will perform only the necessary work:

- resize frames to a small resolution;
- sample at a low, fixed frequency;
- translate raw controls into our small action vector;
- normalize images and mouse movement;
- sample contiguous training sequences;
- split whole episodes between training and validation; and
- visualize frames with their associated actions to verify synchronization.

Both public data and our future recorder will be adapted to one small canonical
episode format. This is the only shared data abstraction we need.

### 3.3 Tiny latent world model

The model will use a small convolutional encoder, a small action-conditioned
dynamics network, and a small convolutional decoder.

A single image does not reveal whether the player is already moving. We keep
the visual autoencoder simple by encoding each frame independently:

$$
e_t = E_\theta(o_t)
$$

The dynamics state is a pair of consecutive visual latents:

$$
z_t=(e_{t-1},e_t)
$$

The dynamics network uses their change plus the action to predict the next
visual latent:

$$
\hat e_{t+1} = F_\phi(e_{t-1},e_t,a_t)
$$

The decoder reconstructs an image from one visual latent:

$$
\hat o_t = D_\psi(e_t)
$$

The initial model should be tiny enough to inspect and train locally:

- input frames: $64 \times 64$ RGB;
- V1 latent: a flat 256-value vector, retained as a baseline;
- current latent: a $16\times16\times16$ spatial feature map (4,096 values);
- encoder and decoder: a few convolutional layers;
- dynamics: a residual MLP;
- deterministic predictions; and
- short rollouts of approximately 1–2 seconds.

We will build it in two understandable stages. First, train the autoencoder and
verify that reconstruction works. Then train the dynamics model to predict the
next latent. Joint fine-tuning is optional and will only be added if the simple
version clearly needs it.

The conceptual training loss is:

$$
\mathcal L =
\lambda_{recon}\lVert D(E(o_t))-o_t\rVert^2
+
\lambda_{latent}\lVert F(e_{t-1},e_t,a_t)-e_{t+1}\rVert^2
+
\lambda_{pixel}\lVert D(F(e_{t-1},e_t,a_t))-o_{t+1}\rVert^2
$$

The terms ask for three things:

1. reconstruct what the model currently sees;
2. predict the next compressed state; and
3. make that predicted state decode into the actual next frame.

We will begin with the smallest subset of these losses needed for stable,
debuggable training and add terms one at a time.

### 3.4 Interactive rollout

The interactive program will load a trained model and start from two real seed
frames. After that, the person supplies keyboard and mouse actions while the
model supplies the images:

```text
two real seed frames
        |
        v
  initial latent z_0
        |
 user presses a key <-------------------+
        |                               |
        v                               |
 predict next latent                    |
        |                               |
        v                               |
 decode and display imagined frame -----+
```

No new Minecraft frame is shown after the seed. Every displayed future frame
comes from the model. A small local browser page shows only the current imagined
view and provides held-key and pointer-lock mouse input. It is served directly
by the Python process with no frontend build or external service.

The rollout will eventually drift or blur. That is expected. We want to see how
long the model remains coherent, how its predictions respond to controls, and
how errors compound when predictions are fed back into the model.

## 4. Data strategy

### Stage A: public data first

We will begin with a small subset of the public OpenAI VPT contractor data. It
provides Minecraft `.mp4` recordings with corresponding `.jsonl` action logs.

The previous project downloaded 24 JSONL files but not their matching videos.
Those files were sufficient for a structured-state experiment, but they are not
sufficient for a visual latent model. This project needs matched pairs:

```text
episode-name.mp4
episode-name.jsonl
```

We will initially use only a few episodes or short clips. More data is not useful
until we have proved that:

- video frames decode correctly;
- actions align with the correct frames;
- the dataset returns contiguous sequences;
- the autoencoder can reconstruct frames; and
- the dynamics model reacts differently to different actions.

The public data is a bootstrap dataset, not a permanent infrastructure
commitment. Its job is to prove the learning loop before we build a recorder.

### Stage B: our own recorder

Once the public-data version works, we will record deliberately simple data in a
controlled Minecraft area. Examples include:

- standing still;
- walking forward on flat ground;
- turning left and right;
- walking while turning; and
- jumping while moving.

Controlled data makes cause and effect easier to learn and lets us balance the
action distribution. The recorder's output will be adapted to the same episode
interface already consumed by the model.

## 5. Milestones

### Milestone 0 — reset and specification

Deliverables:

- one canonical scope document;
- no legacy implementation; and
- no features outside the four-component project.

Status: complete.

### Milestone 1 — public episode viewer

Deliverables:

- a `uv` Python project;
- download one or a few matched VPT video/action episodes;
- parse the raw action format;
- display sampled frames with an action overlay; and
- manually confirm that actions and visual changes are synchronized.

Completion test: we can watch a clip and see the corresponding keyboard and
mouse values beside every sampled frame.

Status: complete. The first official episode contains 6,000 action records and
6,001 video frames at 20 Hz. An annotated preview has been generated and checked.

### Milestone 2 — sequence dataset

Deliverables:

- a canonical episode representation;
- a PyTorch dataset returning contiguous frame/action sequences;
- episode-level train/validation splitting;
- a batch visualization; and
- small automated tests for indexing, shapes, and temporal alignment.

Completion test: one command shows a batch with shapes and an ordered visual
sequence whose actions make sense.

Status: complete. Dataset `vpt_v4` contains 707 public episodes from 705
independent groups, processed into 10 Hz, $64\times64$ canonical episodes. The
group-safe 80/10/10 assignment provides 398,354 training, 52,331 validation,
and 49,906 test eight-step sequences. Validation and test contain 70 independent
groups each; the original narrow held-out session remains in validation because
earlier experiments already used it for model development.

### Milestone 3 — visual autoencoder

Deliverables:

- tiny encoder and decoder;
- reconstruction training command;
- saved checkpoint; and
- side-by-side original/reconstruction images on held-out episodes.

Completion test: held-out reconstructions are recognizable enough that we can
identify the scene and camera direction.

Status: redesigned after interactive testing. The original 1,396,835-parameter
autoencoder compressed each frame to a flat 256-value vector and reached
$28.27$ dB, but decoder-oracle diagnostics showed that it discarded too much
spatial detail. The replacement is a 253,395-parameter convolutional
autoencoder with a $16\times16\times16$ spatial latent. The selected checkpoint
trained on 100,000 V3 frames and reaches held-out MSE $0.000180$, L1 $0.00674$,
PSNR $37.46$ dB, and a $0.974$ edge-energy ratio. See Learning 08 for the
diagnosis and gates.

### Milestone 4 — latent dynamics

Deliverables:

- action-conditioned latent transition model;
- one-step latent and image prediction training;
- validation curves;
- real-versus-predicted next-frame comparisons; and
- an action-sensitivity test.

Completion test: predictions on held-out sequences change with the supplied
action, and shuffling actions makes prediction worse.

Status: complete with the selected spatial model. The versioned additive model
trained on 100,000 V4 transitions. On the final 70-group test split, latent MSE
is $0.003874$ versus $0.007246$ for copy, while pixel L1 is $0.03069$ versus
$0.03749$ for decoded copy. Shuffled actions raise latent MSE to $0.005700$.

### Milestone 5 — open-loop evaluation

Deliverables:

- recursive rollouts over several horizons;
- real and imagined frame strips;
- error-versus-horizon graph; and
- comparisons against copying the previous frame.

Completion test: the model beats the copy-frame baseline for at least a short
horizon and produces visually interpretable rollouts.

Status: complete with the spatial replacement. Across 5,000 final-test windows,
recursive predictions beat frozen decoded copy at every measured horizon. At
20 steps, pixel MSE is $0.02364$ versus $0.02985$ for copy; mismatched actions
are 22.9% worse. Visual predictions remain interpretable briefly but smooth as
recursive error accumulates.

### Milestone 6 — interactive rollout

Deliverables:

- local single-viewport browser frontend;
- real seed-frame selection;
- keyboard and mouse action input; and
- live display of recursively imagined frames.

Completion test: changing the controls produces visibly different imagined
futures without reading new frames from Minecraft.

Status: complete with the spatial checkpoint. The local web frontend accepts
held movement keys and pointer-lock camera controls, shows only the current
imagined view, and runs the spatial dynamics model recursively at 10 Hz. Its
HTTP step path, live seed switching, and scripted mode are verified. Longer
interaction still exposes the deterministic model's gradual blur.

### Milestone 7 — recorder replacement

Deliverables:

- synchronized screen and input recorder;
- episode inspection tool;
- conversion to the canonical episode interface; and
- a model trained on our own controlled recordings.

Completion test: the same training and rollout commands work after switching
from public episodes to our episodes.

## 6. Proposed repository shape

The implementation should stay close to this shape:

```text
README.md
docs/
    README.md
    PROJECT.md
pyproject.toml
src/mcwm/
    dataset.py       # canonical episode and sequence dataset
    vpt.py           # public VPT adapter
    recorder.py      # added only at Milestone 7
    model.py         # encoder, dynamics, decoder
    training.py      # training, metrics, and evaluation visuals
    rollout.py       # offline and interactive rollouts
tests/
```

We will use `uv` for the environment, dependencies, and commands. We will not
create services, databases, configuration frameworks, experiment platforms, or
multiple model families.

## 7. What success looks like

The project succeeds when we can demonstrate this sequence:

1. Show a synchronized Minecraft frame/action sequence.
2. Show that the encoder compresses frames and the decoder reconstructs them.
3. Show that the dynamics model predicts different next states for different
   controls.
4. Show a held-out real trajectory beside the model's imagined trajectory.
5. Start from a real frame and interactively control a short imagined future.
6. Repeat the experiment using data from our own recorder.

The model does not need to produce sharp, long, or generally convincing video.
A blurry but action-responsive short rollout is a successful first world model
because it demonstrates representation, learned dynamics, and imagination in
one end-to-end system.

## 8. Non-goals

The following are explicitly outside the current project:

- autonomous agents;
- rewards, planning, or MPC;
- robotics framing;
- text conditioning;
- high-resolution or photorealistic video generation;
- transformers and large pretrained models;
- full Minecraft mechanics;
- long-horizon coherence;
- inventory and crafting;
- multiplayer support; and
- production-scale data infrastructure.

### Revision: conditional latent diffusion is now in scope

This list originally excluded diffusion models alongside transformers and
pretrained models. That exclusion is lifted for the dynamics network only,
deliberately and for a measured reason.

Learning 09 and Learning 10 established that the rollout blur is caused by the
training objective rather than by the encoder, the architecture, or the amount
of data. Squared error is minimized by the average over plausible next frames,
and that average is smooth. Learning 10 separated the two candidate causes:
training on the model's own predictions removed exposure bias and bought a
consistent horizon-scaling improvement, and the blur remained. That is the
evidence that deterministic regression is the binding constraint.

Modeling a distribution instead of a point estimate is the only change that
addresses it. The scope guard this list exists to enforce is *stay tiny and
stay attributable*, and the revision respects both:

- the frozen spatial autoencoder is unchanged;
- the dynamics keeps its convolutional backbone and gains a timestep
  embedding, denoising the next latent instead of regressing it;
- no transformer, no pretrained weights, no new service or framework; and
- sampling costs about 5 ms per frame against a 100 ms budget at 10 Hz, so the
  interactive rollout stays interactive.

Transformers, pretrained models, and every other item above remain out of
scope. Diffusion over pixels remains out of scope; this is diffusion over the
existing 16x16x16 latent only.

If a proposed feature does not directly help us record sequences, train the
tiny latent model, or interact with its rollout, we should leave it out.
