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
- latent size: 256 values in the current quality-focused version;
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
comes from the model. A small local window, likely using Pygame, is sufficient.

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

Status: complete. Dataset `vpt_v1` contains 14 public episodes from 12
independent sessions, processed into 10 Hz, $64\times64$ canonical episodes. An
explicit manifest assigns 11 groups to training and freezes one group for
validation. This produces 5,605 training sequences and 618 held-out sequences
at an eight-step prediction horizon.

### Milestone 3 — visual autoencoder

Deliverables:

- tiny encoder and decoder;
- reconstruction training command;
- saved checkpoint; and
- side-by-side original/reconstruction images on held-out episodes.

Completion test: held-out reconstructions are recognizable enough that we can
identify the scene and camera direction.

Status: complete. A 1,396,835-parameter convolutional autoencoder compresses
each $64\times64$ RGB frame from 12,288 pixel values to 256 latent values. Its
best held-out checkpoint reached MSE $0.00213$, L1 $0.02517$, and PSNR $26.72$
dB. A 512-latent experiment improved L1 by only about 2%, so the smaller state
was retained for action-conditioned dynamics.

### Milestone 4 — latent dynamics

Deliverables:

- action-conditioned latent transition model;
- one-step latent and image prediction training;
- validation curves;
- real-versus-predicted next-frame comparisons; and
- an action-sensitivity test.

Completion test: predictions on held-out sequences change with the supplied
action, and shuffling actions makes prediction worse.

Status: implementation ready. The frozen-checkpoint training path, one-step
metrics, copy baseline, shuffled-action control, validation curves, and visual
comparisons are implemented. Final training and completion measurements wait
for a dedicated run against the selected 256-feature autoencoder checkpoint.

### Milestone 5 — open-loop evaluation

Deliverables:

- recursive rollouts over several horizons;
- real and imagined frame strips;
- error-versus-horizon graph; and
- comparisons against copying the previous frame.

Completion test: the model beats the copy-frame baseline for at least a short
horizon and produces visually interpretable rollouts.

### Milestone 6 — interactive rollout

Deliverables:

- local interactive window;
- real seed-frame selection;
- keyboard and mouse action input; and
- live display of recursively imagined frames.

Completion test: changing the controls produces visibly different imagined
futures without reading new frames from Minecraft.

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
- transformers, diffusion models, or large pretrained models;
- full Minecraft mechanics;
- long-horizon coherence;
- inventory and crafting;
- multiplayer support; and
- production-scale data infrastructure.

If a proposed feature does not directly help us record sequences, train the
tiny latent model, or interact with its rollout, we should leave it out.
