# Minecraft World Model: First-Principles Roadmap

## Read this first

This project will build a small learned simulator of Minecraft movement and use it to control a player. It is intended to teach world models in the same conceptual setting in which they are used in robotics.

The project has three distinct parts:

~~~text
1. World model:     What will happen if I take this action?
2. Controller:      Which action should I take to reach my goal?
3. State estimator: What is the current state, given my sensors?
~~~

They will be implemented separately before they are combined.

The first complete offline version is:

> Load synchronized public Minecraft state/action trajectories, learn the
> player's short-term movement dynamics, and verify action-conditioned imagined
> rollouts on held-out players.

This first version is already a world-model project. It uses a learned model of explicit player state. A later version will replace the explicit state with a recurrent latent state learned from images.

The next control milestone adds a live bridge and MPC. MPC consumes the world
model; MPC is not required for the learned dynamics to qualify as a world model.

The initial project will not attempt to generate realistic Minecraft video, play open-ended survival, or reproduce large systems such as Oasis, MineWorld, or Dreamer 4.

## 1. What is a world model?

Imagine that Minecraft contains an unknown transition function:

~~~text
next_state = minecraft_dynamics(current_state, action)
~~~

For example:

~~~text
current state:
    moving forward at 3 blocks/second
    looking 20 degrees left
    standing on the ground

action:
    hold forward
    turn camera right

next state:
    new position
    new velocity
    new orientation
    new grounded/collision state
~~~

We collect examples of these transitions and train a neural network:

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

The learned function is a world model. It predicts the consequences of actions.

A world model does not have to generate pixels. World models can operate over:

- Explicit physical state, such as position and velocity
- A hand-designed sensor representation
- A learned latent representation
- Pixels or video tokens

Robotics projects often start with measured state or proprioception before adding vision. Our progression follows the same logic.

### World model versus controller

The world model predicts. A controller decides.

~~~text
                         candidate actions
                                |
                                v
observation --> state --> world model --> predicted futures
                                ^                 |
                                |                 v
Minecraft <-- action <----------+---- controller or policy
~~~

The same world model can support several controllers:

- Model-predictive control
- Random-shooting trajectory optimization
- Cross-entropy-method planning
- A policy trained from imagined experience

Changing the controller does not change whether the learned dynamics are a world model.

### Explicit state versus latent state

An explicit-state model receives quantities whose meanings we chose:

~~~text
position, velocity, yaw, pitch, grounded, collision
~~~

A latent world model learns its own internal state:

~~~text
recent images and actions --> encoder --> latent state
latent state + action --> predicted future latent state
~~~

The explicit model is easier to inspect. The latent model is more powerful because it can represent details that we did not manually specify. We will build both, in that order.

### Forward dynamics model

A forward dynamics model predicts the state that results from applying an action in the current state:

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

where:

- $s_t$ is the current state at time $t$
- $a_t$ is the action applied after time $t$
- $f_\theta$ is a learned model with parameters $\theta$
- $\hat{s}_{t+1}$ is the model's prediction of the next state
- $s_{t+1}$ is the next state that Minecraft actually produces

For Minecraft, $s_t$ may contain position, velocity, orientation, and contact flags, while $a_t$ contains movement keys, jump, sprint, and camera changes.

Instead of predicting the complete next state directly, our first model will predict a change in state:

$$
\widehat{\Delta s_t} = f_\theta(s_t, a_t)
$$

$$
\hat{s}_{t+1} = s_t + \widehat{\Delta s_t}
$$

This residual form is often easier to learn because many state components change only slightly over one short control interval.

Training compares the predicted next state with the state observed from Minecraft. A simple continuous-state objective is:

$$
\mathcal{L}_{\text{dynamics}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
s_{t+1}^{(i)}
-
f_\theta\left(s_t^{(i)}, a_t^{(i)}\right)
\right\|_2^2
$$

In practice, different state components receive separate normalization and loss weights. Boolean predictions such as grounded or collided use a classification loss rather than squared error.

The model can imagine several steps by feeding each prediction back into itself:

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t)
$$

$$
\hat{s}_{t+2} = f_\theta(\hat{s}_{t+1}, a_{t+1})
$$

$$
\hat{s}_{t+H}
=
f_\theta^{(H)}
\left(
s_t,
a_{t:t+H-1}
\right)
$$

The difference between $\hat{s}_{t+H}$ and the real $s_{t+H}$ is multi-step rollout error. MPC uses these imagined rollouts to compare candidate action sequences.

A forward dynamics model follows the causal direction:

~~~text
current state + action --> predicted next state
~~~

An inverse dynamics model solves a different problem:

$$
\hat{a}_t = g_\phi(s_t, s_{t+1})
$$

It infers which action probably caused an observed transition. VPT uses inverse dynamics to infer actions from gameplay video. Our core model is forward dynamics because planning requires predicting the consequences of proposed future actions.

## 2. How model-predictive control uses a world model

Model-predictive control, or MPC, searches over possible future actions.

Suppose the player must reach a waypoint. At each real control step, MPC:

1. Observes the current state.
2. Creates many candidate action sequences.
3. Rolls each candidate forward inside the learned world model.
4. Scores the predicted trajectories.
5. Executes only the first action from the best sequence.
6. Observes the real result and replans.

Example candidates:

~~~text
forward, forward, turn-left
forward, jump, forward
turn-right, forward, forward
~~~

Executing only the first action is important. Learned models make errors, and those errors compound over long imagined rollouts. Frequent replanning lets real observations correct the model.

MPC is one way of using a world model. Dreamer instead trains an actor and critic on trajectories imagined by its world model. We will start with MPC because its behavior is easier to inspect, then consider an imagination-trained policy later.

## 3. Relationship to robotics

| Robotics concept | Minecraft equivalent |
| --- | --- |
| RGB camera | Rendered first-person frame |
| Proprioception | Position, velocity, orientation, grounded state |
| Range or depth sensing | Ray distances or local block observations |
| Motor command | Movement keys, jump, sprint, camera deltas |
| Dynamics | Gravity, acceleration, collisions, terrain effects |
| State estimation | Inferring motion and geometry from recent observations |
| Forward model | Predicting the next state from state and action |
| MPC | Simulating and scoring candidate Minecraft actions |
| Receding-horizon feedback | Acting once, observing, and replanning |
| Offline robot data | Recorded Minecraft state-action trajectories |

Minecraft is not a realistic physics simulator. It is still useful because it contains partial observability, contact events, gravity, terrain, action delays, compounding prediction errors, and closed-loop control.

It also provides privileged internal state for debugging. Real robots may obtain comparable signals from motion capture, joint encoders, force sensors, or calibrated tracking systems.

## 4. What the literature does

There are two related research families that are both called world models.

### Control-oriented world models

These models learn representations and dynamics for decision-making. They do not necessarily generate photorealistic video.

#### PlaNet

[PlaNet: Learning Latent Dynamics for Planning from Pixels](https://planetrl.github.io/) learns a recurrent stochastic state-space model from images and chooses actions by planning with the cross-entropy method in latent space.

What we borrow:

- A recurrent latent state for partial observability
- Multi-step latent prediction
- Online planning over candidate actions
- Reward prediction

PlaNet is the closest conceptual ancestor of our eventual visual-plus-MPC system.

#### DreamerV3

[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) uses a recurrent state-space model with deterministic memory and discrete stochastic representations. It reconstructs observations and predicts rewards and episode continuation. An actor and critic learn entirely from latent imagined trajectories.

Its Minecraft experiment learned online from pixels and sparse rewards, used MineRL with a simplified categorical action space and abstract crafting actions, and ran for up to 100 million environment steps. This is useful conceptual evidence but not an appropriate scale or task definition for this project.

What we borrow later:

- Posterior state estimation versus prior-only imagination
- Reward and continuation heads
- Actor-critic training inside the model
- Evaluation of model usefulness through agent performance

#### DayDreamer

[DayDreamer](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf) applied Dreamer's recurrent world model to physical robots. It fused camera observations and proprioceptive sensors, learned from a replay buffer of real experience, and optimized behaviors from latent imagined rollouts.

This is the closest match to our robotics motivation.

What we borrow:

- Treat Minecraft state as robot proprioception
- Record multiple synchronized sensor modalities
- Learn from replayed trajectory sequences
- Separate real data collection from imagined behavior learning
- Use reconstructed observations for inspection, not as the only success metric

#### TD-MPC2

[TD-MPC2](https://www.tdmpc2.com/) performs trajectory optimization inside a learned, decoder-free latent world model. It demonstrates that a model can be useful for control without reconstructing every observation.

What we borrow:

- Judge representations by control usefulness
- Plan in a compact state space
- Do not require pixel generation when prediction heads are sufficient

### Interactive Minecraft video world models

These systems focus on generating the next visible frame accurately and interactively. They are impressive, but their compute and objective differ from ours.

#### Oasis

[Oasis](https://oasis-model.github.io/) uses a Transformer spatial autoencoder and a latent diffusion Transformer. It autoregressively generates frames conditioned on keyboard inputs. The released model has 500 million parameters.

Its central goal is a playable neural video renderer. It is not primarily a small robotics-control system.

What we borrow:

- Per-step action conditioning
- Autoregressive rollout evaluation
- Awareness that small visual errors compound

#### MineWorld

[MineWorld](https://arxiv.org/abs/2504.08388) converts 224×384 frames into 336 VQ-VAE tokens and each action into 11 tokens, then trains a 300-million to 1.2-billion-parameter autoregressive Transformer. Its reported training used 32 A100 GPUs for 200,000 steps.

MineWorld evaluates both visual quality and whether an inverse dynamics model can recover the requested actions from generated frames.

What we borrow:

- Correct-action versus shuffled-action tests
- Explicit controllability evaluation
- Evaluation by action type

What we do not borrow:

- The large token model
- High-resolution autoregressive rendering
- Large-scale distributed training

#### Dreamer 4

[Dreamer 4](https://danijar.com/project/dreamer4/) trains a scalable video world model and then trains behaviors inside it. Its Minecraft work uses offline VPT gameplay data, action-conditioned generation, learned rewards, and imagination training. The paper reports experiments using roughly 2,541 hours of VPT video.

Dreamer 4 is the best long-term conceptual reference for combining a Minecraft simulator model with agent training, but it is many orders of magnitude larger than this learning project.

What we borrow:

- Separate world knowledge from action grounding
- Learn reward signals for imagined trajectories
- Train behavior offline inside the model

### VPT as a data source

[OpenAI Video PreTraining](https://github.com/openai/video-pre-training) is primarily a policy-learning and inverse-dynamics project, not a world-model architecture. Its released contractor data is nevertheless valuable because it contains synchronized video, keyboard and mouse actions, timestamps, and Minecraft metadata.

VPT action/state recordings are the primary implemented V0 dataset. Video can
later become auxiliary training data for visual representation learning.

## 5. What we are building

The project is divided into versions and internal phases.

### Project V0: explicit-state offline world model — complete

V0 proves the perception-free system-identification portion of the
robotics-style loop:

~~~text
public data --> estimate state --> train dynamics --> imagine --> evaluate
~~~

The implemented V0 contains phases 0A and 0B. Live planning and action execution
move to V0.1 so the first learned model stays small and independently testable.

#### Phase 0A: contracts and synthetic dynamics

Before integrating Minecraft:

- Define state, action, transition, and episode schemas.
- Implement coordinate transforms and angle wrapping.
- Generate simple synthetic movement trajectories.
- Train persistence, constant-velocity, and MLP baselines.
- Test open-loop rollouts.

This validates the learning interface without Minecraft installation or synchronization problems.

#### Phase 0B: Minecraft locomotion dynamics

Load public VPT Minecraft trajectories and learn:

$$
\hat{s}_{t+1}=f_\theta(s_t,a_t)
$$

The implemented state contains:

- Global position for integration and goal scoring
- Causally derived world-frame velocity
- Yaw and pitch

The first action contains:

- Forward, back, left, right
- Jump
- Sprint
- Sneak
- Yaw delta
- Pitch delta

The model predicts residual changes:

- World-frame position delta
- Velocity delta
- Camera deltas are integrated directly from commands

Start with a small MLP. Add a GRU only if experiments show that the current measured state does not contain enough history.

### Project V0.1: live bridge and MPC

#### Phase 0C: MPC on open terrain

Use the learned model to reach nearby targets on open ground and across simple surface changes.

Open-terrain navigation is deliberately first because proprioception alone does not describe nearby walls. The goal here is to validate planning, integration, and replanning, not obstacle perception.

The first planner uses random shooting:

- 64–256 candidate sequences
- 5–15 model steps
- Discrete movement macro-actions
- Binned camera changes
- Execute one action and replan

The trajectory cost can include:

- Distance to target
- Heading error
- Unexpected jump or fall penalty
- Control-change penalty

#### Phase 0D: local geometry and obstacles

Before asking the controller to navigate walls, the observation must contain geometry.

Add a simple robot-like exteroceptive sensor:

- A small set of horizontal and downward ray distances; or
- A compact egocentric local occupancy grid

The world model then predicts both player motion and how those local observations change.

This stage teaches sensor fusion and collision-aware prediction without yet requiring an image encoder. It also prevents the model from merely memorizing a fixed arena using absolute coordinates.

V0.1 is complete when the learned model supports useful closed-loop planning in
a controlled environment. The offline V0 is complete because it beats simple
dynamics baselines on held-out trajectories and remains action-sensitive during
recursive rollouts.

### Project V1: visual and latent world model

V1 replaces hand-designed sensing with learned visual state.

#### Phase 1A: supervised visual state estimation

Record RGB frames alongside all V0 data. Train a multi-frame encoder:

$$
\hat{s}_t=E(o_{t-k:t},a_{t-k:t-1})
$$

Use privileged Minecraft state as supervision for:

- Velocity
- Orientation change
- Grounded and collision state
- Ray distances or local occupancy

Test the controller with:

1. Ground-truth state and learned dynamics
2. Visually estimated state and learned dynamics

This separates perception error from dynamics error.

#### Phase 1B: recurrent latent world model

Replace the fixed explicit bottleneck with a small recurrent state-space model:

- Image encoder
- Deterministic recurrent belief state
- Small stochastic latent state
- Action-conditioned transition prior
- Posterior that incorporates real observations
- Observation, state, reward, and continuation heads

During training, the posterior corrects the latent state using real observations. During imagination, only the transition prior and proposed actions are available.

The preferred initial stochastic state is a diagonal Gaussian because it is easier to understand and debug. Categorical latents can be considered later.

MPC can continue planning in the latent state. Pixel-perfect generation is not required if the model predicts state and rewards accurately enough for control.

### Project V2: imagination-trained behavior

V2 adds a learned actor and critic:

- Start imagined trajectories from states encoded from real data.
- Generate future latent states using the world model.
- Predict rewards and continuation.
- Train an actor to choose high-return actions.
- Train a critic to estimate long-term return.
- Evaluate the learned policy only in the real Minecraft environment.

At this point we can compare MPC with Dreamer-style behavior learning.

Only after this works should the task expand toward block interactions, richer terrain, or longer-horizon goals.

## 6. Data strategy

### Primary implemented data: public VPT action/state trajectories

V0 uses official OpenAI VPT JSONL recordings. The tested subset has 24
recordings, 10 player/session groups, 25,067 aggregated transitions, and about
102 minutes of usable gameplay.

Reasons to begin here:

- It provides real Minecraft positions, orientations, timestamps, and actions.
- It avoids building a live bridge before testing the learning fundamentals.
- Multiple human sessions support held-out-player generalization tests.
- JSONL state/action files are small enough for CPU experiments.

Controlled collection becomes primary for V0.1, when repeatable resets, targeted
maneuvers, reward evaluation, and live action execution are necessary.

### Canonical transition

Every sample represents:

~~~text
state[t]
action[t]
dt[t]
state[t + 1]
~~~

The non-negotiable timing invariant is:

> action[t] is the control aligned with the observed change from state[t] to
> state[t + 1].

VPT rows are approximately 20 Hz. V0 combines four native rows into a typical
200 ms model interval while retaining the actual timestamp-derived duration.

### Later controlled-collection mixture

Purely random key presses are poor training data. Use a mixture of:

- Structured random action primitives held for sensible durations
- Scripted acceleration, stopping, turning, strafing, and jumping
- Deliberate collisions and recovery
- Human trajectories
- Later, trajectories generated by the current planner

New data should be targeted at failure cases discovered in evaluation.

### Implemented and later sizes

- Synthetic pipeline smoke test: a few thousand transitions
- Implemented public model: 25,067 transitions
- First controlled dynamics model: 20,000–50,000 transitions
- More robust V0: around 100,000 transitions
- Initial visual model: download or collect RGB only when needed

At 5 Hz, 50,000 transitions is about 2.8 hours of play. Coverage matters more
than raw duration.

### Later VPT video use

The VPT contractor release contains 360p video sampled at 20 Hz, JSONL keyboard and mouse actions, timestamps, player position and orientation, inventory and statistics metadata, and clips of up to five minutes.

Potential later video uses:

- Pretrain the visual encoder on more varied Minecraft scenes
- Test representation transfer
- Compare controlled data with uncontrolled human play

VPT is sufficient for offline V0 but cannot provide reproducible closed-loop
evaluation. That limitation matters at V0.1, not for proving the forward model.

### Episode format

Raw recordings remain immutable. The in-memory V0 episode contract contains:

~~~text
source: string
states: float32 [T, 8]
actions: float32 [T - 1, 9]
dts: float32 [T - 1]
~~~

Targets are derived on demand from adjacent states. Filtering and segmentation
happen while importing the immutable JSONL source.

Split complete recognized player/session groups:

- 70% training
- 15% validation
- 15% test

Never randomly split neighboring transitions across train and test.

## 7. Training and baselines

### V0 baselines

1. Persistence: state remains unchanged.
2. Constant velocity: integrate the current velocity.
3. Learned residual MLP dynamics.
4. Shuffled-movement-action ablation.

Train six continuous movement residuals with Smooth L1 loss. Compute
normalization statistics from the training split only.

Train one-step predictions, then evaluate recursive unrolls at 1, 5, 10, and 20
steps.

### V1 model progression

1. Visual encoder with supervised state heads
2. Deterministic recurrent latent transition
3. Stochastic recurrent state-space model
4. Reward and continuation heads
5. Latent MPC
6. Imagination-trained actor and critic

Each step must demonstrate a measurable benefit before the next is added.

## 8. Evaluation

### Prediction evaluation

Report:

- One-step error for every state component
- Position and orientation drift at multiple horizons
- Error separated by action and maneuver type
- Error on unseen episodes and starting locations
- Correct-action versus shuffled-action prediction loss
- Predicted and real trajectory plots

Correct actions should predict the future better than shuffled or zeroed actions. Otherwise the model may be ignoring control inputs.

### Control evaluation

Report:

- Waypoint success rate
- Final distance to target
- Time or actions required
- Collision count
- Recovery after model error
- Performance versus planning horizon

Compare against:

- Random actions
- A simple reactive controller
- MPC using constant-velocity dynamics
- MPC using learned dynamics

A reactive controller may solve trivial open-ground tasks. Early open-ground MPC is an educational integration test; later terrain and sensor-based tasks must make prediction genuinely useful.

### Completion gates

V0 is successful when:

- The data alignment contract is verified.
- The learned dynamics beat persistence and constant velocity.
- Correct movement actions beat shuffled movement actions.
- Multi-step rollouts remain useful over a short imagination horizon.
- Results repeat across held-out player/session splits.
- Training, evaluation, and checkpoint reload are reproducible.

V0.1 is successful when:

- A live bridge applies synchronized actions and observes resulting states.
- MPC with learned dynamics improves at least one controlled task.
- The system replans successfully after moderate prediction errors.

V1 is successful when:

- Visual state estimates support closed-loop control.
- Prior-only latent imagination remains action-sensitive.
- Latent planning performs above non-model baselines.
- Privileged state is no longer required as model input.

## 9. Milestones

| Milestone | Deliverable | Exit gate |
| --- | --- | --- |
| M0 | Package, schemas, math utilities, synthetic trajectories | Synthetic episode passes through training and rollout |
| M1 | Public VPT importer and audit | State/action timing is inspectable and unambiguous |
| M2 | Persistence, constant-velocity, and MLP dynamics | Learned model beats baselines on held-out data |
| M3 | Multi-step dynamics evaluation | Rollouts remain useful over the planning horizon |
| M4 | Live bridge and open-terrain MPC | Closed-loop target reaching works |
| M5 | Local geometry sensing | Model predicts collision-aware motion |
| M6 | Sensor-based obstacle task | Learned-model controller beats simple baselines |
| M7 | Visual state estimator | Estimated state supports closed-loop control |
| M8 | Recurrent latent world model | Prior-only action-conditioned imagination works |
| M9 | Imagination-trained policy | Policy trained in imagination succeeds in Minecraft |

M0 through M3 constitute offline project V0 and are complete. M4 through M6
constitute V0.1. M7 and M8 constitute V1. M9 constitutes the initial V2.

## 10. Current V0 repository structure

~~~text
minecraft-world-model/
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── README.md
│   ├── RESULTS_V0.md
│   ├── ROADMAP.md
│   ├── RUN_V0.md
│   └── V0.md
├── src/mcwm/
│   ├── data/
│   │   ├── features.py
│   │   ├── schema.py
│   │   ├── synthetic.py
│   │   └── vpt.py
│   ├── models/
│   │   ├── baselines.py
│   │   └── dynamics.py
│   ├── cli.py
│   ├── download.py
│   ├── evaluation.py
│   ├── math.py
│   └── training.py
├── tests/
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_math.py
│   ├── test_training.py
│   └── test_vpt.py
├── artifacts/            # generated and ignored
└── data/
    └── raw/vpt/          # downloaded and ignored
~~~

A future Minecraft bridge should implement a small environment interface.
Minecraft-specific details must not leak into model or controller APIs.

## 11. Main risks

### State-action misalignment

Incorrect timing can make the model appear to ignore actions. Alignment inspection and tests come before model tuning.

### Insufficient environment observation

The model cannot predict walls or terrain that are absent from its inputs. Tasks must match the available sensors. Local geometry sensing comes before obstacle navigation.

### Dataset imbalance

Straight walking can dominate the data. Collection must deliberately cover turns, stops, jumps, collisions, and recovery.

### Open-loop drift

Small errors compound. Use multi-step training, short planning horizons, and frequent replanning.

### Planner exploitation

A planner may discover unrealistic predictions that score well. Constrain actions, shorten horizons, compare predictions with reality, and later add uncertainty penalties.

### Memorizing the arena

Absolute position can let a model memorize fixed obstacles. Use held-out starts or arena variants, local-coordinate targets, and explicit local sensors.

### Moving to pixels too early

Vision combines perception and dynamics failures. Complete the explicit-state control loop first.

### Building a video model instead of a control model

High-quality pixels are not the goal. State, reward, event, and control performance are the primary measures.

## 12. Decisions deferred until implementation

- Exact Minecraft bridge technology
- Exact control interval after timing experiments
- Ray sensor versus local occupancy grid
- MLP versus GRU beyond the baseline
- Exact arena layout
- RGB resolution and HUD treatment
- Gaussian versus categorical latent variables

Each decision should be made with the smallest experiment that reveals the tradeoff.

## 13. First implementation slice

After this roadmap is approved, begin with M0:

1. Initialize the Python package and minimal dependencies with uv.
2. Define typed state, action, transition, and episode contracts.
3. Implement angle wrapping, coordinate transforms, normalization, and state integration.
4. Generate a synthetic movement dataset.
5. Implement persistence, constant-velocity, and MLP models.
6. Add tiny-overfit and multi-step rollout tests.

M0 validates the model interface before Minecraft integration introduces installation and synchronization complexity. M1 will then be a small technical spike to select and prove the Minecraft bridge.

The detailed equations, data fields, training objective, compute requirements, and uv commands for this slice are specified in [V0: Explicit-State Minecraft World Model](V0.md).
