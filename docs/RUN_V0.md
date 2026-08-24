# Running and Understanding V0

This is the shortest hands-on path through the current implementation.

## 1. Create the environment

From the repository root:

~~~bash
uv sync
~~~

This creates an isolated Python environment from **pyproject.toml** and **uv.lock**.

## 2. Verify the known-dynamics pipeline

~~~bash
uv run mcwm synthetic-v0 --epochs 30 --output artifacts/synthetic
~~~

This generates small trajectories from equations already known to us, trains the same MLP used for Minecraft, and compares it with simple baselines.

This proves that the data, model, loss, checkpoint, and rollout machinery work. It does not prove that the model learned Minecraft.

## 3. Download public Minecraft trajectories

~~~bash
uv run mcwm download-vpt --limit 24
~~~

Only JSONL action/state recordings are downloaded for V0. MP4 video is not needed until the visual model.

The files are official OpenAI VPT contractor recordings. They remain under **data/** and are ignored by Git.

## 4. Audit the dataset and alignment

~~~bash
uv run mcwm audit-vpt
~~~

This writes **artifacts/v0/data-audit.json** and reports recording/session
counts, usable duration, action coverage, time-step quantiles, camera alignment,
and the exact train/validation/test recording assignments.

List a filename:

~~~bash
ls data/raw/vpt/episodes
~~~

Then inspect it:

~~~bash
uv run mcwm inspect-vpt data/raw/vpt/episodes/RECORDING.jsonl
~~~

The command reports valid segments, transitions, duration, and correlations between commanded and observed camera changes at offsets from -2 to +2.

The correct alignment should peak at offset zero:

$$
a_t \longrightarrow s_{t+1}-s_t
$$

If a nearby offset were stronger, the action/state timeline would need correction before training.

## 5. Train the real V0 model

~~~bash
uv run mcwm train-v0 --epochs 80
~~~

By default, training combines four native 20 Hz VPT rows into one interval. Held
buttons are averaged over the interval and camera deltas are summed. This trains
the model at approximately 5 Hz, where movement actions have a measurable effect.

Recordings are split by recognized player/session group:

- Training groups fit model parameters and normalization.
- Validation groups choose the best epoch.
- Test groups are used only for the final report.

Clips with the same player/session identifier cannot occur in different splits.

Each transition is:

$$
(s_t,a_t,\Delta t)\longrightarrow\Delta s_t
$$

The numerical state is position, derived velocity, yaw, and pitch. Inputs omit absolute position and instead use velocity, $\sin(\text{yaw})$, $\cos(\text{yaw})$, pitch, elapsed time, and action.

## 6. Read the result

Open:

- **artifacts/v0/metrics.json**
- **artifacts/v0/rollout.png**

The report compares:

- Persistence: predicts no change.
- Constant velocity: continues current motion and camera command.
- Learned MLP: predicts corrections to the constant-velocity delta from state and action.
- Shuffled actions: gives the learned model incorrect actions.

The learned model should beat the simple baselines, and correct actions should beat shuffled actions.

The tested results are recorded in [RESULTS_V0.md](RESULTS_V0.md).

## 7. Reload and independently evaluate the checkpoint

~~~bash
uv run mcwm evaluate-v0
~~~

This starts a new process, restores **artifacts/v0/model.pt**, loads its exact
held-out recording manifest, repeats the one-step/action-ablation/
rollout evaluation, and writes **artifacts/v0/reloaded/evaluation.json**.

Rollout errors are reported after 1, 5, 10, and 20 recursive steps. With the
default action repeat these correspond to roughly 0.2, 1, 2, and 4 seconds,
although actual timestamps are used.

## 8. Read the code in learning order

1. **src/mcwm/data/schema.py** — what an episode contains
2. **src/mcwm/data/vpt.py** — conversion from VPT JSONL
3. **src/mcwm/data/features.py** — model inputs, targets, normalization
4. **src/mcwm/models/baselines.py** — minimum comparisons
5. **src/mcwm/models/dynamics.py** — the small neural network
6. **src/mcwm/training.py** — optimization and checkpoints
7. **src/mcwm/evaluation.py** — metrics and recursive rollouts
8. **src/mcwm/cli.py** — commands that connect the pieces

## 9. Useful experiments

Train for fewer epochs:

~~~bash
uv run mcwm train-v0 --epochs 5 --output artifacts/five-epochs
~~~

Use a smaller model:

~~~bash
uv run mcwm train-v0 --hidden-dim 32 --output artifacts/small-model
~~~

Compare the resulting metrics and rollout plots. This makes model capacity and underfitting visible.
