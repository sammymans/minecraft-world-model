# V0 Public-Data Results

This document records what the implemented V0 actually learned. It separates
measured evidence from future plans.

## Result

V0 is a functional, action-conditioned forward dynamics model for short-horizon
Minecraft player motion. It was trained on public OpenAI VPT state/action
recordings and evaluated on player/session groups excluded from training.

Across three deterministic splits and model initializations, the learned model
beat the constant-velocity baseline in every experiment:

| Seed | Position RMSE, learned | Baseline | Velocity RMSE, learned | Baseline | 20-step error, learned | Baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.169 blocks | 0.272 | 0.836 blocks/s | 1.342 | 1.909 blocks | 7.759 |
| 17 | 0.140 blocks | 0.197 | 0.675 blocks/s | 0.960 | 1.492 blocks | 4.201 |
| 29 | 0.115 blocks | 0.186 | 0.551 blocks/s | 0.896 | 1.183 blocks | 4.076 |
| Mean | 0.142 blocks | 0.219 | 0.687 blocks/s | 1.066 | 1.528 blocks | 5.345 |

On average, this is a 35% reduction in one-step position error, a 36%
reduction in one-step velocity error, and a 71% reduction in 20-step rollout
position error relative to constant velocity.

With the default four-frame aggregation, 20 steps are approximately four
seconds. The code always integrates using recorded timestamps rather than
assuming an exact frequency.

## What the model learns

The numerical state is:

$$
s_t=
\begin{bmatrix}
x_t & y_t & z_t & v_t^x & v_t^y & v_t^z & \psi_t & \phi_t
\end{bmatrix}^{\mathsf T}
$$

Velocity is a causal backward difference:

$$
v_t=\frac{p_t-p_{t-1}}{t_t-t_{t-1}}
$$

It never uses $p_{t+1}$ to construct the input at time $t$.

The action contains forward, back, left, right, jump, sprint, sneak, commanded
yaw change, and commanded pitch change. Four native VPT rows are combined into
one model interval. Held controls are averaged and camera deltas are summed.

The kinematic baseline predicts:

$$
\Delta p_t^{\mathrm{kin}}=v_t\Delta t
$$

The neural network predicts a six-dimensional movement residual:

$$
r_\theta=f_\theta
\left(
v_t,\sin\psi_t,\cos\psi_t,\phi_t,\Delta t,a_t
\right)
$$

and the final movement prediction is:

$$
\widehat{\Delta s}_t^{\mathrm{move}}
=
\Delta s_t^{\mathrm{kin}}+r_\theta
$$

Camera rotation is integrated directly from the mouse command. This is an
explicit dynamics path: the public-data audit found offset-zero command/
observation correlations of 0.976 for yaw and 0.997 for pitch.

## Public dataset audit

The tested local subset contains:

- 24 official VPT action/state recordings
- 10 player/session groups
- 314 clean contiguous segments
- 25,067 aggregated transitions
- 6,136 seconds, or about 102 minutes, of usable gameplay
- 14/6/4 recordings in the primary train/validation/test split

All recordings belonging to one recognized player/session identifier remain in
one split. This prevents nearly adjacent clips from the same player leaking
between training and testing.

Two official files contain legacy non-UTF-8 bytes in an unused keyboard text
field. The importer replaces those text bytes while preserving the valid JSON,
numerical state, and key/mouse action fields. GUI-open rows, timestamp gaps,
teleports, and malformed state rows are excluded or used as segment boundaries.

Run and save the full audit with:

~~~bash
uv run mcwm audit-vpt
~~~

## Evidence that actions matter

Beating a motion baseline alone does not prove action conditioning. Evaluation
therefore permutes only the held movement controls while leaving each current
state, timestamp, and camera command intact.

For the primary seed-7 test split:

| Input actions | Position RMSE | Velocity RMSE |
|---|---:|---:|
| Correct movement actions | 0.169 blocks | 0.836 blocks/s |
| Shuffled movement actions | 0.328 blocks | 1.489 blocks/s |

Incorrect movement controls almost double both errors. The network therefore
uses the proposed action rather than only copying the current velocity.

## Recursive rollout test

Starting from a real state, the predictor is applied recursively:

$$
\hat{s}_{t+k+1}
=
\operatorname{integrate}
\left(
\hat{s}_{t+k},
\hat{\Delta s}_{t+k}
\right)
$$

Only recorded future actions and elapsed times are supplied. Intermediate
ground-truth states are not given to the model, so error compounds as it will in
short imagined trajectories.

The primary model checkpoint was also loaded in a separate process and
reevaluated. Its predictions and metrics exactly matched the training report.

## Verification performed

- Tests for angle wrapping, delta integration, schemas, normalization, causal
  velocity, action aggregation, legacy-byte handling, split leakage, synthetic
  learning, and checkpoint round trips
- Ruff linting
- Synthetic known-dynamics training
- Public-data timing and camera-alignment audit
- Three public-data training runs with different held-out groups and seeds
- Persistence, constant-velocity, and shuffled-action comparisons
- One-step and recursive rollout metrics
- Saved-checkpoint reload and reevaluation
- Visual inspection of the generated rollout plot

Run the main gates with:

~~~bash
uv run pytest
uv run ruff check .
uv run mcwm evaluate-v0
~~~

## Honest scope

V0 models player locomotion from structured telemetry. It is a small robotics-
style system-identification world model, not a visual Minecraft simulator.

It does not yet:

- Observe blocks or images
- Predict inventory, mobs, mining, or crafting
- Represent uncertainty or multiple possible futures
- Run closed-loop MPC inside a live Minecraft environment
- Generalize indefinitely in open-loop rollouts

Those are later milestones. V0 establishes synchronized state/action ingestion,
learned action-conditioned dynamics, short imagination, baseline-based
evaluation, and reproducible training on limited CPU compute.
