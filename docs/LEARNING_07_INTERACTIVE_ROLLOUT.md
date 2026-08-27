# Learning 07 — Interactive latent rollout

This milestone turns the trained world model into something we can control.
It does not connect to a running copy of Minecraft. Instead, it seeds the model
with two real held-out observations and then lets the model imagine everything
that follows.

## What is interactive

The offline evaluator used recorded actions:

$$
(z_{t-1},z_t,a_t^{recorded})\rightarrow\hat z_{t+1}.
$$

The playground uses an action assembled from our live controls:

$$
(z_{t-1},z_t,a_t^{user})\rightarrow\hat z_{t+1}.
$$

After the first prediction, it shifts its internal state forward:

$$
(z_t,\hat z_{t+1},a_{t+1}^{user})\rightarrow\hat z_{t+2}.
$$

The viewer never reads $x_{t+1}$ or any later real frame. The only real inputs
are the seed pair $x_{t-1},x_t$.

## Why two seed frames

A single screenshot cannot reveal whether the player is already turning,
falling, or moving. The model was therefore trained with both a current latent
and a latent motion estimate:

$$
m_t=z_t-z_{t-1}.
$$

The two seeds initialize that motion context. This is why the window displays
`real seed t-1`, `real seed t`, and the current imagined frame separately.

## Action representation

The playground constructs the exact same nine-value vector used in training:

$$
a_t=[W,A,S,D,\text{jump},\text{sprint},\text{sneak},
\Delta x_{mouse},\Delta y_{mouse}].
$$

The first seven controls are toggles. They remain active at every 10 Hz model
step until toggled off. Camera movement is accumulated and then consumed once,
matching a mouse delta over one transition.

The controls are:

| key | effect |
|---|---|
| `W/A/S/D` | toggle a movement direction |
| `E` | toggle sprint |
| `C` | toggle sneak |
| `Space` | toggle jump |
| `H/J/K/L` | add look left/down/up/right to the next step |
| left-mouse drag | add a raw camera delta to the next step |
| `N` | imagine exactly one step |
| `P` | start or pause continuous 10 Hz imagination |
| `X` | clear all active controls |
| `R` | reset to the original seed pair |
| `G` | save the current viewer canvas |
| `Q` | quit |

Movement uses toggles because OpenCV's small cross-platform window reliably
reports key presses but not key-release state. Starting paused and pressing `N`
is the easiest way to see exactly which action produces each prediction.

## Run it

The V2 checkpoints and validation data are the defaults:

```bash
uv run mcwm play-rollout
```

Choose another clean held-out starting point with:

```bash
uv run mcwm play-rollout --sample-index 143
```

There are currently 1,034 clean held-out seed transitions. The command prints
the selected episode and exact step so a run is reproducible.

## Scripted mode

The same engine can run without a GUI. A comma separates successive action
specifications, `+` combines controls, and `*N` repeats an action for $N$ model
steps:

```bash
uv run mcwm play-rollout \
  --sample-index 143 \
  --script 'w+sprint*10,w+sprint+look_right*5,idle*5' \
  --output artifacts/interactive-rollout/scripted-rollout.png
```

This means:

1. walk forward while sprinting for one second;
2. continue forward while turning right for half a second; and
3. provide no controls for half a second.

Supported scripted camera tokens are `look_left`, `look_right`, `look_up`, and
`look_down`. Each produces a raw mouse delta of 30 by default. Override it with
`--camera-step`.

Scripted mode is useful for debugging because the exact counterfactual action
sequence can be repeated after every new checkpoint.

## What the first result means

The verified scripted run successfully:

1. loaded the paired autoencoder and dynamics checkpoints;
2. selected a clean held-out seed;
3. encoded only the two real seed frames;
4. applied 20 user-specified actions recursively; and
5. decoded and saved every imagined latent.

A same-seed counterfactual check after ten recursive steps measured:

| branches compared | latent MSE between branches | pixel MSE between branches |
|---|---:|---:|
| idle versus forward + sprint | 0.003150 | 0.002044 |
| forward + sprint versus forward + sprint + turn | 0.008827 | 0.004800 |

The nonzero differences prove that live actions reach the model and alter its
imagined state. Whether each difference looks like the intended Minecraft
motion is still a qualitative model-quality question, which is why the live
viewer remains useful.

The rollout remains stable, but it is visibly blurred and changes less than an
actual Minecraft video would. This agrees with the recursive evaluation: the
model has learned useful average dynamics, but it tends to predict conservative
futures and loses sharp details.

The important distinction is:

- **software success:** the interactive closed loop works correctly;
- **model quality:** action responses are currently subtle and blurry.

More data may improve the second point. The V3 experiment can now be evaluated
with exactly the same interactive command rather than inventing a new demo.

## A useful manual experiment

Use one seed and compare three runs, resetting with `R` between them:

1. idle for ten single steps;
2. toggle `W`, then take ten single steps; and
3. toggle `W`, add `L` before each step, then take ten steps.

If all three imagined futures are identical, the model is ignoring our action.
If they differ slightly but plausibly, the action-conditioning path works. If
they immediately become unstable, recursive distribution shift is the main
problem.

The aggregate held-out evaluation already shows that correct recorded actions
beat mismatched actions. This manual experiment makes that statistical result
tangible.
