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

The two seeds initialize that motion context. They stay internal; the browser
shows only the current real or imagined frame.

## Action representation

The playground constructs the exact same nine-value vector used in training:

$$
a_t=[W,A,S,D,\text{jump},\text{sprint},\text{sneak},
\Delta x_{mouse},\Delta y_{mouse}].
$$

Held keys are sampled at every model step. While paused, each movement-key press
advances exactly once. Continuous playback defaults to one step per second so
the short coherent horizon can be inspected; the speed button can restore the
recorded 10 Hz rate. Camera movement is accumulated and consumed once per step.

The controls are:

| key | effect |
|---|---|
| `W/A/S/D` | hold a movement direction |
| `Shift` | sprint while held |
| `Control` | sneak while held |
| `Space` | jump while held |
| arrow keys | look left/down/up/right while held |
| click the viewport | capture the mouse for relative camera control |
| `Escape` | release the captured mouse |
| `P` or Pause button | pause or resume; starting this way produces an idle rollout |
| `N` or Step once button | advance exactly one frame using the currently held controls |
| speed button | toggle between one step per second and the recorded model rate |
| `R` or Reset button | reset to the original seed pair |
| previous/random/next seed | choose a different held-out scene and reset the rollout |
| Stop server button | stop the local model process |

The lightweight browser frontend reports both key presses and releases, so the
controls behave normally. It shows only the current $64\times64$ frame enlarged
inside one viewport; the two seed frames remain internal model state. The page
waits at the real current seed until an action key or Step once is pressed.
Continuous playback sends actions at the selected rate.

The page is vanilla HTML, CSS, and JavaScript served by Python's standard HTTP
server on `127.0.0.1`. It requires no internet connection, frontend build,
Node.js, React, WebSocket service, or external account. The Python process owns
the model and returns one newly decoded PNG after each action.

## Run it

The selected spatial checkpoints and broad V4 validation data are the defaults:

```bash
uv run mcwm play-rollout
```

The command opens `http://127.0.0.1:8765` automatically. Use `--no-open` if you
want to open that address yourself. Stop it with the page button or `Ctrl-C` in
the terminal.

Choose an exact clean held-out starting point with:

```bash
uv run mcwm play-rollout --sample-index 20000
```

There are currently 65,073 clean held-out seed transitions. The page can switch
seeds without restarting or reloading the checkpoints. The command prints the
selected episode and exact step so a run is reproducible.

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

## What the current result means

The selected spatial model beats frozen copy at every measured recursive
horizon through 20 steps. On final-test windows, mismatched actions are 22.9%
worse at step 20, proving that live controls enter a learned action path rather
than a decorative UI path.

The rollout still becomes smooth after several recursive steps. This agrees
with the offline evaluation: the model has learned useful average dynamics, but
a deterministic squared-error predictor loses sharp details as uncertainty
accumulates.

The important distinction is:

- **software success:** the interactive closed loop works correctly;
- **model quality:** action responses are currently subtle and blurry.

Seed switching makes it easy to check that conclusion across many different
Minecraft scenes with the same loaded model.

## A useful manual experiment

Use one seed and compare three runs, resetting with `R` between them:

1. click `Start idle` and let the idle model run for one second;
2. reset, then hold `W` for one second; and
3. reset, then hold `W` and the right arrow together for one second.

If all three imagined futures are identical, the model is ignoring our action.
If they differ slightly but plausibly, the action-conditioning path works. If
they immediately become unstable, recursive distribution shift is the main
problem.

The aggregate held-out evaluation already shows that correct recorded actions
beat mismatched actions. This manual experiment makes that statistical result
tangible.
