# Learning 06: Multi-Step Recursive Evaluation

## Why one-step success is not enough

The current model predicts one next latent:

$$
\hat e_{t+1}=F(e_{t-1},e_t,a_t)
$$

During one-step evaluation, both visual inputs are real encoder outputs. This is
sometimes called teacher-forced evaluation: the model always starts from the
correct history.

An interactive world model cannot request the real next Minecraft frame. It
must feed its own prediction back into itself:

$$
\hat e_{t+2}=F(e_t,\hat e_{t+1},a_{t+1})
$$

$$
\hat e_{t+3}=F(\hat e_{t+1},\hat e_{t+2},a_{t+2})
$$

Only the first two latents are real. Every later state depends on earlier model
predictions. This is an open-loop or recursive rollout.

## The exact evaluation protocol

Select a clean held-out sequence:

$$
(o_{t-1},o_t,a_t,o_{t+1},a_{t+1},\ldots,o_{t+H})
$$

Encode only the two seed frames:

$$
e_{t-1}=E(o_{t-1}),\qquad e_t=E(o_t)
$$

Then repeat for $k=0,\ldots,H-1$:

$$
\hat e_{t+k+1}
=F(\tilde e_{t+k-1},\tilde e_{t+k},a_{t+k})
$$

where:

$$
\tilde e_j=
\begin{cases}
e_j & j\leq t\\
\hat e_j & j>t
\end{cases}
$$

The recorded action sequence is allowed because actions are inputs to the world
model. Real future observations are used only as evaluation targets; they are
never fed back into the recursive prediction.

At each horizon, decode the predicted latent:

$$
\hat o_{t+k}=D(\hat e_{t+k})
$$

and compare it with the real held-out frame $o_{t+k}$.

## Why errors compound

Suppose the first prediction has a small error:

$$
\epsilon_1=\hat e_{t+1}-e_{t+1}
$$

The next call to $F$ receives $\hat e_{t+1}$ instead of the real $e_{t+1}$.
The model may never have seen that slightly incorrect state during training.
Its second error can therefore contain both a new prediction mistake and an
amplified version of the first:

$$
\epsilon_{k+1}
\approx J_F\epsilon_k+\epsilon_{new}
$$

Here $J_F$ informally represents how sensitive the dynamics function is to
errors in its latent input. If that sensitivity is large, rollout error can
grow rapidly even when one-step metrics look good.

## Horizons we will measure

The processed dataset runs at 10 Hz, so:

| horizon | imagined time |
|---:|---:|
| 1 step | 0.1 seconds |
| 2 steps | 0.2 seconds |
| 5 steps | 0.5 seconds |
| 10 steps | 1.0 second |
| 20 steps | 2.0 seconds |

The goal is not indefinite Minecraft simulation. A coherent one- or two-second
action-responsive rollout is a successful first latent world model.

## Metrics and baselines

### Recursive learned error

Measure latent and decoded pixel error at every horizon:

$$
\operatorname{MSE}_{pixel}(k)
=\frac{1}{N}\lVert D(\hat e_{t+k})-o_{t+k}\rVert_2^2
$$

Plot the mean over many held-out starting points. A curve shows whether error
grows gradually or explodes.

### Teacher-forced one-step reference

For each future position, predict using the real two-frame history. This does
not simulate interaction, but it separates ordinary one-step difficulty from
recursive error accumulation.

If teacher-forced error stays flat while recursive error rises, feeding back
predictions is the problem.

### Frozen-copy baseline

Keep decoding the initial current latent:

$$
\hat e_{t+k}^{copy}=e_t
$$

The learned rollout must beat this baseline for a useful short horizon.

### Decoder oracle

Encode and decode each real future frame:

$$
o_{t+k}^{oracle}=D(E(o_{t+k}))
$$

This is not a dynamics prediction. It measures error that comes from the visual
bottleneck itself and gives a lower bound for the present decoder.

### Shuffled-action rollout

Run the same seeds with actions taken from other validation sequences. Correct
actions should produce lower error:

$$
\operatorname{MSE}_{correct}(k)
<\operatorname{MSE}_{shuffled}(k)
$$

This tells us whether control remains important after repeated prediction.

## Visual evidence

The evaluator should create two artifacts.

First, an error-growth graph:

```text
pixel error
    ^                 shuffled actions
    |             ___/
    |       model/
    |  copy ----/
    +------------------------> rollout horizon
       1   2      5   10   20
```

Second, rollout filmstrips:

```text
real:       t      t+1      t+2      t+5      t+10      t+20
predicted:  t      p+1      p+2      p+5      p+10      p+20
copy:       t      copy     copy     copy     copy      copy
error:      0      err1     err2     err5     err10     err20
```

We should include movement, camera rotation, jumping, and near-idle examples.
A single attractive rollout is not sufficient; the graph aggregates many
held-out sequences.

## What we expect to see

Some degradation is normal:

- fine texture may blur first;
- block edges may shift slightly;
- the scene may gradually lose contrast;
- camera movement may be under- or over-estimated; and
- errors may become obvious after 10 or 20 recursive steps.

The concerning failures are:

- freezing into nearly the same decoded image regardless of actions;
- explosive latent values or sudden unrelated colors;
- action branches that remain identical;
- immediate loss of scene geometry after two or three steps; or
- performing no better than frozen copy at every horizon.

## Completion gate

Milestone 5 passes when, on held-out sequences:

1. the learned rollout beats frozen copy for a non-trivial short horizon;
2. correct actions beat shuffled actions across that horizon;
3. error grows gradually rather than becoming unstable immediately;
4. decoded frames remain recognizable for approximately 1–2 seconds; and
5. different action sequences create visibly different imagined futures.

We will not choose an arbitrary metric threshold before observing the full
curves. Baseline-relative results and stability across examples matter more
than one absolute pixel number.

## What happens if recursion fails

The first correction is short-horizon rollout training, not a large new model.
Starting from two real latents, feed predictions back during training and
optimize every future target:

$$
\mathcal L_{rollout}
=\sum_{k=1}^{H}w_k
\left(
\lambda_{latent}\lVert\hat e_{t+k}-e_{t+k}\rVert_2^2
+
\lambda_{pixel}\lVert D(\hat e_{t+k})-o_{t+k}\rVert_2^2
\right)
$$

We would begin with $H=4$ or $H=8$. This exposes the model to its own imperfect
states and teaches it to recover rather than drift. Only if that fails should
we consider a recurrent dynamics network or jointly learning a more predictable
latent representation.

## How this becomes interactive

Offline recursive evaluation and the interactive viewer use the same loop. The
only difference is where actions come from:

```text
offline evaluation: recorded held-out actions
interactive viewer: live keyboard and mouse actions
```

The viewer will:

1. load two real seed frames;
2. encode them;
3. read one user action;
4. predict and decode the next latent;
5. display the imagined frame;
6. shift the latent pair forward; and
7. repeat without reading another real Minecraft frame.

Recursive evaluation is therefore the offline safety check for the exact loop
that will power interaction.

## Planned command and outputs

The next implementation will add a command shaped like:

```bash
uv run mcwm evaluate-rollout \
  --processed-dir data/processed/vpt_v2 \
  --manifest data/manifests/vpt_v2.jsonl \
  --autoencoder-checkpoint artifacts/autoencoder-v2/best.pt \
  --dynamics-checkpoint artifacts/dynamics-v2-new-ae/best.pt \
  --output-dir artifacts/rollout-v2 \
  --horizons 1 2 5 10 20
```

This command is a specification for the next milestone; it is not implemented
yet. Expected outputs are:

```text
artifacts/rollout-v2/metrics.json
artifacts/rollout-v2/error-vs-horizon.png
artifacts/rollout-v2/rollout-filmstrips.png
```

## Check your understanding

1. Why are real future actions allowed while real future latents are forbidden
   during recursive evaluation?
2. What does it mean if teacher-forced error stays low but recursive error rises
   quickly?
3. Why do we compare with both frozen copy and decoder oracle?
4. How is the recursive evaluator almost identical to the interactive viewer?
5. Why would rollout training help with states the one-step model never saw?
