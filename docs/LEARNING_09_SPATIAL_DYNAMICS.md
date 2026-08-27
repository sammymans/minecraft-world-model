# Learning 09: making the action move the picture

Learning 08 fixed the *representation*. The spatial autoencoder reaches 37.46 dB
on held-out episodes, so the decoder is no longer the thing throwing detail
away. This lesson builds the piece that consumes that representation:

$$
\hat e_{t+1}=F_\phi(e_{t-1},e_t,a_t),
\qquad e\in\mathbb{R}^{16\times16\times16},
\qquad a\in\mathbb{R}^{9}.
$$

The action vector is
$a_t=[\text{W},\text{A},\text{S},\text{D},\text{jump},\text{sprint},\text{sneak},\Delta x,\Delta y]$,
where the last two are raw mouse deltas.

## The obvious architecture, and a diagnosis that was half right

Learning 08 ended by proposing the natural design: broadcast the action across
the grid as nine extra channels and let a convolutional residual network
predict a change.

$$
\hat e_{t+1}=e_t+\Delta_\phi(e_{t-1},e_t,a_t)
$$

We built exactly that, initialized $\Delta_\phi$ to zero so training starts at
the copy baseline, and trained it on 256 transitions. It failed:

| measurement | residual network, 256 transitions | decoded copy |
|---|---:|---:|
| held-out latent MSE | 0.005565 | 0.007160 |
| held-out pixel L1 | 0.038835 | **0.033883** |

It beat copying in latent space and lost in pixel space — the prediction was
blurrier than simply showing the previous frame again. Shuffling the actions
cost it only 2.7% of its own error. The action was decorative.

**We read this as an architecture failure. It was a data failure.** That
mistake is the most useful thing in this lesson, so the rest of this document
keeps both the investigation it triggered and the ablation that corrected it.

## What one latent transition actually contains

Over 3,000 clean `vpt_v3` training transitions:

| probe | inner-region MSE | variance explained |
|---|---:|---:|
| copy $e_t$ unchanged | 0.009977 | — |
| warp $e_t$ by a global shift read straight off $(\Delta x,\Delta y)$ | 0.008344 | **16.4%** |
| warp $e_t$ by the best global shift, chosen with hindsight | 0.006507 | **34.8%** |
| constant velocity, $2e_t-e_{t-1}$ | 0.027719 | $-178\%$ |

Three things fall out of this table.

**A parameter-free warp was competitive with the trained network.** Sliding the
current latent map by a displacement computed from nothing but the two mouse
deltas removes 16.4% of the error. The 255,376-parameter residual network
trained on 256 transitions removed 22%. At that data scale, almost everything
the network had learned, a single linear camera constant already knew.

**The displacements are small and fractional.** The median best shift is 0.5
latent cells, the 90th percentile is 2.5 cells, and the fitted camera constant
is $0.0102$ cells per unit of mouse delta. Half a cell is the key number. A
$3\times3$ convolution predicting an additive residual can only produce a
half-cell translation by mixing each cell with its neighbours — which *is*
blurring. The architecture's cheapest way to reduce squared error was the exact
artifact we were trying to remove.

**The motion input is nearly useless as velocity.** Extrapolating linearly from
$e_{t-1}$ and $e_t$ is 2.8 times worse than not moving at all. Latent motion at
10 Hz is not smooth, so $e_t-e_{t-1}$ is closer to noise than to a velocity.

## The change: predict where things go, not what to add

Instead of asking the network for the *content* of the next latent, ask it
where each cell of the current latent *moves*, then sample there:

$$
\hat e_{t+1}=\mathcal{W}\big(e_t,\;f_\phi(e_{t-1},e_t,a_t)\big)+r_\phi(e_{t-1},e_t,a_t)
$$

where $f_\phi\in\mathbb{R}^{2\times16\times16}$ is a displacement field in latent
cells and $\mathcal{W}$ is bilinear resampling. Bilinear resampling performs a
fractional translation *by construction*, so half-cell motion costs the network
nothing and destroys no detail. The residual $r_\phi$ is then left to explain
only what a translation cannot: sky entering at the edge, new terrain revealed,
lighting changes.

Three details matter:

1. **A direct action-to-flow path.** The displacement field is
   $f_\phi = \text{conv}(h) + W a$, where $Wa$ is a single linear map from the
   normalized action to one global displacement. Camera motion therefore
   reaches the warp without having to survive the convolutional trunk.
2. **FiLM conditioning in every block.** The action also produces a per-channel
   scale and shift applied inside each residual block, so its influence is
   multiplicative and repeated with depth rather than nine constant channels
   glued to the input.
3. **Zero initialization everywhere.** The local flow, the global flow, and the
   residual head all start at zero, so the model begins *exactly* at the copy
   baseline (to within bilinear float precision) and has to earn every change.

## The overfit gate, and how we nearly misread it

The first capacity check trained on 256 transitions for 80 epochs. The loss
stalled at 0.23 and looked like an architecture failure. It was not. Each
transition has $16\cdot16\cdot16=4{,}096$ target values, so 256 transitions is
$1{,}048{,}576$ numbers to memorize with 260,390 parameters. Memorization was
information-theoretically impossible; the plateau was arithmetic, not a bug.

Re-run with 16 transitions — 65,536 targets, comfortably under the parameter
count — the same model drives training loss from 0.754 to 0.0037, a 200-fold
reduction. The optimizer and architecture are healthy.

The lesson generalizes: an overfit gate only proves something if the target
count is smaller than the parameter count.

## The pilot result

30,000 bounded transitions from `vpt_v3`, 87,375 encoded frames, 260,390
parameters, 20 epochs, evaluated on the 1,034 frozen held-out transitions:

| measurement | prediction | decoded copy | decoder oracle |
|---|---:|---:|---:|
| latent MSE | **0.003902** | 0.007160 | 0 |
| pixel L1 | **0.028623** | 0.033883 | 0.006611 |

The gate passes. The prediction beats copying by 46% in latent space and 16% in
pixel space, and shuffling the actions raises the latent error by $+0.001689$ —
43% of the model's own error, against 2.7% for the additive residual network.
The action is now load-bearing.

Splitting by how much the scene actually moved shows the win is not an artifact
of easy, static frames — it is *larger* where there is motion:

| | prediction | decoded copy |
|---|---:|---:|
| low-motion transitions (n=682) | 0.01657 | 0.01799 |
| high-motion transitions (n=352) | **0.05198** | 0.06467 |

## The ablation that corrected us

The pilot passes, but "we changed the architecture and it started working" is
not a measurement — the failing run also had 117 times less data. So we trained
all three variants on the *same* 30,000 transitions for the same 20 epochs,
changing nothing else:

| variant | latent MSE | pixel L1 | action effect |
|---|---:|---:|---:|
| original: additive residual, action as input channels only | 0.003941 | 0.029529 | 37.6% |
| + FiLM conditioning in every block | 0.003950 | 0.029614 | 39.9% |
| + FiLM + explicit warp (the pilot) | **0.003902** | **0.028623** | **43.1%** |

The original architecture, given data, already beats decoded copy and already
has a 37.6% action effect. **Data scale was the whole story.** FiLM is within
noise on error and buys a little action sensitivity; the warp is worth 1.2% in
latent MSE and 3.3% in pixel L1. Both are real, consistent, and small.

The 256-transition failure was never evidence about architecture. Each
transition carries $16\cdot16\cdot16=4{,}096$ target values, so 256 of them is
a million numbers — the model was not choosing a blurry hedge because it lacked
a warp, it was choosing one because it had seen almost nothing.

We keep the warp, because it wins on every metric and because the decomposition
below shows it does genuine work. But the honest ranking of levers, so far, is
**data, then architecture**. That points at the next task: the on-disk latent
cache, which is what makes the remaining 215,000 transitions reachable.

## Blur is reduced, not solved

The aggregate numbers pass, but the prediction grid still looks soft on
fast-motion frames, and a sharpness measurement agrees with the eye:

| image | mean absolute gradient |
|---|---:|
| real next frame | 0.02240 |
| decoder oracle | 0.02182 |
| decoded copy | 0.02188 |
| warp only, no residual | 0.01710 |
| full prediction | **0.01330** |

The decoder is not the problem. The oracle sits at 97% of the real frame's edge
energy, so handing the decoder a *real* latent gives a sharp image. The blur is
already present in the latent the dynamics model produces, and the decoder
renders it faithfully. Measuring cell-to-cell variation inside the latent map
itself tracks the image almost one-to-one:

| latent map | spatial detail | share of real |
|---|---:|---:|
| real next latent | 0.06074 | 100.0% |
| current latent (the copy) | 0.06097 | 100.4% |
| after warp, before residual | 0.04094 | 67.4% |
| final prediction | 0.03577 | 58.9% |

It is tempting to blame the warp for the 100% to 67% step, since bilinear
resampling is a low-pass filter and a half-cell offset is its worst case. That
reading is wrong. The no-warp ablation, which has no resampling anywhere,
arrives at the same place:

| model | pixel L1 | image sharpness | latent detail |
|---|---:|---:|---:|
| warp | 0.028623 | 61.0% of oracle | 58.9% of real |
| no-warp | 0.029614 | 60.7% of oracle | 59.1% of real |

Two structurally different models land within 0.3 points of each other. The
architecture does not choose the sharpness — **the objective does**. Squared
error is minimized by the average over every plausible next latent, and that
average is spatially smooth. The warp reaches that average by resampling and
the residual reaches it by convolution, but the loss picked the destination.

This also predicts what will *not* help. More data makes the conditional mean
more accurate; it does not stop it being a mean. The ablation ranked data above
architecture for one-step error, and that still holds — but neither lever moves
sharpness.

The warp is genuinely doing work: warping alone, before the residual is added,
already cuts latent MSE to 0.005397 from the copy baseline's 0.007160 — better
than the 0.008344 that the parameter-free camera warp achieved. The learned
mean displacement is 0.26 cells, matching the measured distribution.

Since sharpness is set by the objective, the remaining blur has to be attacked
in the loss, so
`--edge-weight` adds the same image-gradient penalty the spatial autoencoder
uses. It defaults to zero, which reproduces the numbers above. Note the scale:
the normalized latent term is O(0.4) and the gradient term is O(0.005), so a
weight that actually changes the outcome is in the tens.

## Running it

```bash
uv run mcwm train-spatial-dynamics \
  --processed-dir data/processed/vpt_v3 \
  --manifest data/manifests/vpt_v3.jsonl \
  --autoencoder-checkpoint artifacts/spatial-autoencoder-v3/best.pt \
  --output-dir artifacts/spatial-dynamics-v3-pilot \
  --maximum-transitions 30000 --epochs 20
```

```bash
uv run mcwm evaluate-spatial-dynamics \
  --dynamics-checkpoint artifacts/spatial-dynamics-v3-pilot/best.pt \
  --output-dir artifacts/spatial-dynamics-v3-pilot/evaluation
```

`--maximum-transitions` bounds the latent cache deliberately. The encoder runs
once over whole episodes and the resulting maps are held in memory as float16;
without a bound, the full 245,087-sequence `vpt_v3` split would build a 6–12 GiB
cache while the architecture is still being debugged. The pilot stays near
1 GiB. An on-disk cache is the next step, not this one.

Every run writes a checkpoint, `metrics.json`, a training curve, and a
seven-column prediction grid: previous, current, real next, decoded copy,
predicted next, decoder oracle, and a $4\times$ error map. The decoded-copy and
decoder-oracle columns are the two honest bracketing baselines — the prediction
must beat the first, and cannot beat the second.

## What comes next

In priority order, argued from the measurements above:

1. **Scale the data.** The pilot used 30,000 of 245,087 available `vpt_v3`
   transitions and was still improving when it hit epoch 20. This is the lever
   the ablation says matters most, and it needs the on-disk latent cache — the
   in-memory cache is bounded on purpose and does not go further.
2. **Attack blur through the loss.** `--edge-weight` is wired and defaults to
   zero. The prediction sits at 59% of the real frame's edge energy, so this is
   the measurable target.
3. **Then recursive rollout**, where bilinear resampling's mild smoothing
   compounds every step, and only then reconnect the browser frontend.

The Minecraft recorder remains the final milestone.
