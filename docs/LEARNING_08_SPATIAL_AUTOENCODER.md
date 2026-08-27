# Learning 08: fixing the visual state before fixing dynamics

## Why we stopped and redesigned the representation

The first interactive rollout was useful because it exposed a real model
failure: after one imagined step the image became blurry, and holding `W` did
not produce convincing forward motion. This was not primarily a frontend bug.

For one held-out transition, the measurements were:

| path | next-frame pixel MSE |
|---|---:|
| reconstruct the current real frame | 0.002133 |
| decode the real next-frame latent (decoder oracle) | 0.002305 |
| predict and decode the next latent | 0.003780 |

The decoder oracle matters. It gives the decoder the *correct* next latent, so
the dynamics model is not involved. Because even this image was already soft,
training dynamics for longer could not restore details that the representation
had discarded.

Sharpness told the same story:

| image | gradient energy |
|---|---:|
| real next frame | 1284 |
| decoded oracle | 266 |
| dynamics prediction | 212 |

The old 256-value vector autoencoder therefore became the bottleneck. We keep
it as a reproducible V1 baseline, but we do not build the next dynamics model
on top of it.

## What changed

The old encoder collapsed the complete image into a flat vector:

$$
o_t \in \mathbb{R}^{3\times64\times64}
\xrightarrow{E}
e_t \in \mathbb{R}^{256}.
$$

A flat vector can contain spatial information, but the network must learn where
everything is while passing through a severe 48-to-1 bottleneck. Minecraft
motion is highly spatial: camera motion shifts edges across the screen, blocks
occupy neighboring locations, and the HUD remains near the bottom.

The new encoder keeps a small feature map:

$$
o_t \in \mathbb{R}^{3\times64\times64}
\xrightarrow{E}
e_t \in \mathbb{R}^{16\times16\times16}.
$$

This is 4,096 latent values. The raw image contains $3\cdot64\cdot64=12,288$
values, so the representation still compresses the image by a factor of three:

$$
\text{compression ratio} = \frac{12,288}{4,096}=3.
$$

That is intentionally a gentler bottleneck. Our immediate goal is a world
model that can retain and move visual structure, not the smallest possible
code.

The 253,395-parameter network is fully convolutional:

```text
RGB 3x64x64
  -> 32x32x32
  -> 64x16x16
  -> 128x16x16
  -> latent 16x16x16
  -> 128x16x16
  -> 64x16x16
  -> 32x32x32
  -> RGB 3x64x64
```

Convolutions preserve the idea of locality: nearby latent cells describe
nearby image regions. This will also let the next dynamics model predict local
changes with convolutions instead of treating every location as unrelated.

## Training objective

Pixel MSE rewards correct colours and is the main term:

$$
\mathcal{L}_{\mathrm{MSE}}
=\frac{1}{N}\sum_i(\hat{o}_{t,i}-o_{t,i})^2.
$$

Pixel L1 makes the objective less dominated by a few large errors:

$$
\mathcal{L}_{\mathrm{L1}}
=\frac{1}{N}\sum_i\left|\hat{o}_{t,i}-o_{t,i}\right|.
$$

We also compare horizontal and vertical finite differences. For example,

$$
\nabla_x o_{c,y,x}=o_{c,y,x+1}-o_{c,y,x}.
$$

The edge term is

$$
\mathcal{L}_{\mathrm{edge}}
=\frac{1}{2}\left(
\lVert\nabla_x\hat{o}_t-\nabla_xo_t\rVert_1
+\lVert\nabla_y\hat{o}_t-\nabla_yo_t\rVert_1
\right).
$$

The implemented objective is:

$$
\mathcal{L}_{AE}
=\mathcal{L}_{\mathrm{MSE}}
+0.1\mathcal{L}_{\mathrm{L1}}
+0.25\mathcal{L}_{\mathrm{edge}}.
$$

This does not magically make images sharp. It gives losing block boundaries
and HUD edges an explicit cost in addition to pixel error.

## Gates and measured results

We deliberately tested the architecture in increasing order of cost.

### Gate 1: memorize 32 frames

The first spatial attempt used an $8\times8\times32$ latent and a bounded output
activation. It collapsed toward a dark average image. Removing that output
activation and making MSE the main loss fixed the collapse. Increasing the map
to $16\times16\times16$ then allowed the small network to retain blocks, items,
trees, water, and HUD structure.

After 2,000 optimization steps on the same 32 frames:

| metric | value |
|---|---:|
| pixel L1 | 0.006326 |
| reconstructed/real gradient energy | 0.896 |

An overfit is a plumbing and capacity test, not proof of generalization. If a
model cannot memorize 32 examples, scaling the dataset is pointless. Passing
this gate justified a held-out experiment.

### Gate 2: unseen frames

A CPU pilot trained on 20,000 sampled V3 training frames for five epochs and
was evaluated on the frozen V3 validation episodes:

| model | held-out L1 | held-out MSE | PSNR | edge-energy ratio |
|---|---:|---:|---:|---:|
| old flat 256-value AE | 0.02128 | 0.001491 | 28.27 dB | not recorded |
| new spatial AE pilot | **0.01613** | **0.000934** | **30.30 dB** | 0.711 |
| new spatial AE, selected | **0.00674** | **0.000180** | **37.46 dB** | **0.974** |

The selected checkpoint trained for 12 epochs on 100,000 deterministically
sampled frames. All 978 validation frames come from the frozen held-out player
session; none were training examples. Its reconstruction grid is nearly
pixel-aligned, including block boundaries, water, foliage, tools, and HUD text.
The train/validation gap is also small (train L1 $0.00639$ versus validation L1
$0.00674$).

This passes the representation gate. It is **not** evidence that the world
model works yet: this experiment tests only $D(E(o_t))$, not future prediction.

## Commands

Run the small memorization gate:

```bash
uv run mcwm sanity-spatial-autoencoder \
  --steps 2000 \
  --output-dir artifacts/spatial-autoencoder-sanity
```

Train on a deterministic sample of 100,000 V3 frames:

```bash
caffeinate -i uv run mcwm train-spatial-autoencoder \
  --output-dir artifacts/spatial-autoencoder-v3 \
  --epochs 12 \
  --batch-size 64 \
  --max-training-frames 100000 \
  --patience 4
```

Evaluate a saved checkpoint again:

```bash
uv run mcwm evaluate-spatial-autoencoder \
  --checkpoint artifacts/spatial-autoencoder-v3/best.pt \
  --output-dir artifacts/spatial-autoencoder-v3/evaluation
```

Each training directory contains a checkpoint, `metrics.json`, a training
curve, and a held-out reconstruction grid.

## What comes next

The next component is a **spatial action-conditioned dynamics model**. It will
receive two consecutive latent maps and the synchronized action:

$$
\hat e_{t+1}=F_\phi(e_{t-1},e_t,a_t).
$$

The action can be broadcast across the $16\times16$ grid as extra channels,
and a small convolutional residual network can predict a change:

$$
\hat e_{t+1}=e_t+\Delta_\phi(e_{t-1},e_t,a_t).
$$

We will then repeat the same honest tests: decoded copy baseline, shuffled
actions, one-step prediction, and recursive rollout. Only after those pass do
we reconnect the browser frontend. The Minecraft recorder remains the final
milestone, after the public-data model is credible.
