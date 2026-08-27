# Learning 04: A Reproducible Local Data Pipeline

## Why we added this now

The first five episodes were downloaded and processed one command at a time.
That was appropriate while proving synchronization, cleaning, and autoencoder
training. It becomes fragile when we increase the dataset:

- nobody can tell which remote episodes define an experiment;
- adding a lexicographically earlier filename could silently change the split;
- a partially downloaded file could be mistaken for a valid input; and
- reproducing the dataset requires copying commands from chat history.

The solution is a small, versioned manifest. It is dataset infrastructure, but
it remains local and directly supports the model rather than introducing cloud
services.

## The three layers

```text
committed manifest
data/manifests/vpt_v1.jsonl
             |
             | dataset-download
             v
ignored immutable raw pairs
data/raw/vpt/{episode}.mp4 + {episode}.jsonl
             |
             | dataset-preprocess
             v
ignored versioned model data
data/processed/vpt_v1/{episode}.npz
             |
             v
autoencoder and dynamics training
```

Only the small manifest belongs in Git. Raw videos, actions, processed arrays,
and model artifacts remain local and ignored.

## What one manifest line means

Each JSONL record describes exactly one synchronized public episode:

```json
{
  "episode": "cheeky-cornflower-setter-032f...-125807",
  "group": "cheeky-cornflower-setter-032f...",
  "split": "training",
  "video_url": "https://.../episode.mp4",
  "actions_url": "https://.../episode.jsonl",
  "video_bytes": 172767401,
  "actions_bytes": 19754285
}
```

The `episode` identifies a matched pair. The `group` identifies consecutive
segments from the same player/session. Every member of one group must have the
same split:

$$
g_i=g_j\implies\operatorname{split}(i)=\operatorname{split}(j)
$$

The loader rejects a manifest that violates this rule. This prevents nearly
identical consecutive recordings from leaking between training and validation.

The expected byte counts are simple integrity checks. A download is first
written as `filename.part`. It becomes the final file only when its size matches
the manifest:

$$
\operatorname{bytes}(f_{downloaded})=\operatorname{bytes}(f_{manifest})
$$

This is not a cryptographic authenticity guarantee, but it reliably detects an
interrupted or truncated transfer for this project.

The expanded run exposed one useful public-data edge case: an official action
record contained a legacy non-UTF-8 byte inside the unused `keyboard.chars`
field. Its local MD5 matched the official blob, so it was source data rather
than download corruption. The parser replaces malformed text bytes but still
strictly parses each JSON object and every action field used by the model. A
regression test preserves that behavior.

## Dataset version `vpt_v1`

The first version deliberately remains bounded:

```text
14 matched episode pairs
12 independent player/session groups
11 training groups
 1 validation group
2.16 GiB expected raw data
```

After processing, the local snapshot contains:

```text
27,157 non-GUI autoencoder training frames
 5,605 clean eight-step dynamics training sequences
   978 frozen autoencoder validation frames
   618 frozen eight-step validation sequences
```

The raw cache occupies approximately 2.3 GB as reported by the operating
system, while the compressed processed cache occupies approximately 200 MB.

The validation group is the same `02e...` session used by the earlier
autoencoder experiments. Freezing it lets us compare new models against old
models on exactly the same held-out images.

The other groups were selected from the official public VPT 10.0 container. We
use one segment from most new groups instead of downloading every consecutive
segment from only a few players. Visual diversity matters more than adjacent
minutes of nearly identical gameplay at this scale.

## Dataset version `vpt_v2`

The scaling experiment uses a live Azure object listing to exclude stale,
missing, or zero-length public blobs before writing a static manifest. With a
fixed seed, it selects one segment from each new session group until reaching
the requested byte budget. The original validation group is preserved exactly.

The verified local `vpt_v2` snapshot contains:

```text
173 episodes across 171 independent groups
171 training episodes from 170 groups
  2 validation episodes from the original frozen group
25.135 GiB raw video and action data
 2.129 GiB processed model-ready data
12.81 hours of training gameplay
461,191 training frames
148,069 clean one-step dynamics examples
117,348 clean eight-step sequences
```

The previous split provided 7,314 one-step examples, so this is a 20.2-times
increase in usable dynamics examples while evaluation stays fixed.

## Dataset version `vpt_v4`: separate validation and test

The earlier frozen evaluation set contained 978 frames from two consecutive
episodes belonging to one session. That is much more than two images, but it is
still only one person's gameplay distribution. It is too narrow for choosing
between models and then honestly claiming generalization.

There is no universal requirement that machine-learning data use exactly
70/15/15. Independence and sufficient coverage matter more than the familiar
ratio. At this scale, an 80/10/10 split leaves roughly 10 GiB in both validation
and test, so we retain more training data without making evaluation small.

The split is performed on complete player/session groups:

$$
g_i=g_j\implies\operatorname{split}(i)=\operatorname{split}(j),
$$

and uses three distinct roles:

- training fits parameters;
- validation chooses checkpoints and architecture; and
- test is opened only after those decisions are frozen.

Generate the assignment manifest without copying or preprocessing data again:

```bash
uv run mcwm dataset-split-manifest \
  --source-manifest data/manifests/vpt_v4.jsonl \
  --output data/manifests/vpt_v4_split.jsonl \
  --validation-fraction 0.10 \
  --test-fraction 0.10 \
  --seed 7
```

The verified result is:

| split | independent groups | episodes | raw size | clean eight-step sequences |
|---|---:|---:|---:|---:|
| training | 565 | 566 | 80.35 GiB | 398,354 |
| validation | 70 | 71 | 9.85 GiB | 52,331 |
| test | 70 | 70 | 9.82 GiB | 49,906 |

The original held-out group remains in validation because prior experiments
already used it for model development. It is never allowed back into training
or relabeled as fresh test evidence. The source inventory manifest remains
unchanged, making the split command deterministic and reproducible.

## Run the complete pipeline

Create another deterministic byte-budgeted manifest with:

```bash
uv run mcwm dataset-expand-manifest \
  --base-manifest data/manifests/vpt_v1.jsonl \
  --output data/manifests/vpt_v2.jsonl \
  --target-gib 25 \
  --seed 7
```

Download or verify every selected raw pair:

```bash
uv run mcwm dataset-download
```

For `vpt_v2`, supply its paths explicitly:

```bash
uv run mcwm dataset-download --manifest data/manifests/vpt_v2.jsonl
uv run mcwm dataset-preprocess \
  --manifest data/manifests/vpt_v2.jsonl \
  --output-dir data/processed/vpt_v2
uv run mcwm dataset-verify \
  --manifest data/manifests/vpt_v2.jsonl \
  --processed-dir data/processed/vpt_v2
```

The command is safe to rerun. Correctly sized files are skipped, and three
episodes download concurrently by default. `--force` explicitly replaces an
incorrect or intentionally refreshed download.

Build the versioned 10 Hz, $64\times64$ processed episodes:

```bash
uv run mcwm dataset-preprocess
```

This is also resumable: existing processed episodes are skipped unless
`--force` is supplied.

Check the entire local snapshot:

```bash
uv run mcwm dataset-verify
uv run mcwm dataset-summary
```

Verification checks:

- manifest JSON and required fields;
- unique episode identifiers;
- group-safe explicit splits;
- existence and exact size of every raw pair;
- existence and readability of every processed episode; and
- agreement between processed metadata and manifest episode identity; and
- the expected 10 Hz, $64\times64$ RGB processed format.

## Two frame-selection policies

The visual autoencoder and dynamics model do not require the same data.

The autoencoder learns:

$$
o_t\longrightarrow E(o_t)\longrightarrow D(E(o_t))
$$

It does not use $a_t$. Mining, using a tool, or pressing a key outside our small
action space does not invalidate the image. The representation learner now uses
all ordinary frames while excluding frames adjacent to a `GUI_OPEN` transition.

The dynamics model will learn:

$$
(e_{t-1},e_t,a_t)\longrightarrow e_{t+1}
$$

Here the action is essential. Dynamics training therefore continues to require
fully supported, contiguous transitions with valid timing. The strict sequence
cleaning was not removed; it was assigned only to the component that needs it.

```text
Autoencoder: all non-GUI training frames
Dynamics:    clean movement/camera sequence windows
Validation:  unchanged clean held-out sequence frames
```

Keeping validation unchanged is important. Otherwise a metric change could be
caused by evaluating different images rather than learning a better model.

## Why not S3 yet?

The manifest separates dataset identity from storage location. Today its URLs
are the official public source and its working cache is local. Later, a recorder
could add an `s3_uri` or another durable source without changing how models read
processed episodes.

We will add remote storage when we have irreplaceable recordings, multiple
training machines, or collaborators who need the same snapshot. Training will
still materialize a local cache first. Deferring S3 keeps the present lesson
about data and world models rather than credentials and cloud operations.

## Check your understanding

1. Why is the manifest committed while `.mp4`, `.jsonl`, and `.npz` files are
   ignored?
2. Why must all episode segments from one group share a split?
3. Why can an attack frame help the autoencoder but invalidate a dynamics
   transition?
4. Why do we keep the original validation group unchanged while adding training
   groups?
