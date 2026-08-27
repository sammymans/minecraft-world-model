# Learning 01: Frames, Actions, and Time

## What we built

Milestone 1 is a deliberately model-free data inspection pipeline. It can:

1. download one official OpenAI VPT demonstration;
2. parse its keyboard and mouse records;
3. compare the number of video frames and action records;
4. print the exact vector the future model will receive; and
5. create a short video with actions drawn over the Minecraft frames.

We started here because a model cannot repair incorrectly synchronized data. If
the action is paired with the wrong image, the model is taught the wrong cause
and effect.

The public VPT repository describes contractor demonstrations as matched video
and JSONL files and processes video frame $i$ with JSON record $i$ in its
reference loader. See the official
[VPT repository](https://github.com/openai/Video-Pre-Training#contractor-demonstrations)
and its
[data loader](https://github.com/openai/Video-Pre-Training/blob/main/data_loader.py).

## The two source files

The downloaded demonstration is one five-minute episode:

```text
data/raw/vpt/
    cheeky-cornflower-setter-...-092639.mp4
    cheeky-cornflower-setter-...-092639.jsonl
```

The MP4 contains the observations:

$$
o_0,o_1,o_2,\ldots
$$

Each line of the JSONL file is an independent JSON object containing the input
recorded at the corresponding timestep:

$$
a_0,a_1,a_2,\ldots
$$

JSONL means "JSON Lines." It is not one large JSON array. Streaming one line at
a time lets us process long episodes without loading the whole file into memory.

The raw record includes far more than this project needs: position, inventory,
statistics, GUI state, timestamps, and other metadata. We intentionally extract
only keyboard and mouse controls. The latent model must learn from pixels rather
than receiving position as a shortcut.

## The action vector

The parser turns a verbose record into:

$$
a_t = [W,A,S,D,\text{jump},\text{sprint},\text{sneak},
\Delta x_{mouse},\Delta y_{mouse}]
$$

The first seven values are binary. For example, $W=1$ means the forward key is
held. Mouse deltas are continuous values describing camera movement during that
timestep.

At source frame 100, the real episode contains:

```text
keys=W  mouse=(+3.0, -5.0)
model vector: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, -5.0)
```

The vector is not a learned representation. It is simply a clean numerical
encoding of the human input. The learned latent state will come from the images.

## Frame/action alignment

The inspected episode contains:

```text
resolution:     640 x 360
source rate:    20 frames per second
video frames:   6,001
action records: 6,000
paired frames:  6,000
duration:       300 seconds
```

We pair only the common prefix, so the extra final video frame is harmless:

$$
N_{paired}=\min(N_{video},N_{actions})=6000
$$

For visualization, JSONL line $i$ is drawn on video frame $i$. For next-frame
training, our initial causal convention will be:

$$
(o_i,a_i)\longrightarrow o_{i+1}
$$

This says: start from the current image, apply the recorded action, and predict
the next image. Before training, the sequence pipeline will compare small timing
offsets and visually check transitions. Capture systems can introduce a
one-frame delay, so causality should be tested rather than assumed solely from
matching file indices.

## Run it yourself

Create the environment and install this milestone's small dependencies:

```bash
uv sync
```

Download the official demonstration pair:

```bash
uv run mcwm download-demo
```

The command is safe to rerun. Existing files are reused unless `--force` is
provided.

All commands accept `--episode NAME` when more than one pair is present. A
second example used by this project is:

```text
cheeky-cornflower-setter-02e496ce4abb-20220421-093149
```

For example:

```bash
uv run mcwm inspect-demo \
  --episode cheeky-cornflower-setter-02e496ce4abb-20220421-093149
```

This second segment takes place in a dark cave and contains mining. Its overlay
shows `ATTACK` when the left mouse button is held. Attack is intentionally not
part of our first model's action vector, so the sequence dataset will initially
exclude attack transitions. Otherwise, block breaking would appear to be an
unexplained visual change. This is an example of using the viewer to decide what
data actually fits the model's declared action space.

Inspect the episode:

```bash
uv run mcwm inspect-demo
```

Inspect one raw timestep after parsing:

```bash
uv run mcwm show-action 100
```

Create a 15-second annotated preview beginning three seconds into the recording:

```bash
uv run mcwm make-preview --start 3 --duration 15
```

On macOS, open it with:

```bash
open artifacts/vpt-preview.mp4
```

While watching, check three things:

1. When `W` appears, the view should move forward.
2. When `JUMP` appears, the camera trajectory should show a jump.
3. Nonzero mouse deltas should coincide with camera rotation.

This is our first debugging visualization. If those relationships look wrong,
we fix the alignment before constructing a dataset or model.

## Where the code lives

```text
src/mcwm/download.py  official demonstration downloader
src/mcwm/vpt.py       JSONL parser and small action vector
src/mcwm/preview.py   episode inspection and annotated video
src/mcwm/cli.py       commands used above
tests/                synthetic alignment and parsing tests
```

The tests use a tiny generated video, not the 188 MB public episode. This keeps
the test suite fast and allows it to run without network access:

```bash
uv run pytest -q
uv run ruff check .
```

## What this milestone proves

It proves that we can obtain synchronized observations and actions and express
them in the form needed by a world model:

$$
(o_t,a_t,o_{t+1})
$$

It does not prove that anything has been learned yet. There is no neural network
in this milestone. The next milestone will turn the paired stream into short,
contiguous sequences suitable for training.

## Check your understanding

Before moving on, you should be able to answer:

1. Why do we need both the MP4 and JSONL file?
2. Why is action $a_t$ paired with a transition rather than treated as an image
   label?
3. Why do we ignore position and inventory even though they exist in the JSONL?
4. Why must train/validation splits eventually happen by episode instead of by
   random frame?
