# Minecraft World Model

A small, robotics-oriented project for learning action-conditioned world models from Minecraft trajectories.

The current V0 learns:

$$
\widehat{\Delta s_t}^{\mathrm{move}}
=
\Delta s_t^{\mathrm{kin}}+f_\theta(s_t,a_t,\Delta t)
$$

from public OpenAI VPT state/action recordings. It uses numerical player state rather than pixels so the dynamics, data alignment, and rollout behavior remain easy to inspect.

## Start here

- [Documentation index](docs/README.md)
- [First-principles roadmap](docs/ROADMAP.md)
- [Detailed V0 math and design](docs/V0.md)
- [Hands-on V0 walkthrough](docs/RUN_V0.md)
- [Measured public-data results](docs/RESULTS_V0.md)

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

~~~bash
uv sync
~~~

No GPU is required for V0.

## Quick workflow

Download the tested 24-recording official VPT subset:

~~~bash
uv run mcwm download-vpt --limit 24
~~~

Audit timing, action coverage, alignment, and leakage-safe splits:

~~~bash
uv run mcwm audit-vpt
~~~

Train and evaluate the model:

~~~bash
uv run mcwm train-v0 --epochs 80
~~~

Reload and independently evaluate the saved checkpoint:

~~~bash
uv run mcwm evaluate-v0
~~~

Training defaults to four native VPT steps per learned transition
(approximately 5 Hz). Use **--action-repeat 1** to experiment with native 20 Hz
transitions.

Outputs are written to **artifacts/v0/**:

- **model.pt** — model, normalization statistics, and exact data manifest
- **metrics.json** — learned and baseline metrics
- **history.json** — training curve data
- **rollout.png** — open-loop real versus predicted trajectory

Run the known-dynamics smoke test without downloading Minecraft data:

~~~bash
uv run mcwm synthetic-v0 --epochs 30
~~~

Run project checks:

~~~bash
uv run pytest
uv run ruff check .
~~~
