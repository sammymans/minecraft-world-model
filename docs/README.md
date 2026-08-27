# Documentation

- [Project plan](PROJECT.md) — what we are building, how it works, the data
  strategy, architecture, milestones, and completion criteria.
- [Learning 01: public data](LEARNING_01_PUBLIC_DATA.md) — what a synchronized
  frame/action episode contains and how to inspect the first real example.
- [Learning 02: sequence dataset](LEARNING_02_DATASET.md) — filtering unsupported
  transitions, aggregating time, processing frames, and sampling valid windows.
- [Learning 03: visual autoencoder](LEARNING_03_AUTOENCODER.md) — compressing
  frames, the sanity-overfit gate, held-out reconstruction, and honest results.
- [Learning 04: local data pipeline](LEARNING_04_LOCAL_DATA_PIPELINE.md) — a
  versioned manifest, resumable local downloads, explicit splits, verification,
  and separate frame policies for representation and dynamics learning.
- [Learning 05: latent dynamics](LEARNING_05_LATENT_DYNAMICS.md) — frozen visual
  latents, action-conditioned residual prediction, one-step losses, baselines,
  and the shuffled-action test.
- [Data-scaling results](RESULTS_DATA_SCALING.md) — the committed experiment
  table, comparison rules, checkpoint pairings, and protocol for larger data.
- [Learning 06: multi-step evaluation](LEARNING_06_MULTI_STEP_EVALUATION.md) —
  recursive open-loop prediction, error growth, baselines, failure modes, and
  the gate before interaction.

`PROJECT.md` is the canonical scope. Features not listed there are out of scope
unless we deliberately revise the plan.
