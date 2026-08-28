# Documentation

- [Project plan](PROJECT.md) — what we are building, how it works, the data
  strategy, architecture, milestones, and completion criteria.
- [Learning 02: sequence dataset](LEARNING_02_DATASET.md) — filtering unsupported
  transitions, aggregating time, processing frames, and sampling valid windows.
- [Data-scaling results](RESULTS_DATA_SCALING.md) — the committed experiment
  table, comparison rules, checkpoint pairings, and protocol for larger data.
- [Learning 07: interactive rollout](LEARNING_07_INTERACTIVE_ROLLOUT.md) — live
  action construction, recursive latent state updates, viewer controls, and a
  reproducible scripted mode.
- [Learning 08: spatial autoencoder](LEARNING_08_SPATIAL_AUTOENCODER.md) — why
  the first interactive model blurred, decoder-oracle diagnosis, the spatial
  latent redesign, edge-aware reconstruction, and the gates before retraining
  dynamics.
- [Learning 09: spatial dynamics](LEARNING_09_SPATIAL_DYNAMICS.md) — spatial
  residual prediction, action-sensitivity evidence, deterministic blur, the
  broad V4 split, and the next retraining gate.
- [Learning 10: multi-step training](LEARNING_10_MULTI_STEP_TRAINING.md) — why
  recursive predictions drift, five-step unrolled training, and the measured
  result: a horizon-scaling gain (-7.6% at five steps, -21% at twenty) that
  leaves blur as the dominant failure and points at deterministic regression.
- [Rejected visual experiments](REJECTED_VISUAL_EXPERIMENTS.md) — concise results
  from the direct-diffusion, discrete-tokenizer, and video-flow pilots.
- [V2 action-conditioned latent diffusion](V2_ACTION_CONDITIONED_LATENT_DIFFUSION.md)
  — the literature-backed architecture, staged implementation, and visual and
  action-conditioning gates for the next model.

`PROJECT.md` is the canonical scope. Features not listed there are out of scope
unless we deliberately revise the plan.
