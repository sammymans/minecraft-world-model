# Documentation

Start here:

1. [First-Principles Roadmap](ROADMAP.md) — the full project, literature context, and V0–V2 progression.
2. [V0: Explicit-State World Model](V0.md) — the data, model, training math, evaluation, MPC, compute requirements, and implementation sequence for the first version.
3. [Running V0](RUN_V0.md) — the commands, expected outputs, and recommended code-reading order.
4. [V0 Public-Data Results](RESULTS_V0.md) — the measured held-out results, ablations, and current limitations.

## Document roles

- **ROADMAP.md** answers: What are we building, why does it resemble robotics, and how do the versions fit together?
- **V0.md** answers: What exactly goes into the first model, what comes out, and how will we train and evaluate it?
- **RESULTS_V0.md** answers: Did the implementation actually work on public Minecraft data, and what evidence supports that conclusion?

## Learning workflow

This is a learning project, so implementation will be collaborative:

1. Before a component is implemented, its purpose and inputs/outputs will be stated.
2. After a component is implemented, its behavior and tests will be explained.
3. At natural checkpoints, short questions will test understanding.
4. Confusing concepts will be clarified before more complexity is added.

The goal is not only to obtain working code. You should be able to explain why each component exists and how data moves through the system.
