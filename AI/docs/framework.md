# DriftCast Reinforcement Learning Framework

This document will capture the evolving RL system design for Oceans Four DriftCast.

- **Task A (Model Correction):** Agents observe forecast vs. observed drift errors and issue parameter nudges (e.g., diffusivity, windage) to minimize future discrepancies.
- **Task B (Cleanup Optimization):** Agents coordinate cleanup vessels, booms, or autonomous surface vehicles to intercept predicted plastic accumulations while respecting energy and safety constraints.

Key sections to flesh out:

1. Data sources and preprocessing.
2. Simulator abstractions and differentiable components.
3. Observation, action, and reward definitions for each task.
4. Training pipeline (baseline algorithms, curriculum, evaluation).

Implemented baselines:

- `rl_drift_correction.py` crafts a Gym PPO agent that applies velocity corrections on top of the differentiable simulator while tracking ensemble spread penalties.
- `rl_cleanup.py` supports single- and multi-agent PPO rollouts for coordinating cleanup fleets with plastic depletion rewards and operational costs.
- `error_utils.py` supplies ensemble, RMSE, and confidence ellipse tooling consumed by both RL tasks and reporting utilities.
- `train_pipeline.py` stages data, policy optimisation, logging, and prediction demos for end-to-end experiments.
- `viz.py` captures trajectory plots, heatmaps, animations, and reporting helpers to communicate outcomes.