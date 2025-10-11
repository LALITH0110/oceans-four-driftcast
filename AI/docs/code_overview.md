# DriftCast Code Overview

This guide gives newcomers a quick tour of the repository structure and explains what happens in each module. Pair it with `README.md` when you want to dive deeper into a specific workflow.

## Root Entry Points

- `main.py`  
  Two modes: `simulate` runs the baseline advection–diffusion model, `train` invokes the full reinforcement-learning pipeline. It wires together data loading, logging, and the simulator.

- `requirements.txt`  
  Consolidated dependency list. Installs the JAX + PyTorch + Stable-Baselines3 stack alongside scientific I/O tooling (xarray, netCDF4, cdsapi).

## `src/` Package

- `__init__.py`  
  Exposes the main modules (`data_loader`, `simulator`, RL helpers, etc.) so external scripts can import them without deep paths.

- `data_loader.py`  
  Loads HYCOM/CMEMS currents, ERA5 winds, and GDP drifter CSVs. Also provides synthetic fields for demos, plus placeholder download helpers that respect environment credentials. Output tensors are converted to JAX arrays to keep the simulator differentiable.

- `simulator.py`  
  The differentiable physics core. Implements particle initialisation, step functions, and multi-step runs using advection–diffusion, windage, and optional turbulence. Batch operations use `jax.vmap` so downstream gradients remain intact.

- `rl_drift_correction.py` (Task A)  
  Defines a Gymnasium environment that nudges velocity corrections to reduce drifter-tracking error. Exposes `train_agent` and `evaluate_correction` functions that wrap Stable-Baselines3 PPO.

- `rl_cleanup.py` (Task B)  
  Models cleanup fleets moving over a plastic concentration field. Includes single-agent and lightweight multi-agent interfaces plus PPO training helpers. Integrates optional corrections from Task A during rollouts.

- `error_utils.py`  
  Houses Monte Carlo ensemble runners, RMSE/confidence ellipse metrics, quick spread estimators, and a textual `predict_location` helper. These functions plug into reward shaping, scripts, and reporting utilities.

- `train_pipeline.py`  
  End-to-end orchestrator. Handles preprocessing, Task A/B training, logging (TensorBoard/W&B), model saving, uncertainty-aware prediction demos, and graceful CPU/GPU selection.

- `viz.py`  
  Generates trajectory plots, plastic heatmaps, quick animations, and Markdown summaries. Hooks into `error_utils` to visualise ensemble spreads in reports.

## Scripts

- `scripts/download_era5_wind.py`  
  Minimal CDS API example that pulls a North Atlantic ERA5 window (u10/v10) to NetCDF.

- `scripts/download_hycom_subset.py`  
  Uses xarray + OPeNDAP to subset HYCOM currents and write them locally for loader tests.

- `scripts/validate_against_drifters.py`  
  Command-line utility comparing simulator runs against observed drifter trajectories and reporting error in km/day.

## Tests

- `tests/test_all.py`  
  Pytest smoke suite that exercises loaders, simulator math, RL environment wiring, and error metrics. It skips automatically if optional heavy dependencies (e.g., JAX) are missing.

## Documentation

- `docs/framework.md`  
  Design notes for the reinforcement-learning framework: tasks, data sources, and future sections to flesh out.

- `docs/code_overview.md` *(this file)*  
  Human-friendly map of the codebase for first-time contributors.

## Typical Workflow

1. Install dependencies (CPU or GPU stack).  
2. Drop real HYCOM/ERA5/drifter samples into `data/raw/`.  
3. Run `python main.py --mode simulate` for quick physics checks.  
4. Run `python main.py --mode train` to update Task A/B policies.  
5. Use `scripts/validate_against_drifters.py` to measure km/day error.  
6. Share plots/reports generated under `demos/` and `runs/`.

Happy drifting! Feel free to open issues or PRs once you explore the modules above.
