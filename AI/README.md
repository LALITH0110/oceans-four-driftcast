# DriftCast: Reinforcement Learning for North Atlantic Plastic Drift

## Overview

DriftCast couples a differentiable advection–diffusion model with reinforcement learning to improve surface-plastic forecasts and optimize cleanup asset routing.  
- **Task A – Drift Correction:** learn velocity adjustments that align HYCOM/ERA5 forcing with NOAA GDP drifter trajectories.  
- **Task B – Cleanup Optimization:** plan asset movements that skim plastic hotspots predicted by the corrected physics.

All forcing data already lives in `data/raw/...`; the pipeline never downloads anything and gracefully skips months that are missing (HYCOM currents ≥ 2022‑06, CMEMS Stokes drift ≥ 2022‑11, winds and drifters intermittently present).

## Data Layout

- Currents: `data/raw/hycom/cmems_currents_YYYY-MM_natl.nc` (`uo`, `vo`)  
- Stokes drift: `data/raw/waves/cmems_stokes_YYYY-MM_natl.nc` (`vsdx`, `vsdy`) – optional fallback to `stokes_scale * winds`  
- Winds: `data/raw/era5/era5_u10_v10_YYYY-MM_natl.nc` (`u10`, `v10`)  
- Drifters: `data/raw/drifters/gdp_drifters_YYYY-MM_YYYY-MM-DD_YYYY-MM-DD_natl_clean.csv`

Longitude axes are normalized to [-180, 180] before subsetting the NATL box (N=60°, S=0°, W=–80°, E=10°).

## How to Run

```powershell
# 1) Setup
python -m venv .venv
. .venv/Scripts/activate            # Windows
pip install -r requirements.txt

# 2) Quick data check (no downloads; just indexes what’s present)
python -m main prepare

# 3) Baseline sim (keeps defaults small and fast)
python -m main simulate --hours 24 --n-particles 2000

# 4) Train RL drift correction (Task A)
python -m main train-a --timesteps 100000

# 5) Evaluate correction
python -m main eval-a

# 6) Train cleanup optimization (Task B)
python -m main train-b --episodes 300

# 7) Predict with uncertainty
python -m main predict --time "2025-10-11T10:10:00" --pos "40,-50" --hours 24

# 8) Tests & style
pytest -q
ruff check .
black --check .
mypy src || true   # be pragmatic on typing
```
> **PowerShell tip:** inline Bash heredocs such as `python - <<'PY'` do not work. Use `python -c "..."` or `@'... '@ | python` instead.

The CLI prints where artefacts live (plots under `docs/plots/`, models under `models/`).

## Components

| Module | Purpose |
|--------|---------|
| `src/config.py` | Pydantic settings, dataset indexing, longitude normalization helpers. |
| `src/data_loader.py` | Chunked xarray readers, nearest-time interpolation, drifter ingestion. |
| `src/simulator.py` | JAX advection–diffusion kernel with windage, optional Stokes drift, and RNG-controlled turbulence. |
| `src/rl_drift_correction.py` | Gymnasium env + PPO wrappers for Task A, kd-tree matching to drifters, checkpoints & eval. |
| `src/rl_cleanup.py` | Simple cleanup env and PPO trainer for Task B, plastic-grid rewards. |
| `src/error_utils.py` | Monte Carlo ensembles, RMSE/ellipse metrics, textual forecasts. |
| `src/viz.py` | Trajectory plots, plastic heatmaps, optional GIF animation (cartopy fallback). |
| `src/train_pipeline.py` | Prepare → simulate → train A → evaluate → train B orchestration with logging. |
| `src/main.py` | CLI subcommands (`prepare`, `simulate`, `train-a`, `eval-a`, `train-b`, `predict`). |
| `tests/test_all.py` | Pytest smoke tests for loaders, simulator, RL env steps, and error metrics. |

## Example Outputs

- `docs/plots/baseline_trajectories.png` – baseline advection vs RL-corrected tracks.  
- `docs/plots/plastic_heatmap.png` – plastic density before/after cleanup.  
- `docs/plots/uncertainty_cloud.png` – ensemble spread after 24 h forecast.

## Limitations & Caveats

- HYCOM and CMEMS wave products are missing before mid‑2022; the pipeline logs warnings and falls back to wind-derived drift.  
- Sparse drifter coverage south of ~10°N limits reward density; Task A down-weights unmatched particles.  
- Mesoscale eddies remain under-resolved; policy defaults target 24–48 h windows to keep computations tractable.  
- GPU acceleration is optional but recommended for long PPO runs (see requirements header for CUDA installs).
