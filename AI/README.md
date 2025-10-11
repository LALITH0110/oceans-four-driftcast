## Oceans Four DriftCast ![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg) ![GitHub Repo stars](https://img.shields.io/github/stars/Prana/oceans-four-driftcast?style=social)

### Overview

DriftCast is an ocean plastic drift sandbox that marries differentiable advection-diffusion physics with reinforcement learning. Two primary tasks are explored:

- **Task A – Model Correction:** Learn velocity corrections that reduce forecast error against drifting buoy observations.
- **Task B – Cleanup Optimization:** Coordinate cleanup assets to intercept predicted plastic hotspots under realistic forcing.

See `docs/framework.md` for the evolving end-to-end architecture and design notes.

### Physics Background

Plastic particles advance via an advection-diffusion scheme that couples surface currents, winds, and a surface Stokes approximation:

```
dx = (u + alpha*w + s)*dt + sqrt(2kappa)*dW
```

where `u` are currents, `w` winds, `s ~= 0.01*w` Stokes drift, and `kappa` a tunable diffusivity. Differentiability via JAX keeps the simulator gradient-friendly for hybrid ML workflows.

### Setup

1. **Environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Credentials**
   Export provider secrets before requesting fresh data:
   - `CDS_API_KEY` (Copernicus ERA5)
   - `CDS_API_URL` (optional override)
   - `CMEMS_USERNAME`, `CMEMS_PASSWORD` (Copernicus Marine currents)
   - Any additional downstream API tokens (e.g., drifter feeds)
3. **Optional Extras**
   - Install `stable-baselines3` GPU builds for faster RL training.
   - Configure wandb or TensorBoard credentials if remote logging is desired.

### Usage

- **Baseline simulation**
  ```bash
  python main.py --mode simulate --steps 240 --particles 500
  ```
- **Train both RL tasks**
  ```bash
  python main.py --mode train
  # or explicitly
  python -m src.train_pipeline --data-dir data
  ```
- **Validate against drifters**
  ```bash
  python scripts/validate_against_drifters.py --drifters data/drifters.csv
  ```
- **Run regression tests** (requires JAX + test dependencies)
  ```bash
  pytest tests/test_all.py
  ```

Artifacts (models, plots, reports) are written to `models/`, `demos/`, and `runs/` by default.

### Project Structure

- `src/` – Simulation core, RL environments, training pipeline, visualization helpers.
- `data/` – NetCDF/CSV forcing data (ignored for large binaries).
- `models/` – Saved RL checkpoints.
- `docs/` – Framework documentation and research notes.
- `scripts/` – Data downloaders and validation utilities.
- `tests/` – Pytest smoke suite for loaders, simulator, RL wrappers, and analytics.

### Data Sources

- **CMEMS HYCOM / GLOBAL_ANALYSIS_FORECAST_PHY** for surface currents.
- **ERA5 Single Levels** (u10, v10) for wind forcing.
- **NOAA / GDP Drifter CSVs** (or equivalent) for trajectory validation.
- Synthetic scaffolds are provided for quick experimentation when live data are unavailable.

### Results

- Quicklook particle trajectories (`docs/quicklook.png`) highlight raw vs. corrected drift.
- Cleanup heatmaps from `src/viz.py` illustrate plastic concentration decay under trained policies.
- Ensemble uncertainty plots via `error_utils.plot_uncertainty_cloud` summarize forecast spread.

### Limitations

- Mesoscale eddies and sub-grid turbulence are approximated; high-frequency variability may be under-resolved.
- Live CMEMS/ERA5 pulls require consistent credentials and may introduce dataset latency.
- Stable-Baselines3 CPU runs are slow; GPU acceleration is recommended for full-scale training.

### Future Work

- Real-time data assimilation from drifting sensors to refresh state estimates.
- Domain randomization and transfer learning for broader ocean basins.
- Multi-objective optimization balancing cleanup efficacy and energy consumption.

### Deployment Suggestion

Wrap `error_utils.predict_location` inside a lightweight Flask API to support operational forecasts:

```python
from flask import Flask, request, jsonify
from src import error_utils

app = Flask(__name__)

@app.get("/predict")
def predict():
    time_str = request.args.get("time", "2025-10-11T10:10:00")
    lat, lon = map(float, request.args.get("pos", "40,-50").split(","))
    result = error_utils.predict_location(time_str, (lat, lon), hours_ahead=24)
    return jsonify({"summary": result})

if __name__ == "__main__":
    app.run()
```

This endpoint could back a government-facing dashboard (`/predict?time=10:10&pos=40,-50`) to deliver hourly location updates with uncertainty bounds.
