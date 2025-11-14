# Driftcast local run and visuals checklist

## 0. TL;DR
- Create and activate the Conda env: `conda env create -f environment.yml` then `conda activate driftcast`.
- Install the package editable with `pip install -e .` after narrowing package discovery to `driftcast/`.
- Generate demo assets with `python scripts/generate_synthetic_inputs.py` and `python -m driftcast.cli run ...`.
- Render graphs/videos via `python -m driftcast.cli animate preview` and `python -m driftcast.cli animate final`.
- Ensure `ffmpeg` is on `PATH`; verify with `python -c "from driftcast.viz.ffmpeg import detect_ffmpeg; print(detect_ffmpeg())"`.
- See troubleshooting for the current setuptools build failure and Typer extra warning.

## 1. Prerequisites
- Conda (miniconda or mambaforge) with ability to create Python 3.10 envs.
- System dependency: FFmpeg (`scoop install ffmpeg` on Windows, `brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu).
- Git configured so `git rev-parse HEAD` works (manifests capture commit metadata).
- (Optional) Dask scheduler if you plan to run `driftcast sweep --cluster ...`.

## 2. Installing dependencies with Conda
```powershell
conda env create -f environment.yml          # or: conda env update -f environment.yml --prune
conda activate driftcast
pip install -e .                             # requires package discovery fix below
make precommit                               # optional: installs git hooks
```

### Current failure you will see
The repo is a flat layout with several top-level directories. `setuptools` refuses to build:
```
error: Multiple top-level packages discovered in a flat-layout:
['data', 'configs', 'results', 'schemas', 'driftcast', 'notebooks']
```
Because `pip install -e .` triggers a build step, the same error repeats until package discovery is narrowed.

### Quick fixes
1. **Narrow package discovery (recommended).** Add the following to `pyproject.toml` so only the `driftcast` package is included:
   ```toml
   [tool.setuptools.packages.find]
   include = ["driftcast*"]
   ```
   After saving, re-run `pip install -e .`.
2. **Adopt a `src/` layout.** Move the Python package into `src/driftcast/` and update `pyproject.toml` with `package-dir = {"" = "src"}`. This is the long-term best practice but involves path updates.
3. **Temporary workaround without installing.** Until the metadata is fixed, activate the environment (`conda activate driftcast`) and run modules directly with `python -m driftcast.cli ...` (Python automatically prefers the local package on `PYTHONPATH`).

### Typer extra warning
Recent Typer releases (>=0.20.0) no longer expose the `all` extra:
```
warning: The package typer==0.20.0 does not have an extra named "all"
```
- If you need the optional extras, pin to an earlier release: set `typer[all]==0.12.3` in `pyproject.toml`.
- Otherwise ignore the warning; the base `typer` dependency still installs.

### Optional: using uv instead
If you later switch back to `uv`, run:
```powershell
uv python pin 3.11
uv venv
uv sync --editable .
```
The package discovery fix above is still required.

## 3. End-to-end workflow once dependencies resolve
Activate the environment:
```powershell
conda activate driftcast    # adjust if you chose a different env name
```

Generate synthetic inputs so downstream notebooks/animations have data:
```powershell
python scripts/generate_synthetic_inputs.py
python -m driftcast.cli ingest normalize data/raw/mock_crowd.json --out-dir data/crowd/processed
```

Run the baseline simulation and build outputs:
```powershell
python -m driftcast.cli run --config configs/natl_subtropical_gyre.yaml --seed 42
```
Outputs land under `results/outputs/` (NetCDF by default). Manifest JSON sidecars include environment metadata.

Render visuals and videos:
```powershell
# Quick preview MP4 (lower resolution/trail settings)
python -m driftcast.cli animate preview --config configs/natl_coastal.yaml --seed 42

# Final competition-cut MP4
python -m driftcast.cli animate final --config configs/natl_coastal.yaml --seed 42

# Highlight animations
python -m driftcast.cli animate gyre --config configs/natl_subtropical_gyre.yaml --days 180 --preset microplastic_default --seed 42
python -m driftcast.cli animate sources --config configs/natl_subtropical_gyre.yaml --days 90 --legend-fade-in --seed 42
python -m driftcast.cli animate beaching --config configs/natl_subtropical_gyre.yaml --days 90 --seed 42
python -m driftcast.cli animate backtrack --config configs/natl_subtropical_gyre.yaml --days-back 30 --seed 42
python -m driftcast.cli animate long --config configs/natl_subtropical_gyre.yaml --minutes 5 --seed 42
python -m driftcast.cli animate sweep --config configs/natl_subtropical_gyre.yaml --param windage=0.001,0.005,0.01 --param Kh=15,30,60 --seed 21

# Judge-ready bundle: MP4 + hero PNG + docs/onepager.pdf
python -m driftcast.cli judge --config configs/natl_subtropical_gyre.yaml --seed 42

# Figure gallery (publication PNG/SVGs)
python -m driftcast.cli plots all --run results/outputs/simulation.nc --config configs/natl_subtropical_gyre.yaml --sweep results/batch
python -m driftcast.cli plots key --run results/outputs/simulation.nc --config configs/natl_subtropical_gyre.yaml

# Validation golden numbers
python -m driftcast.cli validate run --run results/outputs/simulation.nc --out results/validation/report.json

# Release bundle for judges
python -m driftcast.cli publish bundle --out release/
```
Generated assets:
- `results/videos/preview.mp4` and `results/videos/final_cut.mp4`
- `results/videos/gyre_convergence.mp4`, `sources_mix.mp4`, `beaching_timelapse.mp4`, `backtrack_from_gyre.mp4`, `natl_longcut.mp4`, `parameter_sweep.mp4`
- `results/figures/hero.png`
- `results/figures/accumulation_heatmap_<run>.png` (and 15 peers) mirrored under `docs/assets/`
- `docs/onepager.pdf`

## 4. Verifying FFmpeg and plots
- Confirm detection:
  ```powershell
  python -c "from driftcast.viz.ffmpeg import detect_ffmpeg; print(detect_ffmpeg())"
  ```
- If the command fails with "FFmpeg binary not found on PATH", install FFmpeg and update `%PATH%` (see the commented PATH override in `.env.example`).
- Matplotlib defaults to an interactive backend on Windows; if you are scripting or running headless, set `MPLBACKEND=Agg` (see `.env.example`).

## 5. Graph/notebook pointers
- Static figures are saved in `results/figures/` during animation routines. You can regenerate or customize via `driftcast/viz/animate.py`.
- Example notebooks for exploratory plots live in `notebooks/` (open in Jupyter Lab once the environment is ready).
- For quick inspection of outputs, use:
  ```powershell
  python -m xarray.open_dataset results/outputs/natl_subtropical_gyre.nc
  python -m matplotlib.pyplot results/figures/hero.png
  ```

## 6. Troubleshooting matrix
- **Package discovery error persists.** Double-check that you committed the `[tool.setuptools.packages.find]` section and re-ran `uv sync`.
- **`pip install -e .` fails the same way.** The issue is metadata-driven; pip and uv both invoke `setuptools`. Apply the fix above.
- **`ModuleNotFoundError` when running `python -m driftcast.cli`.** Ensure you are in the repo root and virtual environment is active; otherwise set `PYTHONPATH=%CD%`.
- **FFmpeg not detected even after install.** On Windows, restart the terminal so the updated `%PATH%` is loaded; alternatively, set `FFMPEG_BINARY` in `.env`.
- **Animations show blank coastlines.** Confirm `cartopy` downloaded its shapefiles (first run may fetch them); allow the CLI to reach the internet or pre-populate `%USERPROFILE%\AppData\Local\Cartopy`.

## 7. Next steps
- Export `.env.example` to `.env`, adjust paths, and reload your shell so tooling picks it up.
- Consider running `make precommit` after installation to sync hooks.
- Once the package metadata fix lands, switch back to `uv run driftcast --help` for a clean CLI experience.
