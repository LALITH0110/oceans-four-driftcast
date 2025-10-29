File Summary:
- Landing page for driftcast documentation describing approach and workflow.
- Covers model equations, data flow, and extension pathways.
- Serves as root page for the lightweight HTML docs build.

# Driftcast Documentation

Driftcast simulates synthetic plastic drift across the North Atlantic using an analytic two-gyre streamfunction, configurable particle sources, and modular physics kernels. Outputs feed directly into Matplotlib-based animations for storytelling and outreach.

![Preview Drift](assets/preview.gif)

## Governing Equation

Surface particle motion is integrated using an Euler-Maruyama scheme for the stochastic differential equation

```math
\mathrm{d}\mathbf{x} =
\left[\mathbf{u}_{\text{gyre}} +
\alpha_w \mathbf{u}_{10} +
\alpha_s \mathbf{u}_{\text{Stokes}}\right] \, \mathrm{d}t +
\sqrt{2K_h} \, \mathrm{d}\mathbf{W}(t),
```

where \(K_h\) is the horizontal diffusivity and \(\mathrm{d}\mathbf{W}\) represents a Wiener process. Shoreline interactions provide probabilistic beaching with optional resuspension.

## Data Flow

```
YAML Scenario
     |
     v
Config Loader --> Sources (rivers, shipping, coastal)
     |              |
     |              v
     |        Particle Ensembles
     |              |
     v              v
Gyre/Wind/Stokes Fields ---> Integrators ---> xarray Outputs
                                              |
                                   +----------+-----------+
                                   |                      |
                                   v                      v
                          Post-processing          Animations (MP4/GIF)
                                   |
                              Analytics
```

## Judge Workflow Snapshot

- `driftcast judge --seed 42` runs the calibrated subtropical scenario, writes reproducible NetCDF/Zarr plus manifest metadata, renders the 1080p final cut, exports the hero PNG, and refreshes `docs/onepager.pdf`.
- The command prints absolute paths for the video, hero frame, and PDF so judges can drop them directly into the review portal.
- Preview the pacing script in [docs/judging.md](judging.md) to keep narration aligned with the animation cues.

## Module Map

- `driftcast.fields`: Synthetic gyre, wind, and Stokes generators.
- `driftcast.particles`: Particle class catalogues and physics kernels.
- `driftcast.sources`: Poisson seeding along rivers, shipping lanes, and coastlines.
- `driftcast.sim`: Integrators, orchestration, and Dask batch utilities.
- `driftcast.post`: Density rasters and hotspot reports.
- `driftcast.viz`: Styling, cartopy basemap helpers, and animation builders.

## Extending to Real Data

1. Replace analytic velocities with reanalysis (e.g., HYCOM, CMEMS) in `driftcast.fields`.
2. Plug real winds or wave Stokes drift into `seasonal_wind_field` and `stokes_drift_velocity`.
3. Stream near real-time crowdsourced reports through `driftcast ingest` to update parquet archives.
4. Update configs with new sources and rerun `driftcast sweep` for sensitivity studies.

For further architectural background, consult `Implementation_Plan.md` and the reorganized content from the legacy `AI/` and `cs/` directories referenced in `docs/credits.md`.
