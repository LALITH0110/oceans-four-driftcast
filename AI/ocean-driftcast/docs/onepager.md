"""
File Summary:
- One-page printable summary outlining Driftcast goals, workflow, and impact.
- Used to generate docs/onepager.pdf for judges and stakeholders.
- Highlights innovation, technical stack, and future roadmap.
"""

# Driftcast One-Pager

## Mission

- Simulate and visualize synthetic plastic drift across the North Atlantic.
- Provide a reproducible, contributor-friendly foundation for crowdsourced and reanalysis integrations.
- Deliver high-impact storytelling assets (MP4, GIF) for the Illinois Tech Grainger Computing Innovation Prize.

## Innovation Highlights

- **Analytic twin-gyre dynamics** with seasonal modulation for subtropical and subpolar circulation.
- **Modular sources**: riverine, shipping lanes, and uniform coastal seeding with Poisson arrivals.
- **Stochastic physics**: windage, Stokes drift, diffusion, and shoreline beach/resuspension toggles.
- **Distributed-ready**: Dask-powered sweeps for laptops or clusters.
- **Media pipeline**: title cards, fading trails, and credits rendered via Matplotlib + FFmpeg.

## Workflow Overview

1. Configure scenarios in YAML (domain, physics, sources, outputs).
2. Run simulations with `driftcast run` or batch sweeps with `driftcast sweep`.
3. Generate diagnostic density maps and hotspot metrics.
4. Render preview and final-cut videos (`driftcast animate preview|final`).
5. Ingest community data through JSON schema validation (`driftcast ingest`).

## Tech Stack

- **Core**: Python 3.10, NumPy, SciPy, Xarray, Dask, Pydantic.
- **Geospatial**: Cartopy, Shapely, PyProj.
- **Visualization**: Matplotlib, ImageIO, MoviePy (optional).
- **QA**: Pytest, Coverage, Ruff, Black, Isort, Docformatter, Pre-commit.

## Roadmap

- Swap analytic fields with ocean reanalysis (HYCOM/CMEMS) and ERA5 winds.
- Integrate wave hindcasts for refined Stokes drift and windage tuning.
- Expand crowdsourced data ingestion API with authentication and dashboards.
- Deploy scalable compute via Dask Gateway or Kubernetes.
- Add vertical mixing and size-class dependent buoyancy in v2.

## Contact

- Oceans Four Driftcast | Illinois Tech Grainger Computing Innovation Prize
- Repository: https://github.com/oceans-four/driftcast
- Email: oceans-four@example.com
