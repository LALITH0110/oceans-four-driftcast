# Oceans-Four DriftCast - North Atlantic Plastic Drift Visualization

**DISCLAIMER: All results are synthetic/simulated for demonstration purposes only.**

## Quick Start

### Requirements
```bash
pip install numpy matplotlib cartopy shapely scipy imageio imageio-ffmpeg
```

### Generate All Outputs
```bash
python make_all.py
```

This will create:
- **20 publication-style figures** in `outputs/figures/`
- **1 looping animation (MP4)** in `outputs/animations/driftcast.mp4`
- **1 teaser animation (GIF)** in `outputs/animations/driftcast_teaser.gif`

Runtime: ~5-10 minutes depending on your system.

## What Gets Generated

### Figures (1920×1080 PNG)
1. **01_current_field.png** - Base map with current vectors and gyre labels
2. **02_source_map.png** - Particle release zones by source region
3. **03_dispersion_us_30d.png** - 30-day dispersion from US sources
4. **04_dispersion_eu_30d.png** - 30-day dispersion from European sources
5. **05_gyre_accumulation_180d.png** - Subtropical gyre hotspot accumulation
6. **06_fate_timeseries.png** - Particle fate evolution over time
7. **07_distance_histogram.png** - Travel distance distributions
8. **08_gyre_entry_kde.png** - Gyre arrival time probability by source
9. **09_beaching_hotspots.png** - Coastal beaching intensity map
10. **10_sankey_sources_to_fates.png** - Flow diagram from sources to fates
11. **11_storm_comparison.png** - Before/after storm event comparison
12. **12_trajectory_spaghetti.png** - Sample particle trajectories
13. **13_quiver_streamplot.png** - Current field visualization overlay
14. **14_pareto_contributions.png** - Source contributions to gyre (Pareto chart)
15. **15_seasonal_comparison.png** - Seasonal variation in dispersion
16. **16_residence_time_map.png** - Average particle age by region
17. **17_velocity_magnitude.png** - Ocean current speed map
18. **18_concentration_gradients.png** - Particle convergence zones
19. **19_transit_time_histograms.png** - Time to beaching by region
20. **20_source_fate_matrix.png** - Source-fate relationship heatmap

### Animations
- **driftcast.mp4** - 10-second looping animation (1080p, 30 fps) showing 365 days of drift
- **driftcast_teaser.gif** - 20-second teaser (lower resolution for quick sharing)

## Customization

### Adjust Particle Counts
Edit `make_all.py` around line 30:
```python
N_PARTICLES_BASE = 5000  # Increase for more particles
```

Or modify individual source counts in `SOURCE_REGIONS` dictionary (line ~100).

### Change Simulation Duration
```python
SIM_DAYS = 365  # Change simulation length
```

### Modify Physics Parameters
```python
TURBULENT_DIFFUSION = 0.02  # Increase for more spreading
WINDAGE_FACTOR = 0.02  # Wind-driven drift component
BEACHING_DISTANCE_KM = 20  # Distance threshold for beaching
SINKING_PROB_PER_DAY = 0.001  # Daily sinking probability
```

### Adjust Animation Settings
In the `generate_all_outputs()` function:
```python
create_drift_animation(snapshots, anim_mp4, duration_seconds=10, fps=30)
```
Change `duration_seconds` for longer/shorter videos.

## Technical Details

### Synthetic Physics Model
- **Idealized gyre circulation** with subtropical and subpolar components
- **RK4 advection** with hourly timesteps
- **Turbulent diffusion** using Gaussian random walk
- **Windage component** for surface drift
- **Beaching detection** within 20 km of coastlines
- **Sinking and fragmentation** processes

### Source Regions
11 release zones including:
- US East Coast, Gulf Coast, Mississippi Delta
- Caribbean islands
- St. Lawrence Seaway
- European rivers (Thames, Seine, Rhine)
- Iberian Peninsula
- Northwest African coast

### Output Styling
- Dark ocean background (#0a1628)
- Mid-gray land (#404040)
- White coastlines
- Orange/magenta particles with alpha blending
- Consistent fonts and subtle gridlines
- "Demo, synthetic" watermark on all outputs

## File Structure
```
.
├── make_all.py          # Main generation script
├── README.md            # This file
└── outputs/
    ├── figures/         # 20 PNG figures
    └── animations/      # MP4 and GIF files
```

## Notes

- **Reproducibility**: Fixed random seed (SEED=42) ensures identical results
- **No internet required**: All computation is local
- **Memory usage**: ~2-4 GB RAM for default settings
- **Disk space**: ~50-100 MB for all outputs

## Disclaimer

**These are illustrative synthetic results for demonstration purposes only.** The simulation uses idealized ocean currents and simplified physics. Results should not be used for scientific analysis, policy decisions, or real-world predictions of ocean plastic transport.

For actual ocean drift modeling, see operational systems like NOAA's GNOME, ECMWF's marine forecasts, or research models like HYCOM/ROMS.