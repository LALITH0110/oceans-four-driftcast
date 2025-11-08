# Oceans-Four DriftCast: North Atlantic Plastic Drift Simulator

**DISCLAIMER: All results produced by this system are synthetic and generated for demonstration purposes only. They are not based on real ocean current data or actual plastic pollution measurements.**

## Overview

This project generates synthetic visualizations of plastic debris drift patterns in the North Atlantic Ocean. It produces publication-quality figures and a long-form animation showing particle dispersion over time using idealized ocean current models.

## Requirements

Install required Python packages:

```bash
pip install numpy matplotlib cartopy shapely scipy imageio imageio-ffmpeg
```

## Quick Start

1. **Generate all outputs (figures + animation):**
   ```bash
   python make_all.py
   ```

2. **Outputs will be saved to:**
   - `outputs/figures/` - 20 PNG figures (1920×1080)
   - `outputs/animations/` - MP4 video and GIF teaser

## Execution Time

- **Figures:** ~2-3 minutes
- **Animation:** ~15-25 minutes (depending on system)
- **Total:** ~20-30 minutes

## Customization

Edit parameters at the top of `make_all.py`:

### Particle Counts by Source
```python
SOURCE_COUNTS = {
    'US_East': 5000,      # US East Coast particles
    'US_Gulf': 3000,      # US Gulf Coast particles
    'Caribbean': 2000,    # Caribbean particles
    'Europe_North': 2500, # Northern Europe particles
    'Europe_South': 1500, # Southern Europe particles
    'St_Lawrence': 800,   # St. Lawrence River particles
}
```

### Animation Duration
```python
ANIMATION_DURATION_SEC = 480  # Duration in seconds (300-600 range)
```
- 300 seconds = 5 minutes
- 480 seconds = 8 minutes (default)
- 600 seconds = 10 minutes

### Simulation Physics
```python
DIFFUSION_KM = 0.5              # Turbulent diffusion per hour
WINDAGE_FACTOR = 0.02           # Additional wind-driven velocity
BEACHING_DISTANCE_KM = 20       # Distance to coast for beaching
SINKING_PROB_PER_DAY = 0.0005   # Daily probability of sinking
FRAGMENTATION_PROB_PER_DAY = 0.001  # Daily probability of fragmentation
```

### Random Seed
```python
SEED = 42  # Change for different results
```

## Output Description

### Figures (outputs/figures/)

1. **01_base_map.png** - Base map with current vectors and gyre labels
2. **02_sources.png** - Particle release zones with counts
3. **03_us_dispersion_30d.png** - US East Coast 30-day dispersion heatmap
4. **04_eu_dispersion_30d.png** - European 30-day dispersion heatmap
5. **05_accumulation_180d.png** - 180-day accumulation with gyre hotspot
6. **06_fate_timeseries.png** - Time series of particle fates
7. **07_distance_histogram.png** - Travel distance distributions
8. **08_arrival_kde.png** - Gyre arrival time probability by source
9. **09_beaching_hotspots.png** - Coastal beaching intensity map
10. **10_sankey_flow.png** - Source-to-fate flow diagram
11. **11_storm_comparison.png** - Normal vs storm conditions
12. **12_trajectory_spaghetti.png** - Individual particle trajectories
13. **13_quiver_streamplot.png** - Current vectors with streamlines
14. **14_pareto_sources.png** - Source contributions (Pareto chart)
15. **15_monthly_release.png** - Seasonal release effect (small multiples)
16. **16_size_distribution.png** - Particle size after fragmentation
17. **17_residence_time.png** - Residence time by zone
18. **18_velocity_histogram.png** - Current velocity distribution
19. **19_concentration_profile.png** - Latitudinal concentration
20. **20_age_distribution.png** - Particle age by status

### Animation (outputs/animations/)

- **driftcast.mp4** - Full animation (8 minutes at 30 fps)
  - Single continuous simulation timeline
  - Shows particle movement with current vectors
  - Day counter and status tracking
  - 1920×1080 resolution

- **driftcast_teaser.gif** - Short preview (25 seconds)
  - First 25 seconds of full animation
  - Smaller file size for quick sharing

## Technical Details

### Ocean Current Model

The synthetic current field includes:
- **Subtropical Gyre:** Westward North Equatorial Current (10-20°N), northward Gulf Stream, eastward North Atlantic Drift, southward Canary Current
- **Subpolar Gyre:** Counter-clockwise circulation near Iceland
- **Mesoscale Variability:** Added noise for realism

### Particle Physics

- **Advection:** 4th-order Runge-Kutta integration
- **Diffusion:** Gaussian random walk (turbulent mixing)
- **Windage:** Additional surface drift (2% of wind speed)
- **Fates:**
  - Floating (active transport)
  - Beached (within 20 km of coast)
  - Sunk (random probability)
  - Gyre-trapped (>30 days in gyre region)
- **Fragmentation:** Size reduction over time

### Geographic Extent

- Longitude: -100° to 20° (West to East)
- Latitude: 0° to 70° (Equator to Arctic)
- Projection: PlateCarree (equirectangular)

## Interpretation Notes

This is a **simplified educational model**. Real ocean drift involves:
- Complex 3D circulation patterns
- Seasonal and interannual variability
- Wind-driven Ekman transport
- Stokes drift from surface waves
- Biofouling and density changes
- Detailed coastline and bathymetry

For research-grade simulations, use validated ocean models (e.g., HYCOM, OSCAR, CMEMS) with real forcing data.

## File Structure

```
.
├── make_all.py                    # Main script
├── README.md                      # This file
└── outputs/
    ├── figures/                   # 20 PNG figures
    │   ├── 01_base_map.png
    │   ├── 02_sources.png
    │   └── ...
    └── animations/                # Video and GIF
        ├── driftcast.mp4
        └── driftcast_teaser.gif
```

## Troubleshooting

### Memory Issues
If the animation generation runs out of memory:
- Reduce `ANIMATION_DURATION_SEC` (e.g., to 300)
- Reduce particle counts in `SOURCE_COUNTS`
- Close other applications

### Slow Rendering
- Animation rendering is CPU-intensive
- Expected time: 15-25 minutes for 8-minute video
- Progress updates printed every 100 frames

### Missing Dependencies
```bash
# If cartopy installation fails:
conda install -c conda-forge cartopy

# Or use system packages:
sudo apt-get install libgeos-dev
pip install cartopy
```

## Citation

If using these methods as inspiration for research:

> Synthetic ocean drift visualization system inspired by North Atlantic 
> circulation patterns. All data generated synthetically for demonstration 
> purposes. For actual research, consult peer-reviewed oceanographic models.

## License

This demonstration code is provided as-is for educational purposes.

---

**Remember:** These are illustrative synthetic results created for a demonstration and should not be used for scientific conclusions about real ocean plastic pollution.