# Oceans-Four DriftCast: Synthetic Demo

Publication-quality figures and animations for demonstrating North Atlantic plastic drift using synthetic physics-based simulation.

**DISCLAIMER: All results are synthetic and illustrative for demonstration purposes only.**

## Quick Start

### Prerequisites

Install required Python packages:

```bash
pip install numpy matplotlib cartopy shapely scipy imageio imageio-ffmpeg
```

Or using conda:

```bash
conda install numpy matplotlib cartopy shapely scipy
pip install imageio imageio-ffmpeg
```

### Running the Demo

Simply run the main script:

```bash
python make_all.py
```

This will:
1. Generate 20 publication-style figures (PNG, 1920×1080) in `outputs/figures/`
2. Create one continuous animation video (8 minutes, 30fps MP4) in `outputs/animations/driftcast.mp4`
3. Generate a short teaser GIF (25 seconds) in `outputs/animations/driftcast_teaser.gif`

**Expected runtime:** 10-30 minutes depending on your system (most time is spent on animation rendering).

## Output Files

### Figures (outputs/figures/)

The script generates 20 publication-quality figures:

1. **fig01_base_map_currents.png** - Ocean current vectors and gyre structure
2. **fig02_source_map.png** - Particle release locations by source region
3. **fig03_dispersion_us_30d.png** - 30-day dispersion from US sources
4. **fig04_dispersion_eu_30d.png** - 30-day dispersion from European sources
5. **fig05_accumulation_180d.png** - 180-day accumulation density with gyre hotspot
6. **fig06_time_series_fates.png** - Time series of particle fates
7. **fig07_travel_distance_hist.png** - Travel distance distributions
8. **fig08_gyre_arrival_kde.png** - Gyre entry time by source region
9. **fig09_beaching_hotspots.png** - Coastline beaching locations
10. **fig10_source_fate_flow.png** - Source-to-fate flow diagram
11. **fig11_storm_scenario.png** - Normal vs storm week comparison
12. **fig12_trajectory_spaghetti.png** - Individual particle trajectories
13. **fig13_quiver_streamplot.png** - Current vectors and streamlines
14. **fig14_pareto_contribution.png** - Source contributions to gyre
15. **fig15_small_multiples_month.png** - Release month variations
16. **fig16_age_histogram.png** - Particle age distribution
17. **fig17_distance_vs_density.png** - Distance traveled vs local density
18. **fig18_hovmoller_latitude.png** - Time-latitude diagram
19. **fig19_residence_time_map.png** - Mean residence time by location
20. **fig20_curvature_map.png** - Flow curvature (vorticity) map

### Animations (outputs/animations/)

- **driftcast.mp4** - Main animation (480 seconds / 8 minutes, 1920×1080, 30fps)
  - Shows continuous particle drift over 180-day simulation
  - Includes current vectors, particle positions, and faint trails
  - Single file, constant frame rate, no repetition

- **driftcast_teaser.gif** - Short preview (25 seconds, 10fps)
  - Quick preview of the simulation
  - Suitable for presentations and social media

## Customization

### Adjusting Particle Counts

Edit the `SOURCE_COUNTS` dictionary in `make_all.py` (around line 46):

```python
SOURCE_COUNTS = {
    'US_East_Coast': 5000,    # Increase for denser visualization
    'US_Gulf_Coast': 3000,
    'Caribbean': 2000,
    'St_Lawrence': 1000,
    'Mississippi': 1500,
    'European_Rivers': 4000,
    'UK_Coast': 2000,
}
```

### Adjusting Animation Duration

Change `ANIMATION_DURATION_SEC` in `make_all.py` (around line 60):

```python
ANIMATION_DURATION_SEC = 480  # Must be between 300-600 seconds
```

Valid range: 300-600 seconds (5-10 minutes).

### Adjusting Simulation Parameters

Key parameters (around lines 35-55):

- `SIMULATION_DAYS` - Total simulation time (default: 180 days)
- `DT_HOURS` - Timestep resolution (default: 1.0 hour)
- `WINDAGE_FACTOR` - Wind-driven drift component (default: 0.02 = 2%)
- `DIFFUSION_COEF` - Turbulent diffusion strength (default: 0.001)
- `BEACHING_THRESHOLD` - Distance to coast for beaching (default: 0.2 degrees ≈ 20km)
- `SINKING_PROB_PER_DAY` - Daily sinking probability (default: 0.001)

### Changing Visual Style

Styling parameters (around lines 68-80):

- `OCEAN_COLOR` - Background ocean color (default: '#0a1929')
- `LAND_COLOR` - Land fill color (default: '#4a4a4a')
- `PLASTIC_COLOR` - Particle color (default: '#ff6b35')
- `PLASTIC_ALPHA` - Particle transparency (default: 0.6)

## Physics Model

The simulation uses a simplified but plausible North Atlantic circulation model:

### Ocean Currents
- **Gulf Stream**: Strong northward western boundary current along US East Coast
- **North Atlantic Drift**: Northeast flow toward Europe
- **Canary Current**: Southward flow along Europe/Africa
- **North Equatorial Current**: Westward return flow
- **Subtropical Gyre**: Clockwise circulation centered near 28°N, 40°W
- **Subpolar Gyre**: Weaker cyclonic circulation near Iceland

### Particle Dynamics
- **Advection**: 4th-order Runge-Kutta integration of velocity field
- **Diffusion**: Random walk representing sub-grid turbulence
- **Windage**: Small eastward/poleward drift component (2% of wind speed)
- **Beaching**: Particles within ~20km of coast are marked as beached
- **Sinking**: Small daily probability of vertical sinking
- **Gyre Trapping**: Particles remaining in gyre center for >60 days

### Source Regions
- US East Coast (Florida to Maine)
- US Gulf Coast
- Caribbean Islands
- St. Lawrence River
- Mississippi River Delta
- European Rivers (Thames, Seine, Rhine, Tagus, Ebro)
- UK and Irish Coasts

## Technical Details

### Map Projection
- Plate Carrée (equirectangular) projection
- Extent: longitude -100° to 20°E, latitude 0° to 70°N
- Covers full North Atlantic basin

### Data Structure
- Particle positions tracked as (lon, lat) pairs
- Status flags: 0=floating, 1=beached, 2=sunk, 3=gyre-trapped
- Metadata: source region, age, distance traveled

### Random Seed
Fixed seed (42) ensures reproducible results across runs.

### Performance Tips
- Reducing `SOURCE_COUNTS` values speeds up simulation
- Increasing `DT_HOURS` (to 2 or 3) reduces computation time
- Animation rendering is the slowest step (~20-25 minutes for 480 seconds)
- Figures generate quickly (~2-3 minutes total)

## File Structure

```
.
├── make_all.py              # Main script
├── README.md                # This file
└── outputs/
    ├── figures/             # 20 publication-quality PNG figures
    │   ├── fig01_base_map_currents.png
    │   ├── fig02_source_map.png
    │   └── ...
    └── animations/          # Video and GIF outputs
        ├── driftcast.mp4            # Main 8-minute animation
        └── driftcast_teaser.gif     # 25-second preview
```

## Troubleshooting

### ImportError: No module named 'cartopy'
Install cartopy: `conda install cartopy` or `pip install cartopy`

### ImportError: No module named 'imageio'
Install imageio: `pip install imageio imageio-ffmpeg`

### Animation skipped
If imageio is not available, figures will still be generated but animations will be skipped.

### Memory issues
If you run out of memory:
- Reduce particle counts in `SOURCE_COUNTS`
- Reduce `ANIMATION_DURATION_SEC`
- Increase `save_interval_days` to reduce stored snapshots

### Slow performance
- Reduce total particles (currently ~18,000)
- Increase `DT_HOURS` to 2 or 3
- Reduce `SIMULATION_DAYS` to 90 or 120
- Use a smaller animation duration (300 seconds minimum)

## Citation

If you use these visualizations in presentations or publications, please include:

> Visualizations generated using synthetic particle drift simulation for demonstration purposes.
> Source: Oceans-Four DriftCast Demo (synthetic data).

## License

This demonstration code is provided for educational and illustrative purposes.

## Acknowledgments

Synthetic ocean circulation model inspired by:
- North Atlantic subtropical gyre dynamics
- Gulf Stream and North Atlantic Drift pathways
- Observed plastic accumulation patterns in the Sargasso Sea region

**Remember: These are synthetic results for demonstration purposes only and should not be used for scientific conclusions or policy decisions.**
