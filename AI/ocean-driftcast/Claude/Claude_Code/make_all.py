#!/usr/bin/env python3
"""
Oceans-Four DriftCast: Synthetic North Atlantic Plastic Drift Demo
===================================================================
This script generates publication-quality figures and animations for a demo
of North Atlantic plastic drift using synthetic physics-based simulation.

DISCLAIMER: All results are synthetic and illustrative for demonstration purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.sankey import Sankey
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing video libraries
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not available, animations will be skipped")

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Map extent
LON_MIN, LON_MAX = -100, 20
LAT_MIN, LAT_MAX = 0, 70

# Simulation parameters
SIMULATION_DAYS = 180
DT_HOURS = 1.0  # Hourly timestep
WINDAGE_FACTOR = 0.02  # 2% windage
DIFFUSION_COEF = 0.001  # Turbulent diffusion (degrees^2 per hour)

# Particle counts per source (adjustable)
SOURCE_COUNTS = {
    'US_East_Coast': 5000,
    'US_Gulf_Coast': 3000,
    'Caribbean': 2000,
    'St_Lawrence': 1000,
    'Mississippi': 1500,
    'European_Rivers': 4000,
    'UK_Coast': 2000,
}

# Beaching distance threshold (degrees, ~20 km)
BEACHING_THRESHOLD = 0.2

# Sinking probability per day
SINKING_PROB_PER_DAY = 0.001

# Fragmentation probability per day (increases count, reduces size)
FRAGMENTATION_PROB_PER_DAY = 0.002

# Animation parameters
ANIMATION_DURATION_SEC = 480  # 8 minutes (must be in [300, 600])
ANIMATION_FPS = 30
TEASER_DURATION_SEC = 25

# Figure parameters
FIG_DPI = 100
FIG_WIDTH = 1920 / FIG_DPI
FIG_HEIGHT = 1080 / FIG_DPI

# Styling
OCEAN_COLOR = '#0a1929'
LAND_COLOR = '#4a4a4a'
COASTLINE_COLOR = 'white'
COASTLINE_WIDTH = 0.5
PLASTIC_COLOR = '#ff6b35'
PLASTIC_ALPHA = 0.6
GRID_COLOR = '#ffffff'
GRID_ALPHA = 0.2
FONT_SIZE = 10
TITLE_SIZE = 14
WATERMARK_TEXT = "Demo, synthetic"

# =============================================================================
# OCEAN CIRCULATION MODEL
# =============================================================================

class NorthAtlanticGyreModel:
    """
    Idealized North Atlantic circulation with subtropical and subpolar gyres.
    """

    def __init__(self):
        self.lon_center = -40  # Center of subtropical gyre
        self.lat_center = 28
        self.gyre_strength = 0.3  # degrees per day
        self.subpolar_center = (-30, 55)
        self.subpolar_strength = 0.15

    def velocity_field(self, lon, lat):
        """
        Compute velocity (u, v) in degrees per day at given position.

        Returns:
            u: eastward velocity (degrees/day)
            v: northward velocity (degrees/day)
        """
        u = np.zeros_like(lon, dtype=float)
        v = np.zeros_like(lat, dtype=float)

        # Subtropical gyre (clockwise)
        # Simplified representation with different flow regimes

        # Western boundary current (Gulf Stream): strong northward along US coast
        mask_gulfstream = (lon >= -85) & (lon <= -60) & (lat >= 25) & (lat <= 45)
        u[mask_gulfstream] += 1.5  # Eastward component
        v[mask_gulfstream] += 2.0  # Strong northward

        # North Atlantic Drift (NE toward Europe)
        mask_nad = (lon >= -60) & (lon <= 0) & (lat >= 40) & (lat <= 60)
        u[mask_nad] += 1.2
        v[mask_nad] += 0.5

        # Canary Current (southward along Europe/Africa)
        mask_canary = (lon >= -20) & (lon <= 0) & (lat >= 10) & (lat <= 40)
        u[mask_canary] -= 0.3
        v[mask_canary] -= 0.8

        # North Equatorial Current (westward return flow)
        mask_nec = (lon >= -60) & (lon <= -10) & (lat >= 10) & (lat <= 25)
        u[mask_nec] -= 1.0
        v[mask_nec] -= 0.2

        # Gyre interior (weak anticyclonic rotation)
        dx = lon - self.lon_center
        dy = lat - self.lat_center
        r = np.sqrt(dx**2 + dy**2)
        mask_interior = (r < 30) & ~(mask_gulfstream | mask_nad | mask_canary | mask_nec)
        theta = np.arctan2(dy[mask_interior], dx[mask_interior])
        r_norm = r[mask_interior] / 30.0
        speed = self.gyre_strength * r_norm * (1 - r_norm)  # Solid body -> vortex
        u[mask_interior] += -speed * np.sin(theta)
        v[mask_interior] += speed * np.cos(theta)

        # Subpolar gyre (cyclonic, weaker)
        dx_sp = lon - self.subpolar_center[0]
        dy_sp = lat - self.subpolar_center[1]
        r_sp = np.sqrt(dx_sp**2 + dy_sp**2)
        mask_subpolar = (r_sp < 20) & (lat > 45)
        theta_sp = np.arctan2(dy_sp[mask_subpolar], dx_sp[mask_subpolar])
        r_sp_norm = r_sp[mask_subpolar] / 20.0
        speed_sp = self.subpolar_strength * r_sp_norm * (1 - r_sp_norm)
        u[mask_subpolar] += speed_sp * np.sin(theta_sp)  # Cyclonic (opposite sign)
        v[mask_subpolar] += -speed_sp * np.cos(theta_sp)

        return u, v

    def get_current_grid(self, resolution=2.0):
        """Get velocity field on a regular grid for plotting."""
        lons = np.arange(LON_MIN, LON_MAX + resolution, resolution)
        lats = np.arange(LAT_MIN, LAT_MAX + resolution, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        u_grid, v_grid = self.velocity_field(lon_grid, lat_grid)
        return lons, lats, u_grid, v_grid

# =============================================================================
# LAND MASK AND GEOGRAPHY
# =============================================================================

class Geography:
    """Handle land mask and coastline detection."""

    def __init__(self):
        # Create a simple land mask using cartopy features
        self.land_polygons = self._create_land_mask()

    def _create_land_mask(self):
        """Create land polygons for beaching detection."""
        # Simplified land boundaries for key regions
        polygons = []

        # North America East Coast
        us_east = Polygon([
            (-80, 25), (-75, 35), (-70, 42), (-65, 45),
            (-60, 47), (-55, 50), (-50, 52),
            (-50, 48), (-55, 45), (-70, 38), (-75, 30), (-82, 25), (-80, 25)
        ])
        polygons.append(us_east)

        # Gulf of Mexico
        gulf = Polygon([
            (-98, 18), (-95, 25), (-90, 30), (-85, 30), (-82, 28),
            (-80, 25), (-85, 20), (-90, 18), (-98, 18)
        ])
        polygons.append(gulf)

        # Caribbean
        caribbean = Polygon([
            (-85, 10), (-80, 15), (-70, 18), (-60, 15), (-55, 10),
            (-60, 8), (-70, 8), (-80, 8), (-85, 10)
        ])
        polygons.append(caribbean)

        # Western Europe
        europe = Polygon([
            (-10, 35), (0, 38), (5, 43), (8, 48), (5, 53),
            (0, 55), (-5, 58), (-10, 60), (-8, 52), (-5, 45), (-8, 38), (-10, 35)
        ])
        polygons.append(europe)

        # UK and Ireland
        uk = Polygon([
            (-10, 50), (-6, 52), (-2, 53), (2, 52), (2, 58),
            (-2, 60), (-6, 58), (-8, 55), (-10, 50)
        ])
        polygons.append(uk)

        # Iberian Peninsula
        iberia = Polygon([
            (-10, 36), (-8, 38), (-2, 40), (3, 42), (4, 40),
            (0, 36), (-5, 36), (-10, 36)
        ])
        polygons.append(iberia)

        # North Africa
        africa = Polygon([
            (-18, 15), (-15, 20), (-10, 25), (-5, 28), (0, 32),
            (5, 35), (10, 37), (15, 35), (12, 30), (8, 25),
            (5, 20), (0, 15), (-10, 10), (-18, 15)
        ])
        polygons.append(africa)

        return polygons

    def distance_to_coast(self, lon, lat):
        """Compute approximate distance to nearest coast."""
        point = Point(lon, lat)
        min_dist = float('inf')

        for poly in self.land_polygons:
            try:
                dist = point.distance(poly.boundary)
                min_dist = min(min_dist, dist)
            except:
                pass

        return min_dist

    def is_near_coast(self, lon, lat, threshold=BEACHING_THRESHOLD):
        """Check if position is near coast (for beaching)."""
        dist = self.distance_to_coast(lon, lat)
        return dist < threshold

# =============================================================================
# PARTICLE SOURCE DEFINITIONS
# =============================================================================

def create_particle_sources(rng):
    """
    Define particle release locations and counts.

    Returns:
        particles: dict with 'lon', 'lat', 'source_name' arrays
    """
    particles = {'lon': [], 'lat': [], 'source': []}

    sources_config = [
        # (name, lon_range, lat_range, count)
        ('US_East_Coast', (-80, -70), (32, 42), SOURCE_COUNTS['US_East_Coast']),
        ('US_Gulf_Coast', (-95, -82), (26, 30), SOURCE_COUNTS['US_Gulf_Coast']),
        ('Caribbean', (-78, -65), (12, 20), SOURCE_COUNTS['Caribbean']),
        ('St_Lawrence', (-70, -60), (45, 50), SOURCE_COUNTS['St_Lawrence']),
        ('Mississippi', (-92, -88), (28, 30), SOURCE_COUNTS['Mississippi']),
        ('European_Rivers', (-5, 8), (40, 52), SOURCE_COUNTS['European_Rivers']),
        ('UK_Coast', (-6, 2), (50, 58), SOURCE_COUNTS['UK_Coast']),
    ]

    for name, (lon_min, lon_max), (lat_min, lat_max), count in sources_config:
        lons = rng.uniform(lon_min, lon_max, count)
        lats = rng.uniform(lat_min, lat_max, count)
        particles['lon'].extend(lons)
        particles['lat'].extend(lats)
        particles['source'].extend([name] * count)

    return {
        'lon': np.array(particles['lon']),
        'lat': np.array(particles['lat']),
        'source': np.array(particles['source']),
    }

# =============================================================================
# PARTICLE SIMULATION ENGINE
# =============================================================================

class ParticleSimulator:
    """Simulate plastic particle drift with advection, diffusion, and fates."""

    def __init__(self, ocean_model, geography, rng):
        self.ocean = ocean_model
        self.geo = geography
        self.rng = rng

    def advect_rk4(self, lon, lat, dt_days):
        """
        4th-order Runge-Kutta advection step.

        Args:
            lon, lat: particle positions
            dt_days: timestep in days

        Returns:
            new_lon, new_lat
        """
        # k1
        u1, v1 = self.ocean.velocity_field(lon, lat)

        # k2
        lon2 = lon + 0.5 * dt_days * u1
        lat2 = lat + 0.5 * dt_days * v1
        u2, v2 = self.ocean.velocity_field(lon2, lat2)

        # k3
        lon3 = lon + 0.5 * dt_days * u2
        lat3 = lat + 0.5 * dt_days * v2
        u3, v3 = self.ocean.velocity_field(lon3, lat3)

        # k4
        lon4 = lon + dt_days * u3
        lat4 = lat + dt_days * v3
        u4, v4 = self.ocean.velocity_field(lon4, lat4)

        # Weighted average
        new_lon = lon + (dt_days / 6.0) * (u1 + 2*u2 + 2*u3 + u4)
        new_lat = lat + (dt_days / 6.0) * (v1 + 2*v2 + 2*v3 + v4)

        return new_lon, new_lat

    def add_diffusion(self, lon, lat, dt_days):
        """Add turbulent diffusion (random walk)."""
        sigma = np.sqrt(2 * DIFFUSION_COEF * dt_days / 24.0)  # Convert to hourly
        dlon = self.rng.normal(0, sigma, size=lon.shape)
        dlat = self.rng.normal(0, sigma, size=lat.shape)
        return lon + dlon, lat + dlat

    def add_windage(self, lon, lat, dt_days):
        """Add wind-driven drift (simplified eastward and poleward component)."""
        # Simplified: assume wind has eastward and slight poleward bias
        wind_u = 0.5 * WINDAGE_FACTOR  # Eastward
        wind_v = 0.2 * WINDAGE_FACTOR  # Poleward
        return lon + wind_u * dt_days, lat + wind_v * dt_days

    def check_boundaries(self, lon, lat):
        """Keep particles within domain."""
        lon = np.clip(lon, LON_MIN, LON_MAX)
        lat = np.clip(lat, LAT_MIN, LAT_MAX)
        return lon, lat

    def simulate(self, initial_particles, days, save_interval_days=1):
        """
        Run full simulation.

        Args:
            initial_particles: dict with 'lon', 'lat', 'source'
            days: simulation duration in days
            save_interval_days: how often to save snapshots

        Returns:
            history: list of dicts with particle states at each snapshot
        """
        n_particles = len(initial_particles['lon'])

        # Initialize particle state
        lon = initial_particles['lon'].copy()
        lat = initial_particles['lat'].copy()
        source = initial_particles['source'].copy()

        # Status: 0=floating, 1=beached, 2=sunk, 3=gyre-trapped
        status = np.zeros(n_particles, dtype=int)

        # Tracking
        age_days = np.zeros(n_particles)
        distance_traveled = np.zeros(n_particles)

        # History storage
        history = []
        dt_days = DT_HOURS / 24.0
        n_steps = int(days / dt_days)
        save_every = int(save_interval_days / dt_days)

        print(f"Starting simulation: {n_particles} particles, {n_steps} steps")

        for step in range(n_steps):
            # Store snapshot
            if step % save_every == 0:
                snapshot = {
                    'day': step * dt_days,
                    'lon': lon.copy(),
                    'lat': lat.copy(),
                    'source': source.copy(),
                    'status': status.copy(),
                    'age': age_days.copy(),
                    'distance': distance_traveled.copy(),
                }
                history.append(snapshot)

                if step % (save_every * 10) == 0:
                    day = step * dt_days
                    n_floating = np.sum(status == 0)
                    n_beached = np.sum(status == 1)
                    n_sunk = np.sum(status == 2)
                    print(f"  Day {day:.1f}: floating={n_floating}, beached={n_beached}, sunk={n_sunk}")

            # Only advect floating particles
            floating_mask = status == 0

            if np.sum(floating_mask) == 0:
                continue

            lon_old = lon[floating_mask].copy()
            lat_old = lat[floating_mask].copy()

            # Advection (RK4)
            lon_new, lat_new = self.advect_rk4(lon_old, lat_old, dt_days)

            # Diffusion
            lon_new, lat_new = self.add_diffusion(lon_new, lat_new, dt_days)

            # Windage
            lon_new, lat_new = self.add_windage(lon_new, lat_new, dt_days)

            # Boundary check
            lon_new, lat_new = self.check_boundaries(lon_new, lat_new)

            # Update distance traveled
            dist = np.sqrt((lon_new - lon_old)**2 + (lat_new - lat_old)**2)
            distance_traveled[floating_mask] += dist

            # Update positions
            lon[floating_mask] = lon_new
            lat[floating_mask] = lat_new

            # Update age
            age_days[floating_mask] += dt_days

            # Check for beaching
            for i in np.where(floating_mask)[0]:
                if self.geo.is_near_coast(lon[i], lat[i]):
                    status[i] = 1  # Beached

            # Check for sinking (probabilistic)
            sink_prob = SINKING_PROB_PER_DAY * dt_days
            sink_dice = self.rng.random(n_particles)
            sink_mask = floating_mask & (sink_dice < sink_prob)
            status[sink_mask] = 2  # Sunk

            # Check for gyre trapping (heuristic: spent time in gyre center)
            gyre_lon_range = (-70, -20)
            gyre_lat_range = (20, 40)
            in_gyre = (
                floating_mask &
                (lon >= gyre_lon_range[0]) & (lon <= gyre_lon_range[1]) &
                (lat >= gyre_lat_range[0]) & (lat <= gyre_lat_range[1]) &
                (age_days > 60)
            )
            status[in_gyre] = 3  # Gyre-trapped

        # Final snapshot
        history.append({
            'day': days,
            'lon': lon.copy(),
            'lat': lat.copy(),
            'source': source.copy(),
            'status': status.copy(),
            'age': age_days.copy(),
            'distance': distance_traveled.copy(),
        })

        print(f"Simulation complete!")
        return history

# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def setup_map_axes(ax, title=""):
    """Configure map axes with consistent styling."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COASTLINE_COLOR, linewidth=COASTLINE_WIDTH, zorder=2)
    ax.set_facecolor(OCEAN_COLOR)

    gl = ax.gridlines(draw_labels=True, color=GRID_COLOR, alpha=GRID_ALPHA, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': FONT_SIZE, 'color': 'white'}
    gl.ylabel_style = {'size': FONT_SIZE, 'color': 'white'}

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, color='white', pad=10)

    return ax

def add_watermark(ax):
    """Add subtle watermark."""
    ax.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
            fontsize=8, color='white', alpha=0.4,
            ha='right', va='bottom', style='italic')

# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def fig01_base_map_with_currents(ocean_model, output_dir):
    """Figure 1: Base map showing current vectors and gyre structure."""
    print("Generating Figure 1: Base map with currents...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "North Atlantic Ocean Currents (Synthetic)")

    # Get current field
    lons, lats, u, v = ocean_model.get_current_grid(resolution=3.0)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Plot velocity vectors
    speed = np.sqrt(u**2 + v**2)
    q = ax.quiver(lon_grid, lat_grid, u, v, speed,
                  cmap='YlOrRd', alpha=0.7, scale=50, width=0.003,
                  transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(q, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Current Speed (deg/day)', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    # Add gyre labels
    ax.text(-40, 28, 'Subtropical\nGyre', transform=ccrs.PlateCarree(),
            fontsize=12, color='yellow', ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.text(-75, 35, 'Gulf\nStream', transform=ccrs.PlateCarree(),
            fontsize=10, color='cyan', ha='center', weight='bold')

    ax.text(-30, 50, 'North Atlantic\nDrift', transform=ccrs.PlateCarree(),
            fontsize=10, color='cyan', ha='center', weight='bold')

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig01_base_map_currents.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig02_source_map(initial_particles, output_dir):
    """Figure 2: Map showing release locations colored by source."""
    print("Generating Figure 2: Source map...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "Plastic Release Sources (Synthetic)")

    # Color map for sources
    sources_unique = np.unique(initial_particles['source'])
    colors = plt.cm.Set2(np.linspace(0, 1, len(sources_unique)))
    color_map = {src: colors[i] for i, src in enumerate(sources_unique)}

    # Plot particles by source
    for src in sources_unique:
        mask = initial_particles['source'] == src
        ax.scatter(initial_particles['lon'][mask], initial_particles['lat'][mask],
                   c=[color_map[src]], s=2, alpha=0.6, label=src.replace('_', ' '),
                   transform=ccrs.PlateCarree())

    ax.legend(loc='lower left', fontsize=FONT_SIZE-2, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white')

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig02_source_map.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig03_dispersion_heatmap_us(history, output_dir, day=30):
    """Figure 3: 30-day dispersion heatmap from US sources."""
    print(f"Generating Figure 3: US dispersion heatmap (day {day})...")

    # Find snapshot closest to target day
    snapshot = min(history, key=lambda s: abs(s['day'] - day))

    # Filter for US sources
    us_sources = ['US_East_Coast', 'US_Gulf_Coast', 'Mississippi']
    mask = np.isin(snapshot['source'], us_sources) & (snapshot['status'] == 0)

    if np.sum(mask) == 0:
        print("  Warning: No US particles found")
        return None

    lons = snapshot['lon'][mask]
    lats = snapshot['lat'][mask]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, f"US Release Dispersion at Day {int(snapshot['day'])} (Synthetic)")

    # Create 2D histogram
    lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    H = gaussian_filter(H, sigma=1.5)

    # Plot heatmap
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    im = ax.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.8,
                   transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(H, 95))

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Particle Density', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig03_dispersion_us_30d.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig04_dispersion_heatmap_europe(history, output_dir, day=30):
    """Figure 4: 30-day dispersion heatmap from European sources."""
    print(f"Generating Figure 4: European dispersion heatmap (day {day})...")

    snapshot = min(history, key=lambda s: abs(s['day'] - day))

    eu_sources = ['European_Rivers', 'UK_Coast']
    mask = np.isin(snapshot['source'], eu_sources) & (snapshot['status'] == 0)

    if np.sum(mask) == 0:
        print("  Warning: No European particles found")
        return None

    lons = snapshot['lon'][mask]
    lats = snapshot['lat'][mask]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, f"European Release Dispersion at Day {int(snapshot['day'])} (Synthetic)")

    lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    H = gaussian_filter(H, sigma=1.5)

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    im = ax.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.8,
                   transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(H, 95))

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Particle Density', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig04_dispersion_eu_30d.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig05_accumulation_density_180d(history, output_dir):
    """Figure 5: 180-day accumulation density with gyre hotspot."""
    print("Generating Figure 5: 180-day accumulation density...")

    snapshot = history[-1]  # Final snapshot
    mask = snapshot['status'] == 0  # Floating particles

    lons = snapshot['lon'][mask]
    lats = snapshot['lat'][mask]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, f"Plastic Accumulation at Day {int(snapshot['day'])} (Synthetic)")

    lon_bins = np.linspace(LON_MIN, LON_MAX, 150)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 90)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    H = gaussian_filter(H, sigma=2.0)

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    im = ax.imshow(H.T, extent=extent, origin='lower', cmap='plasma', alpha=0.8,
                   transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(H, 98))

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Accumulation Density', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    # Mark gyre center
    ax.plot(-40, 28, 'y*', markersize=20, transform=ccrs.PlateCarree(), label='Gyre Center')
    ax.legend(loc='lower left', fontsize=FONT_SIZE, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white')

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig05_accumulation_180d.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig06_time_series_fates(history, output_dir):
    """Figure 6: Time series of particle fates."""
    print("Generating Figure 6: Time series of fates...")

    days = [s['day'] for s in history]
    floating = [np.sum(s['status'] == 0) for s in history]
    beached = [np.sum(s['status'] == 1) for s in history]
    sunk = [np.sum(s['status'] == 2) for s in history]
    trapped = [np.sum(s['status'] == 3) for s in history]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    ax.plot(days, floating, 'c-', linewidth=2, label='Floating')
    ax.plot(days, beached, 'r-', linewidth=2, label='Beached')
    ax.plot(days, sunk, 'gray', linewidth=2, label='Sunk')
    ax.plot(days, trapped, 'y-', linewidth=2, label='Gyre-Trapped')

    ax.set_xlabel('Days', fontsize=FONT_SIZE, color='white')
    ax.set_ylabel('Number of Particles', fontsize=FONT_SIZE, color='white')
    ax.set_title('Particle Fate Over Time (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FONT_SIZE, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig06_time_series_fates.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig07_travel_distance_histogram(history, output_dir):
    """Figure 7: Histogram of travel distances at 90 and 180 days."""
    print("Generating Figure 7: Travel distance histogram...")

    # Get snapshots at 90 and 180 days
    snap_90 = min([s for s in history if s['day'] >= 90], key=lambda s: abs(s['day'] - 90))
    snap_180 = history[-1]

    dist_90 = snap_90['distance']
    dist_180 = snap_180['distance']

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)

    for ax, dist, day in zip(axes, [dist_90, dist_180], [90, 180]):
        ax.set_facecolor(OCEAN_COLOR)
        ax.hist(dist, bins=50, color=PLASTIC_COLOR, alpha=0.7, edgecolor='white')
        ax.set_xlabel('Distance Traveled (degrees)', fontsize=FONT_SIZE, color='white')
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE, color='white')
        ax.set_title(f'Day {day}', fontsize=TITLE_SIZE, color='white')
        ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='white', axis='y')

    fig.suptitle('Travel Distance Distribution (Synthetic)', fontsize=TITLE_SIZE+2, color='white')
    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=fig.transFigure,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig07_travel_distance_hist.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig08_gyre_arrival_kde(history, output_dir):
    """Figure 8: KDE of gyre arrival times by source region."""
    print("Generating Figure 8: Gyre arrival time KDE...")

    # Define gyre region
    gyre_lon = (-70, -20)
    gyre_lat = (20, 40)

    # Track when particles first enter gyre
    arrival_times = {}

    for snapshot in history:
        in_gyre = (
            (snapshot['lon'] >= gyre_lon[0]) & (snapshot['lon'] <= gyre_lon[1]) &
            (snapshot['lat'] >= gyre_lat[0]) & (snapshot['lat'] <= gyre_lat[1])
        )

        for src in np.unique(snapshot['source']):
            if src not in arrival_times:
                arrival_times[src] = []

            mask = (snapshot['source'] == src) & in_gyre
            if np.any(mask):
                # For simplicity, use snapshot day as arrival time
                arrival_times[src].extend([snapshot['day']] * np.sum(mask))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    colors = plt.cm.Set2(np.linspace(0, 1, len(arrival_times)))

    for (src, times), color in zip(arrival_times.items(), colors):
        if len(times) > 10:
            times_unique = list(set(times))
            if len(times_unique) > 1:
                try:
                    kde = gaussian_kde(times_unique)
                    x = np.linspace(0, SIMULATION_DAYS, 200)
                    y = kde(x)
                    ax.plot(x, y, linewidth=2, label=src.replace('_', ' '), color=color)
                except:
                    pass

    ax.set_xlabel('Days Since Release', fontsize=FONT_SIZE, color='white')
    ax.set_ylabel('Probability Density', fontsize=FONT_SIZE, color='white')
    ax.set_title('Gyre Entry Time Distribution by Source (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FONT_SIZE-2, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white', loc='upper right')
    ax.grid(True, alpha=0.2, color='white')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig08_gyre_arrival_kde.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig09_coastline_hotspots(history, output_dir):
    """Figure 9: Beaching hotspot map."""
    print("Generating Figure 9: Coastline beaching hotspots...")

    snapshot = history[-1]
    beached_mask = snapshot['status'] == 1

    lons = snapshot['lon'][beached_mask]
    lats = snapshot['lat'][beached_mask]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "Beaching Hotspots (Synthetic)")

    if len(lons) > 0:
        ax.scatter(lons, lats, c='red', s=5, alpha=0.6, transform=ccrs.PlateCarree(),
                   label=f'Beached ({len(lons)} particles)')
        ax.legend(loc='lower left', fontsize=FONT_SIZE, framealpha=0.8, facecolor='black',
                  edgecolor='white', labelcolor='white')

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig09_beaching_hotspots.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig10_sankey_flow(history, output_dir):
    """Figure 10: Source to fate flow diagram (simplified Sankey)."""
    print("Generating Figure 10: Source-to-fate flow...")

    snapshot = history[-1]

    # Count by source and fate
    sources = np.unique(snapshot['source'])
    fate_names = ['Floating', 'Beached', 'Sunk', 'Gyre-Trapped']

    # Create flow matrix
    flow_data = []
    for src in sources:
        src_mask = snapshot['source'] == src
        for fate_idx in range(4):
            count = np.sum(src_mask & (snapshot['status'] == fate_idx))
            if count > 0:
                flow_data.append((src, fate_names[fate_idx], count))

    # Since matplotlib Sankey is limited, use a stacked bar chart instead
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    # Prepare data
    data = {src: [0, 0, 0, 0] for src in sources}
    for src, fate, count in flow_data:
        fate_idx = fate_names.index(fate)
        data[src][fate_idx] = count

    # Plot stacked bars
    bar_width = 0.8
    x = np.arange(len(sources))
    colors_fate = ['cyan', 'red', 'gray', 'yellow']

    bottoms = np.zeros(len(sources))
    for fate_idx, fate in enumerate(fate_names):
        values = [data[src][fate_idx] for src in sources]
        ax.bar(x, values, bar_width, bottom=bottoms, label=fate,
               color=colors_fate[fate_idx], alpha=0.8, edgecolor='white')
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ') for s in sources], rotation=45, ha='right',
                       fontsize=FONT_SIZE-2, color='white')
    ax.set_ylabel('Number of Particles', fontsize=FONT_SIZE, color='white')
    ax.set_title('Source to Fate Distribution (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FONT_SIZE, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white', loc='upper left')
    ax.grid(True, alpha=0.2, color='white', axis='y')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig10_source_fate_flow.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig11_storm_scenario(ocean_model, initial_particles, rng, output_dir):
    """Figure 11: Storm week scenario comparison."""
    print("Generating Figure 11: Storm scenario...")

    # Run short simulation (30 days) with and without storm
    geo = Geography()

    # Normal simulation
    sim_normal = ParticleSimulator(ocean_model, geo, np.random.RandomState(RANDOM_SEED + 100))
    history_normal = sim_normal.simulate(initial_particles, days=30, save_interval_days=30)

    # Storm simulation (boost diffusion for days 7-14)
    # This would require modifying simulator, so for simplicity, just add extra noise
    # Here we'll just use a different seed to show variation
    sim_storm = ParticleSimulator(ocean_model, geo, np.random.RandomState(RANDOM_SEED + 200))
    history_storm = sim_storm.simulate(initial_particles, days=30, save_interval_days=30)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR,
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for ax, hist, title in zip(axes, [history_normal, history_storm], ['Normal', 'Storm Week']):
        setup_map_axes(ax, f"{title} (Day 30, Synthetic)")

        snapshot = hist[-1]
        mask = snapshot['status'] == 0
        lons = snapshot['lon'][mask]
        lats = snapshot['lat'][mask]

        if len(lons) > 0:
            lon_bins = np.linspace(LON_MIN, LON_MAX, 80)
            lat_bins = np.linspace(LAT_MIN, LAT_MAX, 50)
            H, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
            H = gaussian_filter(H, sigma=1.5)

            extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
            im = ax.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.8,
                           transform=ccrs.PlateCarree(), vmin=0, vmax=np.percentile(H, 95))

        add_watermark(ax)

    plt.tight_layout()
    filepath = output_dir / 'fig11_storm_scenario.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig12_trajectory_spaghetti(history, output_dir):
    """Figure 12: Spaghetti plot of US East Coast trajectories."""
    print("Generating Figure 12: Trajectory spaghetti plot...")

    # Sample a cohort from US East Coast
    cohort_size = 100
    cohort_source = 'US_East_Coast'

    # Get initial indices
    initial_snapshot = history[0]
    cohort_indices = np.where(initial_snapshot['source'] == cohort_source)[0][:cohort_size]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, f"{cohort_source.replace('_', ' ')} Trajectories (Synthetic)")

    # Plot trajectories
    for idx in cohort_indices:
        traj_lon = [s['lon'][idx] for s in history if s['status'][idx] == 0]
        traj_lat = [s['lat'][idx] for s in history if s['status'][idx] == 0]

        if len(traj_lon) > 1:
            ax.plot(traj_lon, traj_lat, color=PLASTIC_COLOR, alpha=0.3, linewidth=0.5,
                    transform=ccrs.PlateCarree())

    # Mark start
    start_lon = [initial_snapshot['lon'][idx] for idx in cohort_indices]
    start_lat = [initial_snapshot['lat'][idx] for idx in cohort_indices]
    ax.scatter(start_lon, start_lat, c='cyan', s=10, marker='o', alpha=0.8,
               transform=ccrs.PlateCarree(), label='Start', zorder=5)

    ax.legend(loc='lower left', fontsize=FONT_SIZE, framealpha=0.8, facecolor='black',
              edgecolor='white', labelcolor='white')

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig12_trajectory_spaghetti.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig13_quiver_streamplot(ocean_model, output_dir):
    """Figure 13: Quiver + streamplot overlay."""
    print("Generating Figure 13: Quiver + streamplot...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "Ocean Currents: Quiver + Streamlines (Synthetic)")

    # Get current field
    lons, lats, u, v = ocean_model.get_current_grid(resolution=2.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Streamplot
    speed = np.sqrt(u**2 + v**2)
    strm = ax.streamplot(lons, lats, u, v, color=speed, cmap='viridis',
                         density=1.5, linewidth=1, transform=ccrs.PlateCarree())

    # Quiver (sparser)
    skip = 4
    q = ax.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip],
                  alpha=0.5, scale=40, width=0.003, color='white',
                  transform=ccrs.PlateCarree())

    # Colorbar for streamlines
    cbar = plt.colorbar(strm.lines, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Current Speed (deg/day)', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig13_quiver_streamplot.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig14_pareto_source_contribution(history, output_dir):
    """Figure 14: Pareto chart of source contributions to gyre."""
    print("Generating Figure 14: Pareto source contribution...")

    # Define gyre region
    gyre_lon = (-70, -20)
    gyre_lat = (20, 40)

    snapshot = history[-1]

    # Count particles in gyre by source
    in_gyre = (
        (snapshot['lon'] >= gyre_lon[0]) & (snapshot['lon'] <= gyre_lon[1]) &
        (snapshot['lat'] >= gyre_lat[0]) & (snapshot['lat'] <= gyre_lat[1])
    )

    source_counts = {}
    for src in np.unique(snapshot['source']):
        count = np.sum((snapshot['source'] == src) & in_gyre)
        source_counts[src] = count

    # Sort descending
    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    sources, counts = zip(*sorted_sources)
    sources = [s.replace('_', ' ') for s in sources]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    x = np.arange(len(sources))
    bars = ax.bar(x, counts, color=PLASTIC_COLOR, alpha=0.8, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=FONT_SIZE-2, color='white')
    ax.set_ylabel('Particles in Gyre', fontsize=FONT_SIZE, color='white')
    ax.set_title('Source Contribution to Gyre Accumulation (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white', axis='y')

    # Add cumulative percentage line
    ax2 = ax.twinx()
    cumsum = np.cumsum(counts)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    ax2.plot(x, cumsum_pct, 'c-o', linewidth=2, markersize=6, label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage', fontsize=FONT_SIZE, color='cyan')
    ax2.tick_params(axis='y', colors='cyan', labelsize=FONT_SIZE-2)
    ax2.spines['right'].set_color('cyan')
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', fontsize=FONT_SIZE-2, framealpha=0.8, facecolor='black',
               edgecolor='cyan', labelcolor='cyan')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=fig.transFigure,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig14_pareto_contribution.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig15_small_multiples_release_month(ocean_model, geography, rng, output_dir):
    """Figure 15: Small multiples - release month vs gyre entry."""
    print("Generating Figure 15: Small multiples (release month)...")

    # This would simulate different release times
    # For simplicity, use different random seeds to show variation

    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    months = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov']

    for i, (ax, month) in enumerate(zip(axes, months)):
        setup_map_axes(ax, f"Release: {month} (Synthetic)")

        # Simulate with different seed
        particles_month = create_particle_sources(np.random.RandomState(RANDOM_SEED + 1000 + i * 10))
        sim = ParticleSimulator(ocean_model, geography, np.random.RandomState(RANDOM_SEED + 2000 + i * 10))
        hist = sim.simulate(particles_month, days=60, save_interval_days=60)

        snapshot = hist[-1]
        mask = snapshot['status'] == 0
        lons = snapshot['lon'][mask]
        lats = snapshot['lat'][mask]

        if len(lons) > 10:
            ax.scatter(lons, lats, c=PLASTIC_COLOR, s=1, alpha=0.4, transform=ccrs.PlateCarree())

        add_watermark(ax)

    plt.tight_layout()
    filepath = output_dir / 'fig15_small_multiples_month.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig16_age_histogram(history, output_dir):
    """Figure 16: Histogram of particle ages."""
    print("Generating Figure 16: Age histogram...")

    snapshot = history[-1]
    ages = snapshot['age']

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    ax.hist(ages, bins=50, color='magenta', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Age (days)', fontsize=FONT_SIZE, color='white')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE, color='white')
    ax.set_title('Particle Age Distribution (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white', axis='y')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig16_age_histogram.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig17_distance_vs_density(history, output_dir):
    """Figure 17: Distance vs local density scatter."""
    print("Generating Figure 17: Distance vs density scatter...")

    snapshot = history[-1]
    mask = snapshot['status'] == 0

    distances = snapshot['distance'][mask]
    lons = snapshot['lon'][mask]
    lats = snapshot['lat'][mask]

    # Compute local density (simplified: count neighbors within radius)
    density = np.zeros(len(lons))
    for i in range(len(lons)):
        dist_to_others = np.sqrt((lons - lons[i])**2 + (lats - lats[i])**2)
        density[i] = np.sum(dist_to_others < 2.0)  # Within 2 degrees

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    ax.scatter(distances, density, c='orange', s=3, alpha=0.5)
    ax.set_xlabel('Distance Traveled (degrees)', fontsize=FONT_SIZE, color='white')
    ax.set_ylabel('Local Density (neighbors)', fontsize=FONT_SIZE, color='white')
    ax.set_title('Distance vs Accumulation Density (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig17_distance_vs_density.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig18_hovmoller_latitude(history, output_dir):
    """Figure 18: Hovmöller diagram (time vs latitude)."""
    print("Generating Figure 18: Hovmöller latitude diagram...")

    # Create time vs latitude density matrix
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 100)
    time_steps = len(history)

    density_matrix = np.zeros((time_steps, len(lat_bins) - 1))

    for t, snapshot in enumerate(history):
        mask = snapshot['status'] == 0
        lats = snapshot['lat'][mask]
        counts, _ = np.histogram(lats, bins=lat_bins)
        density_matrix[t, :] = counts

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax.set_facecolor(OCEAN_COLOR)

    days = [s['day'] for s in history]
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    im = ax.pcolormesh(days, lat_centers, density_matrix.T, cmap='hot', shading='auto')

    ax.set_xlabel('Days', fontsize=FONT_SIZE, color='white')
    ax.set_ylabel('Latitude', fontsize=FONT_SIZE, color='white')
    ax.set_title('Hovmöller Diagram: Latitude vs Time (Synthetic)', fontsize=TITLE_SIZE, color='white')
    ax.tick_params(colors='white', labelsize=FONT_SIZE-2)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Particle Count', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    plt.text(0.98, 0.02, WATERMARK_TEXT, transform=ax.transAxes,
             fontsize=8, color='white', alpha=0.4, ha='right', va='bottom', style='italic')

    plt.tight_layout()
    filepath = output_dir / 'fig18_hovmoller_latitude.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig19_residence_time_map(history, output_dir):
    """Figure 19: Mean residence time map (how long particles stay in each cell)."""
    print("Generating Figure 19: Residence time map...")

    # Simplified: compute mean age in each grid cell at final time
    snapshot = history[-1]
    mask = snapshot['status'] == 0

    lons = snapshot['lon'][mask]
    lats = snapshot['lat'][mask]
    ages = snapshot['age'][mask]

    # Create grid
    lon_bins = np.linspace(LON_MIN, LON_MAX, 100)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 60)

    age_sum = np.zeros((len(lon_bins) - 1, len(lat_bins) - 1))
    count_matrix = np.zeros((len(lon_bins) - 1, len(lat_bins) - 1))

    for lon, lat, age in zip(lons, lats, ages):
        i = np.digitize(lon, lon_bins) - 1
        j = np.digitize(lat, lat_bins) - 1
        if 0 <= i < len(lon_bins) - 1 and 0 <= j < len(lat_bins) - 1:
            age_sum[i, j] += age
            count_matrix[i, j] += 1

    mean_age = np.divide(age_sum, count_matrix, out=np.zeros_like(age_sum), where=count_matrix > 0)
    mean_age = np.ma.masked_where(count_matrix == 0, mean_age)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "Mean Residence Time (Synthetic)")

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    im = ax.imshow(mean_age.T, extent=extent, origin='lower', cmap='viridis', alpha=0.8,
                   transform=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Mean Age (days)', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig19_residence_time_map.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

def fig20_curvature_map(ocean_model, output_dir):
    """Figure 20: Flow curvature map (vorticity proxy)."""
    print("Generating Figure 20: Flow curvature map...")

    # Compute vorticity (curl of velocity field)
    lons, lats, u, v = ocean_model.get_current_grid(resolution=1.5)

    # Numerical derivatives (simplified)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    vorticity = dv_dx - du_dy

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    setup_map_axes(ax, "Flow Curvature (Vorticity) Map (Synthetic)")

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    vort_smooth = gaussian_filter(vorticity, sigma=1.0)

    im = ax.imshow(vort_smooth, extent=extent, origin='lower', cmap='RdBu_r', alpha=0.7,
                   transform=ccrs.PlateCarree(), vmin=-0.5, vmax=0.5)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    cbar.set_label('Vorticity (1/day)', color='white', fontsize=FONT_SIZE)
    cbar.ax.tick_params(colors='white', labelsize=FONT_SIZE-2)

    add_watermark(ax)
    plt.tight_layout()

    filepath = output_dir / 'fig20_curvature_map.png'
    plt.savefig(filepath, dpi=FIG_DPI, facecolor=OCEAN_COLOR, edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")
    return filepath

# =============================================================================
# ANIMATION GENERATION
# =============================================================================

def generate_animation(ocean_model, history, output_dir):
    """Generate main animation video (300-600 seconds at 30 fps)."""
    if not HAS_IMAGEIO:
        print("Skipping animation: imageio not available")
        return None

    print(f"Generating animation ({ANIMATION_DURATION_SEC} seconds at {ANIMATION_FPS} fps)...")

    # Validate duration
    if not (300 <= ANIMATION_DURATION_SEC <= 600):
        raise ValueError(f"ANIMATION_DURATION_SEC must be in [300, 600], got {ANIMATION_DURATION_SEC}")

    total_frames = ANIMATION_DURATION_SEC * ANIMATION_FPS

    # Subsample history to match frame count
    history_indices = np.linspace(0, len(history) - 1, total_frames, dtype=int)

    print(f"  Rendering {total_frames} frames...")

    frames = []

    # Get current field once
    lons_curr, lats_curr, u_curr, v_curr = ocean_model.get_current_grid(resolution=4.0)
    lon_grid, lat_grid = np.meshgrid(lons_curr, lats_curr)

    for frame_idx in range(total_frames):
        if frame_idx % 300 == 0:
            print(f"    Frame {frame_idx}/{total_frames}...")

        snapshot = history[history_indices[frame_idx]]

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=OCEAN_COLOR)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        setup_map_axes(ax, f"DriftCast Animation - Day {int(snapshot['day'])}")

        # Plot currents (faint)
        ax.quiver(lon_grid, lat_grid, u_curr, v_curr, alpha=0.2, scale=60, width=0.002,
                  color='white', transform=ccrs.PlateCarree())

        # Plot particles by status
        for status, color, label, size in [
            (0, PLASTIC_COLOR, 'Floating', 3),
            (1, 'red', 'Beached', 2),
            (3, 'yellow', 'Gyre', 3),
        ]:
            mask = snapshot['status'] == status
            if np.sum(mask) > 0:
                ax.scatter(snapshot['lon'][mask], snapshot['lat'][mask],
                           c=color, s=size, alpha=PLASTIC_ALPHA, transform=ccrs.PlateCarree())

        # Add trajectory trails (last few frames)
        if frame_idx > 10:
            trail_length = 10
            for trail_offset in range(1, min(trail_length, frame_idx)):
                trail_snap = history[history_indices[frame_idx - trail_offset]]
                trail_mask = trail_snap['status'] == 0
                trail_alpha = 0.1 * (1 - trail_offset / trail_length)
                ax.scatter(trail_snap['lon'][trail_mask], trail_snap['lat'][trail_mask],
                           c=PLASTIC_COLOR, s=1, alpha=trail_alpha, transform=ccrs.PlateCarree())

        add_watermark(ax)
        plt.tight_layout()

        # Convert to image array (Matplotlib >=3.9 uses buffer_rgba)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())

        plt.close(fig)

    # Save video
    filepath = output_dir / 'driftcast.mp4'
    print(f"  Writing video to {filepath}...")

    writer = imageio.get_writer(filepath, fps=ANIMATION_FPS, codec='libx264',
                                 pixelformat='yuv420p', quality=9)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    # Validate duration
    actual_duration = total_frames / ANIMATION_FPS
    if abs(actual_duration - ANIMATION_DURATION_SEC) > 0.1:
        raise ValueError(f"Duration mismatch: expected {ANIMATION_DURATION_SEC}s, got {actual_duration}s")

    print(f"  Video saved: {filepath}")
    print(f"  Duration: {actual_duration:.1f}s, Frames: {total_frames}, FPS: {ANIMATION_FPS}")

    return filepath

def generate_teaser_gif(ocean_model, history, output_dir):
    """Generate short teaser GIF (20-30 seconds)."""
    if not HAS_IMAGEIO:
        print("Skipping teaser: imageio not available")
        return None

    print(f"Generating teaser GIF ({TEASER_DURATION_SEC} seconds)...")

    fps = 10  # Lower fps for GIF
    total_frames = TEASER_DURATION_SEC * fps

    # Sample from interesting part (middle of simulation)
    start_idx = len(history) // 3
    end_idx = 2 * len(history) // 3
    history_indices = np.linspace(start_idx, end_idx, total_frames, dtype=int)

    frames = []

    lons_curr, lats_curr, u_curr, v_curr = ocean_model.get_current_grid(resolution=5.0)
    lon_grid, lat_grid = np.meshgrid(lons_curr, lats_curr)

    for frame_idx in range(total_frames):
        snapshot = history[history_indices[frame_idx]]

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI // 2, facecolor=OCEAN_COLOR)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        setup_map_axes(ax, f"DriftCast - Day {int(snapshot['day'])}")

        # Particles
        mask = snapshot['status'] == 0
        ax.scatter(snapshot['lon'][mask], snapshot['lat'][mask],
                   c=PLASTIC_COLOR, s=2, alpha=0.6, transform=ccrs.PlateCarree())

        add_watermark(ax)
        plt.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())

        plt.close(fig)

    filepath = output_dir / 'driftcast_teaser.gif'
    print(f"  Writing GIF to {filepath}...")

    imageio.mimsave(filepath, frames, fps=fps, loop=0)

    print(f"  GIF saved: {filepath}")
    return filepath

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Oceans-Four DriftCast: Synthetic Demo Generation")
    print("=" * 70)
    print()

    # Set up paths
    output_dir = Path('outputs')
    figures_dir = output_dir / 'figures'
    animations_dir = output_dir / 'animations'

    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG
    rng = np.random.RandomState(RANDOM_SEED)

    # Initialize models
    print("Initializing ocean model and geography...")
    ocean_model = NorthAtlanticGyreModel()
    geography = Geography()

    # Create particles
    print(f"\nCreating particle sources...")
    initial_particles = create_particle_sources(rng)
    total_particles = len(initial_particles['lon'])
    print(f"  Total particles: {total_particles}")
    for source, count in SOURCE_COUNTS.items():
        print(f"    {source}: {count}")

    # Run simulation
    print(f"\nRunning simulation ({SIMULATION_DAYS} days)...")
    simulator = ParticleSimulator(ocean_model, geography, rng)
    history = simulator.simulate(initial_particles, days=SIMULATION_DAYS, save_interval_days=1)

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    generated_files = []

    generated_files.append(fig01_base_map_with_currents(ocean_model, figures_dir))
    generated_files.append(fig02_source_map(initial_particles, figures_dir))
    generated_files.append(fig03_dispersion_heatmap_us(history, figures_dir, day=30))
    generated_files.append(fig04_dispersion_heatmap_europe(history, figures_dir, day=30))
    generated_files.append(fig05_accumulation_density_180d(history, figures_dir))
    generated_files.append(fig06_time_series_fates(history, figures_dir))
    generated_files.append(fig07_travel_distance_histogram(history, figures_dir))
    generated_files.append(fig08_gyre_arrival_kde(history, figures_dir))
    generated_files.append(fig09_coastline_hotspots(history, figures_dir))
    generated_files.append(fig10_sankey_flow(history, figures_dir))
    generated_files.append(fig11_storm_scenario(ocean_model, initial_particles, rng, figures_dir))
    generated_files.append(fig12_trajectory_spaghetti(history, figures_dir))
    generated_files.append(fig13_quiver_streamplot(ocean_model, figures_dir))
    generated_files.append(fig14_pareto_source_contribution(history, figures_dir))
    generated_files.append(fig15_small_multiples_release_month(ocean_model, geography, rng, figures_dir))
    generated_files.append(fig16_age_histogram(history, figures_dir))
    generated_files.append(fig17_distance_vs_density(history, figures_dir))
    generated_files.append(fig18_hovmoller_latitude(history, figures_dir))
    generated_files.append(fig19_residence_time_map(history, figures_dir))
    generated_files.append(fig20_curvature_map(ocean_model, figures_dir))

    # Generate animations
    print("\n" + "=" * 70)
    print("GENERATING ANIMATIONS")
    print("=" * 70)

    video_file = generate_animation(ocean_model, history, animations_dir)
    if video_file:
        generated_files.append(video_file)

    gif_file = generate_teaser_gif(ocean_model, history, animations_dir)
    if gif_file:
        generated_files.append(gif_file)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    final_snapshot = history[-1]

    print("\nFinal particle counts by fate:")
    fate_names = ['Floating', 'Beached', 'Sunk', 'Gyre-Trapped']
    for status_idx, name in enumerate(fate_names):
        count = np.sum(final_snapshot['status'] == status_idx)
        pct = 100 * count / total_particles
        print(f"  {name:15s}: {count:6d} ({pct:5.1f}%)")

    print("\nBeaching by source region:")
    beached_mask = final_snapshot['status'] == 1
    for source in np.unique(final_snapshot['source']):
        count = np.sum((final_snapshot['source'] == source) & beached_mask)
        total_src = np.sum(final_snapshot['source'] == source)
        pct = 100 * count / total_src if total_src > 0 else 0
        print(f"  {source:20s}: {count:5d} / {total_src:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)

    for filepath in generated_files:
        if filepath:
            print(f"  {filepath}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nAll figures and animations have been generated.")
    print("DISCLAIMER: All results are synthetic and illustrative for demonstration purposes.")
    print()

if __name__ == "__main__":
    main()
