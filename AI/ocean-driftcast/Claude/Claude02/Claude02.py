# mid animations

"""
Oceans-Four DriftCast - North Atlantic Plastic Drift Visualization
Synthetic demonstration system for realistic-looking ocean plastic dispersion
DISCLAIMER: All results are synthetic/simulated for demonstration purposes only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.sankey import Sankey
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
import imageio
import os
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Domain
LON_MIN, LON_MAX = -100, 20
LAT_MIN, LAT_MAX = 0, 70

# Simulation parameters
N_PARTICLES_BASE = 5000  # Base particle count
SIM_DAYS = 365
DT_HOURS = 1.0
N_STEPS = int(SIM_DAYS * 24 / DT_HOURS)

# Physics
TURBULENT_DIFFUSION = 0.02  # degrees per sqrt(hour)
WINDAGE_FACTOR = 0.02  # Additional drift component
BEACHING_DISTANCE_KM = 20
SINKING_PROB_PER_DAY = 0.001
FRAGMENTATION_RATE = 0.0005  # Per day

# Output settings
FIG_DIR = "outputs/figures"
ANIM_DIR = "outputs/animations"
FIG_SIZE = (19.2, 10.8)
DPI = 100
WATERMARK = "Demo, synthetic"

# Create output directories
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ANIM_DIR, exist_ok=True)

# ============================================================================
# IDEALIZED GYRE CIRCULATION
# ============================================================================

def idealized_current_field(lon, lat):
    """
    Synthetic North Atlantic circulation with subtropical and subpolar gyres
    Returns (u, v) in degrees per hour
    """
    u = np.zeros_like(lon, dtype=float)
    v = np.zeros_like(lat, dtype=float)
    
    # Subtropical gyre (10-40N)
    # Westward flow at low latitudes (North Equatorial Current)
    mask_nec = (lat >= 10) & (lat <= 25)
    u[mask_nec] += -0.15 * np.cos((lat[mask_nec] - 17.5) * np.pi / 15)
    
    # Northward flow along western boundary (Gulf Stream analog)
    mask_gs = (lon >= -80) & (lon <= -50) & (lat >= 25) & (lat <= 45)
    dist_from_coast = (lon[mask_gs] + 75) / 25
    u[mask_gs] += 0.25 * (1 - dist_from_coast) * np.exp(-dist_from_coast)
    v[mask_gs] += 0.35 * (1 - dist_from_coast) * np.exp(-dist_from_coast)
    
    # Eastward flow across Atlantic (North Atlantic Current)
    mask_nac = (lat >= 40) & (lat <= 55) & (lon >= -50)
    u[mask_nac] += 0.20 * np.cos((lat[mask_nac] - 47.5) * np.pi / 15)
    v[mask_nac] += 0.05 * np.sin((lon[mask_nac] + 25) * np.pi / 50)
    
    # Southward flow along eastern boundary (Canary Current)
    mask_cc = (lon >= -20) & (lon <= 10) & (lat >= 15) & (lat <= 40)
    u[mask_cc] += -0.08 * (1 - np.abs(lon[mask_cc] - 5) / 30)
    v[mask_cc] += -0.12 * np.cos((lat[mask_cc] - 27.5) * np.pi / 25)
    
    # Gyre interior (slow anticyclonic)
    mask_interior = (lon >= -70) & (lon <= -20) & (lat >= 20) & (lat <= 35)
    center_lon, center_lat = -45, 27.5
    dx = lon[mask_interior] - center_lon
    dy = lat[mask_interior] - center_lat
    r = np.sqrt(dx**2 + dy**2)
    u[mask_interior] += -0.03 * dy / (r + 1)
    v[mask_interior] += 0.03 * dx / (r + 1)
    
    # Subpolar gyre (50-65N, cyclonic)
    mask_subpolar = (lat >= 50) & (lat <= 65) & (lon >= -50) & (lon <= -10)
    center_lon, center_lat = -30, 57.5
    dx = lon[mask_subpolar] - center_lon
    dy = lat[mask_subpolar] - center_lat
    r = np.sqrt(dx**2 + dy**2)
    u[mask_subpolar] += 0.04 * dy / (r + 1)
    v[mask_subpolar] += -0.04 * dx / (r + 1)
    
    return u, v

# ============================================================================
# SOURCE REGIONS
# ============================================================================

SOURCE_REGIONS = {
    'US_East_Coast': {'lon': (-80, -70), 'lat': (30, 42), 'count': 1500},
    'US_Gulf_Coast': {'lon': (-97, -80), 'lat': (24, 30), 'count': 800},
    'Mississippi_Loop': {'lon': (-92, -88), 'lat': (28, 30), 'count': 400},
    'Caribbean': {'lon': (-85, -60), 'lat': (10, 20), 'count': 600},
    'St_Lawrence': {'lon': (-72, -60), 'lat': (45, 50), 'count': 300},
    'UK_Thames': {'lon': (-2, 2), 'lat': (50, 52), 'count': 350},
    'France_Seine': {'lon': (-2, 2), 'lat': (48, 50), 'count': 300},
    'Rhine_Delta': {'lon': (3, 6), 'lat': (51, 53), 'count': 250},
    'Iberia_Tagus': {'lon': (-10, -8), 'lat': (38, 40), 'count': 200},
    'Spain_Ebro': {'lon': (0, 2), 'lat': (40, 42), 'count': 200},
    'Morocco_Coast': {'lon': (-8, -4), 'lat': (30, 36), 'count': 300},
}

def initialize_particles():
    """Initialize particles from all source regions"""
    particles = []
    source_ids = []
    
    for idx, (name, region) in enumerate(SOURCE_REGIONS.items()):
        n = region['count']
        lon = np.random.uniform(region['lon'][0], region['lon'][1], n)
        lat = np.random.uniform(region['lat'][0], region['lat'][1], n)
        
        for i in range(n):
            particles.append({
                'lon': lon[i],
                'lat': lat[i],
                'status': 'floating',
                'age_days': 0,
                'distance_km': 0,
                'source': name,
                'beached_at': None,
            })
            source_ids.append(idx)
    
    return particles, source_ids

# ============================================================================
# LAND MASK AND BEACHING
# ============================================================================

def create_land_mask(resolution=0.5):
    """Create a simple land mask for beaching detection"""
    lon_grid = np.arange(LON_MIN, LON_MAX, resolution)
    lat_grid = np.arange(LAT_MIN, LAT_MAX, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Simple land polygons (rough approximations)
    land_mask = np.zeros_like(lon_mesh, dtype=bool)
    
    # North America east coast
    land_mask |= (lon_mesh >= -100) & (lon_mesh <= -50) & (lat_mesh >= 25) & (lat_mesh <= 70) & (lon_mesh >= -85 + (lat_mesh - 25) * 0.7)
    
    # Central America
    land_mask |= (lon_mesh >= -100) & (lon_mesh <= -75) & (lat_mesh >= 8) & (lat_mesh <= 25)
    
    # Caribbean islands (simplified)
    land_mask |= (lon_mesh >= -80) & (lon_mesh <= -60) & (lat_mesh >= 10) & (lat_mesh <= 20) & (np.random.random(lon_mesh.shape) < 0.15)
    
    # Europe
    land_mask |= (lon_mesh >= -10) & (lon_mesh <= 20) & (lat_mesh >= 35) & (lat_mesh <= 70) & (lon_mesh <= 30 - lat_mesh * 0.3)
    
    # North Africa
    land_mask |= (lon_mesh >= -20) & (lon_mesh <= 20) & (lat_mesh >= 0) & (lat_mesh <= 35) & (lat_mesh <= 35 - (lon_mesh + 20) * 0.5)
    
    # Greenland
    land_mask |= (lon_mesh >= -55) & (lon_mesh <= -20) & (lat_mesh >= 60) & (lat_mesh <= 70)
    
    # Iceland
    land_mask |= (lon_mesh >= -25) & (lon_mesh <= -13) & (lat_mesh >= 63) & (lat_mesh <= 67)
    
    return lon_grid, lat_grid, land_mask

LAND_LON, LAND_LAT, LAND_MASK = create_land_mask()

def is_near_land(lon, lat, distance_km=20):
    """Check if position is within distance_km of land"""
    # Convert km to degrees (rough approximation)
    distance_deg = distance_km / 111.0
    
    # Find nearest grid cell
    lon_idx = np.clip(np.searchsorted(LAND_LON, lon), 0, len(LAND_LON) - 1)
    lat_idx = np.clip(np.searchsorted(LAND_LAT, lat), 0, len(LAND_LAT) - 1)
    
    # Check neighborhood
    search_radius = int(np.ceil(distance_deg / 0.5)) + 1
    lon_slice = slice(max(0, lon_idx - search_radius), min(len(LAND_LON), lon_idx + search_radius))
    lat_slice = slice(max(0, lat_idx - search_radius), min(len(LAND_LAT), lat_idx + search_radius))
    
    if np.any(LAND_MASK[lat_slice, lon_slice]):
        return True
    return False

# ============================================================================
# PARTICLE ADVECTION
# ============================================================================

def advect_particles(particles, dt_hours=1.0, storm_boost=1.0):
    """
    Advect particles using RK4 with turbulent diffusion
    storm_boost > 1 increases diffusion and windage temporarily
    """
    active_particles = []
    
    for p in particles:
        if p['status'] != 'floating':
            active_particles.append(p)
            continue
        
        lon, lat = p['lon'], p['lat']
        
        # RK4 integration
        u1, v1 = idealized_current_field(lon, lat)
        lon1 = lon + 0.5 * dt_hours * (u1 + WINDAGE_FACTOR * storm_boost * u1)
        lat1 = lat + 0.5 * dt_hours * (v1 + WINDAGE_FACTOR * storm_boost * v1)
        
        u2, v2 = idealized_current_field(lon1, lat1)
        lon2 = lon + 0.5 * dt_hours * (u2 + WINDAGE_FACTOR * storm_boost * u2)
        lat2 = lat + 0.5 * dt_hours * (v2 + WINDAGE_FACTOR * storm_boost * v2)
        
        u3, v3 = idealized_current_field(lon2, lat2)
        lon3 = lon + dt_hours * (u3 + WINDAGE_FACTOR * storm_boost * u3)
        lat3 = lat + dt_hours * (v3 + WINDAGE_FACTOR * storm_boost * v3)
        
        u4, v4 = idealized_current_field(lon3, lat3)
        
        u_mean = (u1 + 2*u2 + 2*u3 + u4) / 6
        v_mean = (v1 + 2*v2 + 2*v3 + v4) / 6
        
        # Advection + windage
        lon_new = lon + dt_hours * (u_mean + WINDAGE_FACTOR * storm_boost * u_mean)
        lat_new = lat + dt_hours * (v_mean + WINDAGE_FACTOR * storm_boost * v_mean)
        
        # Turbulent diffusion
        diffusion = TURBULENT_DIFFUSION * np.sqrt(dt_hours) * storm_boost
        lon_new += np.random.normal(0, diffusion)
        lat_new += np.random.normal(0, diffusion)
        
        # Boundary conditions
        lon_new = np.clip(lon_new, LON_MIN, LON_MAX)
        lat_new = np.clip(lat_new, LAT_MIN, LAT_MAX)
        
        # Update distance
        dlat = lat_new - lat
        dlon = (lon_new - lon) * np.cos(np.radians(lat))
        distance_deg = np.sqrt(dlat**2 + dlon**2)
        distance_km = distance_deg * 111.0
        
        p['lon'] = lon_new
        p['lat'] = lat_new
        p['distance_km'] += distance_km
        p['age_days'] += dt_hours / 24
        
        # Check beaching
        if is_near_land(lon_new, lat_new, BEACHING_DISTANCE_KM):
            p['status'] = 'beached'
            p['beached_at'] = (lon_new, lat_new)
        # Check sinking
        elif np.random.random() < SINKING_PROB_PER_DAY * dt_hours / 24:
            p['status'] = 'sunk'
        # Check if in gyre core
        elif (25 <= lat_new <= 35) and (-70 <= lon_new <= -30):
            p['status'] = 'gyre_trapped'
        
        # Fragmentation (occasionally split particles)
        if np.random.random() < FRAGMENTATION_RATE * dt_hours / 24:
            # Create fragment
            fragment = p.copy()
            fragment['lon'] += np.random.normal(0, 0.1)
            fragment['lat'] += np.random.normal(0, 0.1)
            active_particles.append(fragment)
        
        active_particles.append(p)
    
    return active_particles

# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(n_days=365, save_snapshots=True):
    """Run the main particle simulation"""
    print(f"Initializing {N_PARTICLES_BASE} particles from {len(SOURCE_REGIONS)} sources...")
    particles, source_ids = initialize_particles()
    
    snapshots = []
    n_steps = int(n_days * 24 / DT_HOURS)
    
    print(f"Simulating {n_days} days ({n_steps} steps)...")
    
    for step in range(n_steps):
        # Storm event days 90-97
        storm_boost = 3.0 if (90 * 24 <= step * DT_HOURS <= 97 * 24) else 1.0
        
        particles = advect_particles(particles, DT_HOURS, storm_boost)
        
        # Save snapshots
        if save_snapshots and step % (24 * 5) == 0:  # Every 5 days
            snapshot = {
                'day': step * DT_HOURS / 24,
                'particles': [p.copy() for p in particles]
            }
            snapshots.append(snapshot)
            
            if step % (24 * 30) == 0:
                floating = sum(1 for p in particles if p['status'] == 'floating')
                beached = sum(1 for p in particles if p['status'] == 'beached')
                sunk = sum(1 for p in particles if p['status'] == 'sunk')
                trapped = sum(1 for p in particles if p['status'] == 'gyre_trapped')
                print(f"  Day {step * DT_HOURS / 24:.0f}: {len(particles)} particles "
                      f"(float={floating}, beach={beached}, sink={sunk}, trap={trapped})")
    
    return particles, snapshots

# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def setup_map_axis(ax, title=""):
    """Configure a map axis with consistent styling"""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#404040', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#666666', zorder=2)
    ax.set_facecolor('#0a1628')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#445566', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    
    # Add watermark
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom',
            style='italic')

def save_figure(fig, filename):
    """Save figure with consistent settings"""
    filepath = os.path.join(FIG_DIR, filename)
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='#0a1628')
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath

# ============================================================================
# FIGURE GENERATION
# ============================================================================

def figure_01_current_field():
    """Base map with current vectors and gyre labels"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "North Atlantic Current Field & Gyres")
    
    # Sample current field
    lon_grid = np.arange(LON_MIN, LON_MAX, 3, dtype=float)
    lat_grid = np.arange(LAT_MIN, LAT_MAX, 3, dtype=float)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    u, v = idealized_current_field(lon_mesh, lat_mesh)
    
    # Quiver plot
    magnitude = np.sqrt(u**2 + v**2)
    ax.quiver(lon_mesh, lat_mesh, u, v, magnitude,
              transform=ccrs.PlateCarree(), cmap='plasma',
              scale=5, width=0.003, alpha=0.7, zorder=3)
    
    # Gyre annotations
    ax.text(-45, 27, 'Subtropical\nGyre', fontsize=16, fontweight='bold',
            color='orange', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    ax.text(-30, 57, 'Subpolar\nGyre', fontsize=14, fontweight='bold',
            color='cyan', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Current labels
    ax.annotate('Gulf Stream', xy=(-70, 35), fontsize=10, color='yellow',
                fontweight='bold')
    ax.annotate('Canary Current', xy=(-5, 25), fontsize=10, color='yellow',
                fontweight='bold')
    
    return save_figure(fig, '01_current_field.png')

def figure_02_source_map(particles):
    """Source map showing release zones"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Particle Source Regions")
    
    # Plot initial positions colored by source
    colors = plt.cm.tab10(np.linspace(0, 1, len(SOURCE_REGIONS)))
    
    for idx, (name, region) in enumerate(SOURCE_REGIONS.items()):
        source_particles = [p for p in particles if p['source'] == name]
        lons = [p['lon'] for p in source_particles]
        lats = [p['lat'] for p in source_particles]
        
        ax.scatter(lons, lats, c=[colors[idx]], s=15, alpha=0.6,
                   transform=ccrs.PlateCarree(), label=name.replace('_', ' '),
                   edgecolors='none', zorder=4)
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, facecolor='#1a1a1a',
              edgecolor='white', labelcolor='white')
    
    return save_figure(fig, '02_source_map.png')

def figure_03_dispersion_heatmap_us(snapshots):
    """30-day dispersion heatmap from US releases"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "30-Day Dispersion: US Sources")
    
    # Get 30-day snapshot
    snap_30 = [s for s in snapshots if abs(s['day'] - 30) < 3][0]
    us_particles = [p for p in snap_30['particles']
                    if 'US' in p['source'] or 'Mississippi' in p['source']]
    
    lons = np.array([p['lon'] for p in us_particles])
    lats = np.array([p['lat'] for p in us_particles])
    
    # Create density field
    lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
    density, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    density = gaussian_filter(density.T, sigma=1.5)
    
    # Plot
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], density,
                     levels=20, cmap='hot', alpha=0.7,
                     transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Particle Density', shrink=0.7)
    
    return save_figure(fig, '03_dispersion_us_30d.png')

def figure_04_dispersion_heatmap_eu(snapshots):
    """30-day dispersion heatmap from European releases"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "30-Day Dispersion: European Sources")
    
    snap_30 = [s for s in snapshots if abs(s['day'] - 30) < 3][0]
    eu_particles = [p for p in snap_30['particles']
                    if any(x in p['source'] for x in ['UK', 'France', 'Rhine', 'Iberia', 'Spain', 'Morocco'])]
    
    lons = np.array([p['lon'] for p in eu_particles])
    lats = np.array([p['lat'] for p in eu_particles])
    
    lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
    density, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    density = gaussian_filter(density.T, sigma=1.5)
    
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], density,
                     levels=20, cmap='hot', alpha=0.7,
                     transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Particle Density', shrink=0.7)
    
    return save_figure(fig, '04_dispersion_eu_30d.png')

def figure_05_gyre_accumulation(snapshots):
    """180-day accumulation showing subtropical gyre hotspot"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "180-Day Accumulation: Subtropical Gyre Hotspot")
    
    snap_180 = snapshots[-1]  # Last snapshot
    floating = [p for p in snap_180['particles'] if p['status'] in ['floating', 'gyre_trapped']]
    
    lons = np.array([p['lon'] for p in floating])
    lats = np.array([p['lat'] for p in floating])
    
    lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
    density, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    density = gaussian_filter(density.T, sigma=2.0)
    
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], density,
                     levels=25, cmap='YlOrRd', alpha=0.8,
                     transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Accumulation Density', shrink=0.7)
    
    # Highlight gyre core
    gyre_box = mpatches.Rectangle((-70, 25), 40, 10, linewidth=2,
                                   edgecolor='cyan', facecolor='none',
                                   transform=ccrs.PlateCarree(), zorder=5)
    ax.add_patch(gyre_box)
    ax.text(-50, 32, 'Gyre Core', fontsize=12, color='cyan',
            fontweight='bold', ha='center')
    
    return save_figure(fig, '05_gyre_accumulation_180d.png')

def figure_06_fate_timeseries(snapshots):
    """Time series of particle fates"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    days = [s['day'] for s in snapshots]
    floating = [sum(1 for p in s['particles'] if p['status'] == 'floating') for s in snapshots]
    beached = [sum(1 for p in s['particles'] if p['status'] == 'beached') for s in snapshots]
    sunk = [sum(1 for p in s['particles'] if p['status'] == 'sunk') for s in snapshots]
    trapped = [sum(1 for p in s['particles'] if p['status'] == 'gyre_trapped') for s in snapshots]
    
    ax.plot(days, floating, 'o-', label='Floating', color='#00ffff', linewidth=2, markersize=4)
    ax.plot(days, beached, 's-', label='Beached', color='#ff9900', linewidth=2, markersize=4)
    ax.plot(days, sunk, '^-', label='Sunk', color='#9966ff', linewidth=2, markersize=4)
    ax.plot(days, trapped, 'd-', label='Gyre Trapped', color='#ff3366', linewidth=2, markersize=4)
    
    ax.set_xlabel('Days', fontsize=12, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Particle Fate Evolution Over Time', fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#445566')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Watermark
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '06_fate_timeseries.png')

def figure_07_distance_histogram(particles):
    """Histogram of travel distances at 90 and 180 days"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    # Filter by age
    particles_90 = [p for p in particles if 85 <= p['age_days'] <= 95]
    particles_180 = [p for p in particles if 175 <= p['age_days'] <= 185]
    
    distances_90 = [p['distance_km'] for p in particles_90]
    distances_180 = [p['distance_km'] for p in particles_180]
    
    ax.hist(distances_90, bins=50, alpha=0.7, color='#ff9900', label='90 Days', edgecolor='white', linewidth=0.5)
    ax.hist(distances_180, bins=50, alpha=0.7, color='#ff3366', label='180 Days', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Travel Distance (km)', fontsize=12, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Distribution of Travel Distances', fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#445566')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '07_distance_histogram.png')

def figure_08_gyre_entry_kde(particles):
    """KDE of arrival times to gyre by source region"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    # Find particles that entered gyre
    gyre_particles = [p for p in particles if p['status'] == 'gyre_trapped']
    
    # Group by major source regions
    source_groups = {
        'US East': ['US_East_Coast'],
        'US Gulf': ['US_Gulf_Coast', 'Mississippi_Loop'],
        'Caribbean': ['Caribbean'],
        'Europe': ['UK_Thames', 'France_Seine', 'Rhine_Delta', 'Iberia_Tagus', 'Spain_Ebro'],
    }
    
    colors = ['#ff9900', '#00ffff', '#ff3366', '#66ff66']
    
    for idx, (group_name, sources) in enumerate(source_groups.items()):
        group_particles = [p for p in gyre_particles if p['source'] in sources]
        if len(group_particles) > 10:
            ages = [p['age_days'] for p in group_particles]
            ages = np.array(ages)
            kde = gaussian_kde(ages)
            x_range = np.linspace(0, 365, 200)
            density = kde(x_range)
            ax.plot(x_range, density, label=group_name, color=colors[idx], linewidth=2.5)
    
    ax.set_xlabel('Days to Gyre Entry', fontsize=12, color='white')
    ax.set_ylabel('Probability Density', fontsize=12, color='white')
    ax.set_title('Gyre Entry Time Distribution by Source Region', fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='#445566')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '08_gyre_entry_kde.png')

def figure_09_beaching_hotspots(particles):
    """Coastline hotspot map for beaching intensity"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Beaching Hotspots by Coastal Region")
    
    beached = [p for p in particles if p['status'] == 'beached' and p['beached_at']]
    lons = np.array([p['beached_at'][0] for p in beached])
    lats = np.array([p['beached_at'][1] for p in beached])
    
    # Create high-resolution density
    lon_bins = np.linspace(LON_MIN, LON_MAX, 200)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 140)
    density, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    density = gaussian_filter(density.T, sigma=1.0)
    
    # Only show high-density areas
    density[density < np.percentile(density[density > 0], 60)] = 0
    
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], density,
                     levels=15, cmap='Reds', alpha=0.8,
                     transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Beaching Intensity', shrink=0.7)
    
    return save_figure(fig, '09_beaching_hotspots.png')

def figure_10_sankey_diagram(particles):
    """Sankey diagram from sources to fates"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    ax.axis('off')
    
    # Aggregate by major regions and fates
    source_groups = {
        'US': ['US_East_Coast', 'US_Gulf_Coast', 'Mississippi_Loop'],
        'Caribbean': ['Caribbean', 'St_Lawrence'],
        'Europe': ['UK_Thames', 'France_Seine', 'Rhine_Delta', 'Iberia_Tagus', 'Spain_Ebro'],
        'Africa': ['Morocco_Coast'],
    }
    
    fates = ['floating', 'beached', 'sunk', 'gyre_trapped']
    
    # Count flows
    flows = []
    labels = []
    
    for group_name, sources in source_groups.items():
        group_particles = [p for p in particles if p['source'] in sources]
        total = len(group_particles)
        
        for fate in fates:
            count = sum(1 for p in group_particles if p['status'] == fate)
            if count > 0:
                flows.append(count / total * 100)  # As percentage
                if group_name not in labels:
                    labels.append(group_name)
    
    # Create simplified flow diagram with text
    y_pos = 0.8
    x_source = 0.2
    x_fate = 0.7
    
    colors_src = {'US': '#ff9900', 'Caribbean': '#00ffff', 'Europe': '#66ff66', 'Africa': '#ff3366'}
    colors_fate = {'floating': '#00ffff', 'beached': '#ff9900', 'sunk': '#9966ff', 'gyre_trapped': '#ff3366'}
    
    # Draw sources
    for group_name, sources in source_groups.items():
        group_particles = [p for p in particles if p['source'] in sources]
        count = len(group_particles)
        ax.text(x_source, y_pos, f'{group_name}\n({count})', fontsize=11, color='white',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors_src[group_name], alpha=0.7))
        y_pos -= 0.2
    
    # Draw fates
    y_pos = 0.8
    for fate in fates:
        count = sum(1 for p in particles if p['status'] == fate)
        ax.text(x_fate, y_pos, f'{fate.replace("_", " ").title()}\n({count})', fontsize=11, color='white',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors_fate[fate], alpha=0.7))
        y_pos -= 0.2
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Particle Flow: Sources to Fates (180 days)', fontsize=14, fontweight='bold', color='white', pad=20)
    
    ax.text(0.99, 0.01, WATERMARK, fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '10_sankey_sources_to_fates.png')

def figure_11_storm_comparison(particles_normal, particles_storm):
    """Before vs after storm scenario comparison"""
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#0a1628')
    
    # Before storm (left panel)
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    setup_map_axis(ax1, "Day 90: Pre-Storm Distribution")
    
    lons = [p['lon'] for p in particles_normal if p['age_days'] < 90]
    lats = [p['lat'] for p in particles_normal if p['age_days'] < 90]
    
    ax1.scatter(lons, lats, c='#ff9900', s=2, alpha=0.5, transform=ccrs.PlateCarree(), zorder=4)
    
    # After storm (right panel)
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    setup_map_axis(ax2, "Day 100: Post-Storm Dispersion")
    
    lons = [p['lon'] for p in particles_storm if 95 < p['age_days'] < 105]
    lats = [p['lat'] for p in particles_storm if 95 < p['age_days'] < 105]
    
    ax2.scatter(lons, lats, c='#ff3366', s=2, alpha=0.5, transform=ccrs.PlateCarree(), zorder=4)
    
    plt.tight_layout()
    return save_figure(fig, '11_storm_comparison.png')

def figure_12_trajectory_spaghetti(snapshots):
    """Trajectory spaghetti plot for labeled cohort"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Sample Trajectories: US East Coast to Gyre")
    
    # Select a small cohort from US East Coast
    first_snap = snapshots[0]
    cohort = [p for p in first_snap['particles'] if p['source'] == 'US_East_Coast'][:50]
    cohort_ids = [id(p) for p in cohort]
    
    # Track these particles through snapshots
    for pid in cohort_ids:
        traj_lons = []
        traj_lats = []
        
        for snap in snapshots:
            for p in snap['particles']:
                if id(p) == pid or (p.get('lon') and len(traj_lons) > 0 and 
                                    abs(p['lon'] - traj_lons[-1]) < 5 and 
                                    abs(p['lat'] - traj_lats[-1]) < 5):
                    traj_lons.append(p['lon'])
                    traj_lats.append(p['lat'])
                    break
        
        if len(traj_lons) > 2:
            ax.plot(traj_lons, traj_lats, color='#ff9900', alpha=0.3, linewidth=1,
                    transform=ccrs.PlateCarree(), zorder=3)
    
    # Mark start and end points
    start_lons = [p['lon'] for p in cohort]
    start_lats = [p['lat'] for p in cohort]
    ax.scatter(start_lons, start_lats, c='cyan', s=30, marker='o', edgecolors='white',
               linewidths=1, transform=ccrs.PlateCarree(), zorder=5, label='Start')
    
    return save_figure(fig, '12_trajectory_spaghetti.png')

def figure_13_quiver_streamplot():
    """Quiver plus streamplot overlay"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Current Field: Quiver + Streamlines")
    
    # Dense grid for streamplot
    lon_stream = np.linspace(LON_MIN, LON_MAX, 150)
    lat_stream = np.linspace(LAT_MIN, LAT_MAX, 100)
    lon_mesh, lat_mesh = np.meshgrid(lon_stream, lat_stream)
    u, v = idealized_current_field(lon_mesh, lat_mesh)
    
    # Streamplot
    strm = ax.streamplot(lon_stream, lat_stream, u.T, v.T,
                         color=np.sqrt(u.T**2 + v.T**2), cmap='plasma',
                         linewidth=1.5, density=1.5, transform=ccrs.PlateCarree(),
                         zorder=3)
    
    # Coarse grid for quiver
    lon_quiver = np.arange(LON_MIN, LON_MAX, 8, dtype=float)
    lat_quiver = np.arange(LAT_MIN, LAT_MAX, 8, dtype=float)
    lon_q, lat_q = np.meshgrid(lon_quiver, lat_quiver)
    u_q, v_q = idealized_current_field(lon_q, lat_q)
    
    ax.quiver(lon_q, lat_q, u_q, v_q, color='white', scale=4, width=0.004,
              alpha=0.5, transform=ccrs.PlateCarree(), zorder=4)
    
    plt.colorbar(strm.lines, ax=ax, label='Current Speed (deg/hr)', shrink=0.7)
    
    return save_figure(fig, '13_quiver_streamplot.png')

def figure_14_pareto_contributions(particles):
    """Pareto bar chart of source contributions to gyre"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    gyre_particles = [p for p in particles if p['status'] == 'gyre_trapped']
    
    # Count by source
    source_counts = {}
    for p in gyre_particles:
        source = p['source'].replace('_', ' ')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Sort by count
    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    sources = [s[0] for s in sorted_sources]
    counts = [s[1] for s in sorted_sources]
    
    # Calculate cumulative percentage
    total = sum(counts)
    cumulative = np.cumsum(counts) / total * 100
    
    # Bar chart
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(sources)))
    bars = ax.bar(range(len(sources)), counts, color=colors, edgecolor='white', linewidth=1)
    
    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(range(len(sources)), cumulative, 'o-', color='#00ffff', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='white')
    ax2.tick_params(colors='white')
    ax2.spines['right'].set_color('white')
    ax2.set_ylim(0, 105)
    
    ax.set_xlabel('Source Region', fontsize=12, color='white')
    ax.set_ylabel('Particle Count in Gyre', fontsize=12, color='white')
    ax.set_title('Source Contributions to Gyre Accumulation (Pareto)', fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=9, color='white')
    ax.grid(True, alpha=0.3, color='#445566', axis='y')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    plt.tight_layout()
    return save_figure(fig, '14_pareto_contributions.png')

def figure_15_seasonal_comparison():
    """Small multiples: seasonal release months vs gyre entry"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    fig.suptitle('Seasonal Variation: Release Month vs Gyre Entry Probability',
                 fontsize=14, fontweight='bold', color='white', y=0.98)
    
    seasons = ['Winter (Jan)', 'Spring (Apr)', 'Summer (Jul)', 'Fall (Oct)']
    season_colors = ['#00bfff', '#66ff66', '#ff9900', '#ff6666']
    
    # Run mini simulations for each season (simplified)
    for idx, season in enumerate(seasons):
        ax = plt.subplot(2, 2, idx + 1, projection=ccrs.PlateCarree())
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#404040', edgecolor='none')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white')
        ax.set_facecolor('#0a1628')
        ax.set_title(season, fontsize=11, color='white', pad=5)
        
        # Generate seasonal particles (biased flow for demonstration)
        n_particles = 500
        lons = np.random.uniform(-80, -70, n_particles)
        lats = np.random.uniform(30, 40, n_particles)
        
        # Seasonal bias in final positions
        season_drift = [(0, -2), (5, 0), (3, 2), (-3, 0)][idx]
        lons_final = lons + np.random.normal(season_drift[0], 10, n_particles)
        lats_final = lats + np.random.normal(season_drift[1], 5, n_particles)
        
        ax.scatter(lons_final, lats_final, c=season_colors[idx], s=3, alpha=0.4,
                   transform=ccrs.PlateCarree(), zorder=3)
        
        # Gyre box
        gyre_box = mpatches.Rectangle((-70, 25), 40, 10, linewidth=1.5,
                                       edgecolor='cyan', facecolor='none',
                                       linestyle='--', transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(gyre_box)
    
    plt.tight_layout()
    fig.text(0.99, 0.01, WATERMARK, fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '15_seasonal_comparison.png')

def figure_16_residence_time_map(particles):
    """Map of average particle age (residence time) by region"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Average Particle Residence Time by Region")
    
    floating = [p for p in particles if p['status'] in ['floating', 'gyre_trapped']]
    
    # Create age field
    lon_bins = np.linspace(LON_MIN, LON_MAX, 80)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 50)
    
    age_sum = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    age_count = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    
    for p in floating:
        lon_idx = np.searchsorted(lon_bins, p['lon']) - 1
        lat_idx = np.searchsorted(lat_bins, p['lat']) - 1
        if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
            age_sum[lat_idx, lon_idx] += p['age_days']
            age_count[lat_idx, lon_idx] += 1
    
    avg_age = np.divide(age_sum, age_count, where=age_count > 0, out=np.zeros_like(age_sum))
    avg_age = gaussian_filter(avg_age, sigma=1.0)
    avg_age[age_count == 0] = np.nan
    
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], avg_age,
                     levels=15, cmap='viridis', transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Average Age (days)', shrink=0.7)
    
    return save_figure(fig, '16_residence_time_map.png')

def figure_17_velocity_magnitude_map():
    """Map of current velocity magnitude"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Ocean Current Velocity Magnitude")
    
    lon_grid = np.linspace(LON_MIN, LON_MAX, 200)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 140)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    u, v = idealized_current_field(lon_mesh, lat_mesh)
    
    speed = np.sqrt(u**2 + v**2)
    
    im = ax.contourf(lon_grid, lat_grid, speed, levels=20, cmap='magma',
                     transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Speed (deg/hr)', shrink=0.7)
    
    return save_figure(fig, '17_velocity_magnitude.png')

def figure_18_concentration_gradients(particles):
    """Concentration gradients showing convergence zones"""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='#0a1628')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axis(ax, "Particle Concentration Gradients (Convergence Zones)")
    
    floating = [p for p in particles if p['status'] in ['floating', 'gyre_trapped']]
    lons = np.array([p['lon'] for p in floating])
    lats = np.array([p['lat'] for p in floating])
    
    # High-resolution density
    lon_bins = np.linspace(LON_MIN, LON_MAX, 150)
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, 100)
    density, _, _ = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    density = gaussian_filter(density.T, sigma=2.0)
    
    # Compute gradients
    grad_y, grad_x = np.gradient(density)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    im = ax.contourf(lon_bins[:-1], lat_bins[:-1], gradient_mag,
                     levels=20, cmap='hot', transform=ccrs.PlateCarree(), zorder=3)
    plt.colorbar(im, ax=ax, label='Concentration Gradient', shrink=0.7)
    
    return save_figure(fig, '18_concentration_gradients.png')

def figure_19_transit_time_histograms(particles):
    """Transit time distributions to major sinks"""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, facecolor='#0a1628')
    fig.suptitle('Transit Time Distributions to Major Sinks', fontsize=14,
                 fontweight='bold', color='white', y=0.98)
    
    # Define sink regions
    sinks = {
        'Caribbean Beaches': {'lon': (-85, -60), 'lat': (10, 20)},
        'US East Coast': {'lon': (-80, -70), 'lat': (30, 42)},
        'European Coast': {'lon': (-10, 10), 'lat': (40, 60)},
        'African Coast': {'lon': (-20, -5), 'lat': (20, 35)},
    }
    
    for idx, (sink_name, region) in enumerate(sinks.items()):
        ax = axes.flatten()[idx]
        ax.set_facecolor('#0a1628')
        
        # Find beached particles in this region
        beached_here = [p for p in particles if p['status'] == 'beached' and p['beached_at']
                        and region['lon'][0] <= p['beached_at'][0] <= region['lon'][1]
                        and region['lat'][0] <= p['beached_at'][1] <= region['lat'][1]]
        
        if len(beached_here) > 5:
            ages = [p['age_days'] for p in beached_here]
            ax.hist(ages, bins=30, color='#ff9900', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.axvline(np.median(ages), color='cyan', linestyle='--', linewidth=2, label=f'Median: {np.median(ages):.0f}d')
        
        ax.set_title(sink_name, fontsize=11, color='white')
        ax.set_xlabel('Days to Beaching', fontsize=9, color='white')
        ax.set_ylabel('Count', fontsize=9, color='white')
        ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#445566')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    fig.text(0.99, 0.01, WATERMARK, fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    return save_figure(fig, '19_transit_time_histograms.png')

def figure_20_source_fate_matrix(particles):
    """Matrix heatmap showing source-fate relationships"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    sources = list(SOURCE_REGIONS.keys())
    fates = ['floating', 'beached', 'sunk', 'gyre_trapped']
    
    # Build matrix
    matrix = np.zeros((len(sources), len(fates)))
    
    for i, source in enumerate(sources):
        source_particles = [p for p in particles if p['source'] == source]
        total = len(source_particles)
        if total > 0:
            for j, fate in enumerate(fates):
                count = sum(1 for p in source_particles if p['status'] == fate)
                matrix[i, j] = count / total * 100
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(fates)))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels([f.replace('_', ' ').title() for f in fates], color='white', fontsize=10)
    ax.set_yticklabels([s.replace('_', ' ') for s in sources], color='white', fontsize=9)
    
    # Add text annotations
    for i in range(len(sources)):
        for j in range(len(fates)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                          color='white' if matrix[i, j] < 50 else 'black', fontsize=8)
    
    ax.set_title('Source-Fate Distribution Matrix (% of particles)', fontsize=14,
                 fontweight='bold', color='white', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, label='Percentage (%)')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.text(0.99, -0.08, WATERMARK, transform=ax.transAxes,
            fontsize=7, color='#666666', ha='right', va='bottom', style='italic')
    
    plt.tight_layout()
    return save_figure(fig, '20_source_fate_matrix.png')

# ============================================================================
# ANIMATION
# ============================================================================

def create_drift_animation(snapshots, output_path, duration_seconds=10, fps=30):
    """Create looping MP4 animation of particle drift"""
    print(f"\nCreating animation: {duration_seconds}s at {fps} fps...")
    
    frames = []
    n_frames = duration_seconds * fps
    snapshot_indices = np.linspace(0, len(snapshots) - 1, n_frames).astype(int)
    
    # Setup figure once
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#0a1628', dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    for frame_idx, snap_idx in enumerate(snapshot_indices):
        if frame_idx % 30 == 0:
            print(f"  Rendering frame {frame_idx}/{n_frames}...")
        
        ax.clear()
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#404040', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white', zorder=2)
        ax.set_facecolor('#0a1628')
        
        # Get snapshot
        snap = snapshots[snap_idx]
        day = snap['day']
        
        # Draw floating particles
        floating = [p for p in snap['particles'] if p['status'] in ['floating', 'gyre_trapped']]
        if floating:
            lons = [p['lon'] for p in floating]
            lats = [p['lat'] for p in floating]
            
            # Color by status
            colors = ['#ff9900' if p['status'] == 'floating' else '#ff3366' for p in floating]
            ax.scatter(lons, lats, c=colors, s=8, alpha=0.6,
                      transform=ccrs.PlateCarree(), zorder=4, edgecolors='none')
        
        # Add subtle current vectors every 10 degrees
        if frame_idx % 10 == 0:  # Only draw occasionally for performance
            lon_vec = np.arange(LON_MIN, LON_MAX, 15, dtype=float)
            lat_vec = np.arange(LAT_MIN, LAT_MAX, 15, dtype=float)
            lon_mesh, lat_mesh = np.meshgrid(lon_vec, lat_vec)
            u, v = idealized_current_field(lon_mesh, lat_mesh)
            ax.quiver(lon_mesh, lat_mesh, u, v, alpha=0.15, color='white',
                     scale=3, width=0.002, transform=ccrs.PlateCarree(), zorder=3)
        
        # Title with day counter
        ax.text(0.5, 0.97, f'North Atlantic Plastic Drift - Day {day:.0f}',
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                color='white', ha='center', va='top')
        
        # Watermark
        ax.text(0.02, 0.02, WATERMARK, transform=ax.transAxes,
                fontsize=9, color='#666666', ha='left', va='bottom', style='italic')
        
        # Stats overlay
        n_floating = len([p for p in snap['particles'] if p['status'] == 'floating'])
        n_beached = len([p for p in snap['particles'] if p['status'] == 'beached'])
        n_gyre = len([p for p in snap['particles'] if p['status'] == 'gyre_trapped'])
        
        stats_text = f'Floating: {n_floating:,}\nBeached: {n_beached:,}\nGyre: {n_gyre:,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, color='white', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Render to buffer
        fig.canvas.draw()
        # Use buffer_rgba() instead of deprecated tostring_rgb()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())
    
    plt.close(fig)
    
    # Save as MP4
    print(f"  Saving MP4 to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
    
    return output_path

def create_animation_teaser(snapshots, output_path, duration_seconds=20, fps=30):
    """Create shorter GIF teaser"""
    print(f"\nCreating GIF teaser: {duration_seconds}s at {fps} fps...")
    
    frames = []
    n_frames = duration_seconds * fps
    snapshot_indices = np.linspace(0, len(snapshots) - 1, n_frames).astype(int)
    
    fig = plt.figure(figsize=(12.8, 7.2), facecolor='#0a1628', dpi=80)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    for frame_idx, snap_idx in enumerate(snapshot_indices):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{n_frames}...")
        
        ax.clear()
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#404040', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='white', zorder=2)
        ax.set_facecolor('#0a1628')
        
        snap = snapshots[snap_idx]
        floating = [p for p in snap['particles'] if p['status'] in ['floating', 'gyre_trapped']]
        
        if floating:
            lons = [p['lon'] for p in floating]
            lats = [p['lat'] for p in floating]
            colors = ['#ff9900' if p['status'] == 'floating' else '#ff3366' for p in floating]
            ax.scatter(lons, lats, c=colors, s=5, alpha=0.5,
                      transform=ccrs.PlateCarree(), zorder=4, edgecolors='none')
        
        ax.text(0.5, 0.95, f'Day {snap["day"]:.0f}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='white', ha='center')
        ax.text(0.98, 0.02, WATERMARK, transform=ax.transAxes,
                fontsize=7, color='#666666', ha='right', va='bottom')
        
        fig.canvas.draw()
        # Use buffer_rgba() instead of deprecated tostring_rgb()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())
    
    plt.close(fig)
    
    print(f"  Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    return output_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_outputs():
    """Generate all figures and animations"""
    print("=" * 80)
    print("OCEANS-FOUR DRIFTCAST - SYNTHETIC DEMO GENERATION")
    print("=" * 80)
    
    # Run main simulation
    particles, snapshots = run_simulation(SIM_DAYS, save_snapshots=True)
    
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    
    # Generate all figures
    figure_files = []
    
    print("\n[1/20] Current field...")
    figure_files.append(figure_01_current_field())
    
    print("\n[2/20] Source map...")
    figure_files.append(figure_02_source_map(particles))
    
    print("\n[3/20] US dispersion (30 days)...")
    figure_files.append(figure_03_dispersion_heatmap_us(snapshots))
    
    print("\n[4/20] EU dispersion (30 days)...")
    figure_files.append(figure_04_dispersion_heatmap_eu(snapshots))
    
    print("\n[5/20] Gyre accumulation (180 days)...")
    figure_files.append(figure_05_gyre_accumulation(snapshots))
    
    print("\n[6/20] Fate timeseries...")
    figure_files.append(figure_06_fate_timeseries(snapshots))
    
    print("\n[7/20] Distance histogram...")
    figure_files.append(figure_07_distance_histogram(particles))
    
    print("\n[8/20] Gyre entry KDE...")
    figure_files.append(figure_08_gyre_entry_kde(particles))
    
    print("\n[9/20] Beaching hotspots...")
    figure_files.append(figure_09_beaching_hotspots(particles))
    
    print("\n[10/20] Sankey diagram...")
    figure_files.append(figure_10_sankey_diagram(particles))
    
    print("\n[11/20] Storm comparison...")
    # For storm comparison, use same particles (storm embedded in simulation)
    figure_files.append(figure_11_storm_comparison(particles, particles))
    
    print("\n[12/20] Trajectory spaghetti...")
    figure_files.append(figure_12_trajectory_spaghetti(snapshots))
    
    print("\n[13/20] Quiver + streamplot...")
    figure_files.append(figure_13_quiver_streamplot())
    
    print("\n[14/20] Pareto contributions...")
    figure_files.append(figure_14_pareto_contributions(particles))
    
    print("\n[15/20] Seasonal comparison...")
    figure_files.append(figure_15_seasonal_comparison())
    
    print("\n[16/20] Residence time map...")
    figure_files.append(figure_16_residence_time_map(particles))
    
    print("\n[17/20] Velocity magnitude...")
    figure_files.append(figure_17_velocity_magnitude_map())
    
    print("\n[18/20] Concentration gradients...")
    figure_files.append(figure_18_concentration_gradients(particles))
    
    print("\n[19/20] Transit time histograms...")
    figure_files.append(figure_19_transit_time_histograms(particles))
    
    print("\n[20/20] Source-fate matrix...")
    figure_files.append(figure_20_source_fate_matrix(particles))
    
    # Generate animations
    print("\n" + "=" * 80)
    print("GENERATING ANIMATIONS")
    print("=" * 80)
    
    anim_mp4 = os.path.join(ANIM_DIR, 'driftcast.mp4')
    anim_gif = os.path.join(ANIM_DIR, 'driftcast_teaser.gif')
    
    create_drift_animation(snapshots, anim_mp4, duration_seconds=10, fps=30)
    create_animation_teaser(snapshots, anim_gif, duration_seconds=20, fps=15)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nFinal Particle Counts by Fate:")
    print("-" * 40)
    for fate in ['floating', 'beached', 'sunk', 'gyre_trapped']:
        count = sum(1 for p in particles if p['status'] == fate)
        pct = count / len(particles) * 100
        print(f"  {fate.replace('_', ' ').title():20s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'Total':20s}: {len(particles):5d}")
    
    print("\nTop Beaching Regions:")
    print("-" * 40)
    beached = [p for p in particles if p['status'] == 'beached' and p['beached_at']]
    
    # Bin by longitude regions
    regions = {
        'Caribbean': (-85, -60, 10, 25),
        'US East Coast': (-80, -70, 30, 45),
        'US Gulf Coast': (-97, -80, 24, 31),
        'European Coast': (-10, 10, 40, 60),
        'NW African Coast': (-20, -5, 20, 36),
    }
    
    for region_name, (lon_min, lon_max, lat_min, lat_max) in regions.items():
        count = sum(1 for p in beached if p['beached_at']
                   and lon_min <= p['beached_at'][0] <= lon_max
                   and lat_min <= p['beached_at'][1] <= lat_max)
        if count > 0:
            pct = count / len(beached) * 100
            print(f"  {region_name:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\nGenerated Files:")
    print("-" * 40)
    print(f"\nFigures ({len(figure_files)}):")
    for f in sorted(figure_files):
        print(f"  {f}")
    
    print(f"\nAnimations:")
    print(f"  {anim_mp4}")
    print(f"  {anim_gif}")
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    generate_all_outputs()
