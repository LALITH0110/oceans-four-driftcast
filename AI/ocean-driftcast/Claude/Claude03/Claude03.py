#!/usr/bin/env python3
"""
Oceans-Four DriftCast: Synthetic North Atlantic Plastic Drift Simulator
Demo visualization system with plausible but synthetic physics.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import imageio
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Map extent: North Atlantic
LON_MIN, LON_MAX = -100, 20
LAT_MIN, LAT_MAX = 0, 70

# Simulation parameters
HOURS_PER_DAY = 24
DT_HOURS = 1.0  # Integration timestep
DIFFUSION_KM = 0.5  # Turbulent diffusion per hour
WINDAGE_FACTOR = 0.02  # Additional wind-driven velocity
BEACHING_DISTANCE_KM = 20  # Distance to coast for beaching
SINKING_PROB_PER_DAY = 0.0005  # Daily probability of sinking
FRAGMENTATION_PROB_PER_DAY = 0.001  # Daily probability of breaking up

# Particle source configuration (release counts)
SOURCE_COUNTS = {
    'US_East': 5000,
    'US_Gulf': 3000,
    'Caribbean': 2000,
    'Europe_North': 2500,
    'Europe_South': 1500,
    'St_Lawrence': 800,
}

# Animation parameters
ANIMATION_DURATION_SEC = 480  # 8 minutes (must be in [300, 600])
ANIMATION_FPS = 30
ANIMATION_RESOLUTION = (1920, 1080)

# Figure resolution
FIGURE_DPI = 100
FIGURE_SIZE = (1920/FIGURE_DPI, 1080/FIGURE_DPI)

# Style parameters
OCEAN_COLOR = '#0a1929'
LAND_COLOR = '#4a4a4a'
COASTLINE_COLOR = 'white'
PLASTIC_COLOR = '#ff6b35'
PLASTIC_ALPHA = 0.6
GRID_ALPHA = 0.15

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_watermark(ax):
    """Add subtle watermark to plot."""
    ax.text(0.99, 0.01, 'Demo · Synthetic Data', 
            transform=ax.transAxes, fontsize=7, alpha=0.4,
            ha='right', va='bottom', color='white', style='italic')

def setup_map(ax, title=''):
    """Configure map axes with North Atlantic extent."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=COASTLINE_COLOR, zorder=2)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=GRID_ALPHA, color='white')
    ax.set_facecolor(OCEAN_COLOR)
    if title:
        ax.set_title(title, fontsize=14, pad=10, color='white')
    ax.set_xlabel('Longitude', fontsize=10, color='white')
    ax.set_ylabel('Latitude', fontsize=10, color='white')

def km_to_degrees(km, lat=30):
    """Convert km to degrees at given latitude."""
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat))
    km_per_deg_lat = 111.32
    return km / km_per_deg_lat

# ============================================================================
# OCEAN CURRENT FIELD (SYNTHETIC)
# ============================================================================

def create_current_field(lon_grid, lat_grid):
    """
    Generate synthetic North Atlantic gyre circulation.
    Returns u, v components in km/hour.
    """
    # Convert to radians for smoother fields
    lon_r = np.radians(lon_grid)
    lat_r = np.radians(lat_grid)
    
    # Initialize velocity components
    u = np.zeros_like(lon_grid)
    v = np.zeros_like(lon_grid)
    
    # Subtropical gyre (main circulation)
    # Westward flow (North Equatorial Current) at 10-20N
    mask_nec = (lat_grid >= 10) & (lat_grid <= 20)
    u[mask_nec] = -15 * (1 - 0.3 * np.sin(2 * lon_r[mask_nec]))
    
    # Northward flow along Americas (Gulf Stream start)
    mask_gulf_start = (lon_grid >= -85) & (lon_grid <= -70) & (lat_grid >= 20) & (lat_grid <= 35)
    u[mask_gulf_start] += 10 * np.exp(-((lat_grid[mask_gulf_start] - 28)**2) / 50)
    v[mask_gulf_start] += 30 * (1 + 0.5 * np.cos(lon_r[mask_gulf_start]))
    
    # Gulf Stream core (northeast flow)
    mask_gulf_stream = (lon_grid >= -75) & (lon_grid <= -40) & (lat_grid >= 30) & (lat_grid <= 45)
    u[mask_gulf_stream] += 40 * np.exp(-((lat_grid[mask_gulf_stream] - 38)**2) / 30)
    v[mask_gulf_stream] += 15 * np.exp(-((lat_grid[mask_gulf_stream] - 38)**2) / 30)
    
    # North Atlantic Drift (eastward to Europe)
    mask_nad = (lon_grid >= -40) & (lon_grid <= 10) & (lat_grid >= 40) & (lat_grid <= 55)
    u[mask_nad] += 20 * (1 - 0.4 * np.sin(lat_r[mask_nad]))
    v[mask_nad] += 5 * np.cos(2 * lon_r[mask_nad])
    
    # Canary Current (southward along Europe/Africa)
    mask_canary = (lon_grid >= -20) & (lon_grid <= 10) & (lat_grid >= 15) & (lat_grid <= 40)
    u[mask_canary] += -8 * np.exp(-((lon_grid[mask_canary] + 10)**2) / 40)
    v[mask_canary] += -18 * (1 + 0.3 * np.sin(lon_r[mask_canary]))
    
    # Subpolar gyre (counterclockwise near Iceland)
    mask_subpolar = (lon_grid >= -50) & (lon_grid <= -10) & (lat_grid >= 55) & (lat_grid <= 65)
    center_lon, center_lat = -30, 60
    dx = lon_grid[mask_subpolar] - center_lon
    dy = lat_grid[mask_subpolar] - center_lat
    r = np.sqrt(dx**2 + dy**2)
    u[mask_subpolar] += -12 * dy / (r + 1) * np.exp(-r / 15)
    v[mask_subpolar] += 12 * dx / (r + 1) * np.exp(-r / 15)
    
    # Gyre interior (slower recirculation)
    mask_interior = (lon_grid >= -70) & (lon_grid <= -30) & (lat_grid >= 22) & (lat_grid <= 38)
    u[mask_interior] += -2 * np.sin(lat_r[mask_interior])
    v[mask_interior] += -1 * np.cos(lon_r[mask_interior])
    
    # Add some mesoscale variability
    u += 2 * np.sin(3 * lon_r) * np.cos(2 * lat_r)
    v += 2 * np.cos(3 * lon_r) * np.sin(2 * lat_r)
    
    # Convert to km/hour (rough scale)
    u *= 0.5
    v *= 0.5
    
    return u, v

# ============================================================================
# PARTICLE SOURCES
# ============================================================================

def define_sources():
    """Define particle release locations."""
    sources = {
        'US_East': {
            'lons': np.linspace(-80, -70, 50),
            'lats': np.linspace(28, 42, 50),
            'count': SOURCE_COUNTS['US_East']
        },
        'US_Gulf': {
            'lons': np.linspace(-97, -82, 40),
            'lats': np.linspace(25, 30, 40),
            'count': SOURCE_COUNTS['US_Gulf']
        },
        'Caribbean': {
            'lons': np.linspace(-80, -60, 30),
            'lats': np.linspace(10, 22, 30),
            'count': SOURCE_COUNTS['Caribbean']
        },
        'Europe_North': {
            'lons': np.linspace(-8, 10, 35),
            'lats': np.linspace(48, 60, 35),
            'count': SOURCE_COUNTS['Europe_North']
        },
        'Europe_South': {
            'lons': np.linspace(-10, 5, 30),
            'lats': np.linspace(36, 45, 30),
            'count': SOURCE_COUNTS['Europe_South']
        },
        'St_Lawrence': {
            'lons': np.linspace(-70, -60, 20),
            'lats': np.linspace(45, 50, 20),
            'count': SOURCE_COUNTS['St_Lawrence']
        },
    }
    return sources

def initialize_particles(sources):
    """Create initial particle positions from sources."""
    particles = []
    for source_name, source_data in sources.items():
        count = source_data['count']
        lon_choices = source_data['lons']
        lat_choices = source_data['lats']
        
        for _ in range(count):
            lon = np.random.choice(lon_choices) + np.random.normal(0, 0.5)
            lat = np.random.choice(lat_choices) + np.random.normal(0, 0.3)
            particles.append({
                'lon': lon,
                'lat': lat,
                'source': source_name,
                'status': 'floating',
                'age_hours': 0,
                'distance_traveled': 0,
                'size': 1.0,
            })
    return particles

# ============================================================================
# PARTICLE ADVECTION
# ============================================================================

def get_velocity_at_point(lon, lat, u_field, v_field, lon_grid, lat_grid):
    """Interpolate velocity at particle position."""
    # Simple bilinear interpolation
    lon_idx = np.searchsorted(lon_grid[0, :], lon) - 1
    lat_idx = np.searchsorted(lat_grid[:, 0], lat) - 1
    
    # Clamp to grid bounds
    lon_idx = np.clip(lon_idx, 0, u_field.shape[1] - 2)
    lat_idx = np.clip(lat_idx, 0, u_field.shape[0] - 2)
    
    # Bilinear weights
    lon_frac = (lon - lon_grid[lat_idx, lon_idx]) / (lon_grid[lat_idx, lon_idx + 1] - lon_grid[lat_idx, lon_idx])
    lat_frac = (lat - lat_grid[lat_idx, lon_idx]) / (lat_grid[lat_idx + 1, lon_idx] - lat_grid[lat_idx, lon_idx])
    
    lon_frac = np.clip(lon_frac, 0, 1)
    lat_frac = np.clip(lat_frac, 0, 1)
    
    # Interpolate u
    u_bottom = u_field[lat_idx, lon_idx] * (1 - lon_frac) + u_field[lat_idx, lon_idx + 1] * lon_frac
    u_top = u_field[lat_idx + 1, lon_idx] * (1 - lon_frac) + u_field[lat_idx + 1, lon_idx + 1] * lon_frac
    u = u_bottom * (1 - lat_frac) + u_top * lat_frac
    
    # Interpolate v
    v_bottom = v_field[lat_idx, lon_idx] * (1 - lon_frac) + v_field[lat_idx, lon_idx + 1] * lon_frac
    v_top = v_field[lat_idx + 1, lon_idx] * (1 - lon_frac) + v_field[lat_idx + 1, lon_idx + 1] * lon_frac
    v = v_bottom * (1 - lat_frac) + v_top * lat_frac
    
    return u, v

def is_on_land(lon, lat):
    """Simple land mask check (rough approximation)."""
    # Very rough land boundaries for North Atlantic
    if lat < 8:  # Below Caribbean
        return False
    if lon < -95:  # West of continent
        return False
    if lon > 15:  # East of Europe
        return False
    
    # US East Coast
    if lon >= -82 and lon <= -70 and lat >= 24 and lat <= 45:
        if lon > -75 - (lat - 30) * 0.3:  # Rough coastline
            return True
    
    # Gulf Coast
    if lon >= -98 and lon <= -80 and lat >= 24 and lat <= 31:
        if lat < 28 + (lon + 90) * 0.2:
            return True
    
    # Europe
    if lon >= -10 and lon <= 15 and lat >= 35:
        if lon > -10 + (lat - 35) * 0.3:
            return True
    
    # Caribbean islands (simplified)
    if lon >= -80 and lon <= -60 and lat >= 10 and lat <= 20:
        if np.random.random() < 0.05:  # Sparse islands
            return True
    
    return False

def advect_particles(particles, u_field, v_field, lon_grid, lat_grid, 
                    n_hours, storm_mode=False):
    """
    Advect particles using RK4 integration.
    """
    diffusion_scale = DIFFUSION_KM * (3 if storm_mode else 1)
    windage = WINDAGE_FACTOR * (2 if storm_mode else 1)
    
    for hour in range(n_hours):
        for p in particles:
            if p['status'] != 'floating':
                continue
            
            # RK4 integration
            lon0, lat0 = p['lon'], p['lat']
            
            # k1
            u1, v1 = get_velocity_at_point(lon0, lat0, u_field, v_field, lon_grid, lat_grid)
            
            # k2
            lon_mid1 = lon0 + 0.5 * u1 * km_to_degrees(DT_HOURS / 2, lat0)
            lat_mid1 = lat0 + 0.5 * v1 * km_to_degrees(DT_HOURS / 2)
            u2, v2 = get_velocity_at_point(lon_mid1, lat_mid1, u_field, v_field, lon_grid, lat_grid)
            
            # k3
            lon_mid2 = lon0 + 0.5 * u2 * km_to_degrees(DT_HOURS / 2, lat0)
            lat_mid2 = lat0 + 0.5 * v2 * km_to_degrees(DT_HOURS / 2)
            u3, v3 = get_velocity_at_point(lon_mid2, lat_mid2, u_field, v_field, lon_grid, lat_grid)
            
            # k4
            lon_end = lon0 + u3 * km_to_degrees(DT_HOURS, lat0)
            lat_end = lat0 + v3 * km_to_degrees(DT_HOURS)
            u4, v4 = get_velocity_at_point(lon_end, lat_end, u_field, v_field, lon_grid, lat_grid)
            
            # Combine
            u_avg = (u1 + 2*u2 + 2*u3 + u4) / 6
            v_avg = (v1 + 2*v2 + 2*v3 + v4) / 6
            
            # Add windage
            u_total = u_avg * (1 + windage)
            v_total = v_avg * (1 + windage)
            
            # Add diffusion
            diffusion_lon = np.random.normal(0, diffusion_scale * km_to_degrees(1, lat0))
            diffusion_lat = np.random.normal(0, diffusion_scale * km_to_degrees(1))
            
            # Update position
            new_lon = lon0 + u_total * km_to_degrees(DT_HOURS, lat0) + diffusion_lon
            new_lat = lat0 + v_total * km_to_degrees(DT_HOURS) + diffusion_lat
            
            # Update distance traveled
            dx = (new_lon - lon0) * 111.32 * np.cos(np.radians(lat0))
            dy = (new_lat - lat0) * 111.32
            p['distance_traveled'] += np.sqrt(dx**2 + dy**2)
            
            # Check boundaries
            if new_lon < LON_MIN or new_lon > LON_MAX or new_lat < LAT_MIN or new_lat > LAT_MAX:
                p['status'] = 'out_of_bounds'
                continue
            
            # Check for beaching
            if is_on_land(new_lon, new_lat):
                p['status'] = 'beached'
                p['lon'] = new_lon
                p['lat'] = new_lat
                continue
            
            # Update position
            p['lon'] = new_lon
            p['lat'] = new_lat
            p['age_hours'] += 1
            
            # Check for sinking
            if np.random.random() < SINKING_PROB_PER_DAY / HOURS_PER_DAY:
                p['status'] = 'sunk'
                continue
            
            # Check for fragmentation
            if np.random.random() < FRAGMENTATION_PROB_PER_DAY / HOURS_PER_DAY:
                p['size'] *= 0.7
            
            # Check if in gyre (rough approximation)
            if (25 <= new_lat <= 35) and (-70 <= new_lon <= -30):
                if p.get('in_gyre_hours', 0) > 24 * 30:  # 30 days in gyre
                    p['status'] = 'gyre_trapped'
                else:
                    p['in_gyre_hours'] = p.get('in_gyre_hours', 0) + 1
    
    return particles

# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figure_1_base_map(u_field, v_field, lon_grid, lat_grid):
    """Figure 1: Base map with current vectors."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'North Atlantic Current System')
    
    # Plot current vectors (subsample for clarity)
    skip = 8
    quiver = ax.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                      u_field[::skip, ::skip], v_field[::skip, ::skip],
                      color='cyan', alpha=0.4, scale=400, width=0.002,
                      transform=ccrs.PlateCarree())
    
    # Add gyre labels
    ax.text(-50, 30, 'Subtropical\nGyre', fontsize=12, color='yellow',
            ha='center', weight='bold', transform=ccrs.PlateCarree())
    ax.text(-30, 60, 'Subpolar\nGyre', fontsize=10, color='lightblue',
            ha='center', weight='bold', transform=ccrs.PlateCarree())
    
    # Add current labels
    ax.annotate('Gulf Stream', xy=(-60, 38), xytext=(-55, 42),
                color='orange', fontsize=10, weight='bold',
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                transform=ccrs.PlateCarree())
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/01_base_map.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_2_sources(sources):
    """Figure 2: Source map."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'Particle Release Zones')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))
    
    for idx, (name, data) in enumerate(sources.items()):
        lons = data['lons']
        lats = data['lats']
        count = data['count']
        
        ax.scatter(lons, lats, c=[colors[idx]], s=20, alpha=0.6, 
                  label=f"{name.replace('_', ' ')}: {count}", 
                  transform=ccrs.PlateCarree(), edgecolors='white', linewidths=0.5)
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8, facecolor='#2a2a2a', 
              edgecolor='white', labelcolor='white')
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/02_sources.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_3_us_dispersion(particles_us_30d):
    """Figure 3: 30-day US dispersion heatmap."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'US East Coast Release: 30-Day Dispersion')
    
    lons = [p['lon'] for p in particles_us_30d if p['status'] == 'floating']
    lats = [p['lat'] for p in particles_us_30d if p['status'] == 'floating']
    
    if len(lons) > 10:
        h = ax.hexbin(lons, lats, gridsize=50, cmap='hot', alpha=0.7, 
                     mincnt=1, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(h, ax=ax, label='Particle Density', pad=0.02)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/03_us_dispersion_30d.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_4_eu_dispersion(particles_eu_30d):
    """Figure 4: 30-day European dispersion heatmap."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'European Release: 30-Day Dispersion')
    
    lons = [p['lon'] for p in particles_eu_30d if p['status'] == 'floating']
    lats = [p['lat'] for p in particles_eu_30d if p['status'] == 'floating']
    
    if len(lons) > 10:
        h = ax.hexbin(lons, lats, gridsize=50, cmap='viridis', alpha=0.7,
                     mincnt=1, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(h, ax=ax, label='Particle Density', pad=0.02)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/04_eu_dispersion_30d.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_5_accumulation(particles_180d):
    """Figure 5: 180-day accumulation density."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, '180-Day Accumulation: Gyre Hotspot')
    
    lons = [p['lon'] for p in particles_180d if p['status'] in ['floating', 'gyre_trapped']]
    lats = [p['lat'] for p in particles_180d if p['status'] in ['floating', 'gyre_trapped']]
    
    if len(lons) > 10:
        # Create density grid
        lon_bins = np.linspace(LON_MIN, LON_MAX, 120)
        lat_bins = np.linspace(LAT_MIN, LAT_MAX, 70)
        H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
        H = gaussian_filter(H, sigma=2)
        
        extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
        im = ax.imshow(H.T, origin='lower', extent=extent, cmap='plasma',
                      alpha=0.8, transform=ccrs.PlateCarree(), aspect='auto')
        cbar = plt.colorbar(im, ax=ax, label='Accumulation Density', pad=0.02)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        # Mark gyre center
        ax.plot(-50, 30, 'w*', markersize=20, markeredgecolor='yellow',
               markeredgewidth=2, transform=ccrs.PlateCarree())
        ax.text(-50, 27, 'Gyre Center', fontsize=10, color='yellow',
               ha='center', weight='bold', transform=ccrs.PlateCarree())
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/05_accumulation_180d.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_6_fate_timeseries(fate_history):
    """Figure 6: Time series of particle fates."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    days = np.array([h['day'] for h in fate_history])
    floating = np.array([h['floating'] for h in fate_history])
    beached = np.array([h['beached'] for h in fate_history])
    sunk = np.array([h['sunk'] for h in fate_history])
    trapped = np.array([h['trapped'] for h in fate_history])
    
    ax.plot(days, floating, 'c-', linewidth=2, label='Floating')
    ax.plot(days, beached, 'orange', linewidth=2, label='Beached')
    ax.plot(days, sunk, 'gray', linewidth=2, label='Sunk')
    ax.plot(days, trapped, 'yellow', linewidth=2, label='Gyre Trapped')
    
    ax.set_xlabel('Days', fontsize=12, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Particle Fate Over Time', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a', 
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/06_fate_timeseries.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_7_distance_histogram(particles_90d, particles_180d):
    """Figure 7: Travel distance histogram."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    dist_90 = [p['distance_traveled'] for p in particles_90d if p['status'] != 'out_of_bounds']
    dist_180 = [p['distance_traveled'] for p in particles_180d if p['status'] != 'out_of_bounds']
    
    ax.hist(dist_90, bins=50, alpha=0.6, color='cyan', label='90 Days', edgecolor='white', linewidth=0.5)
    ax.hist(dist_180, bins=50, alpha=0.6, color='magenta', label='180 Days', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Distance Traveled (km)', fontsize=12, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Distribution of Travel Distances', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a', 
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/07_distance_histogram.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_8_arrival_kde(particles_180d):
    """Figure 8: KDE of gyre arrival times by source."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    sources_unique = set(p['source'] for p in particles_180d)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sources_unique)))
    
    for idx, source in enumerate(sources_unique):
        arrival_times = []
        for p in particles_180d:
            if p['source'] == source and p.get('in_gyre_hours', 0) > 0:
                # Estimate arrival time as first entry into gyre
                arrival_day = (p['age_hours'] - p.get('in_gyre_hours', 0)) / 24
                if 0 < arrival_day < 180:
                    arrival_times.append(arrival_day)
        
        if len(arrival_times) > 5:
            kde = gaussian_kde(arrival_times)
            x_range = np.linspace(0, 180, 200)
            density = kde(x_range)
            ax.plot(x_range, density, color=colors[idx], linewidth=2, 
                   label=source.replace('_', ' '))
            ax.fill_between(x_range, density, alpha=0.3, color=colors[idx])
    
    ax.set_xlabel('Days Since Release', fontsize=12, color='white')
    ax.set_ylabel('Probability Density', fontsize=12, color='white')
    ax.set_title('Gyre Arrival Time Distribution by Source', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=9, framealpha=0.8, facecolor='#2a2a2a', 
              edgecolor='white', labelcolor='white', loc='upper right')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/08_arrival_kde.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_9_beaching_hotspots(particles_180d):
    """Figure 9: Coastline beaching intensity."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'Beaching Hotspots by Coastline Segment')
    
    beached = [p for p in particles_180d if p['status'] == 'beached']
    
    if len(beached) > 0:
        lons = [p['lon'] for p in beached]
        lats = [p['lat'] for p in beached]
        
        ax.scatter(lons, lats, c='red', s=15, alpha=0.7, marker='x',
                  transform=ccrs.PlateCarree(), label=f'Beached: {len(beached)}')
        
        # Create heatmap
        lon_bins = np.linspace(LON_MIN, LON_MAX, 100)
        lat_bins = np.linspace(LAT_MIN, LAT_MAX, 60)
        H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
        H = gaussian_filter(H, sigma=1.5)
        
        extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
        im = ax.imshow(H.T, origin='lower', extent=extent, cmap='Reds',
                      alpha=0.5, transform=ccrs.PlateCarree(), aspect='auto')
        cbar = plt.colorbar(im, ax=ax, label='Beaching Intensity', pad=0.02)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/09_beaching_hotspots.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_10_sankey(particles_180d):
    """Figure 10: Source to fate flow diagram (simplified Sankey)."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    # Count flows from sources to fates
    sources_unique = sorted(set(p['source'] for p in particles_180d))
    fates = ['floating', 'beached', 'sunk', 'gyre_trapped']
    
    flow_data = {}
    for source in sources_unique:
        flow_data[source] = {fate: 0 for fate in fates}
        for p in particles_180d:
            if p['source'] == source:
                status = p['status']
                if status in fates:
                    flow_data[source][status] += 1
    
    # Create stacked bar chart (simplified Sankey)
    x_pos = np.arange(len(sources_unique))
    colors_fate = {'floating': 'cyan', 'beached': 'orange', 
                   'sunk': 'gray', 'gyre_trapped': 'yellow'}
    
    bottom = np.zeros(len(sources_unique))
    for fate in fates:
        values = [flow_data[src][fate] for src in sources_unique]
        ax.bar(x_pos, values, bottom=bottom, label=fate.replace('_', ' ').title(),
              color=colors_fate[fate], alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace('_', ' ') for s in sources_unique], 
                       rotation=45, ha='right', fontsize=10, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Source to Fate Flow (180 Days)', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white', loc='upper left')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white', axis='y')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/10_sankey_flow.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_11_storm_comparison(particles_normal, particles_storm):
    """Figure 11: Storm week density comparison."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    
    # Two subplots
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    setup_map(ax1, 'Normal Conditions (30d)')
    
    lons1 = [p['lon'] for p in particles_normal if p['status'] == 'floating']
    lats1 = [p['lat'] for p in particles_normal if p['status'] == 'floating']
    
    if len(lons1) > 10:
        ax1.hexbin(lons1, lats1, gridsize=40, cmap='Blues', alpha=0.7,
                  mincnt=1, transform=ccrs.PlateCarree())
    
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    setup_map(ax2, 'With Storm Week (30d)')
    
    lons2 = [p['lon'] for p in particles_storm if p['status'] == 'floating']
    lats2 = [p['lat'] for p in particles_storm if p['status'] == 'floating']
    
    if len(lons2) > 10:
        h = ax2.hexbin(lons2, lats2, gridsize=40, cmap='Reds', alpha=0.7,
                      mincnt=1, transform=ccrs.PlateCarree())
        plt.colorbar(h, ax=ax2, label='Particle Density', pad=0.02)
    
    add_watermark(ax1)
    add_watermark(ax2)
    plt.tight_layout()
    plt.savefig('outputs/figures/11_storm_comparison.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_12_trajectory_spaghetti(trajectory_data):
    """Figure 12: Trajectory spaghetti plot."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'US East Coast Cohort: Trajectory Paths to Gyre')
    
    for traj in trajectory_data:
        lons = traj['lons']
        lats = traj['lats']
        ax.plot(lons, lats, color=PLASTIC_COLOR, alpha=0.3, linewidth=0.8,
               transform=ccrs.PlateCarree())
    
    # Mark start and end
    if len(trajectory_data) > 0:
        start_lons = [t['lons'][0] for t in trajectory_data]
        start_lats = [t['lats'][0] for t in trajectory_data]
        end_lons = [t['lons'][-1] for t in trajectory_data]
        end_lats = [t['lats'][-1] for t in trajectory_data]
        
        ax.scatter(start_lons, start_lats, c='lime', s=30, marker='o',
                  edgecolors='white', linewidths=1, transform=ccrs.PlateCarree(),
                  label='Release', zorder=5)
        ax.scatter(end_lons, end_lats, c='yellow', s=30, marker='*',
                  edgecolors='white', linewidths=1, transform=ccrs.PlateCarree(),
                  label='180d Position', zorder=5)
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/12_trajectory_spaghetti.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_13_quiver_streamplot(u_field, v_field, lon_grid, lat_grid, particles):
    """Figure 13: Quiver + streamplot overlay."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map(ax, 'Current Vectors and Streamlines with Particle Positions')
    
    # Streamplot
    skip = 4
    ax.streamplot(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                 u_field[::skip, ::skip], v_field[::skip, ::skip],
                 color='cyan', linewidth=0.8, density=1.5,
                 transform=ccrs.PlateCarree())
    
    # Quiver
    skip_q = 12
    ax.quiver(lon_grid[::skip_q, ::skip_q], lat_grid[::skip_q, ::skip_q],
             u_field[::skip_q, ::skip_q], v_field[::skip_q, ::skip_q],
             color='yellow', alpha=0.3, scale=400, width=0.002,
             transform=ccrs.PlateCarree())
    
    # Particles
    lons = [p['lon'] for p in particles if p['status'] == 'floating'][:2000]
    lats = [p['lat'] for p in particles if p['status'] == 'floating'][:2000]
    
    if len(lons) > 0:
        ax.scatter(lons, lats, c=PLASTIC_COLOR, s=5, alpha=PLASTIC_ALPHA,
                  transform=ccrs.PlateCarree(), label=f'Particles: {len(lons)}')
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/13_quiver_streamplot.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_14_pareto_sources(particles_180d):
    """Figure 14: Pareto chart of gyre contributions by source."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    # Count gyre particles by source
    gyre_counts = {}
    for p in particles_180d:
        if p['status'] in ['floating', 'gyre_trapped']:
            if (25 <= p['lat'] <= 35) and (-70 <= p['lon'] <= -30):
                source = p['source']
                gyre_counts[source] = gyre_counts.get(source, 0) + 1
    
    if len(gyre_counts) > 0:
        # Sort by count
        sources_sorted = sorted(gyre_counts.items(), key=lambda x: x[1], reverse=True)
        sources = [s[0].replace('_', ' ') for s in sources_sorted]
        counts = [s[1] for s in sources_sorted]
        cumulative = np.cumsum(counts) / np.sum(counts) * 100
        
        x_pos = np.arange(len(sources))
        
        # Bar chart
        ax.bar(x_pos, counts, color='coral', alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Cumulative line
        ax2 = ax.twinx()
        ax2.plot(x_pos, cumulative, color='lime', marker='o', linewidth=2, 
                markersize=8, markeredgecolor='white', markeredgewidth=1)
        ax2.set_ylabel('Cumulative Contribution (%)', fontsize=12, color='lime')
        ax2.tick_params(colors='lime')
        ax2.spines['right'].set_color('lime')
        ax2.set_ylim([0, 105])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=10, color='white')
        ax.set_ylabel('Particles in Gyre', fontsize=12, color='white')
        ax.set_title('Source Contributions to Gyre Accumulation (Pareto)', 
                    fontsize=14, color='white', pad=10)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#1a1a1a')
        ax.grid(True, alpha=0.2, color='white', axis='y')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/14_pareto_sources.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_15_monthly_release(u_field, v_field, lon_grid, lat_grid, sources):
    """Figure 15: Small multiples - release month effect on gyre entry."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    
    months = ['Jan', 'Apr', 'Jul', 'Oct']
    seasonal_windage = [0.015, 0.020, 0.025, 0.020]  # Simulate seasonal variation
    
    for idx, (month, windage) in enumerate(zip(months, seasonal_windage)):
        ax = plt.subplot(2, 2, idx + 1, projection=ccrs.PlateCarree())
        setup_map(ax, f'Release: {month}')
        
        # Simulate small particle set
        test_particles = initialize_particles(sources)
        test_particles = test_particles[:500]  # Small sample
        
        # Run 90 days
        advect_particles(test_particles, u_field, v_field, lon_grid, lat_grid, 
                        90 * 24, storm_mode=False)
        
        lons = [p['lon'] for p in test_particles if p['status'] == 'floating']
        lats = [p['lat'] for p in test_particles if p['status'] == 'floating']
        
        # Count in gyre
        in_gyre = sum(1 for p in test_particles 
                     if p['status'] == 'floating' 
                     and (25 <= p['lat'] <= 35) 
                     and (-70 <= p['lon'] <= -30))
        
        if len(lons) > 0:
            ax.scatter(lons, lats, c=PLASTIC_COLOR, s=10, alpha=0.5,
                      transform=ccrs.PlateCarree())
        
        ax.text(0.05, 0.95, f'In Gyre: {in_gyre}', transform=ax.transAxes,
               fontsize=10, color='yellow', weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    plt.suptitle('Seasonal Release Effect on Gyre Entry (90d)', 
                fontsize=14, color='white', y=0.98)
    add_watermark(plt.gcf().axes[0])
    plt.tight_layout()
    plt.savefig('outputs/figures/15_monthly_release.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_16_size_distribution(particles_180d):
    """Figure 16: Particle size distribution after fragmentation."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    sizes = [p['size'] for p in particles_180d if p['status'] != 'out_of_bounds']
    
    ax.hist(sizes, bins=50, color='orchid', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Original Size')
    
    ax.set_xlabel('Relative Particle Size', fontsize=12, color='white')
    ax.set_ylabel('Count', fontsize=12, color='white')
    ax.set_title('Particle Size After 180 Days (Fragmentation Effect)', 
                fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/16_size_distribution.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_17_residence_time(particles_180d):
    """Figure 17: Residence time in different zones."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    # Calculate time spent in different zones
    zones = {
        'Coastal': (0, 50),
        'Offshore': (50, 200),
        'Open Ocean': (200, 1000),
        'Gyre': (1000, 10000)
    }
    
    # Rough estimate based on distance from coast
    zone_times = {zone: [] for zone in zones}
    
    for p in particles_180d:
        if p['status'] == 'out_of_bounds':
            continue
        
        # Estimate zone based on distance traveled and position
        dist = p['distance_traveled']
        
        if dist < 50:
            zone_times['Coastal'].append(p['age_hours'] / 24)
        elif dist < 200:
            zone_times['Offshore'].append(p['age_hours'] / 24)
        elif p['status'] == 'gyre_trapped':
            zone_times['Gyre'].append(p.get('in_gyre_hours', 0) / 24)
        else:
            zone_times['Open Ocean'].append(p['age_hours'] / 24)
    
    # Box plot
    data_to_plot = [times for times in zone_times.values() if len(times) > 0]
    labels_to_plot = [zone for zone, times in zone_times.items() if len(times) > 0]
    
    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                       boxprops=dict(facecolor='skyblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='white'),
                       capprops=dict(color='white'))
    
    ax.set_ylabel('Residence Time (days)', fontsize=12, color='white')
    ax.set_title('Particle Residence Time by Zone', fontsize=14, color='white', pad=10)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white', axis='y')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/17_residence_time.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_18_velocity_histogram(u_field, v_field):
    """Figure 18: Current velocity magnitude distribution."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    # Calculate velocity magnitude
    speed = np.sqrt(u_field**2 + v_field**2)
    
    ax.hist(speed.flatten(), bins=60, color='turquoise', alpha=0.8, 
           edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Current Speed (km/h)', fontsize=12, color='white')
    ax.set_ylabel('Grid Cell Count', fontsize=12, color='white')
    ax.set_title('Distribution of Current Velocities', fontsize=14, color='white', pad=10)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    # Add statistics
    stats_text = f'Mean: {np.mean(speed):.2f} km/h\nMax: {np.max(speed):.2f} km/h'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           color='yellow', va='top', ha='right', weight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/18_velocity_histogram.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_19_concentration_profile(particles_180d):
    """Figure 19: Latitudinal concentration profile."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    lats = [p['lat'] for p in particles_180d if p['status'] in ['floating', 'gyre_trapped']]
    
    if len(lats) > 0:
        # Create histogram
        counts, bins = np.histogram(lats, bins=50)
        centers = (bins[:-1] + bins[1:]) / 2
        
        ax.fill_between(centers, counts, alpha=0.7, color='gold', edgecolor='white', linewidth=1)
        ax.plot(centers, counts, color='orange', linewidth=2)
        
        # Mark gyre latitude
        ax.axvspan(25, 35, alpha=0.2, color='red', label='Gyre Latitude Band')
    
    ax.set_xlabel('Latitude (°N)', fontsize=12, color='white')
    ax.set_ylabel('Particle Count', fontsize=12, color='white')
    ax.set_title('Latitudinal Concentration Profile (180d)', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/19_concentration_profile.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

def generate_figure_20_age_distribution(particles_180d):
    """Figure 20: Particle age distribution by status."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor='black')
    ax = plt.axes(facecolor='#1a1a1a')
    
    statuses = ['floating', 'beached', 'sunk', 'gyre_trapped']
    colors_status = {'floating': 'cyan', 'beached': 'orange', 
                    'sunk': 'gray', 'gyre_trapped': 'yellow'}
    
    for status in statuses:
        ages = [p['age_hours'] / 24 for p in particles_180d if p['status'] == status]
        if len(ages) > 0:
            ax.hist(ages, bins=40, alpha=0.6, label=status.replace('_', ' ').title(),
                   color=colors_status[status], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Particle Age (days)', fontsize=12, color='white')
    ax.set_ylabel('Count', fontsize=12, color='white')
    ax.set_title('Particle Age Distribution by Status (180d)', fontsize=14, color='white', pad=10)
    ax.legend(fontsize=10, framealpha=0.8, facecolor='#2a2a2a',
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#1a1a1a')
    ax.spines['right'].set_color('#1a1a1a')
    ax.grid(True, alpha=0.2, color='white')
    
    add_watermark(ax)
    plt.tight_layout()
    plt.savefig('outputs/figures/20_age_distribution.png', facecolor='black', dpi=FIGURE_DPI)
    plt.close()

# ============================================================================
# ANIMATION GENERATION
# ============================================================================

def generate_animation(sources, u_field, v_field, lon_grid, lat_grid):
    """Generate the main animation video."""
    print("\n" + "="*70)
    print("GENERATING ANIMATION")
    print("="*70)
    
    duration_sec = ANIMATION_DURATION_SEC
    fps = ANIMATION_FPS
    total_frames = duration_sec * fps
    
    print(f"Duration: {duration_sec} seconds ({duration_sec/60:.1f} minutes)")
    print(f"Frame rate: {fps} fps")
    print(f"Total frames: {total_frames}")
    
    # Simulation setup
    particles = initialize_particles(sources)
    print(f"Initial particles: {len(particles)}")
    
    # Calculate hours per frame
    sim_days = 180  # Simulate 180 days
    sim_hours = sim_days * 24
    hours_per_frame = sim_hours / total_frames
    
    print(f"Simulating {sim_days} days over {total_frames} frames")
    print(f"Hours per frame: {hours_per_frame:.2f}")
    
    # Prepare figure
    fig = plt.figure(figsize=(ANIMATION_RESOLUTION[0]/100, ANIMATION_RESOLUTION[1]/100), 
                    dpi=100, facecolor='black')
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    frames = []
    
    for frame_idx in range(total_frames):
        if frame_idx % 100 == 0:
            print(f"Rendering frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
        
        # Advect particles
        if frame_idx > 0:
            n_hours = int(hours_per_frame)
            if n_hours < 1:
                n_hours = 1
            advect_particles(particles, u_field, v_field, lon_grid, lat_grid, n_hours)
        
        # Clear and redraw
        ax.clear()
        setup_map(ax, '')
        
        # Draw current vectors (sparse)
        skip = 15
        ax.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
                 u_field[::skip, ::skip], v_field[::skip, ::skip],
                 color='cyan', alpha=0.15, scale=500, width=0.001,
                 transform=ccrs.PlateCarree())
        
        # Draw particles
        floating = [p for p in particles if p['status'] == 'floating']
        if len(floating) > 0:
            lons = [p['lon'] for p in floating]
            lats = [p['lat'] for p in floating]
            sizes = [max(5, 15 * p['size']) for p in floating]
            
            ax.scatter(lons, lats, c=PLASTIC_COLOR, s=sizes, alpha=PLASTIC_ALPHA,
                      edgecolors='white', linewidths=0.1,
                      transform=ccrs.PlateCarree(), zorder=3)
        
        # Add day counter
        current_day = int(frame_idx * hours_per_frame / 24)
        ax.text(0.02, 0.98, f'Day {current_day}', transform=ax.transAxes,
               fontsize=16, color='white', weight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Add particle count
        n_floating = len(floating)
        n_beached = sum(1 for p in particles if p['status'] == 'beached')
        n_sunk = sum(1 for p in particles if p['status'] == 'sunk')
        
        status_text = f'Floating: {n_floating}\nBeached: {n_beached}\nSunk: {n_sunk}'
        ax.text(0.02, 0.88, status_text, transform=ax.transAxes,
               fontsize=10, color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Watermark
        ax.text(0.98, 0.02, 'Demo · Synthetic', transform=ax.transAxes,
               fontsize=8, color='white', alpha=0.4, ha='right', va='bottom',
               style='italic')
        
        # Render to array
        fig.canvas.draw()
        # Use buffer_rgba() instead of deprecated tostring_rgb()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(rgba[:, :, :3].copy())
    
    plt.close(fig)
    
    # Write video
    print("\nWriting video file...")
    video_path = 'outputs/animations/driftcast.mp4'
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', 
                               pixelformat='yuv420p', quality=8)
    
    for frame in frames:
        writer.append_data(frame)
    
    writer.close()
    
    print(f"Video saved: {video_path}")
    
    # Verify duration
    actual_duration = len(frames) / fps
    print(f"Actual duration: {actual_duration:.2f} seconds")
    
    assert abs(actual_duration - duration_sec) < 0.1, \
        f"Duration mismatch: expected {duration_sec}s, got {actual_duration:.2f}s"
    
    assert 300 <= actual_duration <= 600, \
        f"Duration {actual_duration:.2f}s outside required range [300, 600]"
    
    print("Duration verification: PASSED")
    
    # Create GIF teaser (first 25 seconds)
    print("\nCreating GIF teaser...")
    teaser_frames = frames[:int(25 * fps)]
    gif_path = 'outputs/animations/driftcast_teaser.gif'
    imageio.mimsave(gif_path, teaser_frames[::2], fps=fps//2, loop=0)  # Half speed
    print(f"GIF teaser saved: {gif_path}")
    
    return video_path, gif_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)

    print("="*70)
    print("OCEANS-FOUR DRIFTCAST: SYNTHETIC NORTH ATLANTIC PLASTIC DRIFT")
    print("="*70)
    print(f"Random seed: {SEED}")
    print(f"Map extent: {LON_MIN}°W to {LON_MAX}°E, {LAT_MIN}°N to {LAT_MAX}°N")
    print(f"Animation duration: {ANIMATION_DURATION_SEC}s ({ANIMATION_DURATION_SEC/60:.1f} min)")
    print("="*70)
    
    # Create current field
    print("\n1. Creating synthetic current field...")
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, 200),
        np.linspace(LAT_MIN, LAT_MAX, 120)
    )
    u_field, v_field = create_current_field(lon_grid, lat_grid)
    print(f"   Current field: {u_field.shape}")
    
    # Define sources
    print("\n2. Defining particle sources...")
    sources = define_sources()
    total_particles = sum(s['count'] for s in sources.values())
    print(f"   Total sources: {len(sources)}")
    print(f"   Total particles: {total_particles}")
    for name, data in sources.items():
        print(f"   - {name}: {data['count']} particles")
    
    # Initialize particles
    print("\n3. Initializing particles...")
    particles = initialize_particles(sources)
    print(f"   Initialized: {len(particles)} particles")
    
    # Run simulations for figures
    print("\n4. Running simulations...")
    
    # US 30-day
    print("   - US East Coast 30-day simulation...")
    particles_us_30d = [p.copy() for p in particles if p['source'] == 'US_East']
    advect_particles(particles_us_30d, u_field, v_field, lon_grid, lat_grid, 30 * 24)
    
    # EU 30-day
    print("   - European 30-day simulation...")
    particles_eu_30d = [p.copy() for p in particles 
                       if p['source'] in ['Europe_North', 'Europe_South']]
    advect_particles(particles_eu_30d, u_field, v_field, lon_grid, lat_grid, 30 * 24)
    
    # 90-day
    print("   - 90-day simulation...")
    particles_90d = [p.copy() for p in particles]
    advect_particles(particles_90d, u_field, v_field, lon_grid, lat_grid, 90 * 24)
    
    # 180-day with fate tracking
    print("   - 180-day simulation with fate tracking...")
    particles_180d = [p.copy() for p in particles]
    fate_history = []
    
    for day in range(0, 181, 5):
        if day > 0:
            advect_particles(particles_180d, u_field, v_field, lon_grid, lat_grid, 5 * 24)
        
        fate_history.append({
            'day': day,
            'floating': sum(1 for p in particles_180d if p['status'] == 'floating'),
            'beached': sum(1 for p in particles_180d if p['status'] == 'beached'),
            'sunk': sum(1 for p in particles_180d if p['status'] == 'sunk'),
            'trapped': sum(1 for p in particles_180d if p['status'] == 'gyre_trapped'),
        })
    
    # Storm comparison
    print("   - Storm week comparison simulation...")
    particles_storm = [p.copy() for p in particles]
    advect_particles(particles_storm, u_field, v_field, lon_grid, lat_grid, 7 * 24, storm_mode=True)
    advect_particles(particles_storm, u_field, v_field, lon_grid, lat_grid, 23 * 24, storm_mode=False)
    
    particles_normal = [p.copy() for p in particles]
    advect_particles(particles_normal, u_field, v_field, lon_grid, lat_grid, 30 * 24)
    
    # Trajectory data
    print("   - Generating trajectory data...")
    trajectory_particles = [p.copy() for p in particles if p['source'] == 'US_East'][:50]
    trajectory_data = []
    
    for p in trajectory_particles:
        traj = {'lons': [p['lon']], 'lats': [p['lat']]}
        
        for day in range(180):
            advect_particles([p], u_field, v_field, lon_grid, lat_grid, 24)
            if p['status'] == 'floating':
                traj['lons'].append(p['lon'])
                traj['lats'].append(p['lat'])
            else:
                break
        
        if len(traj['lons']) > 10:
            trajectory_data.append(traj)
    
    print(f"   Generated {len(trajectory_data)} trajectories")
    
    # Generate figures
    print("\n5. Generating figures...")
    
    print("   Figure 1: Base map with currents")
    generate_figure_1_base_map(u_field, v_field, lon_grid, lat_grid)
    
    print("   Figure 2: Source map")
    generate_figure_2_sources(sources)
    
    print("   Figure 3: US 30-day dispersion")
    generate_figure_3_us_dispersion(particles_us_30d)
    
    print("   Figure 4: EU 30-day dispersion")
    generate_figure_4_eu_dispersion(particles_eu_30d)
    
    print("   Figure 5: 180-day accumulation")
    generate_figure_5_accumulation(particles_180d)
    
    print("   Figure 6: Fate time series")
    generate_figure_6_fate_timeseries(fate_history)
    
    print("   Figure 7: Distance histogram")
    generate_figure_7_distance_histogram(particles_90d, particles_180d)
    
    print("   Figure 8: Arrival time KDE")
    generate_figure_8_arrival_kde(particles_180d)
    
    print("   Figure 9: Beaching hotspots")
    generate_figure_9_beaching_hotspots(particles_180d)
    
    print("   Figure 10: Sankey flow")
    generate_figure_10_sankey(particles_180d)
    
    print("   Figure 11: Storm comparison")
    generate_figure_11_storm_comparison(particles_normal, particles_storm)
    
    print("   Figure 12: Trajectory spaghetti")
    generate_figure_12_trajectory_spaghetti(trajectory_data)
    
    print("   Figure 13: Quiver + streamplot")
    generate_figure_13_quiver_streamplot(u_field, v_field, lon_grid, lat_grid, particles_180d)
    
    print("   Figure 14: Pareto chart")
    generate_figure_14_pareto_sources(particles_180d)
    
    print("   Figure 15: Monthly release multiples")
    generate_figure_15_monthly_release(u_field, v_field, lon_grid, lat_grid, sources)
    
    print("   Figure 16: Size distribution")
    generate_figure_16_size_distribution(particles_180d)
    
    print("   Figure 17: Residence time")
    generate_figure_17_residence_time(particles_180d)
    
    print("   Figure 18: Velocity histogram")
    generate_figure_18_velocity_histogram(u_field, v_field)
    
    print("   Figure 19: Concentration profile")
    generate_figure_19_concentration_profile(particles_180d)
    
    print("   Figure 20: Age distribution")
    generate_figure_20_age_distribution(particles_180d)
    
    print("   All 20 figures generated!")
    
    # Generate animation
    print("\n6. Generating animation...")
    video_path, gif_path = generate_animation(sources, u_field, v_field, lon_grid, lat_grid)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (180-day simulation)")
    print("="*70)
    
    status_counts = {}
    for p in particles_180d:
        status = p['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nParticle Fates:")
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(particles_180d)
        print(f"  {status:15s}: {count:6d} ({pct:5.1f}%)")
    
    # Beaching by region
    print("\nTop Beaching Regions:")
    beached_regions = {}
    for p in particles_180d:
        if p['status'] == 'beached':
            # Rough region classification
            lon, lat = p['lon'], p['lat']
            if lon < -70 and lat > 25:
                region = "US East Coast"
            elif lon < -80 and lat < 30:
                region = "US Gulf Coast"
            elif lon > -20 and lat > 40:
                region = "European Coast"
            elif lon > -20 and lat < 40:
                region = "African Coast"
            else:
                region = "Caribbean/Other"
            
            beached_regions[region] = beached_regions.get(region, 0) + 1
    
    for region, count in sorted(beached_regions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region:20s}: {count:4d}")
    
    # List all output files
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    
    print("\nFigures:")
    for i in range(1, 21):
        fname = f"outputs/figures/{i:02d}_*.png"
        import glob
        files = glob.glob(fname)
        if files:
            print(f"  {files[0]}")
    
    print("\nAnimations:")
    print(f"  {video_path}")
    print(f"  {gif_path}")
    
    print("\n" + "="*70)
    print("DRIFTCAST GENERATION COMPLETE")
    print("="*70)
    print("\nREMINDER: All results are synthetic and for demonstration purposes only.")
    print("="*70)

if __name__ == "__main__":
    main()
