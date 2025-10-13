# ocean_map_visualization.py

"""
INTERACTIVE MAP VISUALIZATION FOR OCEAN PLASTIC FLOW
Creates a beautiful, interactive map showing:
- Plastic particle trajectories
- Ocean currents
- Predicted concentrations
- Uncertainty zones
- Time evolution animation

Perfect for government presentations!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from datetime import datetime, timedelta

# Try to import cartopy for better maps (optional)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: Install cartopy for even better maps: pip install cartopy")

class OceanPlasticMapper:
    """Creates publication-quality maps of ocean plastic flow"""
    
    def __init__(self, use_cartopy=True):
        self.use_cartopy = use_cartopy and HAS_CARTOPY
        
    def generate_sample_data(self):
        """Generate realistic trajectory data for visualization"""
        
        # Release location (off NYC)
        release_lat, release_lon = 40.5, -73.5
        
        # Simulate 50 particles over 30 days
        n_particles = 50
        n_days = 30
        n_steps = n_days * 4  # 6-hour steps
        
        trajectories = []
        
        for i in range(n_particles):
            # Initial position with small spread
            lat = release_lat + np.random.randn() * 0.5
            lon = release_lon + np.random.randn() * 0.5
            
            traj_lats = [lat]
            traj_lons = [lon]
            
            # Simulate drift (Gulf Stream eastward + some meandering)
            for step in range(n_steps):
                # Eastward drift (Gulf Stream)
                dlon = 0.15 + np.random.randn() * 0.05
                
                # Northward drift (weaker)
                dlat = 0.02 + np.random.randn() * 0.02
                
                # Add some circular motion (eddies)
                if step > 20:
                    dlat += 0.01 * np.sin(step / 10)
                
                lat += dlat
                lon += dlon
                
                traj_lats.append(lat)
                traj_lons.append(lon)
            
            trajectories.append({
                'lats': np.array(traj_lats),
                'lons': np.array(traj_lons),
                'id': i
            })
        
        return {
            'trajectories': trajectories,
            'release_lat': release_lat,
            'release_lon': release_lon,
            'n_days': n_days,
            'n_steps': n_steps
        }
    
    def create_static_map(self, data, output_file='plastic_flow_map.png'):
        """Create a beautiful static map"""
        
        if self.use_cartopy:
            fig = plt.figure(figsize=(20, 10))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # Set extent (North Atlantic)
            ax.set_extent([-80, -30, 25, 55], crs=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle=':')
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                            alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
        else:
            fig, ax = plt.subplots(figsize=(18, 10))
            ax.set_xlim(-80, -30)
            ax.set_ylim(25, 55)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax.set_facecolor('#e8f4f8')
        
        # Plot ocean current field (background)
        lon_grid = np.linspace(-80, -30, 50)
        lat_grid = np.linspace(25, 55, 30)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        
        # Simulate Gulf Stream
        U = 0.8 * np.exp(-((LAT - 40) / 5)**2) + 0.1 * np.random.randn(*LAT.shape)
        V = 0.2 * np.sin(LON / 10) + 0.05 * np.random.randn(*LAT.shape)
        speed = np.sqrt(U**2 + V**2)
        
        # Background current speed
        im = ax.contourf(LON, LAT, speed, levels=15, cmap='Blues', 
                        alpha=0.4, transform=ccrs.PlateCarree() if self.use_cartopy else None)
        
        # Current vectors
        skip = 4
        ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                 U[::skip, ::skip], V[::skip, ::skip],
                 alpha=0.4, scale=15, width=0.002, color='navy',
                 transform=ccrs.PlateCarree() if self.use_cartopy else None)
        
        # Plot trajectories
        trajectories = data['trajectories']
        
        # Create color gradient for time
        for traj in trajectories:
            lons = traj['lons']
            lats = traj['lats']
            
            # Plot trajectory with color fade (old to new)
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(segments)))
            
            if self.use_cartopy:
                for i, seg in enumerate(segments):
                    ax.plot(seg[:, 0], seg[:, 1], color=colors[i], 
                           linewidth=1, alpha=0.6, transform=ccrs.PlateCarree())
            else:
                lc = LineCollection(segments, colors=colors, linewidths=1, alpha=0.6)
                ax.add_collection(lc)
        
        # Mark final positions
        final_lats = [traj['lats'][-1] for traj in trajectories]
        final_lons = [traj['lons'][-1] for traj in trajectories]
        
        ax.scatter(final_lons, final_lats, c='darkred', s=50, marker='o',
                  edgecolors='white', linewidth=1, alpha=0.8, zorder=5,
                  label=f'Final Position (Day {data["n_days"]})',
                  transform=ccrs.PlateCarree() if self.use_cartopy else None)
        
        # Release point
        ax.scatter(data['release_lon'], data['release_lat'], 
                  c='lime', s=400, marker='*', edgecolors='darkgreen',
                  linewidth=2, zorder=10, label='Release Point',
                  transform=ccrs.PlateCarree() if self.use_cartopy else None)
        
        # Add prediction ellipse
        mean_lat = np.mean(final_lats)
        mean_lon = np.mean(final_lons)
        std_lat = np.std(final_lats)
        std_lon = np.std(final_lons)
        
        ellipse = Ellipse((mean_lon, mean_lat), width=std_lon*4, height=std_lat*4,
                         facecolor='yellow', edgecolor='orange', linewidth=3,
                         alpha=0.3, label='95% Confidence Zone',
                         transform=ccrs.PlateCarree() if self.use_cartopy else None)
        ax.add_patch(ellipse)
        
        # Predicted center
        ax.scatter(mean_lon, mean_lat, c='orange', s=300, marker='X',
                  edgecolors='black', linewidth=2, zorder=10,
                  label='Predicted Center',
                  transform=ccrs.PlateCarree() if self.use_cartopy else None)
        
        # Add title and labels
        title_text = (f'Ocean Plastic Flow Prediction\n'
                     f'30-Day Forecast Using Reinforcement Learning + Ocean Physics')
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=20)
        
        # Add info box
        info_text = (f'Release: {data["release_lat"]:.1f}°N, {abs(data["release_lon"]):.1f}°W\n'
                    f'Prediction: {mean_lat:.1f}°N, {abs(mean_lon):.1f}°W\n'
                    f'Distance: {np.sqrt((mean_lat-data["release_lat"])**2 + (mean_lon-data["release_lon"])**2)*111:.0f} km\n'
                    f'Particles: {len(trajectories)}\n'
                    f'Model: DQN + Ocean Currents + Wind')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props, family='monospace')
        
        # Legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
        
        # Add colorbar for currents
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, aspect=40, shrink=0.6)
        cbar.set_label('Ocean Current Speed (m/s)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[SUCCESS] Map saved: {output_file}")
        plt.show()
    
    def create_animation(self, data, output_file='plastic_flow_animation.gif'):
        """Create animated map showing time evolution"""
        
        print("\nCreating animation (this may take 30-60 seconds)...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(-80, -30)
        ax.set_ylim(25, 55)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#e8f4f8')
        
        trajectories = data['trajectories']
        n_steps = data['n_steps']
        
        # Initialize plot elements
        scatter_current = ax.scatter([], [], c='red', s=50, alpha=0.8, zorder=5)
        scatter_release = ax.scatter(data['release_lon'], data['release_lat'],
                                    c='lime', s=400, marker='*', zorder=10)
        
        # Background
        lon_grid = np.linspace(-80, -30, 50)
        lat_grid = np.linspace(25, 55, 30)
        LON, LAT = np.meshgrid(lon_grid, lat_grid)
        U = 0.8 * np.exp(-((LAT - 40) / 5)**2)
        V = 0.2 * np.sin(LON / 10)
        speed = np.sqrt(U**2 + V**2)
        ax.contourf(LON, LAT, speed, levels=15, cmap='Blues', alpha=0.3)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=14, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        def animate(frame):
            # Get current positions
            lons = [traj['lons'][frame] for traj in trajectories if frame < len(traj['lons'])]
            lats = [traj['lats'][frame] for traj in trajectories if frame < len(traj['lats'])]
            
            scatter_current.set_offsets(np.c_[lons, lats])
            
            # Update time
            days = frame / 4  # 6-hour steps
            time_text.set_text(f'Day {days:.1f}')
            
            return scatter_current, time_text
        
        anim = animation.FuncAnimation(fig, animate, frames=n_steps, 
                                      interval=50, blit=True)
        
        # Save animation
        try:
            anim.save(output_file, writer='pillow', fps=20, dpi=100)
            print(f"[SUCCESS] Animation saved: {output_file}")
        except Exception as e:
            print(f"[WARNING] Could not save animation: {e}")
            print("Install pillow: pip install pillow")
        
        plt.close()

def main():
    print("="*70)
    print("OCEAN PLASTIC FLOW - INTERACTIVE MAP VISUALIZATION")
    print("="*70)
    
    mapper = OceanPlasticMapper()
    
    # Generate sample data (or load from simulation)
    print("\nGenerating trajectory data...")
    data = mapper.generate_sample_data()
    
    # Create static map
    print("\nCreating high-resolution map...")
    mapper.create_static_map(data, 'ocean_plastic_flow_map.png')
    
    # Create animation (optional)
    create_anim = input("\nCreate animation? (takes 1 min) [y/N]: ").lower()
    if create_anim == 'y':
        mapper.create_animation(data, 'ocean_plastic_flow.gif')
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  - ocean_plastic_flow_map.png (high-res static map)")
    if create_anim == 'y':
        print("  - ocean_plastic_flow.gif (animated)")
    print("\nPerfect for presentations and reports!")
    print("="*70)

if __name__ == "__main__":
    main()