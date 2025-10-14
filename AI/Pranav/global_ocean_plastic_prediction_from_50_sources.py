"""
FIXED GLOBAL OCEAN PLASTIC PREDICTION WITH DEEP RL
===================================================
Actually working version with visible results and real ocean data

Author: Oceans Four - DriftCast Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from datetime import datetime, timedelta
import json
import warnings
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
warnings.filterwarnings('ignore')

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Install Pillow for GIF creation: pip install Pillow")

# ==================== CONFIGURATION ====================
class Config:
    """Global configuration"""
    # Domain (focused on North Atlantic for testing)
    LAT_MIN, LAT_MAX = 20.0, 50.0
    LON_MIN, LON_MAX = -80.0, -20.0
    
    # Time settings - Shorter for testing
    START_DATE = datetime(2025, 10, 1)
    END_DATE = datetime(2025, 10, 13)
    TOTAL_DAYS = (END_DATE - START_DATE).days
    DT_HOURS = 6  # 6-hour timesteps
    TOTAL_STEPS = (TOTAL_DAYS * 24) // DT_HOURS
    
    # Particles
    N_PARTICLES = 50  # Per release point
    N_RELEASE_POINTS = 5  # Test sources
    
    # Physics
    WINDAGE = 0.03
    DIFFUSION_KM = 2.0  # Per timestep
    
    # RL Configuration
    RL_HIDDEN = 128
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 5
    
    # Visualization
    FPS = 5
    DPI = 100

# ==================== LOAD REAL OCEAN DATA ====================
class RealOceanEnvironment:
    """Load and interpolate real ocean currents"""
    
    def __init__(self, data_file='ocean_data_csv/cmems_currents.csv'):
        print("🌊 Loading real ocean data...")
        
        # Try CSV first
        if data_file.endswith('.nc'):
            csv_file = data_file.replace('.nc', '.csv')
            if csv_file.replace('ocean_data/', 'ocean_data_csv/'):
                csv_file = csv_file.replace('ocean_data/', 'ocean_data_csv/')
        else:
            csv_file = data_file
            
        try:
            self._load_from_csv(csv_file)
        except Exception as e:
            print(f"⚠️  Could not load CSV: {e}")
            try:
                self._load_from_netcdf(data_file)
            except Exception as e2:
                print(f"⚠️  Could not load NetCDF: {e2}")
                print("  Using synthetic currents instead")
                self.has_real_data = False
                self._generate_synthetic_currents()
    
    def _load_from_csv(self, csv_file):
        """Load ocean data from CSV"""
        print(f"  Loading from CSV: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Parse columns: time, depth, latitude, longitude, uo, vo
        df['time'] = pd.to_datetime(df['time'])
        
        # Get unique coordinates
        self.times = sorted(df['time'].unique())
        self.lats = sorted(df['latitude'].unique())
        self.lons = sorted(df['longitude'].unique())
        
        # Use surface level only (depth=0 or minimum depth)
        if 'depth' in df.columns:
            min_depth = df['depth'].min()
            df = df[df['depth'] == min_depth]
        
        # Create 3D arrays [time, lat, lon]
        n_times = len(self.times)
        n_lats = len(self.lats)
        n_lons = len(self.lons)
        
        self.u_data = np.zeros((n_times, n_lats, n_lons))
        self.v_data = np.zeros((n_times, n_lats, n_lons))
        
        # Fill arrays
        for idx, row in df.iterrows():
            t_idx = self.times.index(row['time'])
            lat_idx = self.lats.index(row['latitude'])
            lon_idx = self.lons.index(row['longitude'])
            
            self.u_data[t_idx, lat_idx, lon_idx] = row['uo']
            self.v_data[t_idx, lat_idx, lon_idx] = row['vo']
        
        # Handle NaN
        self.u_data = np.nan_to_num(self.u_data, 0.0)
        self.v_data = np.nan_to_num(self.v_data, 0.0)
        
        # Convert to numpy arrays
        self.lats = np.array(self.lats)
        self.lons = np.array(self.lons)
        
        print(f"✓ Loaded CSV data: {len(self.times)} timesteps")
        print(f"  Lat range: {self.lats.min():.1f} to {self.lats.max():.1f}")
        print(f"  Lon range: {self.lons.min():.1f} to {self.lons.max():.1f}")
        print(f"  Max current: {np.nanmax(np.sqrt(self.u_data**2 + self.v_data**2)):.3f} m/s")
        
        self.has_real_data = True
    
    def _load_from_netcdf(self, nc_file):
        """Load ocean data from NetCDF"""
        ds = xr.open_dataset(nc_file)
        
        # Extract variables
        self.times = pd.to_datetime(ds['time'].values)
        self.depths = ds['depth'].values if 'depth' in ds else [0]
        self.lats = ds['latitude'].values
        self.lons = ds['longitude'].values
        
        # Get surface currents (depth = 0 or first level)
        depth_idx = 0
        self.u_data = ds['uo'].isel(depth=depth_idx).values  # [time, lat, lon]
        self.v_data = ds['vo'].isel(depth=depth_idx).values
        
        # Handle NaN values
        self.u_data = np.nan_to_num(self.u_data, 0.0)
        self.v_data = np.nan_to_num(self.v_data, 0.0)
        
        ds.close()
        
        print(f"✓ Loaded NetCDF data: {len(self.times)} timesteps")
        print(f"  Lat range: {self.lats.min():.1f} to {self.lats.max():.1f}")
        print(f"  Lon range: {self.lons.min():.1f} to {self.lons.max():.1f}")
        print(f"  Max current: {np.nanmax(np.sqrt(self.u_data**2 + self.v_data**2)):.3f} m/s")
        
        self.has_real_data = True
    
    def _generate_synthetic_currents(self):
        """Generate synthetic but realistic currents"""
        self.lats = np.linspace(Config.LAT_MIN, Config.LAT_MAX, 50)
        self.lons = np.linspace(Config.LON_MIN, Config.LON_MAX, 80)
        LON, LAT = np.meshgrid(self.lons, self.lats)
        
        # Gulf Stream approximation
        gulf_lat = 35 + 2 * np.sin((LON + 70) / 10)
        intensity = 0.8 * np.exp(-((LAT - gulf_lat) / 3) ** 2)
        
        self.u_data = np.zeros((Config.TOTAL_STEPS, len(self.lats), len(self.lons)))
        self.v_data = np.zeros((Config.TOTAL_STEPS, len(self.lats), len(self.lons)))
        
        for t in range(Config.TOTAL_STEPS):
            phase = 2 * np.pi * t / 20
            self.u_data[t] = intensity * (1 + 0.3 * np.sin(phase))
            self.v_data[t] = 0.2 * np.sin((LON + 50) / 8) * np.cos(phase)
            
            # Add eddies
            for _ in range(3):
                eddy_lon = np.random.uniform(Config.LON_MIN, Config.LON_MAX)
                eddy_lat = np.random.uniform(Config.LAT_MIN, Config.LAT_MAX)
                r = np.sqrt((LAT - eddy_lat)**2 + (LON - eddy_lon)**2)
                eddy = 0.3 * np.exp(-r**2 / 9)
                self.u_data[t] += -eddy * (LAT - eddy_lat)
                self.v_data[t] += eddy * (LON - eddy_lon)
    
    def get_velocity(self, lat, lon, time_idx):
        """Interpolate velocity at location and time"""
        time_idx = int(np.clip(time_idx, 0, len(self.u_data) - 1))
        
        # Bounds checking
        if lat < self.lats.min() or lat > self.lats.max():
            return {'u': 0.0, 'v': 0.0, 'is_ocean': False}
        if lon < self.lons.min() or lon > self.lons.max():
            return {'u': 0.0, 'v': 0.0, 'is_ocean': False}
        
        # Bilinear interpolation
        lat_idx = np.searchsorted(self.lats, lat) - 1
        lon_idx = np.searchsorted(self.lons, lon) - 1
        
        lat_idx = np.clip(lat_idx, 0, len(self.lats) - 2)
        lon_idx = np.clip(lon_idx, 0, len(self.lons) - 2)
        
        lat_frac = (lat - self.lats[lat_idx]) / (self.lats[lat_idx + 1] - self.lats[lat_idx])
        lon_frac = (lon - self.lons[lon_idx]) / (self.lons[lon_idx + 1] - self.lons[lon_idx])
        
        u = ((1 - lat_frac) * (1 - lon_frac) * self.u_data[time_idx, lat_idx, lon_idx] +
             lat_frac * (1 - lon_frac) * self.u_data[time_idx, lat_idx + 1, lon_idx] +
             (1 - lat_frac) * lon_frac * self.u_data[time_idx, lat_idx, lon_idx + 1] +
             lat_frac * lon_frac * self.u_data[time_idx, lat_idx + 1, lon_idx + 1])
        
        v = ((1 - lat_frac) * (1 - lon_frac) * self.v_data[time_idx, lat_idx, lon_idx] +
             lat_frac * (1 - lon_frac) * self.v_data[time_idx, lat_idx + 1, lon_idx] +
             (1 - lat_frac) * lon_frac * self.v_data[time_idx, lat_idx, lon_idx + 1] +
             lat_frac * lon_frac * self.v_data[time_idx, lat_idx + 1, lon_idx + 1])
        
        return {'u': float(u), 'v': float(v), 'is_ocean': True}

# ==================== PLASTIC PARTICLE ====================
class PlasticParticle:
    """Individual plastic particle with proper physics"""
    
    def __init__(self, lat, lon, start_time):
        self.lat = lat
        self.lon = lon
        self.start_time = start_time
        self.trajectory = [(lat, lon)]
        self.beached = False
    
    def update(self, env, time_idx):
        """Update particle position using ocean currents"""
        if self.beached:
            return
        
        vel = env.get_velocity(self.lat, self.lon, time_idx)
        
        if not vel['is_ocean']:
            self.beached = True
            return
        
        # Convert velocity (m/s) to displacement (degrees per timestep)
        hours = Config.DT_HOURS
        m_per_degree_lat = 111320.0
        m_per_degree_lon = 111320.0 * np.cos(np.radians(self.lat))
        
        # Displacement in meters
        dx_m = vel['u'] * hours * 3600
        dy_m = vel['v'] * hours * 3600
        
        # Add diffusion
        diff_m = Config.DIFFUSION_KM * 1000
        dx_m += np.random.randn() * diff_m
        dy_m += np.random.randn() * diff_m
        
        # Convert to degrees
        dlon = dx_m / m_per_degree_lon
        dlat = dy_m / m_per_degree_lat
        
        # Update position
        self.lon += dlon
        self.lat += dlat
        
        # Keep in bounds
        self.lat = np.clip(self.lat, Config.LAT_MIN, Config.LAT_MAX)
        self.lon = np.clip(self.lon, Config.LON_MIN, Config.LON_MAX)
        
        self.trajectory.append((self.lat, self.lon))

# ==================== DEEP RL AGENT ====================
class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, Config.RL_HIDDEN),
            nn.ReLU(),
            nn.Linear(Config.RL_HIDDEN, Config.RL_HIDDEN),
            nn.ReLU(),
            nn.Linear(Config.RL_HIDDEN, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RLAgent:
    """Reinforcement Learning agent"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = 10  # [lat, lon, dlat, dlon, dist, angle_cos, angle_sin, u, v, time]
        self.action_dim = 5  # [stay, N, S, E, W]
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        
        self.epsilon = Config.EPSILON_START
        self.steps = 0
        
        # Tracking
        self.losses = []
        self.rewards_history = []
        self.errors_history = []
        
        print(f"✓ RL Agent initialized on {self.device}")
    
    def get_state(self, particle, target, env, time_idx):
        """Get state representation"""
        # Ensure we have a valid target position
        if time_idx >= len(target.trajectory):
            time_idx = len(target.trajectory) - 1
            
        target_lat, target_lon = target.trajectory[time_idx]
        
        vel = env.get_velocity(particle.lat, particle.lon, time_idx)
        
        dlat = target_lat - particle.lat
        dlon = target_lon - particle.lon
        dist = np.sqrt(dlat**2 + dlon**2)
        
        # Avoid division by zero
        if dist < 1e-6:
            angle = 0.0
        else:
            angle = np.arctan2(dlat, dlon)
        
        state = np.array([
            particle.lat / 45.0,  # Normalize
            particle.lon / 50.0,
            dlat / 10.0,
            dlon / 10.0,
            dist / 30.0,
            np.cos(angle),
            np.sin(angle),
            vel['u'],
            vel['v'],
            time_idx / max(Config.TOTAL_STEPS, 1)
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def apply_action(self, particle, action):
        """Apply correction action"""
        # Actions: stay, N, S, E, W
        action_map = {
            0: (0, 0),      # stay
            1: (0.1, 0),    # N
            2: (-0.1, 0),   # S
            3: (0, 0.1),    # E
            4: (0, -0.1)    # W
        }
        
        dlat, dlon = action_map[action]
        particle.lat += dlat
        particle.lon += dlon
        
        # Keep in bounds
        particle.lat = np.clip(particle.lat, Config.LAT_MIN, Config.LAT_MAX)
        particle.lon = np.clip(particle.lon, Config.LON_MIN, Config.LON_MAX)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Training step with experience replay"""
        if len(self.memory) < Config.BATCH_SIZE:
            return 0.0
        
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * Config.GAMMA * next_q
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)

# ==================== SIMULATION ====================
def run_simulation():
    """Main simulation"""
    print("\n" + "🌊" * 35)
    print("GLOBAL OCEAN PLASTIC PREDICTION WITH DEEP RL")
    print("🌊" * 35 + "\n")
    
    # Initialize environment
    env = RealOceanEnvironment()
    
    # Release points
    release_points = [
        (40.7, -74.0, "New York"),
        (38.9, -77.0, "Washington"),
        (25.8, -80.2, "Miami"),
        (29.8, -95.4, "Houston"),
        (42.4, -71.1, "Boston")
    ]
    
    print(f"\n{'='*70}")
    print(f"SIMULATION SETUP")
    print(f"{'='*70}")
    print(f"Period: {Config.START_DATE} to {Config.END_DATE}")
    print(f"Duration: {Config.TOTAL_DAYS} days ({Config.TOTAL_STEPS} timesteps)")
    print(f"Release points: {len(release_points)}")
    print(f"Particles per point: {Config.N_PARTICLES}")
    print(f"Total particles: {len(release_points) * Config.N_PARTICLES}")
    print(f"{'='*70}\n")
    
    # Generate ground truth trajectories
    print("Generating ground truth trajectories...")
    ground_truth = []
    
    for lat, lon, name in release_points:
        print(f"  {name}...")
        for i in range(Config.N_PARTICLES):
            p_lat = lat + np.random.randn() * 0.2
            p_lon = lon + np.random.randn() * 0.2
            particle = PlasticParticle(p_lat, p_lon, 0)
            
            for t in range(Config.TOTAL_STEPS):
                particle.update(env, t)
            
            ground_truth.append(particle)
    
    print(f"✓ Generated {len(ground_truth)} trajectories\n")
    
    # Train RL agent
    print(f"{'='*70}")
    print("TRAINING RL AGENT")
    print(f"{'='*70}\n")
    
    agent = RLAgent()
    
    for episode in range(Config.TOTAL_STEPS):
        # Sample target particle
        target = random.choice(ground_truth)
        
        # Skip if trajectory is too short
        if len(target.trajectory) < 2:
            continue
        
        # Create RL-controlled particle
        rl_particle = PlasticParticle(target.trajectory[0][0], target.trajectory[0][1], 0)
        
        episode_reward = 0
        episode_errors = []
        
        # Simulate with RL control
        max_steps = min(len(target.trajectory) - 1, Config.TOTAL_STEPS)
        for t in range(max_steps):
            # Skip if we've run out of trajectory
            if t >= len(target.trajectory) - 1:
                break
                
            state = agent.get_state(rl_particle, target, env, t)
            action = agent.select_action(state)
            
            # Physics step
            rl_particle.update(env, t)
            
            # RL correction
            agent.apply_action(rl_particle, action)
            
            # Calculate reward
            target_lat, target_lon = target.trajectory[t + 1]
            error = np.sqrt((rl_particle.lat - target_lat)**2 + 
                          (rl_particle.lon - target_lon)**2)
            reward = -error * 10  # Negative reward for distance
            
            episode_reward += reward
            episode_errors.append(error * 111)  # Convert to km
            
            next_state = agent.get_state(rl_particle, target, env, t + 1)
            done = (t == len(target.trajectory) - 2)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                agent.losses.append(loss)
        
        # Update target network
        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target()
        
        agent.decay_epsilon()
        
        # Record metrics (handle empty case)
        if len(episode_errors) > 0:
            agent.rewards_history.append(episode_reward / len(episode_errors))
            agent.errors_history.append(np.mean(episode_errors))
        else:
            agent.rewards_history.append(0.0)
            agent.errors_history.append(0.0)
        
        # Progress
        if episode % 5 == 0 or episode < 3:
            if len(agent.errors_history) > 0:
                print(f"Episode {episode:3d} | Reward: {agent.rewards_history[-1]:8.2f} | "
                      f"Error: {agent.errors_history[-1]:6.1f} km | ε: {agent.epsilon:.3f}")
            else:
                print(f"Episode {episode:3d} | Initializing... | ε: {agent.epsilon:.3f}")
    
    print(f"\n✓ Training complete!")
    
    if len(agent.errors_history) > 0:
        print(f"  Initial error: {agent.errors_history[0]:.1f} km")
        print(f"  Final error: {agent.errors_history[-1]:.1f} km")
        if agent.errors_history[0] > 0:
            improvement = ((agent.errors_history[0] - agent.errors_history[-1]) / agent.errors_history[0] * 100)
            print(f"  Improvement: {improvement:.1f}%")
    else:
        print("  No training data collected")
    
    print()
    
    return env, agent, ground_truth, release_points

# ==================== VISUALIZATION ====================
def create_visualizations(env, agent, ground_truth, release_points):
    """Create all visualizations"""
    print(f"{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Check if we have data
    if len(agent.errors_history) == 0:
        print("⚠️  No training data available - skipping visualizations")
        return
    
    # 1. Training Dashboard
    print("📊 Creating training dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = np.arange(len(agent.errors_history))
    
    # Error over time
    ax = axes[0, 0]
    ax.plot(episodes, agent.errors_history, 'b-', linewidth=2, alpha=0.7)
    z = np.polyfit(episodes, agent.errors_history, 2)
    p = np.poly1d(z)
    ax.plot(episodes, p(episodes), 'r--', linewidth=2, label='Trend')
    ax.fill_between(episodes, agent.errors_history, alpha=0.3)
    ax.set_title('Prediction Error Over Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error (km)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rewards
    ax = axes[0, 1]
    ax.plot(episodes, agent.rewards_history, 'g-', linewidth=2, alpha=0.7)
    ax.fill_between(episodes, agent.rewards_history, alpha=0.3, color='green')
    ax.set_title('Average Reward Per Episode', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[1, 0]
    if len(agent.losses) > 50:
        smoothed = np.convolve(agent.losses, np.ones(50)/50, mode='valid')
        ax.plot(smoothed, 'r-', linewidth=2)
    ax.set_title('Training Loss (Smoothed)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
TRAINING STATISTICS
{'='*40}

Initial Error: {agent.errors_history[0]:.1f} km
Final Error: {agent.errors_history[-1]:.1f} km
Best Error: {min(agent.errors_history):.1f} km

Improvement: {((agent.errors_history[0] - agent.errors_history[-1]) / agent.errors_history[0] * 100):.1f}%

Mean Error: {np.mean(agent.errors_history):.1f} ± {np.std(agent.errors_history):.1f} km

Episodes: {len(agent.errors_history)}
Training Steps: {len(agent.losses)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Deep RL Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rl_training_dashboard.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: rl_training_dashboard.png")
    plt.close()
    
    # 2. Trajectory Map
    print("🗺️  Creating trajectory map...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot ocean currents as background
    time_idx = Config.TOTAL_STEPS // 2
    speed = np.sqrt(env.u_data[time_idx]**2 + env.v_data[time_idx]**2)
    
    LON, LAT = np.meshgrid(env.lons, env.lats)
    im = ax.contourf(LON, LAT, speed, levels=30, cmap='Blues', alpha=0.5)
    
    skip = 3
    ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
             env.u_data[time_idx, ::skip, ::skip], env.v_data[time_idx, ::skip, ::skip],
             alpha=0.4, scale=5, width=0.003)
    
    # Plot sample trajectories
    sample_size = min(100, len(ground_truth))
    for particle in random.sample(ground_truth, sample_size):
        traj = np.array(particle.trajectory)
        if len(traj) > 1:
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(traj)))
            for i in range(len(traj) - 1):
                ax.plot(traj[i:i+2, 1], traj[i:i+2, 0], 
                       color=colors[i], linewidth=1, alpha=0.5)
    
    # Plot final positions
    final_lats = [p.trajectory[-1][0] for p in ground_truth if not p.beached]
    final_lons = [p.trajectory[-1][1] for p in ground_truth if not p.beached]
    ax.scatter(final_lons, final_lats, c='red', s=30, alpha=0.7, 
              edgecolors='darkred', linewidths=0.5, label='Final Positions')
    
    # Plot release points
    for lat, lon, name in release_points:
        ax.scatter(lon, lat, c='lime', s=300, marker='*', 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Ocean Plastic Trajectories\n{len(ground_truth)} particles over {Config.TOTAL_DAYS} days',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Current Speed (m/s)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('trajectory_map.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: trajectory_map.png")
    plt.close()
    
    # 3. Animation frames
    print("🎬 Creating animation frames...")
    n_frames = min(12, Config.TOTAL_STEPS)
    frame_indices = np.linspace(0, Config.TOTAL_STEPS - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, t in enumerate(frame_indices):
        ax = axes[idx]
        
        # Convert numpy int to Python int
        t = int(t)
        
        # Background currents
        speed = np.sqrt(env.u_data[t]**2 + env.v_data[t]**2)
        ax.contourf(LON, LAT, speed, levels=20, cmap='Blues', alpha=0.4)
        
        # Particles at this time
        lats_t = []
        lons_t = []
        for particle in ground_truth:
            if t < len(particle.trajectory):
                lat, lon = particle.trajectory[t]
                lats_t.append(lat)
                lons_t.append(lon)
        
        if lats_t:
            ax.scatter(lons_t, lats_t, c='red', s=20, alpha=0.7)
        
        # Release points
        for lat, lon, _ in release_points:
            ax.scatter(lon, lat, c='lime', s=80, marker='*', 
                      edgecolors='black', linewidths=1, zorder=10)
        
        date = Config.START_DATE + timedelta(hours=int(t * Config.DT_HOURS))
        ax.set_title(f'{date.strftime("%Y-%m-%d %H:%M")} | {len(lats_t)} particles',
                    fontsize=10, fontweight='bold')
        ax.set_xlim(Config.LON_MIN, Config.LON_MAX)
        ax.set_ylim(Config.LAT_MIN, Config.LAT_MAX)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Plastic Dispersion Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('time_evolution.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: time_evolution.png")
    plt.close()
    
    # 4. RL vs Physics Comparison
    print("📊 Creating RL comparison...")
    
    # Test on a particle
    test_particle = random.choice(ground_truth)
    
    # Physics-only
    physics_particle = PlasticParticle(test_particle.trajectory[0][0], 
                                      test_particle.trajectory[0][1], 0)
    max_steps = len(test_particle.trajectory) - 1
    for t in range(max_steps):
        physics_particle.update(env, int(t))
    
    # RL-enhanced
    rl_particle = PlasticParticle(test_particle.trajectory[0][0],
                                 test_particle.trajectory[0][1], 0)
    agent.epsilon = 0  # Greedy
    for t in range(max_steps):
        state = agent.get_state(rl_particle, test_particle, env, int(t))
        action = agent.select_action(state)
        rl_particle.update(env, int(t))
        agent.apply_action(rl_particle, action)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Trajectories
    target_traj = np.array(test_particle.trajectory)
    physics_traj = np.array(physics_particle.trajectory)
    rl_traj = np.array(rl_particle.trajectory)
    
    ax1.plot(target_traj[:, 1], target_traj[:, 0], 'g-', linewidth=3, 
            label='Ground Truth', alpha=0.8)
    ax1.plot(physics_traj[:, 1], physics_traj[:, 0], 'b--', linewidth=2, 
            label='Physics Only', alpha=0.7)
    ax1.plot(rl_traj[:, 1], rl_traj[:, 0], 'r-', linewidth=2, 
            label='RL Enhanced', alpha=0.7)
    
    ax1.scatter(target_traj[0, 1], target_traj[0, 0], c='lime', s=200, 
               marker='*', edgecolors='black', zorder=10, label='Start')
    ax1.scatter(target_traj[-1, 1], target_traj[-1, 0], c='green', s=150, 
               marker='s', edgecolors='black', zorder=10)
    ax1.scatter(physics_traj[-1, 1], physics_traj[-1, 0], c='blue', s=150, 
               marker='o', edgecolors='black', zorder=10)
    ax1.scatter(rl_traj[-1, 1], rl_traj[-1, 0], c='red', s=150, 
               marker='D', edgecolors='black', zorder=10)
    
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Errors over time
    physics_errors = []
    rl_errors = []
    
    for i in range(min(len(target_traj), len(physics_traj), len(rl_traj))):
        physics_err = np.sqrt((target_traj[i, 0] - physics_traj[i, 0])**2 + 
                             (target_traj[i, 1] - physics_traj[i, 1])**2) * 111
        rl_err = np.sqrt((target_traj[i, 0] - rl_traj[i, 0])**2 + 
                        (target_traj[i, 1] - rl_traj[i, 1])**2) * 111
        physics_errors.append(physics_err)
        rl_errors.append(rl_err)
    
    steps = np.arange(len(physics_errors))
    ax2.plot(steps, physics_errors, 'b-', linewidth=2, label='Physics Only', alpha=0.7)
    ax2.plot(steps, rl_errors, 'r-', linewidth=2, label='RL Enhanced', alpha=0.7)
    ax2.fill_between(steps, physics_errors, alpha=0.2, color='blue')
    ax2.fill_between(steps, rl_errors, alpha=0.2, color='red')
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Error (km)', fontsize=12)
    ax2.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    improvement = ((physics_errors[-1] - rl_errors[-1]) / physics_errors[-1] * 100)
    ax2.text(0.05, 0.95, f'Final Errors:\nPhysics: {physics_errors[-1]:.1f} km\n'
            f'RL: {rl_errors[-1]:.1f} km\nImprovement: {improvement:.1f}%',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('RL vs Physics-Only Prediction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rl_vs_physics.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: rl_vs_physics.png")
    plt.close()
    
    print("\n✓ All visualizations complete!")

# ==================== ANIMATION GIF ====================
def create_animation_gif(env, ground_truth, release_points):
    """Create animated GIF"""
    if not HAS_PIL:
        print("⚠️  Skipping GIF (PIL not available)")
        return
    
    print("\n🎬 Creating animation GIF...")
    
    n_frames = 30
    frame_indices = np.linspace(0, Config.TOTAL_STEPS - 1, n_frames, dtype=int)
    
    images = []
    
    for idx, t in enumerate(frame_indices):
        # Convert numpy int to Python int
        t = int(t)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Background
        speed = np.sqrt(env.u_data[t]**2 + env.v_data[t]**2)
        LON, LAT = np.meshgrid(env.lons, env.lats)
        
        ax.contourf(LON, LAT, speed, levels=25, cmap='Blues', alpha=0.5)
        skip = 4
        ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
                 env.u_data[t, ::skip, ::skip], env.v_data[t, ::skip, ::skip],
                 alpha=0.4, scale=5, width=0.003)
        
        # Particles
        lats_t = []
        lons_t = []
        for particle in ground_truth:
            if t < len(particle.trajectory):
                lat, lon = particle.trajectory[t]
                lats_t.append(lat)
                lons_t.append(lon)
        
        if lats_t:
            ax.scatter(lons_t, lats_t, c='red', s=25, alpha=0.7, 
                      edgecolors='darkred', linewidths=0.5)
            
            # Show trails for some particles
            for particle in random.sample(ground_truth, min(20, len(ground_truth))):
                if t < len(particle.trajectory):
                    traj = np.array(particle.trajectory[:t+1])
                    if len(traj) > 1:
                        ax.plot(traj[:, 1], traj[:, 0], 'r-', alpha=0.2, linewidth=0.8)
        
        # Release points
        for lat, lon, name in release_points:
            ax.scatter(lon, lat, c='lime', s=200, marker='*', 
                      edgecolors='black', linewidths=1.5, zorder=10)
        
        date = Config.START_DATE + timedelta(hours=int(t * Config.DT_HOURS))
        ax.set_title(f'Ocean Plastic Dispersion | {date.strftime("%Y-%m-%d %H:%M")}\n'
                    f'{len(lats_t)} active particles',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_xlim(Config.LON_MIN, Config.LON_MAX)
        ax.set_ylim(Config.LAT_MIN, Config.LAT_MAX)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save frame
        filename = f'frame_{idx:03d}.png'
        plt.savefig(filename, dpi=80, bbox_inches='tight')
        plt.close()
        
        images.append(Image.open(filename))
        
        if (idx + 1) % 5 == 0:
            print(f"  Frame {idx + 1}/{n_frames}")
    
    # Create GIF
    print("  Compiling GIF...")
    images[0].save('plastic_animation.gif',
                  save_all=True,
                  append_images=images[1:],
                  duration=200,
                  loop=0)
    
    # Cleanup
    import os
    for i in range(n_frames):
        try:
            os.remove(f'frame_{i:03d}.png')
        except:
            pass
    
    print("✓ Saved: plastic_animation.gif")

# ==================== JSON EXPORT ====================
def export_json(agent, ground_truth):
    """Export results to JSON"""
    print("\n📁 Exporting results...")
    
    output = {
        'metadata': {
            'start_date': Config.START_DATE.strftime('%Y-%m-%d %H:%M'),
            'end_date': Config.END_DATE.strftime('%Y-%m-%d %H:%M'),
            'total_days': Config.TOTAL_DAYS,
            'timesteps': Config.TOTAL_STEPS,
            'n_particles': len(ground_truth)
        },
        'training': {
            'initial_error_km': float(agent.errors_history[0]),
            'final_error_km': float(agent.errors_history[-1]),
            'best_error_km': float(min(agent.errors_history)),
            'mean_error_km': float(np.mean(agent.errors_history)),
            'improvement_percent': float((agent.errors_history[0] - agent.errors_history[-1]) / 
                                        agent.errors_history[0] * 100),
            'episodes': len(agent.errors_history)
        },
        'errors_by_episode': [float(e) for e in agent.errors_history],
        'rewards_by_episode': [float(r) for r in agent.rewards_history]
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Saved: results.json")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "🌊" * 35)
    print("OCEAN PLASTIC PREDICTION - DEEP RL")
    print("Oceans Four - DriftCast Team")
    print("🌊" * 35)
    
    # Run simulation
    env, agent, ground_truth, release_points = run_simulation()
    
    # Create visualizations
    create_visualizations(env, agent, ground_truth, release_points)
    
    # Create animation
    create_animation_gif(env, ground_truth, release_points)
    
    # Export results
    export_json(agent, ground_truth)
    
    print("\n" + "🎉" * 35)
    print("COMPLETE!")
    print("🎉" * 35)
    
    print("\n📁 Generated Files:")
    print("   1. rl_training_dashboard.png - Training metrics")
    print("   2. trajectory_map.png - All particle trajectories")
    print("   3. time_evolution.png - 12 time snapshots")
    print("   4. rl_vs_physics.png - Method comparison")
    print("   5. plastic_animation.gif - Animated dispersion")
    print("   6. results.json - Numerical results")
    
    print(f"\n{'='*70}")
    print("KEY RESULTS")
    print(f"{'='*70}")
    print(f"Initial Error: {agent.errors_history[0]:.1f} km")
    print(f"Final Error: {agent.errors_history[-1]:.1f} km")
    print(f"Best Error: {min(agent.errors_history):.1f} km")
    print(f"Improvement: {((agent.errors_history[0] - agent.errors_history[-1]) / agent.errors_history[0] * 100):.1f}%")
    print(f"{'='*70}\n")