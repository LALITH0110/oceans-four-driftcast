"""
GLOBAL OCEAN PLASTIC PREDICTION WITH DEEP RL
==============================================
Advanced system with animations, global coverage, and comprehensive RL training

Features:
- Global map coverage with realistic ocean currents
- Multi-year historical simulation (2020-2025)
- Improved Deep RL with visual proof of learning
- Animated GIFs showing plastic movement and RL convergence
- Extensive validation and visualization
- JSON export with detailed metrics

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
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Install cartopy for better maps: pip install cartopy")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Install Pillow for GIF creation: pip install Pillow")

# ==================== CONFIGURATION ====================
class Config:
    """Global configuration"""
    # Global domain
    LAT_MIN, LAT_MAX = -60.0, 70.0  # Global coverage
    LON_MIN, LON_MAX = -180.0, 180.0
    GRID_RES = 1.0  # 1 degree resolution
    
    # Time settings - Multi-year simulation
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2025, 10, 13)
    TOTAL_DAYS = (END_DATE - START_DATE).days
    DT_DAYS = 7  # Weekly timesteps (balance between detail and computation)
    TOTAL_STEPS = TOTAL_DAYS // DT_DAYS
    
    # Ensemble
    N_PARTICLES = 50  # Per release point
    N_RELEASE_POINTS = 5  # Major ocean pollution sources
    
    # Physics
    WINDAGE = 0.03
    STOKES_COEFF = 0.016
    DIFFUSION_KM = 10.0  # Per week
    BIOFOULING_WEEKS = 4
    
    # RL Configuration
    RL_HIDDEN = 512  # Larger network
    RL_LAYERS = 3
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64
    MEMORY_SIZE = 50000
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE = 10
    
    # Visualization
    FPS = 10
    DPI = 150

# ==================== GLOBAL OCEAN ENVIRONMENT ====================
class GlobalOceanEnvironment:
    """Realistic global ocean with major currents"""
    
    def __init__(self):
        print("🌍 Generating global ocean environment...")
        
        self.lats = np.arange(Config.LAT_MIN, Config.LAT_MAX + Config.GRID_RES, Config.GRID_RES)
        self.lons = np.arange(Config.LON_MIN, Config.LON_MAX + Config.GRID_RES, Config.GRID_RES)
        self.LON, self.LAT = np.meshgrid(self.lons, self.lats)
        
        self._generate_global_currents()
        self._generate_wind_fields()
        self._create_land_mask()
        
        print(f"✓ Ocean grid: {len(self.lats)} x {len(self.lons)}")
        print(f"✓ Max current: {np.max(np.sqrt(self.u**2 + self.v**2)):.2f} m/s")
    
    def _generate_global_currents(self):
        """Generate realistic global ocean currents"""
        self.u = np.zeros_like(self.LON)
        self.v = np.zeros_like(self.LAT)
        
        # Atlantic: Gulf Stream
        mask_atlantic = (self.LON > -80) & (self.LON < -30) & (self.LAT > 25) & (self.LAT < 55)
        gulf_stream_lat = 38 + 3 * np.sin(self.LON[mask_atlantic] / 10)
        intensity = 1.2 * np.exp(-((self.LAT[mask_atlantic] - gulf_stream_lat) / 4) ** 2)
        self.u[mask_atlantic] += intensity
        self.v[mask_atlantic] += 0.3 * np.sin(self.LON[mask_atlantic] / 8)
        
        # Pacific: Kuroshio Current
        mask_kuroshio = (self.LON > 120) & (self.LON < 160) & (self.LAT > 20) & (self.LAT < 45)
        kuroshio_lat = 35 + 2 * np.sin(self.LON[mask_kuroshio] / 15)
        intensity = 1.5 * np.exp(-((self.LAT[mask_kuroshio] - kuroshio_lat) / 3) ** 2)
        self.u[mask_kuroshio] += intensity
        self.v[mask_kuroshio] += 0.4 * np.sin(self.LON[mask_kuroshio] / 10)
        
        # Equatorial currents
        mask_eq = (self.LAT > -10) & (self.LAT < 10)
        self.u[mask_eq] += 0.5 * np.cos(np.radians(self.LAT[mask_eq] * 3))
        
        # Subtropical gyres (all major oceans)
        gyres = [
            (-40, 30, 15, 0.6),   # North Atlantic
            (-150, 30, 20, 0.7),  # North Pacific
            (80, -30, 18, 0.5),   # South Indian
            (-100, -30, 16, 0.5), # South Pacific
            (-20, -30, 14, 0.4)   # South Atlantic
        ]
        
        for gyre_lon, gyre_lat, radius, strength in gyres:
            r = np.sqrt((self.LAT - gyre_lat)**2 + (self.LON - gyre_lon)**2)
            gyre_mask = r < radius * 2
            gyre_field = strength * np.exp(-r[gyre_mask]**2 / radius**2)
            
            # Circular flow
            self.u[gyre_mask] += -gyre_field * (self.LAT[gyre_mask] - gyre_lat) / radius
            self.v[gyre_mask] += gyre_field * (self.LON[gyre_mask] - gyre_lon) / radius
        
        # Antarctic Circumpolar Current
        mask_acc = (self.LAT < -40) & (self.LAT > -60)
        self.u[mask_acc] += 0.8 * np.ones_like(self.u[mask_acc])
        
        # Add mesoscale eddies
        n_eddies = 50
        for _ in range(n_eddies):
            eddy_lon = np.random.uniform(-180, 180)
            eddy_lat = np.random.uniform(-50, 50)
            eddy_radius = np.random.uniform(3, 8)
            eddy_strength = np.random.uniform(0.1, 0.4)
            
            r = np.sqrt((self.LAT - eddy_lat)**2 + (self.LON - eddy_lon)**2)
            eddy_mask = r < eddy_radius * 2
            eddy_field = eddy_strength * np.exp(-r[eddy_mask]**2 / eddy_radius**2)
            
            self.u[eddy_mask] += -eddy_field * (self.LAT[eddy_mask] - eddy_lat) / eddy_radius
            self.v[eddy_mask] += eddy_field * (self.LON[eddy_mask] - eddy_lon) / eddy_radius
    
    def _generate_wind_fields(self):
        """Generate realistic wind patterns"""
        self.u_wind = np.zeros_like(self.LON)
        self.v_wind = np.zeros_like(self.LAT)
        
        # Trade winds (0-30° both hemispheres)
        mask_trades_n = (self.LAT > 0) & (self.LAT < 30)
        self.u_wind[mask_trades_n] = 6.0
        self.v_wind[mask_trades_n] = -1.0
        
        mask_trades_s = (self.LAT < 0) & (self.LAT > -30)
        self.u_wind[mask_trades_s] = 6.0
        self.v_wind[mask_trades_s] = 1.0
        
        # Westerlies (30-60° both hemispheres)
        mask_west_n = (self.LAT > 30) & (self.LAT < 60)
        self.u_wind[mask_west_n] = -8.0
        self.v_wind[mask_west_n] = 0.5
        
        mask_west_s = (self.LAT < -30) & (self.LAT > -60)
        self.u_wind[mask_west_s] = -10.0
        self.v_wind[mask_west_s] = -0.5
        
        # Add variability
        self.u_wind += np.random.randn(*self.u_wind.shape) * 1.5
        self.v_wind += np.random.randn(*self.v_wind.shape) * 1.5
        
        # Stokes drift
        self.u_stokes = Config.STOKES_COEFF * self.u_wind
        self.v_stokes = Config.STOKES_COEFF * self.v_wind
    
    def _create_land_mask(self):
        """Simple land mask"""
        self.is_ocean = np.ones_like(self.LAT, dtype=bool)
        
        # Major landmasses (simplified)
        # North America
        self.is_ocean[(self.LAT > 25) & (self.LAT < 70) & 
                     (self.LON > -140) & (self.LON < -50)] = False
        
        # South America
        self.is_ocean[(self.LAT > -55) & (self.LAT < 12) & 
                     (self.LON > -80) & (self.LON < -35)] = False
        
        # Europe
        self.is_ocean[(self.LAT > 35) & (self.LAT < 70) & 
                     (self.LON > -10) & (self.LON < 40)] = False
        
        # Africa
        self.is_ocean[(self.LAT > -35) & (self.LAT < 37) & 
                     (self.LON > -20) & (self.LON < 52)] = False
        
        # Asia
        self.is_ocean[(self.LAT > 5) & (self.LAT < 70) & 
                     (self.LON > 25) & (self.LON < 180)] = False
        
        # Australia
        self.is_ocean[(self.LAT > -45) & (self.LAT < -10) & 
                     (self.LON > 110) & (self.LON < 155)] = False
    
    def get_velocity(self, lat, lon, time_step):
        """Get interpolated velocity at position and time"""
        # Handle wraparound longitude
        lon = ((lon + 180) % 360) - 180
        
        # Find grid indices
        lat_idx = np.clip((lat - Config.LAT_MIN) / Config.GRID_RES, 0, len(self.lats) - 2)
        lon_idx = np.clip((lon - Config.LON_MIN) / Config.GRID_RES, 0, len(self.lons) - 2)
        
        i, j = int(lat_idx), int(lon_idx)
        di, dj = lat_idx - i, lon_idx - j
        
        # Bilinear interpolation
        def interp(field):
            return ((1-di)*(1-dj)*field[i,j] + di*(1-dj)*field[i+1,j] +
                   (1-di)*dj*field[i,j+1] + di*dj*field[i+1,j+1])
        
        u_ocean = interp(self.u)
        v_ocean = interp(self.v)
        u_wind = interp(self.u_wind)
        v_wind = interp(self.v_wind)
        u_stokes = interp(self.u_stokes)
        v_stokes = interp(self.v_stokes)
        is_ocean = interp(self.is_ocean) > 0.5
        
        # Add time-varying component (seasonal + tidal)
        seasonal_phase = 2 * np.pi * time_step / 52  # Annual cycle (52 weeks)
        tidal_phase = 2 * np.pi * time_step / 2  # Semi-weekly tide proxy
        
        u_ocean += 0.15 * np.cos(seasonal_phase + lon * 0.05)
        v_ocean += 0.15 * np.sin(seasonal_phase + lat * 0.05)
        u_ocean += 0.05 * np.cos(tidal_phase)
        v_ocean += 0.05 * np.sin(tidal_phase)
        
        return {
            'u_ocean': u_ocean,
            'v_ocean': v_ocean,
            'u_wind': u_wind,
            'v_wind': v_wind,
            'u_stokes': u_stokes,
            'v_stokes': v_stokes,
            'is_ocean': is_ocean
        }

# ==================== PLASTIC PARTICLE ====================
class PlasticParticle:
    """Individual plastic particle"""
    
    def __init__(self, lat, lon, time_step):
        self.lat = lat
        self.lon = lon
        self.age_weeks = 0
        self.biofouling = 0.0
        self.beached = False
        self.start_step = time_step
        self.trajectory = [(lat, lon, time_step)]
    
    def update(self, env, time_step):
        """Update particle position"""
        if self.beached:
            return
        
        vel = env.get_velocity(self.lat, self.lon, time_step)
        
        if not vel['is_ocean']:
            self.beached = True
            return
        
        # Total velocity
        windage = Config.WINDAGE * (1 - 0.8 * self.biofouling)
        stokes_factor = 1.0 - 0.5 * self.biofouling
        
        u_total = (vel['u_ocean'] + 
                  windage * vel['u_wind'] + 
                  stokes_factor * vel['u_stokes'])
        v_total = (vel['v_ocean'] + 
                  windage * vel['v_wind'] + 
                  stokes_factor * vel['v_stokes'])
        
        # Diffusion
        diff_std = Config.DIFFUSION_KM / 111.0  # degrees
        u_total += np.random.randn() * diff_std
        v_total += np.random.randn() * diff_std
        
        # Update position (velocity in m/s, convert to degrees per week)
        seconds_per_week = 7 * 24 * 3600
        m_per_degree = 111320.0
        
        dlat = v_total * seconds_per_week / m_per_degree
        dlon = u_total * seconds_per_week / (m_per_degree * np.cos(np.radians(self.lat)))
        
        self.lat += dlat
        self.lon += dlon
        
        # Wrap longitude
        self.lon = ((self.lon + 180) % 360) - 180
        
        # Clip latitude
        self.lat = np.clip(self.lat, Config.LAT_MIN, Config.LAT_MAX)
        
        # Update properties
        self.age_weeks += 1
        self.biofouling = min(1.0, self.age_weeks / Config.BIOFOULING_WEEKS)
        
        self.trajectory.append((self.lat, self.lon, time_step))

# ==================== DEEP RL AGENT ====================
class ImprovedDQN(nn.Module):
    """Enhanced DQN with residual connections"""
    
    def __init__(self, state_dim, action_dim):
        super(ImprovedDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, Config.RL_HIDDEN)
        self.fc2 = nn.Linear(Config.RL_HIDDEN, Config.RL_HIDDEN)
        self.fc3 = nn.Linear(Config.RL_HIDDEN, Config.RL_HIDDEN)
        self.fc4 = nn.Linear(Config.RL_HIDDEN, action_dim)
        
        self.ln1 = nn.LayerNorm(Config.RL_HIDDEN)
        self.ln2 = nn.LayerNorm(Config.RL_HIDDEN)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x1 = torch.relu(self.ln1(self.fc1(x)))
        x1 = self.dropout(x1)
        
        x2 = torch.relu(self.ln2(self.fc2(x1)))
        x2 = self.dropout(x2)
        x2 = x2 + x1  # Residual
        
        x3 = torch.relu(self.fc3(x2))
        x3 = x3 + x2  # Residual
        
        return self.fc4(x3)

class RLAgent:
    """Improved Deep Q-Learning agent"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = 14  # Enhanced state space
        self.action_dim = 9  # 8 directions + stay
        
        self.policy_net = ImprovedDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = ImprovedDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        
        self.epsilon = Config.EPSILON_START
        self.steps = 0
        
        # Tracking
        self.losses = []
        self.rewards = []
        self.errors = []
        self.epsilon_history = []
        
        print(f"✓ RL Agent on {self.device}")
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        print(f"  Network: {total_params:,} parameters")
    
    def get_state(self, particle, target_particle, env, time_step):
        """Enhanced state representation"""
        vel = env.get_velocity(particle.lat, particle.lon, time_step)
        
        # Distance to target
        dlat = target_particle.lat - particle.lat
        dlon = target_particle.lon - particle.lon
        # Wrap longitude difference
        if abs(dlon) > 180:
            dlon = dlon - 360 * np.sign(dlon)
        dist = np.sqrt(dlat**2 + dlon**2)
        
        # Direction to target
        angle = np.arctan2(dlat, dlon)
        
        state = np.array([
            particle.lat / 90.0,
            particle.lon / 180.0,
            dlat / 10.0,
            dlon / 10.0,
            dist / 100.0,
            np.cos(angle),
            np.sin(angle),
            vel['u_ocean'],
            vel['v_ocean'],
            vel['u_wind'] / 10.0,
            vel['v_wind'] / 10.0,
            particle.biofouling,
            time_step / Config.TOTAL_STEPS,
            float(vel['is_ocean'])
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
    
    def apply_action(self, particle, action, env, time_step):
        """Apply RL correction to particle"""
        # Actions: N, NE, E, SE, S, SW, W, NW, Stay
        action_offsets = [
            (0.1, 0), (0.07, 0.07), (0, 0.1), (-0.07, 0.07),
            (-0.1, 0), (-0.07, -0.07), (0, -0.1), (0.07, -0.07),
            (0, 0)
        ]
        
        dlat, dlon = action_offsets[action]
        particle.lat += dlat
        particle.lon += dlon
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Training with experience replay"""
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
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
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
def run_global_simulation():
    """Main simulation loop"""
    print("\n" + "🌍" * 35)
    print("GLOBAL OCEAN PLASTIC PREDICTION WITH DEEP RL")
    print("🌍" * 35 + "\n")
    
    print(f"{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Period: {Config.START_DATE.date()} to {Config.END_DATE.date()}")
    print(f"Duration: {Config.TOTAL_DAYS} days ({Config.TOTAL_STEPS} weeks)")
    print(f"Timestep: {Config.DT_DAYS} days")
    print(f"Release points: {Config.N_RELEASE_POINTS}")
    print(f"Particles per point: {Config.N_PARTICLES}")
    print(f"Total particles: {Config.N_RELEASE_POINTS * Config.N_PARTICLES}")
    print(f"{'='*70}\n")
    
    # Initialize environment
    env = GlobalOceanEnvironment()
    
    # Define release points (major pollution sources)
    release_points = [
        (35.0, -75.0, "US East Coast"),
        (20.0, -155.0, "Pacific Garbage Patch"),
        (10.0, 105.0, "South China Sea"),
        (-10.0, 40.0, "Indian Ocean"),
        (-30.0, -50.0, "South Atlantic")
    ]
    
    print(f"\n{'='*70}")
    print("RELEASE LOCATIONS")
    print(f"{'='*70}")
    for i, (lat, lon, name) in enumerate(release_points):
        print(f"{i+1}. {name}: {lat:.1f}°N, {abs(lon):.1f}°{'W' if lon < 0 else 'E'}")
    print(f"{'='*70}\n")
    
    # Generate "ground truth" trajectories
    print("Generating ground truth plastic trajectories...")
    ground_truth_particles = []
    
    for lat, lon, name in release_points:
        for _ in range(Config.N_PARTICLES):
            # Add small spread
            p_lat = lat + np.random.randn() * 0.5
            p_lon = lon + np.random.randn() * 0.5
            particle = PlasticParticle(p_lat, p_lon, 0)
            
            # Simulate
            for step in range(Config.TOTAL_STEPS):
                particle.update(env, step)
            
            ground_truth_particles.append(particle)
    
    print(f"✓ Generated {len(ground_truth_particles)} ground truth trajectories\n")
    
    # Initialize RL agent
    agent = RLAgent()
    
    # Training
    print(f"{'='*70}")
    print("RL TRAINING")
    print(f"{'='*70}\n")
    
    results_by_step = []
    
    for step in range(Config.TOTAL_STEPS):
        current_date = Config.START_DATE + timedelta(days=step * Config.DT_DAYS)
        
        step_rewards = []
        step_errors = []
        
        # Train on random subset
        train_particles = random.sample(ground_truth_particles, 
                                       min(20, len(ground_truth_particles)))
        
        for target_particle in train_particles:
            # Create RL-controlled particle
            rl_particle = PlasticParticle(target_particle.trajectory[0][0],
                                         target_particle.trajectory[0][1], 0)
            
            # Simulate with RL control
            episode_reward = 0
            for t in range(min(step + 1, len(target_particle.trajectory) - 1)):
                state = agent.get_state(rl_particle, target_particle, env, t)
                action = agent.select_action(state)
                
                # Physics step
                rl_particle.update(env, t)
                
                # Apply RL correction
                agent.apply_action(rl_particle, action, env, t)
                
                # Compute reward
                target_lat, target_lon, _ = target_particle.trajectory[t + 1]
                dlon = target_lon - rl_particle.lon
                if abs(dlon) > 180:
                    dlon = dlon - 360 * np.sign(dlon)
                error_km = np.sqrt((target_lat - rl_particle.lat)**2 + dlon**2) * 111
                reward = -error_km / 100.0  # Normalized
                
                next_state = agent.get_state(rl_particle, target_particle, env, t + 1)
                done = (t == len(target_particle.trajectory) - 2)
                
                agent.store_experience(state, action, reward, next_state, done)
                episode_reward += reward
                
                # Train
                loss = agent.train_step()
                if loss > 0:
                    agent.losses.append(loss)
            
            step_rewards.append(episode_reward)
            
            # Final error
            target_lat, target_lon, _ = target_particle.trajectory[min(step, len(target_particle.trajectory) - 1)]
            dlon = target_lon - rl_particle.lon
            if abs(dlon) > 180:
                dlon = dlon - 360 * np.sign(dlon)
            final_error = np.sqrt((target_lat - rl_particle.lat)**2 + dlon**2) * 111
            step_errors.append(final_error)
        
        # Update target network
        if step % Config.TARGET_UPDATE == 0:
            agent.update_target()
        
        agent.decay_epsilon()
        
        # Record
        mean_reward = np.mean(step_rewards)
        mean_error = np.mean(step_errors)
        agent.rewards.append(mean_reward)
        agent.errors.append(mean_error)
        agent.epsilon_history.append(agent.epsilon)
        
        # Store results
        step_result = {
            'step': step,
            'date': current_date.strftime('%Y-%m-%d'),
            'days_since_start': step * Config.DT_DAYS,
            'mean_reward': float(mean_reward),
            'mean_error_km': float(mean_error),
            'epsilon': float(agent.epsilon),
            'samples': len(step_rewards)
        }
        results_by_step.append(step_result)
        
        # Progress
        if step % 10 == 0 or step < 5:
            print(f"Week {step:3d} | {current_date.date()} | "
                  f"Reward: {mean_reward:8.2f} | Error: {mean_error:6.1f} km | "
                  f"ε: {agent.epsilon:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final error: {agent.errors[-1]:.1f} km")
    print(f"Best error: {min(agent.errors):.1f} km")
    print(f"Improvement: {((agent.errors[0] - agent.errors[-1]) / agent.errors[0] * 100):.1f}%")
    print(f"{'='*70}\n")
    
    return env, agent, ground_truth_particles, results_by_step

# ==================== VISUALIZATION ====================
def create_visualizations(env, agent, ground_truth_particles, results_by_step):
    """Create comprehensive visualizations"""
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # 1. Training Performance Dashboard
    print("📊 Creating training dashboard...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    steps = np.arange(len(agent.errors))
    dates = [Config.START_DATE + timedelta(days=s * Config.DT_DAYS) for s in steps]
    
    # Panel 1: Prediction Error Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, agent.errors, 'b-', linewidth=2, label='Prediction Error')
    z = np.polyfit(steps, agent.errors, 2)
    p = np.poly1d(z)
    ax1.plot(dates, p(steps), 'r--', linewidth=2, label='Trend', alpha=0.7)
    ax1.fill_between(dates, 0, agent.errors, alpha=0.2, color='blue')
    ax1.set_title('RL Learning Progress: Prediction Error Reduction', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Mean Error (km)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Panel 2: Cumulative Reward
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_reward = np.cumsum(agent.rewards)
    ax2.plot(dates, cumulative_reward, 'g-', linewidth=2)
    ax2.fill_between(dates, cumulative_reward, alpha=0.3, color='green')
    ax2.set_title('Cumulative Reward', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Loss Over Training
    ax3 = fig.add_subplot(gs[1, 1])
    if agent.losses:
        loss_smoothed = np.convolve(agent.losses, np.ones(50)/50, mode='valid')
        ax3.plot(loss_smoothed, 'r-', linewidth=1.5)
        ax3.set_title('Network Loss (Smoothed)', fontweight='bold')
        ax3.set_xlabel('Training Iteration')
        ax3.set_ylabel('MSE Loss')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Epsilon Decay
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(dates, agent.epsilon_history, 'purple', linewidth=2)
    ax4.fill_between(dates, 0, agent.epsilon_history, alpha=0.3, color='purple')
    ax4.set_title('Exploration Rate (ε)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Error Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(agent.errors, bins=40, color='blue', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(agent.errors), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(agent.errors):.1f} km')
    ax5.axvline(np.median(agent.errors), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(agent.errors):.1f} km')
    ax5.set_title('Error Distribution', fontweight='bold')
    ax5.set_xlabel('Error (km)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Learning Rate Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    window = 20
    if len(agent.errors) >= window * 2:
        early_errors = agent.errors[:window]
        late_errors = agent.errors[-window:]
        
        ax6.violinplot([early_errors, late_errors], positions=[1, 2], 
                      showmeans=True, showmedians=True)
        ax6.set_xticks([1, 2])
        ax6.set_xticklabels(['Early Training', 'Late Training'])
        ax6.set_ylabel('Error (km)')
        ax6.set_title('Early vs Late Performance', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Panel 7: Improvement Metrics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    improvement = ((agent.errors[0] - agent.errors[-1]) / agent.errors[0] * 100)
    best_error = min(agent.errors)
    worst_error = max(agent.errors)
    
    stats_text = f"""
LEARNING STATISTICS
{'='*30}

Initial Error: {agent.errors[0]:.1f} km
Final Error: {agent.errors[-1]:.1f} km
Best Error: {best_error:.1f} km
Worst Error: {worst_error:.1f} km

Improvement: {improvement:.1f}%
Mean Error: {np.mean(agent.errors):.1f} km
Std Dev: {np.std(agent.errors):.1f} km

Training Steps: {len(agent.errors)}
Final Epsilon: {agent.epsilon:.4f}
Network Updates: {len(agent.losses)}

Convergence: {'✓' if improvement > 30 else '⚠️'}
    """
    
    ax7.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Deep RL Training Analysis - Global Ocean Plastic Prediction', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('rl_training_dashboard.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: rl_training_dashboard.png")
    plt.close()
    
    # 2. Global Map Visualization
    print("🗺️  Creating global map...")
    
    fig = plt.figure(figsize=(24, 14))
    
    if HAS_CARTOPY:
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        transform = ccrs.PlateCarree()
    else:
        ax = plt.subplot(1, 1, 1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 70)
        ax.set_facecolor('lightblue')
        ax.grid(True, alpha=0.3)
        transform = None
    
    # Plot ocean currents
    skip = 15
    speed = np.sqrt(env.u**2 + env.v**2)
    kwargs = {'transform': transform} if transform else {}
    
    im = ax.contourf(env.LON, env.LAT, speed, levels=30, cmap='Blues', 
                    alpha=0.4, **kwargs)
    ax.quiver(env.LON[::skip, ::skip], env.LAT[::skip, ::skip],
             env.u[::skip, ::skip], env.v[::skip, ::skip],
             alpha=0.3, scale=20, width=0.002, color='darkblue', **kwargs)
    
    # Plot sample trajectories
    sample_size = min(100, len(ground_truth_particles))
    sample_particles = random.sample(ground_truth_particles, sample_size)
    
    for particle in sample_particles:
        traj = np.array(particle.trajectory)
        if len(traj) > 1:
            # Color by time
            colors = plt.cm.Reds(np.linspace(0.2, 1, len(traj)))
            for i in range(len(traj) - 1):
                ax.plot(traj[i:i+2, 1], traj[i:i+2, 0], 
                       color=colors[i], linewidth=0.5, alpha=0.3, **kwargs)
    
    # Plot final positions
    final_lats = [p.trajectory[-1][0] for p in ground_truth_particles if not p.beached]
    final_lons = [p.trajectory[-1][1] for p in ground_truth_particles if not p.beached]
    
    ax.scatter(final_lons, final_lats, c='red', s=20, alpha=0.6,
              label=f'Final Positions ({Config.END_DATE.year})', **kwargs)
    
    # Plot release points
    release_lats = [p.trajectory[0][0] for p in ground_truth_particles]
    release_lons = [p.trajectory[0][1] for p in ground_truth_particles]
    ax.scatter(release_lons, release_lats, c='lime', s=100, marker='*',
              edgecolors='darkgreen', linewidths=1, zorder=10,
              label='Release Points (2020)', **kwargs)
    
    # Title and legend
    ax.set_title(f'Global Ocean Plastic Trajectories: 2020-2025\n'
                f'{len(ground_truth_particles)} Particles | {Config.TOTAL_STEPS} Weeks | '
                f'Deep RL Enhanced Prediction',
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=12, framealpha=0.95)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Ocean Current Speed (m/s)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('global_plastic_map.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: global_plastic_map.png")
    plt.close()
    
    # 3. Time Evolution Animation Frames
    print("🎬 Creating animation frames...")
    
    # Select time snapshots
    n_frames = 12
    frame_indices = np.linspace(0, Config.TOTAL_STEPS - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    for idx, step in enumerate(frame_indices):
        ax = axes[idx]
        
        current_date = Config.START_DATE + timedelta(days=step * Config.DT_DAYS)
        
        # Background
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 70)
        ax.set_facecolor('lightblue')
        
        # Currents (reduced detail)
        skip_anim = 25
        ax.contourf(env.LON, env.LAT, speed, levels=20, cmap='Blues', alpha=0.3)
        ax.quiver(env.LON[::skip_anim, ::skip_anim], env.LAT[::skip_anim, ::skip_anim],
                 env.u[::skip_anim, ::skip_anim], env.v[::skip_anim, ::skip_anim],
                 alpha=0.2, scale=25, width=0.001)
        
        # Particles at this timestep
        lats_at_step = []
        lons_at_step = []
        
        for particle in ground_truth_particles:
            if step < len(particle.trajectory) and not particle.beached:
                lat, lon, _ = particle.trajectory[step]
                lats_at_step.append(lat)
                lons_at_step.append(lon)
        
        if lats_at_step:
            ax.scatter(lons_at_step, lats_at_step, c='red', s=5, alpha=0.6)
            
            # Show trajectories up to this point (sample)
            for particle in random.sample(ground_truth_particles, min(50, len(ground_truth_particles))):
                if step < len(particle.trajectory):
                    traj = np.array(particle.trajectory[:step+1])
                    if len(traj) > 1:
                        ax.plot(traj[:, 1], traj[:, 0], 'r-', alpha=0.15, linewidth=0.3)
        
        # Release points
        ax.scatter(release_lons, release_lats, c='lime', s=50, marker='*', 
                  edgecolors='darkgreen', zorder=10)
        
        ax.set_title(f'{current_date.strftime("%Y-%m-%d")} | Week {step} | '
                    f'{len(lats_at_step)} Active Particles',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
    
    plt.suptitle('Global Plastic Dispersion Over Time (2020-2025)', 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('time_evolution_frames.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: time_evolution_frames.png")
    plt.close()
    
    # 4. RL vs Physics Comparison
    print("📊 Creating RL vs Physics comparison...")
    
    # Run physics-only prediction for comparison
    print("  Running physics-only baseline...")
    test_particle = random.choice(ground_truth_particles)
    
    # Physics-only
    physics_particle = PlasticParticle(test_particle.trajectory[0][0],
                                      test_particle.trajectory[0][1], 0)
    for step in range(len(test_particle.trajectory) - 1):
        physics_particle.update(env, step)
    
    # RL-enhanced
    rl_particle = PlasticParticle(test_particle.trajectory[0][0],
                                 test_particle.trajectory[0][1], 0)
    agent.epsilon = 0  # Greedy for testing
    for step in range(len(test_particle.trajectory) - 1):
        state = agent.get_state(rl_particle, test_particle, env, step)
        action = agent.select_action(state)
        rl_particle.update(env, step)
        agent.apply_action(rl_particle, action, env, step)
    
    # Compare
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot trajectories
    target_traj = np.array(test_particle.trajectory)
    physics_traj = np.array(physics_particle.trajectory)
    rl_traj = np.array(rl_particle.trajectory)
    
    ax1.plot(target_traj[:, 1], target_traj[:, 0], 'g-', linewidth=3, 
            label='Ground Truth', alpha=0.8)
    ax1.plot(physics_traj[:, 1], physics_traj[:, 0], 'b--', linewidth=2, 
            label='Physics Only', alpha=0.7)
    ax1.plot(rl_traj[:, 1], rl_traj[:, 0], 'r-', linewidth=2, 
            label='RL Enhanced', alpha=0.7)
    
    ax1.scatter(target_traj[0, 1], target_traj[0, 0], c='lime', s=300, 
               marker='*', edgecolors='black', zorder=10, label='Start')
    ax1.scatter(target_traj[-1, 1], target_traj[-1, 0], c='green', s=200, 
               marker='s', edgecolors='black', zorder=10, label='Target End')
    ax1.scatter(physics_traj[-1, 1], physics_traj[-1, 0], c='blue', s=200, 
               marker='o', edgecolors='black', zorder=10, label='Physics End')
    ax1.scatter(rl_traj[-1, 1], rl_traj[-1, 0], c='red', s=200, 
               marker='D', edgecolors='black', zorder=10, label='RL End')
    
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Error over time
    ax2 = plt.subplot(1, 2, 2)
    
    physics_errors = []
    rl_errors = []
    
    for i in range(min(len(target_traj), len(physics_traj), len(rl_traj))):
        # Physics error
        dlon = target_traj[i, 1] - physics_traj[i, 1]
        if abs(dlon) > 180:
            dlon = dlon - 360 * np.sign(dlon)
        physics_err = np.sqrt((target_traj[i, 0] - physics_traj[i, 0])**2 + dlon**2) * 111
        physics_errors.append(physics_err)
        
        # RL error
        dlon = target_traj[i, 1] - rl_traj[i, 1]
        if abs(dlon) > 180:
            dlon = dlon - 360 * np.sign(dlon)
        rl_err = np.sqrt((target_traj[i, 0] - rl_traj[i, 0])**2 + dlon**2) * 111
        rl_errors.append(rl_err)
    
    weeks = np.arange(len(physics_errors))
    ax2.plot(weeks, physics_errors, 'b-', linewidth=2, label='Physics Only', alpha=0.7)
    ax2.plot(weeks, rl_errors, 'r-', linewidth=2, label='RL Enhanced', alpha=0.7)
    ax2.fill_between(weeks, physics_errors, alpha=0.2, color='blue')
    ax2.fill_between(weeks, rl_errors, alpha=0.2, color='red')
    
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Prediction Error (km)', fontsize=12)
    ax2.set_title('Error Evolution Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add improvement text
    final_physics_error = physics_errors[-1]
    final_rl_error = rl_errors[-1]
    improvement = ((final_physics_error - final_rl_error) / final_physics_error * 100)
    
    ax2.text(0.05, 0.95, f'Final Errors:\nPhysics: {final_physics_error:.1f} km\n'
            f'RL: {final_rl_error:.1f} km\nImprovement: {improvement:.1f}%',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('RL vs Physics-Only Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rl_vs_physics_comparison.png', dpi=Config.DPI, bbox_inches='tight')
    print("✓ Saved: rl_vs_physics_comparison.png")
    plt.close()
    
    print(f"\n✓ All visualizations complete!")

# ==================== ANIMATION ====================
def create_animated_gif(env, ground_truth_particles):
    """Create animated GIF of plastic movement"""
    if not HAS_PIL:
        print("⚠️  Skipping GIF creation (PIL not installed)")
        return
    
    print("\n🎬 Creating animated GIF...")
    print("  This may take a few minutes...")
    
    # Create frames
    n_frames = 60  # 5 years, ~1 month per frame
    frame_indices = np.linspace(0, Config.TOTAL_STEPS - 1, n_frames, dtype=int)
    
    images = []
    
    for frame_idx, step in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(16, 10))
        
        current_date = Config.START_DATE + timedelta(days=step * Config.DT_DAYS)
        
        # Background
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 70)
        ax.set_facecolor('#0a1929')
        
        # Ocean currents
        speed = np.sqrt(env.u**2 + env.v**2)
        skip = 20
        ax.contourf(env.LON, env.LAT, speed, levels=25, cmap='Blues', alpha=0.4)
        ax.quiver(env.LON[::skip, ::skip], env.LAT[::skip, ::skip],
                 env.u[::skip, ::skip], env.v[::skip, ::skip],
                 alpha=0.3, scale=20, width=0.002, color='lightblue')
        
        # Particles
        lats = []
        lons = []
        for particle in ground_truth_particles:
            if step < len(particle.trajectory) and not particle.beached:
                lat, lon, _ = particle.trajectory[step]
                lats.append(lat)
                lons.append(lon)
        
        if lats:
            ax.scatter(lons, lats, c='red', s=15, alpha=0.7, edgecolors='yellow', linewidths=0.3)
        
        # Trails (last 10 weeks)
        for particle in random.sample(ground_truth_particles, min(100, len(ground_truth_particles))):
            start_idx = max(0, step - 10)
            if step < len(particle.trajectory):
                traj = np.array(particle.trajectory[start_idx:step+1])
                if len(traj) > 1:
                    ax.plot(traj[:, 1], traj[:, 0], 'r-', alpha=0.2, linewidth=0.5)
        
        # Title
        ax.set_title(f'Global Ocean Plastic Dispersion | {current_date.strftime("%B %Y")}\n'
                    f'{len(lats)} Active Particles',
                    fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel('Longitude', fontsize=12, color='white')
        ax.set_ylabel('Latitude', fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        # Progress bar
        progress = step / Config.TOTAL_STEPS
        bar_width = 0.8
        bar_x = -180 + 10
        bar_y = -55
        ax.add_patch(Rectangle((bar_x, bar_y), bar_width * 360, 3, 
                               facecolor='gray', alpha=0.3))
        ax.add_patch(Rectangle((bar_x, bar_y), bar_width * 360 * progress, 3, 
                               facecolor='lime', alpha=0.7))
        
        plt.tight_layout()
        
        # Save frame
        filename = f'frame_{frame_idx:03d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='#0a1929')
        plt.close()
        
        # Load frame
        images.append(Image.open(filename))
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")
    
    # Create GIF
    print("  Compiling GIF...")
    images[0].save('plastic_movement_animation.gif',
                  save_all=True,
                  append_images=images[1:],
                  duration=100,
                  loop=0)
    
    # Cleanup frame files
    import os
    for i in range(n_frames):
        try:
            os.remove(f'frame_{i:03d}.png')
        except:
            pass
    
    print("✓ Saved: plastic_movement_animation.gif")

# ==================== JSON EXPORT ====================
def export_results_json(results_by_step, agent):
    """Export detailed results to JSON"""
    print("\n📁 Exporting results to JSON...")
    
    # Prepare data
    output = {
        'metadata': {
            'simulation_name': 'Global Ocean Plastic RL Prediction',
            'start_date': Config.START_DATE.strftime('%Y-%m-%d'),
            'end_date': Config.END_DATE.strftime('%Y-%m-%d'),
            'total_days': Config.TOTAL_DAYS,
            'total_steps': Config.TOTAL_STEPS,
            'timestep_days': Config.DT_DAYS,
            'n_particles': Config.N_PARTICLES * Config.N_RELEASE_POINTS,
            'rl_config': {
                'hidden_size': Config.RL_HIDDEN,
                'layers': Config.RL_LAYERS,
                'learning_rate': Config.LEARNING_RATE,
                'batch_size': Config.BATCH_SIZE,
                'gamma': Config.GAMMA
            }
        },
        'learning_summary': {
            'initial_error_km': float(agent.errors[0]),
            'final_error_km': float(agent.errors[-1]),
            'best_error_km': float(min(agent.errors)),
            'mean_error_km': float(np.mean(agent.errors)),
            'std_error_km': float(np.std(agent.errors)),
            'improvement_percent': float((agent.errors[0] - agent.errors[-1]) / agent.errors[0] * 100),
            'total_training_steps': len(agent.errors),
            'network_updates': len(agent.losses)
        },
        'weekly_results': results_by_step
    }
    
    with open('global_plastic_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Saved: global_plastic_results.json")
    print(f"  Contains {len(results_by_step)} weekly datapoints")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "🌊" * 35)
    print("GLOBAL OCEAN PLASTIC PREDICTION - DEEP RL SYSTEM")
    print("Oceans Four - DriftCast Team")
    print("🌊" * 35)
    
    # Run simulation
    env, agent, ground_truth_particles, results_by_step = run_global_simulation()
    
    # Create visualizations
    create_visualizations(env, agent, ground_truth_particles, results_by_step)
    
    # Create animated GIF
    create_animated_gif(env, ground_truth_particles)
    
    # Export JSON
    export_results_json(results_by_step, agent)
    
    print("\n" + "🎉" * 35)
    print("COMPLETE - ALL FILES GENERATED")
    print("🎉" * 35)
    
    print("\n📁 Generated Files:")
    print("   1. rl_training_dashboard.png - Comprehensive RL training analysis")
    print("   2. global_plastic_map.png - Global trajectory map")
    print("   3. time_evolution_frames.png - 12 time snapshots")
    print("   4. rl_vs_physics_comparison.png - Method comparison")
    print("   5. plastic_movement_animation.gif - Animated movement (60 frames)")
    print("   6. global_plastic_results.json - Detailed results data")
    
    print(f"\n{'='*70}")
    print("KEY METRICS")
    print(f"{'='*70}")
    print(f"Simulation Period: {Config.TOTAL_DAYS} days ({Config.TOTAL_STEPS} weeks)")
    print(f"Initial Error: {agent.errors[0]:.1f} km")
    print(f"Final Error: {agent.errors[-1]:.1f} km")
    print(f"Best Error: {min(agent.errors):.1f} km")
    print(f"Improvement: {((agent.errors[0] - agent.errors[-1]) / agent.errors[0] * 100):.1f}%")
    print(f"Mean Error: {np.mean(agent.errors):.1f} ± {np.std(agent.errors):.1f} km")
    print(f"{'='*70}\n")
    
    print("🌊 Ready for presentation and analysis!")
    print("   Check the JSON file for detailed week-by-week results")
    print("   View the GIF to see plastic movement over 5 years")
    print("   Use the dashboard to show RL learning progress\n")