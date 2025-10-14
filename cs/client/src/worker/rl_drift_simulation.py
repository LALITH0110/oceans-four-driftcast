#!/Users/lalith/Documents/GitHub/oceans-four-driftcast/AI/igoel/venv/bin/python
"""
RL-based ocean drift simulation for the Ocean Plastic Forecast client
Integrates the AI/igoel reinforcement learning drift environment
"""

import sys
import json
import os
import numpy as np
from pathlib import Path

# Add the AI/igoel directory to Python path
ai_igoel_path = Path(__file__).parent.parent.parent.parent.parent / "AI" / "igoel"
sys.path.insert(0, str(ai_igoel_path))

try:
    from drift_env import DriftEnv, get_physics_velocity, get_true_velocity
    from stable_baselines3 import PPO
    import jax.numpy as jnp
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print("Make sure you have installed the requirements from AI/igoel/requirements.txt", file=sys.stderr)
    sys.exit(1)

class RLDriftSimulator:
    def __init__(self, user_id="lalith"):
        self.user_id = user_id
        self.model_path = ai_igoel_path / "ppo_drift_model.zip"
        self.model = None
        self.env = None
        
        # Load the trained RL model if it exists
        if self.model_path.exists():
            try:
                self.model = PPO.load(str(self.model_path))
                self.env = DriftEnv()
                print(f"Loaded trained RL model for user {self.user_id}", file=sys.stderr)
            except Exception as e:
                print(f"Error loading RL model: {e}", file=sys.stderr)
                self.model = None
        else:
            print(f"No trained model found at {self.model_path}", file=sys.stderr)
    
    def simulate_drift(self, task_data):
        """
        Run the RL-enhanced drift simulation
        """
        try:
            # Parse task parameters
            particle_count = task_data.get('particle_count', 1000)
            time_horizon = task_data.get('parameters', {}).get('time_horizon', 72)
            spatial_bounds = task_data.get('parameters', {}).get('spatial_bounds', {})
            
            print(f"Running RL drift simulation for {particle_count} particles over {time_horizon} hours", file=sys.stderr)
            
            # If we have a trained model, use it for enhanced simulation
            if self.model and self.env:
                results = self._run_rl_enhanced_simulation(particle_count, time_horizon, spatial_bounds)
            else:
                # Fallback to physics-only simulation
                results = self._run_physics_only_simulation(particle_count, time_horizon, spatial_bounds)
            
            return {
                'success': True,
                'results': results,
                'user_id': self.user_id,
                'simulation_type': 'rl_enhanced' if self.model else 'physics_only',
                'metadata': {
                    'particle_count': particle_count,
                    'time_horizon': time_horizon,
                    'model_used': str(self.model_path) if self.model else None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'user_id': self.user_id
            }
    
    def _run_rl_enhanced_simulation(self, particle_count, time_horizon, spatial_bounds):
        """Run simulation using the trained RL model"""
        results = []
        
        # Use the RL environment for enhanced physics
        obs, _ = self.env.reset()
        
        trajectory_points = []
        
        # Run the simulation for the specified time horizon
        for step in range(min(time_horizon, self.env.max_steps)):
            # Get RL correction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Record trajectory point
            agent_pos = obs[:2]  # First two elements are agent position
            trajectory_points.append({
                'x': float(agent_pos[0]),
                'y': float(agent_pos[1]),
                'time_step': step,
                'rl_correction': action.tolist(),
                'reward': float(reward)
            })
            
            if done or truncated:
                break
        
        # Generate results for multiple particles (simulate ensemble)
        for i in range(particle_count):
            # Add some variation for ensemble simulation
            noise_scale = 0.1
            noisy_trajectory = []
            
            for point in trajectory_points:
                noisy_point = {
                    'x': point['x'] + np.random.normal(0, noise_scale),
                    'y': point['y'] + np.random.normal(0, noise_scale),
                    'time_step': point['time_step']
                }
                noisy_trajectory.append(noisy_point)
            
            results.append({
                'particle_id': i,
                'initial_position': {'x': 1.0, 'y': 1.0},  # From drift_env.py
                'final_position': noisy_trajectory[-1] if noisy_trajectory else {'x': 1.0, 'y': 1.0},
                'trajectory': noisy_trajectory,
                'rl_enhanced': True,
                'beached': False,  # Simplified for now
                'distance_traveled': self._calculate_trajectory_distance(noisy_trajectory)
            })
        
        return results
    
    def _run_physics_only_simulation(self, particle_count, time_horizon, spatial_bounds):
        """Fallback physics-only simulation"""
        results = []
        
        print("Running physics-only simulation (no RL model available)", file=sys.stderr)
        
        # Use the physics functions from drift_env.py
        dt = 0.1
        steps = min(int(time_horizon / dt), 1000)  # Limit steps for performance
        
        for i in range(particle_count):
            # Initialize particle
            position = np.array([1.0 + np.random.normal(0, 0.1), 
                               1.0 + np.random.normal(0, 0.1)])
            trajectory = [{'x': float(position[0]), 'y': float(position[1]), 'time_step': 0}]
            
            # Simulate particle movement
            for step in range(steps):
                # Get physics velocity
                physics_vel = np.array(get_physics_velocity(jnp.array(position)))
                
                # Update position
                position = position + physics_vel * dt
                
                # Record trajectory point
                if step % 10 == 0:  # Record every 10th step to reduce data
                    trajectory.append({
                        'x': float(position[0]),
                        'y': float(position[1]),
                        'time_step': step
                    })
            
            results.append({
                'particle_id': i,
                'initial_position': {'x': 1.0, 'y': 1.0},
                'final_position': {'x': float(position[0]), 'y': float(position[1])},
                'trajectory': trajectory,
                'rl_enhanced': False,
                'beached': False,
                'distance_traveled': self._calculate_trajectory_distance(trajectory)
            })
        
        return results
    
    def _calculate_trajectory_distance(self, trajectory):
        """Calculate total distance traveled along trajectory"""
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i]['x'] - trajectory[i-1]['x']
            dy = trajectory[i]['y'] - trajectory[i-1]['y']
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return float(total_distance)

def main():
    """Main function to run the simulation"""
    if len(sys.argv) != 2:
        print("Usage: python rl_drift_simulation.py <task_data_json>", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse task data from command line argument
        task_data = json.loads(sys.argv[1])
        
        # Create simulator instance
        simulator = RLDriftSimulator(user_id="lalith")
        
        # Run simulation
        result = simulator.simulate_drift(task_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'user_id': 'lalith'
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
