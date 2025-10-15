# Tasks Server Gives to Client:
The server creates Ocean Drift Simulation Tasks with the following structure:
```
{
  "task_id": "unique-uuid",
  "simulation_id": "batch-uuid", 
  "particle_count": 10,  // Number of particles to simulate
  "parameters": {
    "time_horizon": 1,  // Hours to simulate
    "spatial_bounds": {
      "min_lat": 25.0,
      "max_lat": 30.0, 
      "min_lon": -95.0,
      "max_lon": -85.0
    },
    "current_strength": 0.5,
    "wind_speed": 10.0,
    "priority": 1
  },
  "input_data": "hex_encoded_ocean_data",
  "deadline": "2024-01-01T12:00:00Z",
  "priority": 1
}
```
# Key Details:
* Task Type: Ocean plastic drift simulation
* Particle Count: Each task simulates 10 particles (configurable)
* Time Horizon: Typically 1-72 hours of simulation
* Spatial Bounds: Geographic region (lat/lon coordinates)
* Auto-Generated: Server automatically creates batches when queue is low
* Load Balancing: Max 5 concurrent tasks per client

# Tasks Client Returns as Results:
The client processes these tasks using Reinforcement Learning-enhanced ocean drift simulation and returns:
```
{
  "success": true,
  "results": [
    {
      "particle_id": 0,
      "initial_position": {"x": 1.0, "y": 1.0},
      "final_position": {"x": 2.5, "y": 1.8},
      "trajectory": [
        {"x": 1.0, "y": 1.0, "time_step": 0},
        {"x": 1.1, "y": 1.05, "time_step": 1},
        // ... more trajectory points
      ],
      "rl_enhanced": true,  // Whether RL model was used
      "beached": false,
      "distance_traveled": 1.85
    }
    // ... results for each particle
  ],
  "user_id": "lalith",
  "simulation_type": "rl_enhanced",  // or "physics_only"
  "metadata": {
    "particle_count": 10,
    "time_horizon": 1,
    "model_used": "/path/to/ppo_drift_model.zip"
  }
}
```

# Key Details:
* RL-Enhanced: Uses trained PPO model from AI/igoel/ for more accurate drift prediction
* Physics Fallback: Falls back to pure physics simulation if no RL model
* Trajectory Data: Complete particle movement paths over time
* Performance Metrics: Execution time, distance traveled, beaching status
* Ensemble Simulation: Adds noise for realistic particle distribution