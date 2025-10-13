from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

# --- JAX-based Physics Simulation ---
# These functions are JIT-compiled by JAX for performance.

@partial(jax.jit, static_argnames=['alpha'])
def get_physics_velocity(position, alpha=0.1):
    x, y = position
    vx = -alpha * y
    vy =  alpha * x
    return jnp.array([vx, vy])

@partial(jax.jit, static_argnames=['alpha'])
def get_true_velocity(position, bias=jnp.array([0.2, 0.0]), alpha=0.1):
    physics_vel = get_physics_velocity(position, alpha)
    return physics_vel + bias


class DriftEnv(gym.Env):
    """A custom Gym environment for the plastic drift simulation."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(DriftEnv, self).__init__()

        self.dt = 0.1
        self.max_steps = 100
        self.current_step = 0

        # Define action and observation space
        # Action: 2D correction vector. We'll use a generous range.
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        # Observation: 6D state [agent_x, agent_y, physics_vx, physics_vy, truth_x, truth_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initial positions
        self.start_pos = np.array([1.0, 1.0], dtype=np.float32)
        self._agent_location = None
        self._true_location = None

    def _get_obs(self):
        """Constructs the observation from the current state."""
        # Note: get_physics_velocity returns a JAX array, so we convert it
        physics_vel = np.array(get_physics_velocity(jnp.array(self._agent_location)))
        return np.concatenate([
            self._agent_location,
            physics_vel,
            self._true_location
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self._agent_location = self.start_pos.copy()
        self._true_location = self.start_pos.copy()
        self.current_step = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Executes one time step in the environment."""
        # Get current velocities from our JAX functions
        physics_vel = np.array(get_physics_velocity(jnp.array(self._agent_location)))
        true_vel = np.array(get_true_velocity(jnp.array(self._true_location)))

        # Update positions
        self._agent_location = self._agent_location + (physics_vel + action) * self.dt
        self._true_location = self._true_location + true_vel * self.dt

        # Calculate reward (negative squared distance, scaled)
        reward = -np.sum((self._agent_location - self._true_location)**2) * 0.01

        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False # Not using a time limit truncation

        # Get next observation
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        pass
