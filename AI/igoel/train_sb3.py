import matplotlib.pyplot as plt
import numpy as np
from drift_env import DriftEnv, get_physics_velocity, get_true_velocity
from stable_baselines3 import PPO


# --- 1. Training ---
def train_agent():
    """Instantiate the environment and train the PPO agent."""
    print("Creating environment...")
    env = DriftEnv()

    # Instantiate the PPO model
    # "MlpPolicy" means it will use a Multi-Layer Perceptron network.
    # verbose=1 will print training logs.
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drift_tensorboard/")

    # Train the model
    # 50,000 timesteps should be enough for this simple problem.
    print("Starting training...")
    model.learn(total_timesteps=50000)

    # Save the trained model
    print("Saving model...")
    model.save("ppo_drift_model")

    return model, env

# --- 2. Evaluation & Visualization ---
def plot_results(model, env):
    """Evaluate the trained agent and plot the results."""
    print("Generating plot...")
    start_pos = env.start_pos
    num_steps = env.max_steps
    dt = env.dt

    # --- Generate True Trajectory ---
    true_positions = [start_pos]
    pos = start_pos.copy()
    for _ in range(num_steps):
        pos = pos + np.array(get_true_velocity(pos)) * dt
        true_positions.append(pos)
    true_positions = np.array(true_positions)

    # --- Generate Physics-Only Trajectory ---
    physics_positions = [start_pos]
    pos = start_pos.copy()
    for _ in range(num_steps):
        pos = pos + np.array(get_physics_velocity(pos)) * dt
        physics_positions.append(pos)
    physics_positions = np.array(physics_positions)

    # --- Generate RL Corrected Trajectory ---
    rl_positions = [start_pos]
    obs, _ = env.reset()
    for _ in range(num_steps):
        # Use the model to predict the action deterministically
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rl_positions.append(obs[:2]) # The first two elements of obs are the agent's position
        if done or truncated:
            break
    rl_positions = np.array(rl_positions)

    # --- Plotting ---
    plt.figure(figsize=(10, 10))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='Ground Truth')
    plt.plot(physics_positions[:, 0], physics_positions[:, 1], 'b--', label='Physics-Only')
    plt.plot(rl_positions[:, 0], rl_positions[:, 1], 'r-', label='RL Corrected (SB3 PPO)')
    plt.scatter(start_pos[0], start_pos[1], c='black', zorder=5, label='Start')
    plt.legend()
    plt.title('Trajectory Comparison (Stable-Baselines3 PPO)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    filename = "trajectory_comparison_sb3.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == '__main__':
    # Train the agent
    trained_model, drift_env = train_agent()

    # Evaluate and plot
    plot_results(trained_model, drift_env)
