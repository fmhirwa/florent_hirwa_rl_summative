import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN, PPO
from environment.custom_env import UrbanSkinExposureEnv

# Initialize separate environments for each model (to record independent trajectories)
env_dqn = UrbanSkinExposureEnv()
env_ppo = UrbanSkinExposureEnv()

# Load both models
model_dqn = DQN.load("models/dqn/dqn_model")
model_ppo = PPO.load("models/ppo/ppo_model")

# Reset environments and record trajectories
obs_dqn, _ = env_dqn.reset()
obs_ppo, _ = env_ppo.reset()
positions_dqn = [env_dqn.agent_pos.copy()]  # trajectory for DQN
positions_ppo = [env_ppo.agent_pos.copy()]    # trajectory for PPO

done_dqn = False
done_ppo = False

# Run until both episodes are done
while not (done_dqn and done_ppo):
    if not done_dqn:
        action_dqn, _ = model_dqn.predict(obs_dqn, deterministic=True)
        obs_dqn, reward_dqn, terminated_dqn, truncated_dqn, _ = env_dqn.step(action_dqn)
        done_dqn = terminated_dqn or truncated_dqn
        positions_dqn.append(env_dqn.agent_pos.copy())
    if not done_ppo:
        action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)
        obs_ppo, reward_ppo, terminated_ppo, truncated_ppo, _ = env_ppo.step(action_ppo)
        done_ppo = terminated_ppo or truncated_ppo
        positions_ppo.append(env_ppo.agent_pos.copy())

# Determine maximum steps for animation (using the longer trajectory)
max_steps = max(len(positions_dqn), len(positions_ppo))

# Create a matplotlib animation
fig, ax = plt.subplots(figsize=(6, 6))

# Display the UVB grid as the background (using one of the environments, assumed similar)
uvb_img = env_dqn.uvb_grid
im = ax.imshow(uvb_img, cmap='coolwarm', origin='upper')
plt.colorbar(im, ax=ax, label="Normalized UVB")

# Mark the goal location on the grid (same for both)
goal_text = ax.text(env_dqn.goal[1], env_dqn.goal[0], "G", color="green",
                    fontsize=18, ha="center", va="center", fontweight="bold")

# Create scatter plots for the agents' positions
agent_scatter_dqn = ax.scatter([], [], s=200, c='blue', marker='o', label="DQN Agent")
agent_scatter_ppo = ax.scatter([], [], s=200, c='red', marker='o', label="PPO Agent")

# Always show the legend
ax.legend(loc="upper right")

# Set grid details
ax.set_xticks(np.arange(-0.5, env_dqn.grid_size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, env_dqn.grid_size, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.set_title("Agent Navigation Simulation")

def init():
    # Initialize offsets for both agents with an empty 2D array shape (0, 2)
    agent_scatter_dqn.set_offsets(np.empty((0, 2)))
    agent_scatter_ppo.set_offsets(np.empty((0, 2)))
    return im, agent_scatter_dqn, agent_scatter_ppo, goal_text

def update(frame_idx):
    # For each agent, if the trajectory is shorter than max_steps, use its last position
    pos_dqn = positions_dqn[frame_idx] if frame_idx < len(positions_dqn) else positions_dqn[-1]
    pos_ppo = positions_ppo[frame_idx] if frame_idx < len(positions_ppo) else positions_ppo[-1]
    # Note: imshow uses (col, row) ordering for coordinates.
    agent_scatter_dqn.set_offsets([[pos_dqn[1], pos_dqn[0]]])
    agent_scatter_ppo.set_offsets([[pos_ppo[1], pos_ppo[0]]])
    ax.set_title(f"Step {frame_idx}")
    return im, agent_scatter_dqn, agent_scatter_ppo, goal_text

# Create the animation (adjust interval for speed; 500ms per frame here)
ani = animation.FuncAnimation(fig, update, frames=max_steps,
                              init_func=init, blit=True, interval=500)

plt.tight_layout()
plt.show()

# Save the animation as a GIF (requires ImageMagick or pillow)
try:
    ani.save("agent_comparison_grid.gif", writer='imagemagick', fps=2)
    print("✅ Animation saved as agent_comparison.gif")
except Exception as e:
    print("⚠️ GIF saving failed, missing ImageMagick or use a different writer.")
    print(e)
