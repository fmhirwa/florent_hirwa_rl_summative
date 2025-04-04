import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO  # Or use DQN if preferred
from environment.custom_env import UrbanSkinExposureEnv

# Initialize environment and load your PPO model
env = UrbanSkinExposureEnv()
model = PPO.load("models/ppo/ppo_model")  # Change to DQN.load if desired

# Reset environment and record the trajectory
obs, _ = env.reset()
positions = [env.agent_pos.copy()]  # Record starting position

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    positions.append(env.agent_pos.copy())

# Create a matplotlib animation
fig, ax = plt.subplots(figsize=(6, 6))

# Display the UVB grid as the background
uvb_img = env.uvb_grid
im = ax.imshow(uvb_img, cmap='coolwarm', origin='upper')
plt.colorbar(im, ax=ax, label="Normalized UVB")

# Mark the goal location on the grid
goal_text = ax.text(env.goal[1], env.goal[0], "G", color="green",
                    fontsize=18, ha="center", va="center", fontweight="bold")

# Create a scatter plot for the agent's position
agent_scatter = ax.scatter([], [], s=200, c='blue', marker='o', label="Agent")

# Set grid details
ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.set_title("Agent Navigation Simulation")
ax.legend(loc="upper right")

def init():
    # Set offsets as an empty 2D array with shape (0,2)
    agent_scatter.set_offsets(np.empty((0, 2)))
    return im, agent_scatter, goal_text

def update(frame_idx):
    pos = positions[frame_idx]
    # Note: imshow uses (col, row) ordering for coordinates.
    agent_scatter.set_offsets([[pos[1], pos[0]]])
    ax.set_title(f"Step {frame_idx}")
    return im, agent_scatter, goal_text

# Create the animation (adjust interval for speed; 500ms per frame here)
ani = animation.FuncAnimation(fig, update, frames=len(positions),
                              init_func=init, blit=True, interval=500)

plt.tight_layout()
plt.show()

# Save the animation as a GIF (requires ImageMagick or pillow)
try:
    ani.save("agent_navigation.gif", writer='imagemagick', fps=2)
    print("✅ Animation saved as agent_navigation.gif")
except Exception as e:
    print("⚠️ GIF saving failed, missing ImageMagick or use a different writer.")
    print(e)
