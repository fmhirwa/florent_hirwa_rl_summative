# environment/custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.load_uvb_data import load_uvb_ascii

class UrbanSkinExposureEnv(gym.Env):

    """
    Custom RL environment simulating UVB exposure in a polluted urban area.
    The agent must reach a skincare center while minimizing UVB damage.
    """

    def __init__(self, grid_size=11, uvb_path="data/56461_UVB3_Mean_UV-B_of_Highest_Month.asc"):
        super(UrbanSkinExposureEnv, self).__init__()

        self.grid_size = grid_size
        self.uvb_grid = load_uvb_ascii(uvb_path, crop_start=(100, 100), crop_size=(grid_size, grid_size))
        self.max_steps = grid_size * 10  # simple limit

        self.action_space = spaces.Discrete(5)  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Wait
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3,),  # x, y (normalized), uvb_level
            dtype=np.float32
        )

        self.agent_pos = None
        self.goal = (grid_size //2, grid_size //2)
        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self._get_obs(), {}
    """
    def reset(self):
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and x > 0:             # Up
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:           # Left
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Right
            y += 1
        # action == 4 is Wait: do nothing

        self.agent_pos = [x, y]
        self.current_step += 1

        uvb_level = self.uvb_grid[x, y]
        done = self.agent_pos == list(self.goal) or self.current_step >= self.max_steps

        reward = -uvb_level  # UVB exposure penalty
        if self.agent_pos == list(self.goal):
            reward += 1.0  # Bonus for reaching the goal

        obs = self._get_obs()
        return obs, reward, done, {}
    """
    def step(self, action):
        # Store old position and compute its distance to the goal
        old_pos = self.agent_pos.copy()
        old_distance = np.linalg.norm(np.array(old_pos) - np.array(self.goal))

        # Process the action and update the agent's position
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        elif action == 4:
            # Wait action: no movement
            pass

        self.agent_pos = [x, y]
        self.current_step += 1

        # Compute new distance from the goal after moving
        new_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))
        # Calculate progress reward (positive if moving closer, negative if moving away)
        progress_reward = (old_distance - new_distance) * 0.1  # Adjust the factor as needed

        # Scale the UVB value to amplify variation in rewards
        scaling_factor = 5.0  # Adjust this factor as needed
        uvb_level = self.uvb_grid[x, y] * scaling_factor

        # Add a constant step penalty to discourage unnecessary moves
        step_penalty = 0.01

        # Compute the total reward: UVB penalty, step penalty, plus progress reward
        reward = -uvb_level - step_penalty + progress_reward

        # Check termination conditions
        terminated = self.agent_pos == list(self.goal)
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += 5.0  # Increased bonus for reaching the goal

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info


    
    """
    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        # action 4 = Wait (no movement)

        self.agent_pos = [x, y]
        self.current_step += 1

        uvb_level = self.uvb_grid[x, y]
        reward = -uvb_level

        terminated = self.agent_pos == list(self.goal)  # reached goal
        truncated = self.current_step >= self.max_steps  # timeout

        if terminated:
            reward += 1.0  # goal bonus

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info
    """

    def _get_obs(self):
        x, y = self.agent_pos
        uvb = self.uvb_grid[x, y]
        return np.array([x / self.grid_size, y / self.grid_size, uvb], dtype=np.float32)

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        x, y = self.agent_pos
        gx, gy = self.goal
        grid[gx, gy] = "G"
        grid[x, y] = "A"
        print("\n".join([" ".join(row) for row in grid]))
        print()  # blank line
