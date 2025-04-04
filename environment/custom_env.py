import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.load_uvb_data import load_uvb_ascii

class UrbanSkinExposureEnv(gym.Env):
    """
    Custom RL environment simulating UVB exposure in a polluted urban area.
    The agent must reach the cell with the lowest UVB (the safest area) while minimizing UVB damage.
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

        # Set goal as the position with the lowest UVB in the grid
        goal_idx = np.unravel_index(np.argmin(self.uvb_grid, axis=None), self.uvb_grid.shape)
        self.goal = goal_idx  # e.g., (row, col) position with lowest UVB

        self.agent_pos = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose a random starting position that is not the goal.
        while True:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if tuple(pos) != self.goal:
                self.agent_pos = pos
                break
        self.current_step = 0
        return self._get_obs(), {}

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
        elif action == 4:
            # Wait action: no movement
            pass

        self.agent_pos = [x, y]
        self.current_step += 1

        # Scale the UVB value to amplify variation in rewards
        scaling_factor = 5.0  # Adjust this factor as needed
        uvb_level = self.uvb_grid[x, y] * scaling_factor

        # Add a constant step penalty to discourage unnecessary moves
        step_penalty = 0.01

        # Compute reward: negative penalty for UVB exposure plus step penalty
        reward = -uvb_level - step_penalty

        # Check termination conditions: goal reached or max steps exceeded
        terminated = self.agent_pos == list(self.goal)
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += 5.0  # Increased bonus for reaching the goal

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

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
