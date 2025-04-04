# florent_hirwa_rl_summative
Summative Assignment - Reinforcement Learning

Video Recording: https://youtu.be/eoBoaxzlJOA 
GitHub Repository: https://github.com/fmhirwa/florent_hirwa_rl_summative
Report: https://docs.google.com/document/d/1rY3WCi_NxhoY0Z76KKj0xHF84FxN6k9n7N5U0nlOtTM/edit?tab=t.0

# Urban Skin Health RL Simulation

This project implements a reinforcement learning environment that simulates the challenges of urban skin aging due to UVB exposure. An agent navigates a grid representing an urban area, with the goal set to the cell with the lowest UVB (i.e., the safest area). The agent starts at a random position, and the reward function combines UVB penalties, step penalties, and progress rewards to encourage efficient navigation.

## Overview

The environment is modeled as an N×N grid where:
- **State:** A vector `[x_norm, y_norm, uvb_level]`, where `x_norm` and `y_norm` are the normalized coordinates and `uvb_level` is the UVB value at the current cell.
- **Actions:** Discrete movements—Up, Down, Left, Right, or Wait.
- **Reward Structure:** 
  - **UVB Penalty:** Proportional to the cell’s UVB value (scaled to amplify differences).
  - **Step Penalty:** A small constant penalty per move.
  - **Progress Reward:** A bonus proportional to the decrease in distance to the goal.
  - **Goal Bonus:** A significant bonus when reaching the cell with the lowest UVB.

The project compares two reinforcement learning methods:
- **Deep Q-Network (DQN)**
- **Proximal Policy Optimization (PPO)**

## Folder Structure

```
project_root/
├── data/
│   └── 56461_UVB3_Mean_UV-B_of_Highest_Month.asc
├── environment/
│   ├── __init__.py
│   ├── custom_env.py
│   └── load_uvb_data.py
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   └── ppo_training.py
├── models/
│   ├── dqn/
│   └── ppo/
├── logs/
│   ├── dqn/
│   └── ppo/
├── main_run_model.py
├── main_visualization.py
├── plot_metrics.py
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Models

- **DQN:**

  ```bash
  python -m training.dqn_training
  ```
  
  This trains the DQN model and saves it to `models/dqn/dqn_model`.

- **PPO:**

  ```bash
  python -m training.ppo_training
  ```
  
  This trains the PPO model and saves it to `models/ppo/ppo_model`.

### Running the Visualization

To view the agent trajectories and generate an animated GIF that compares both models, run:

```bash
python main_visualization.py
```

The script displays both DQN (blue) and PPO (red) agent trajectories on the same grid and saves the output as `agent_comparison.gif`.

### Plotting Metrics

Generate plots for cumulative rewards and training stability with:

```bash
python plot_metrics.py
```

This will create `cumulative_rewards.png` and `training_stability.png` in the project root.

## Environment Details

The custom environment is defined in `environment/custom_env.py` (see citeturn5file0). Key features include:

- **Dynamic Goal:** The goal is automatically set to the cell with the lowest UVB value.
- **Random Start:** The agent begins at a random position (excluding the goal).
- **Reward Shaping:** The reward combines a scaled UVB penalty, a constant step penalty, and a progress reward based on the reduction in Euclidean distance to the goal.

Example reward shaping code:

```python
def step(self, action):
    old_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))
    # Update position based on action...
    self.current_step += 1
    new_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))
    progress_reward = (old_distance - new_distance) * 0.1

    scaling_factor = 5.0
    uvb_level = self.uvb_grid[x, y] * scaling_factor
    step_penalty = 0.01
    reward = -uvb_level - step_penalty + progress_reward

    terminated = self.agent_pos == list(self.goal)
    truncated = self.current_step >= self.max_steps
    if terminated:
        reward += 5.0

    return self._get_obs(), reward, terminated, truncated, {}
```

## Hyperparameter Optimization

### DQN Hyperparameters
- **Learning Rate:** 0.001  
- **Gamma:** 0.95  
- **Replay Buffer Size:** 10,000  
- **Batch Size:** 32  
- **Exploration Strategy:** Epsilon-greedy with exploration_fraction of 0.2

### PPO Hyperparameters
- **Learning Rate:** 0.0003  
- **Gamma:** 0.99  
- **Policy-specific Parameters:** n_steps = 256, ent_coef = 0.01, clip_range = 0.2  
- **Batch Size:** 64

### Metrics Analysis
- **Cumulative Rewards:** See `cumulative_rewards.png` for reward curves.
- **Training Stability:** Loss curves for DQN and policy entropy for PPO are plotted in `training_stability.png`.
- **Episodes to Convergence:** Both methods require tuning; overall, PPO shows more consistency across episodes.
- **Generalization:** Testing on unseen initial states shows PPO’s robust generalization, despite occasional suboptimal episodes.

## Conclusion and Discussion

In our experiments, a test run showed DQN achieving a near-zero reward in one episode, while PPO had a lower reward. However, overall performance testing indicates that PPO consistently reaches the goal more often, likely due to its robust on-policy learning and entropy regularization, which encourage exploration. While DQN can sometimes outperform in individual episodes, PPO’s steady performance across multiple episodes suggests it is more reliable in this environment.

## License

None 

## Author

fmhirwa