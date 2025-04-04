import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cumulative_rewards(ppo_log_path="logs/ppo_eval/evaluations.npz", dqn_log_path="logs/dqn_eval/evaluations.npz"):
    plt.figure(figsize=(10, 6))
    
    # Load and plot PPO evaluation data, if available
    if os.path.exists(ppo_log_path):
        data_ppo = np.load(ppo_log_path)
        timesteps_ppo = data_ppo["timesteps"]
        results_ppo = data_ppo["results"]  # Assuming shape (n_evals, n_episodes)
        mean_rewards_ppo = results_ppo.mean(axis=1)
        std_rewards_ppo = results_ppo.std(axis=1)
        plt.plot(timesteps_ppo, mean_rewards_ppo, label="PPO Mean Reward", color='blue')
        plt.fill_between(timesteps_ppo, mean_rewards_ppo - std_rewards_ppo, mean_rewards_ppo + std_rewards_ppo, alpha=0.3, color='blue')
    else:
        print("PPO evaluation log not found. Simulating data.")
        # Simulated data for demonstration
        timesteps_ppo = np.linspace(0, 20000, 50)
        mean_rewards_ppo = np.linspace(-10, 0, 50)
        std_rewards_ppo = np.random.uniform(0.5, 1.5, 50)
        plt.plot(timesteps_ppo, mean_rewards_ppo, label="PPO Mean Reward (Simulated)", color='blue')
        plt.fill_between(timesteps_ppo, mean_rewards_ppo - std_rewards_ppo, mean_rewards_ppo + std_rewards_ppo, alpha=0.3, color='blue')
    
    # Load and plot DQN evaluation data, if available
    if os.path.exists(dqn_log_path):
        data_dqn = np.load(dqn_log_path)
        timesteps_dqn = data_dqn["timesteps"]
        results_dqn = data_dqn["results"]
        mean_rewards_dqn = results_dqn.mean(axis=1)
        std_rewards_dqn = results_dqn.std(axis=1)
        plt.plot(timesteps_dqn, mean_rewards_dqn, label="DQN Mean Reward", color='red')
        plt.fill_between(timesteps_dqn, mean_rewards_dqn - std_rewards_dqn, mean_rewards_dqn + std_rewards_dqn, alpha=0.3, color='red')
    else:
        print("DQN evaluation log not found. Simulating data.")
        # Simulated data for demonstration
        timesteps_dqn = np.linspace(0, 20000, 50)
        mean_rewards_dqn = np.linspace(-15, -5, 50)
        std_rewards_dqn = np.random.uniform(0.5, 1.5, 50)
        plt.plot(timesteps_dqn, mean_rewards_dqn, label="DQN Mean Reward (Simulated)", color='red')
        plt.fill_between(timesteps_dqn, mean_rewards_dqn - std_rewards_dqn, mean_rewards_dqn + std_rewards_dqn, alpha=0.3, color='red')
    
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Curves for PPO and DQN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cumulative_rewards.png")
    plt.show()
    print("Saved cumulative reward plot as cumulative_rewards.png")

def plot_training_stability(dqn_loss_path="logs/dqn/loss.npy", ppo_entropy_path="logs/ppo/entropy.npy"):
    plt.figure(figsize=(10, 6))
    
    # Plot DQN loss curve if available
    if os.path.exists(dqn_loss_path):
        dqn_loss = np.load(dqn_loss_path)  # Expected to be an array of losses over updates
        plt.plot(dqn_loss, label="DQN Loss", color='red')
    else:
        print("DQN loss log not found. Simulating DQN loss data.")
        # Simulated loss data
        dqn_loss = np.linspace(1.0, 0.1, 100)
        plt.plot(dqn_loss, label="DQN Loss (Simulated)", color='red')
    
    # Plot PPO policy entropy if available
    if os.path.exists(ppo_entropy_path):
        ppo_entropy = np.load(ppo_entropy_path)  # Expected to be an array of entropy values over updates
        plt.plot(ppo_entropy, label="PPO Policy Entropy", color='blue')
    else:
        print("PPO entropy log not found. Simulating PPO entropy data.")
        # Simulated entropy data
        ppo_entropy = np.linspace(0.5, 0.1, 100)
        plt.plot(ppo_entropy, label="PPO Policy Entropy (Simulated)", color='blue')
    
    plt.xlabel("Training Updates")
    plt.ylabel("Value")
    plt.title("Training Stability Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_stability.png")
    plt.show()
    print("Saved training stability plot as training_stability.png")

if __name__ == "__main__":
    plot_cumulative_rewards()
    plot_training_stability()
