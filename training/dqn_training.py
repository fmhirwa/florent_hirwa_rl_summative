# training/dqn_training.py


import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from environment.custom_env import UrbanSkinExposureEnv
import os

# Fix for no module named environment
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import UrbanSkinExposureEnv


def train_dqn(model_path="models/dqn/dqn_model", total_timesteps=5000):
    env = UrbanSkinExposureEnv()

    # Optional: check environment for bugs
    check_env(env, warn=True)

    # Testing HyperParameters 3 configs

    """
    # Base Config hyperparameters

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        gamma=0.95,
        batch_size=32,
        buffer_size=10000,
        exploration_fraction=0.2,
        verbose=1,
        tensorboard_log="./logs/dqn/"
    )

    # Config 1: Slower Learning, More Exploration
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        gamma=0.90,
        batch_size=64,
        buffer_size=20000,
        exploration_fraction=0.3,
        exploration_final_eps=0.1,
        verbose=1,
        tensorboard_log="./logs/dqn_config1/"
    )

    """
    # Config 2: Aggressive Learning, Faster Target Update
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.005,
        gamma=0.95,
        batch_size=32,
        buffer_size=5000,
        exploration_fraction=0.2,
        target_update_interval=500,
        verbose=1,
        tensorboard_log="./logs/dqn_config2/"
    )
    """
    # Config 3: Balanced Settings
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/dqn_config3/"
    )

    """
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n✅ DQN model saved to {model_path}")

if __name__ == "__main__":
    train_dqn()
