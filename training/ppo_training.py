# training/ppo_training.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from environment.custom_env import UrbanSkinExposureEnv
from stable_baselines3.common.env_checker import check_env

def train_ppo(model_path="models/ppo/ppo_model", total_timesteps=20000):
    env = UrbanSkinExposureEnv()

    # Optional check
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=256,
        batch_size=64,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/ppo/"
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\n✅ PPO model saved to {model_path}")

if __name__ == "__main__":
    train_ppo()
