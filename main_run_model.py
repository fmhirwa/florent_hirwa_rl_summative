# main_run_model.py

from stable_baselines3 import DQN, PPO
from environment.custom_env import UrbanSkinExposureEnv
import time
import numpy as np

def run_agent(model, name, env, delay=0.2):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    frames = []

    print(f"\n{name} Agent Starting...\n")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        frame = np.full((env.grid_size, env.grid_size), ".", dtype=str)
        gx, gy = env.goal
        ax, ay = env.agent_pos
        frame[gx, gy] = "G"
        frame[ax, ay] = name[0]  # "D" or "P"
        frames.append([" ".join(row) for row in frame])

        print(f"{name} Agent View:")
        print("\n".join([" ".join(row) for row in frame]))
        print()
        time.sleep(delay)

    print(f"🏁 Total reward ({name}): {total_reward:.4f}")
    return total_reward, frames

if __name__ == "__main__":
    # Load environment and both agents
    env_dqn = UrbanSkinExposureEnv()
    env_ppo = UrbanSkinExposureEnv()

    model_dqn = DQN.load("models/dqn/dqn_model")
    model_ppo = PPO.load("models/ppo/ppo_model")

    # Run both
    reward_dqn, frames_dqn = run_agent(model_dqn, "DQN", env_dqn)
    reward_ppo, frames_ppo = run_agent(model_ppo, "PPO", env_ppo)

    # Optional: compare step-by-step visually (if grid sizes are same)
    print("\n Step-by-step Comparison:\n")
    max_steps = min(len(frames_dqn), len(frames_ppo))
    for i in range(max_steps):
        print(f"Step {i+1}")
        print("DQN:")
        print(frames_dqn[i])
        print("PPO:")
        print(frames_ppo[i])
        print("-" * 30)
        time.sleep(0.5)

    print(f"\n Final Comparison:\nDQN Total Reward: {reward_dqn:.4f}\nPPO Total Reward: {reward_ppo:.4f}")
