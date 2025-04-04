# main_run_model.py

from stable_baselines3 import DQN
from environment.custom_env import UrbanSkinExposureEnv
import time

model_path = "models/dqn/dqn_model"
env = UrbanSkinExposureEnv()
model = DQN.load(model_path)

obs, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.3)  # slows down for visual inspection
    done = terminated or truncated

print(f"\n🏁 Total reward (DQN): {total_reward}")
