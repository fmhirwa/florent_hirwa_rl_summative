# main.py

from environment.custom_env import UrbanSkinExposureEnv

env = UrbanSkinExposureEnv()
obs = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Total reward: {total_reward}")
