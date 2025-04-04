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
        print(f"{name} Position: {env.agent_pos}, Action: {action}")

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

    from PIL import Image, ImageDraw, ImageFont

def text_to_image(lines, font_size=16):
    font = ImageFont.load_default()
    max_width = max([len(line) for line in lines]) * font_size // 2
    img = Image.new("RGB", (max_width, font_size * len(lines)), color="white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((5, i * font_size), line, fill="black", font=font)
    return img

# Generate GIF frames
gif_frames = []
for i in range(max_steps):
    combined = ["DQN:           PPO:"]
    dqn_row = frames_dqn[i]
    ppo_row = frames_ppo[i]
    for d, p in zip(dqn_row, ppo_row):
        combined.append(f"{d}      {p}")
    gif_frame = text_to_image(combined)
    gif_frames.append(gif_frame)

# Save GIF
gif_frames[0].save("agent_comparison.gif", save_all=True, append_images=gif_frames[1:], duration=500, loop=0)
print("✅ agent_comparison.gif saved.")

