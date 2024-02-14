import gymnasium as gym
import tinycarlo
import os

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./segmented_map/config.yaml")
env = gym.make("tinycarlo-v2", config=config_path, render_mode="human")

observation, info = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()