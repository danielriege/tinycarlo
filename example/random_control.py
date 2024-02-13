import gym
import tinycarlo
import os

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./segmented_map/config.yaml")
env = gym.make("tinycarlo-v1", config=config_path)

observation = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        break
env.close()