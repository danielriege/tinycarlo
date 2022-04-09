#!/usr/bin/env python3.9

import gym
import tinycarlo

env = gym.make("tinycarlo-v0", environment="./example_tracks/segmented_track")

print(env.observation_space.shape)
print(env.action_space.shape[0])

observation = env.reset()
for episode in range(10):
    while True:
        env.render()
        action = env.action_space.sample()
        action = [1.0]
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            break
env.close()