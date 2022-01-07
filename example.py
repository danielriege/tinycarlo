#!/usr/bin/env python3.9

import gym
import tinycarlo

env = gym.make("tinycarlo-v0", fps=60)
observation = env.reset()
while True:
    env.render()
    #action = env.action_space.sample()
    action = [0.4,-1.0]
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        break
env.close()