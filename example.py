#!/usr/bin/env python3.9

import gym
import tinycarlo

env = gym.make("tinycarlo-v0")
env.set_fps(30)
observation = env.reset()
while True:
    env.render()
    #action = env.action_space.sample()
    action = [0.2,0]
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        break
env.close()