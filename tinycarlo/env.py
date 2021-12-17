import math

import numpy as np
import gym


class TinyCarloEnv(gym.Env):
    def __init__(self):
        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 200), dtype=np.uint8
        )

        self.wheelbase = 0.15
        self.mass = 0.2

        self.reset()

    def simulate_step(self, fwd_vel, steering_angle):
        dt = 1

        ang_vel = (math.tan(steering_angle) / self.wheelbase) * fwd_vel

        vx = fwd_vel * math.cos(self.car_rotation)
        vy = fwd_vel * math.sin(self.car_rotation)

        dx = (vx * math.cos(self.car_rotation) - vy * math.sin(self.car_rotation)) * dt
        dy = (vx * math.sin(self.car_rotation) + vy * math.cos(self.car_rotation)) * dt
        dyaw = ang_vel * dt

        self.car_position[0] += dx
        self.car_position[1] += dy
        self.car_rotation += dyaw

    def step(self, action):
        observation = None
        reward = 0
        done = False
        info = None

        self.simulate_step(*action)

        return observation, reward, done, info

    def reset(self):
        self.car_position = np.zeros((2,))
        self.car_rotation = 0.0

    def render(self, mode="human", close=False):
        pass
