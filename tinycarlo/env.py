import math
import cv2
import numpy as np
import gym
import time

from  tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from  tinycarlo.track import Track
from tinycarlo.camera import Camera
from tinycarlo.reward_handler import RewardHandler

class TinyCarloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, fps=30, wheelbase=160, track_width=100, camera_resolution=(480,640), car_velocity=0.5, 
    reward_red='done', reward_green=-2, render_realtime=True):
        ####### CONFIGURATION
        self.wheelbase = wheelbase # in mm
        self.track_width = track_width # in mm
        self.mass = 4 # in kg
        self.T = 1/fps
        self.car_velocity = car_velocity

        self.step_limit = 1000
        self.camera_resolution = camera_resolution

        self.reward_red = reward_red
        self.reward_green = reward_green

        self.realtime = render_realtime

        ########
        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.camera_resolution + (3,), dtype=np.uint8
        )
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0
        self.last_steering_angle = 0.0

        self.reward_handler = RewardHandler(reward_red=self.reward_red, reward_green=self.reward_green, reward_tick=1)

        self.track = Track()
        self.car = Car(self.track, self.track_width, self.wheelbase, self.T)

        self.camera = Camera(self.track, self.car, self.camera_resolution)
        self.renderer = Renderer(self.track, self.car, [self.camera])
        self.loop_time = 1

        self.reset()

    def step(self, action):
        self.step_cnt += 1

        # reduce jitter by punishing for big steering angle swings
        steering_angle = action[0]
        steering_angle_diff = self.last_steering_angle - steering_angle
        jitter_reward = -round(abs(steering_angle_diff / 2)**1.5 * 10)
        self.last_steeing_angle = steering_angle

        self.car.step(self.car_velocity, steering_angle)

        # generate new transformed track with car position in center
        self.track.transform(self.car.position, self.car.rotation)

        self.observation = self.camera.capture_frame()
        colission = self.car.check_colission()
        # calculate reward
        reward = self.reward_handler.tick(colission)
        if reward == 'done':
            reward = 0
            self.done = True
        else:
            reward += jitter_reward
            self.reward_sum += reward

        info = {}

        if self.step_cnt > self.step_limit:
            self.done = True

        return self.observation, reward, self.done, info

    def reset(self):
        self.car.reset()

        self.track.transform(self.car.position, self.car.rotation)
        self.observation = self.camera.capture_frame()

        self.last_steering_angle = 0.0
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0

        return self.observation

    def render(self, mode="human", close=False):
        start = time.time()
        camera_view = self.observation # observation is rendered camera view

        overview = self.renderer.render_overview(self.loop_time, self.reward_sum, self.step_cnt)
        overview = cv2.resize(overview, (overview.shape[1]//3, overview.shape[0]//3))

        cv2.imshow('Overview', overview)
        cv2.imshow('Front Camera', camera_view)

        self.loop_time = time.time() - start
        waiting_time = self.T - self.loop_time
        if waiting_time < 0.001 or self.realtime == False:
            waiting_time = 0.001
        if cv2.waitKey(int(waiting_time*1000)) & 0xFF == ord('q'):
            self.close()
            self.done = True
            self.reset()

    def close(self):
        cv2.destroyAllWindows()
