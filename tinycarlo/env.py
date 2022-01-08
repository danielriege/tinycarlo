import math
import cv2
import numpy as np
import gym
import time

from  tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from  tinycarlo.track import Track
from tinycarlo.camera import Camera

class TinyCarloEnv(gym.Env):
    def __init__(self, fps=30):
        ####### CONFIGURATION
        self.wheelbase = 160 # in mm
        self.track_width = 100 # in mm
        self.mass = 4 # in kg
        self.T = 1/fps

        self.camera_resolution = (640,480)

        ########

        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.camera_resolution + (3,), dtype=np.float32
        )
        self.done = False

        self.car = Car(self.track_width, self.wheelbase, self.T)
        self.track = Track()

        self.camera = Camera(self.track, self.car, self.camera_resolution)
        self.renderer = Renderer(self.track, self.car, [self.camera])
        self.loop_time = 1

        self.reset()

    def step(self, action):
        self.observation = self.camera.capture_frame()
        reward = 0
        info = None

        self.car.step(*action)

        return self.observation, reward, self.done, info

    def reset(self):
        self.car.reset()
        self.observation = np.zeros((1,1))

    def render(self, mode="human", close=False):
        start = time.time()
        camera_view = self.observation # observation is rendered camera view

        overview = self.renderer.render_overview(self.loop_time)
        overview = cv2.resize(overview, (overview.shape[1]//3, overview.shape[0]//3))

        cv2.imshow('Overview', overview)
        cv2.imshow('Front Camera', camera_view)

        self.loop_time = time.time() - start
        waiting_time = self.T - self.loop_time
        if waiting_time < 0.001:
            waiting_time = 0.001
        if cv2.waitKey(int(waiting_time*1000)) & 0xFF == ord('q'):
            self.close()
            self.done = True

    def close(self):
        cv2.destroyAllWindows()
