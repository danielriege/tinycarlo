import math
import cv2
import numpy as np
import gym
from  tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from  tinycarlo.track import Track
from tinycarlo.camera import Camera

class TinyCarloEnv(gym.Env):
    def __init__(self):
        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 200), dtype=np.uint8
        )
        self.done = False

        self.wheelbase = 160 # in mm
        self.track_width = 100 # in mm
        self.mass = 6
        self.T = 0.033

        self.camera_resolution = (640,480)

        self.car = Car(self.track_width, self.wheelbase, self.T)
        self.track = Track()

        self.camera = Camera(self.track, self.car, self.camera_resolution)
        self.renderer = Renderer(self.track, self.car)

        self.reset()
    
    def set_fps(self, fps):
        self.T = 1/fps

    def step(self, action):
        self.observation = self.camera.capture_frame()
        reward = 0
        info = None

        self.car.step(*action)

        return self.observation, reward, self.done, info

    def reset(self):
        self.car.reset()
        self.observation = self.renderer.render_overview()

    def render(self, mode="human", close=False):
        camera_view = self.observation # observation is rendered camera view

        overview = self.renderer.render_overview()
        overview = cv2.resize(overview, (overview.shape[1]//3, overview.shape[0]//3))

        cv2.imshow('Overview', overview)
        cv2.imshow('Front Camera', camera_view)

        if cv2.waitKey(int(self.T*1000)) & 0xFF == ord('q'):
            self.close()
            self.done = True

    def close(self):
        cv2.destroyAllWindows()
