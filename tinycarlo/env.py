import math
import cv2
import numpy as np
import gym
import time
import yaml
import os

from tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from tinycarlo.Map import Map
from tinycarlo.camera import Camera
from tinycarlo.reward_handler import RewardHandler

def getEnv(key):
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v.lower() == '1':
            return True
    return False

class TinyCarloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    config can be provided as either path to yaml file or as dictionary
    """
    def __init__(self, config = None):
        self.config_path = None
        if isinstance(config, str):
            try:
                self.config_path = os.path.abspath(os.path.join(config, "config.yaml"))
                with open(self.config_path, "r") as stream:
                    try:
                        config = yaml.safe_load(stream)
                        print(f'Loaded configuration file: {self.config_path}')
                    except yaml.YAMLError as exc:
                        print(exc)
                        exit()
            except:
                print("Error: Could not load config file. Please provide a valid path to a config file.")
                exit()

        self.T = 1/config['sim'].get('fps', 30) # time per frame
        self.step_limit = config['sim'].get('step_limit', 1000)
        self.render_realtime = config['sim'].get('render_realtime', False)
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0
        self.last_steering_angle = 0.0

        self.map = Map(config['maps'], base_path=self.config_path)
        self.car = Car(self.T, self.map, config['car'])

        self.reward_handler = RewardHandler(track=self.map, car=self.car)

        self.cameras = self.__create_cameras(config['cameras'])
        self.renderer = Renderer(self.map, self.car, None)
        self.loop_time = 1

        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        observation_spaces = {}
        for camera in self.cameras:
            observation_spaces[camera.id] = gym.spaces.Box(low=0, high=255, shape=camera.resolution + [3,], dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(observation_spaces)

        self.reset()

    def step(self, action):
        start = time.time()

        self.step_cnt += 1

        steering_angle = action[0]
        self.last_steeing_angle = steering_angle

        self.car.step(0.0001, steering_angle)

        # generate new transformed track with car position in center
        #self.track.transform(self.car.position, self.car.rotation)

        self.observation = {}
        for camera in self.cameras:
            self.observation[camera.id] = camera.capture_frame()

        #colission = self.car.check_colission(self.reward_obstacles)
        # calculate reward
        #reward = self.reward_handler(colission)
        reward = 0 
        if reward == 'done':
            reward = 0
            self.done = True
        else:
            self.reward_sum += reward

        info = {}

        if self.step_cnt > self.step_limit:
            self.done = True

        self.loop_time = time.time() - start
        if getEnv("DEBUG"):
            print(f"Step time: {self.loop_time/1000:.6f} ms")
        return self.observation, reward, self.done, info

    def reset(self):
        self.car.reset()

        #self.map.transform(self.car.position, self.car.rotation)
        self.observation = {}
        for camera in self.cameras:
            self.observation[camera.id] = camera.capture_frame()

        self.last_steering_angle = 0.0
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0

        return self.observation

    def render(self, mode="human", close=False):
        cv2.namedWindow('Map', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
        camera_views = self.observation # observation is rendered camera view

        overview = self.renderer.render_overview()
        overview = cv2.resize(overview, (overview.shape[1]//3, overview.shape[0]//3))

        cv2.imshow('Map', overview)
        for camera_id in camera_views:
            cv2.imshow(camera_id, self.observation[camera_id])

        waiting_time = self.T - self.loop_time
        if waiting_time < 0.001 or self.render_realtime == False:
            waiting_time = 0.001
        if cv2.waitKey(int(waiting_time*1000)) & 0xFF == ord('q'):
            self.done = True
            self.reset()

    def close(self):
        cv2.destroyAllWindows()

    def __create_cameras(self, config):
        cameras = []
        for camera_config in config:
            camera = Camera(self.map, self.car, camera_config)
            cameras.append(camera)
        return cameras