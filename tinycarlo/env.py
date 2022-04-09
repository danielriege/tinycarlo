import math
import cv2
import numpy as np
import gym
import time
import yaml
import os

from  tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from  tinycarlo.track import Track
from tinycarlo.camera import Camera
from tinycarlo.reward_handler import RewardHandler

class TinyCarloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
    environment=None
    ):
        ####### CONFIGURATION
        # Load
        try:
            config_path = os.path.join(environment, "track.yaml")
            with open(config_path, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                    print(f'Loaded configuration file: {config_path}')
                except yaml.YAMLError as exc:
                    print(exc)
        except:
            config = {}

        # SIMULATION
        sim_config = config.get("sim", {})
        self.step_limit = sim_config.get('step_limit', None)
        self.realtime = sim_config.get('render_realtime', True)
        self.fps = sim_config.get('fps', 30)
        self.random_spawn = sim_config.get('random_spawm', True)

        # CAR
        car_config = config.get('car', {})
        self.wheelbase = car_config.get('wheelbase', 160)
        self.track_width = car_config.get('track_width', 100)
        self.mass = 4 # not used by now
        self.T = 1/self.fps
        self.car_velocity = car_config.get('velocity', 0.5)
        self.max_steering_change = car_config.get('max_steering_change', None)

        self.camera_resolution = car_config.get('camera_resolution', [480,640])

        # REWARD DESIGN
        reward_config = config.get('reward_design', {})
        self.reward_red = reward_config.get('reward_red', 'done')
        self.reward_green = reward_config.get('reward_green', -1)
        self.cte_shaping = reward_config.get('cte_shaping', None)

        ########
        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.camera_resolution + [3,], dtype=np.uint8
        )
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0
        self.last_steering_angle = 0.0

        self.track = Track()
        self.car = Car(self.track, self.track_width, self.wheelbase, self.max_steering_change, self.random_spawn, self.T)

        self.reward_handler = RewardHandler(track=self.track, car=self.car, reward_red=self.reward_red, reward_green=self.reward_green)

        self.camera = Camera(self.track, self.car, self.camera_resolution)
        self.renderer = Renderer(self.track, self.car, [self.camera])
        self.loop_time = 1

        self.reset()

    def step(self, action):
        self.step_cnt += 1

        steering_angle = action[0]
        self.last_steeing_angle = steering_angle

        self.car.step(self.car_velocity, steering_angle)

        # generate new transformed track with car position in center
        self.track.transform(self.car.position, self.car.rotation)

        self.observation = self.camera.capture_frame()
        colission = self.car.check_colission()
        # calculate reward
        reward = self.reward_handler.calc_reward(colission, shaping=self.cte_shaping)
        if reward == 'done':
            reward = 0
            self.done = True
        else:
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

        overview = self.renderer.render_overview(self.loop_time, self.reward_sum, self.step_cnt, self.car.steering_input)
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

    ## Additional info about env

    def get_cones(self):
        return self.track.get_cones()
