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
    environment=None,
    cte_shaping=None
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
        self.overview_downscale = sim_config.get('overview_downscale', 1)

        # CAR
        car_config = config.get('car', {})
        self.wheelbase = car_config.get('wheelbase', 160)
        self.track_width = car_config.get('track_width', 100)
        self.mass = 4 # not used by now
        self.T = 1/self.fps
        self.car_velocity = car_config.get('velocity', 500) / 1000 # car velocity is in m/s
        self.max_steering_change = car_config.get('max_steering_change', None)
        
        self.camera_config = config.get('cameras', [])
        ### config is handled in create_cameras() since this config is an array

        # REWARD DESIGN
        reward_config = config.get('reward_design', {})
        self.reward_obstacles = reward_config.get('color_obstacles', [])
        self.cte_shaping = cte_shaping

        cte_config = reward_config.get('cross_track_error', {})
        self.use_cte = cte_config.get('use_cte', False)
        self.trajectory_color = cte_config.get('trajectory_color', [255,255,255])

        # TRACK
        self.track_config = config.get('tracks', [])
        self.number_of_tracks = len(self.track_config)

        ########
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0
        self.last_steering_angle = 0.0

        self.track = Track(self.track_config, environment, self.overview_downscale, self.trajectory_color)
        self.car = Car(self.track, self.track_width, self.wheelbase, self.max_steering_change, self.T)

        self.reward_handler = RewardHandler(track=self.track, car=self.car, reward_obstacles=self.reward_obstacles, use_cte=self.use_cte)

        self.cameras = self.__create_cameras(self.camera_config)
        self.renderer = Renderer(self.track, self.car, self.cameras, self.overview_downscale)
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

        self.car.step(self.car_velocity, steering_angle)

        # generate new transformed track with car position in center
        self.track.transform(self.car.position, self.car.rotation)

        self.observation = {}
        for camera in self.cameras:
            self.observation[camera.id] = camera.capture_frame()

        colission = self.car.check_colission(self.reward_obstacles)
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

        self.loop_time = time.time() - start
        return self.observation, reward, self.done, info

    def reset(self):
        self.car.reset()

        self.track.transform(self.car.position, self.car.rotation)
        self.observation = {}
        for camera in self.cameras:
            self.observation[camera.id] = camera.capture_frame()

        self.last_steering_angle = 0.0
        self.done = False
        self.reward_sum = 0
        self.step_cnt = 0

        return self.observation

    def render(self, mode="human", close=False):
        camera_views = self.observation # observation is rendered camera view

        overview = self.renderer.render_overview(self.loop_time, self.reward_sum, self.step_cnt, self.car.steering_input)
        overview = cv2.resize(overview, (overview.shape[1]//3, overview.shape[0]//3))

        cv2.imshow('Overview', overview)
        for camera_id in camera_views:
            cv2.imshow(camera_id, self.observation[camera_id])

        waiting_time = self.T - self.loop_time
        if waiting_time < 0 and self.step_cnt > 0: # ignore first render since it takes more time
            print(f'Simulation is running behind. Please lower the FPS. Last step took {1/self.loop_time:.0f} FPS.')

        if waiting_time < 0.001 or self.realtime == False:
            waiting_time = 0.001
        if cv2.waitKey(int(waiting_time*1000)) & 0xFF == ord('q'):
            self.close()
            self.done = True
            self.reset()

    def close(self):
        cv2.destroyAllWindows()

    def __create_cameras(self, config):
        cameras = []
        for camera_config in config:
            resolution = camera_config.get('resolution', [480,640])
            roi = camera_config.get('roi', [480,640])
            position = camera_config.get('position', [0,0])
            id_ = camera_config.get('id', 'unkown')
            camera = Camera(self.track,self.car, resolution, roi, position, id_)
            cameras.append(camera)
        return cameras