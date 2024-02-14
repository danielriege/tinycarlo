import cv2
import numpy as np
import time
import yaml
import os
import gymnasium as gym

from tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from tinycarlo.map import Map
from tinycarlo.camera import Camera

def getEnv(key):
    if os.environ.get(key) is not None:
        v = os.environ.get(key)
        if v.lower() == '1':
            return True
    return False

class TinyCarloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode = None, config = None):
        """
        config can be provided as either path to yaml file or as dictionary.
        render_mode can be either "human" or "rgb_array".

        human render mode will open a window and display the map and camera view.
        rgb_array render mode will return the camera view as numpy array in rgb format (independent from observation mode).
        """
        self.config_path = None
        if isinstance(config, str):
            if config.endswith(".yaml"):
                self.config_path = os.path.abspath(config)
            else:
                self.config_path = os.path.abspath(os.path.join(config, "config.yaml"))
            with open(self.config_path, "r") as stream:
                config = yaml.safe_load(stream)
                print(f'Loaded configuration file: {self.config_path}')

        """
        Setting up variables needed for simulation including car, map, camera, reward handler, etc.
        """
        self.fps = config['sim'].get('fps', 30)
        self.T = 1/self.fps
        self.render_realtime = config['sim'].get('render_realtime', False)

        self.map = Map(config['maps'], base_path=self.config_path)
        self.car = Car(self.T, self.map, config['car'])

        self.camera = Camera(self.map, self.car, config['camera'])
        self.renderer = Renderer(self.map, self.car, None)
        self.loop_time = 1
        self.window = None # for human rendering only
        self.window_camera = None # for human rendering only

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        Gym specific setup of action and observation space
        """
        # action space: (velocity, steering angle)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))
        # observation space: camera views
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.camera.resolution + [3,], dtype=np.uint8)

        self.reset()

    def __get_obs(self):
        return self.camera.capture_frame()
    
    def __get_info(self):
        return {}
    
    def __default_reward(self, cte):
        """
        Calculates a default reward solely based on cte with a linear shaping.

        Car track width is used as reference for the shaping. linear function reaches 0 when cte == track_width
        """
        return max((-1/self.car.track_width) * cte + 1, 0)

    def __default_termination(self, cte):
        """
        Default termination condition based on cte.
        """
        return cte > (self.car.track_width * 4)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # reset car position to random spawn point
        self.car.reset(self.np_random)

        observation = self.__get_obs()
        info = self.__get_info()

        if self.render_mode == "human":
            self.__render_frame()

        return observation, info

    def step(self, action):
        start = time.time()

        self.car.step(0.0001, action[0])

        observation = self.__get_obs()
        info = self.__get_info()

        # Cross Track Error
        cte = 0

        """
        This is the default reward and termination condition.
        To change reward and termination, use tinycarlo.wrappers or define a custom wrapper.
        info can be useful to calculate more complex rewards or termination conditions.
        """
        reward = self.__default_reward(cte)
        terminated = self.__default_termination(cte)

        if self.render_mode == "human":
            self.__render_frame()

        # for debugging performance
        self.loop_time = time.time() - start
        if getEnv("DEBUG"):
            print(f"Step time: {self.loop_time/1000:.6f} ms")

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.__render_frame()

    def __render_frame(self):
        if self.window is None and self.render_mode == "human":
            self.window = "Map"
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        if self.window_camera is None and self.render_mode == "human":
            self.window_camera = "Camera"
            cv2.namedWindow(self.window_camera, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
        camera_view = self.camera.get_last_frame()

        if self.render_mode == "human":
            overview = self.renderer.render_overview()

            cv2.imshow(self.window, overview)
            cv2.imshow(self.window_camera, camera_view)
        
            waiting_time = self.T - self.loop_time
            if waiting_time < 0.001 or self.render_realtime == False:
                waiting_time = 0.001
            cv2.waitKey(int(waiting_time*1000))
        else:
            return camera_view
        
    def close(self):
        if self.window is not None:
            cv2.destroyWindow(self.window)
        if self.window_camera is not None:
            cv2.destroyWindow(self.window_camera)