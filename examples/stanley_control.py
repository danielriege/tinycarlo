import gymnasium as gym
import tinycarlo
import os
import math

from tinycarlo.wrapper.reward import CTESparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper

config = {
    "sim": {
        "fps": 30,
        "render_realtime": True,
        "observation_space_format": "rgb", # "rgb" or "classes"
    },
    "car": {
        "wheelbase": 0.06, # distance between front and rear axle in meters
        "track_width": 0.03, # distance between the left and right wheel in meters
        "max_velocity": 0.16, # in m/s
        "max_steering_angle": 35, # in degrees
        "steering_speed": 30, # in deg/s
        "max_acceleration": 0.1, # in m/s^2
        "max_deceleration": 1 # in m/s^2
    },
    "camera": {
        "position": [0, 0, 0.03], # [x,y,z] in m relative to middle of front axle
        "orientation": [20, 0, 0], # [pitch,roll,yaw] in degrees
        "resolution": [480, 640], # [height, width] in pixels
        "fov": 90, # in degrees
        "max_range": 0.5, # in meters
        "line_thickness": 2 # in pixels
    },
    "map": {
        "json_path": os.path.join(os.path.dirname(__file__), "maps/knuffingen.json"),
        "pixel_per_meter": 266
    }
}
env = gym.make("tinycarlo-v2", config=config, render_mode="human")
env = CTESparseRewardWrapper(env, 0.002)
env = LanelineCrossingTerminationWrapper(env, ["outer"])

k = 5
speed = 0.6

observation, info = env.reset(seed=2)

while True:
    cte, heading_error = info["cte"], info["heading_error"]
    # Lateral Control with Stanley Controller
    steering_correction = math.atan2(k * cte, speed)
    steering_angle = (heading_error + steering_correction) * 180 / math.pi / config["car"]["max_steering_angle"]

    action = {"car_control": [speed, steering_angle], "maneuver": 3} # always try to turn left
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()