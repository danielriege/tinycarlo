import gymnasium as gym
import tinycarlo
import os
import math

config = {
    "sim": {
        "fps": 30,
        "render_realtime": True,
        "observation_space_format": "rgb",
        "overview_pixel_per_meter": 100
    },
    "car": {
        "wheelbase": 0.06,
        "track_width": 0.03,
        "max_velocity": 0.0001,
        "max_steering_angle": 35,
        "steering_speed": 30,
        "max_acceleration": 0.01,
        "max_deceleration": 0.0001
    },
    "camera": {
        "position": [0, 0, 0.03],
        "orientation": [20, 0, 0],
        "resolution": [480, 640],
        "fov": 90,
        "max_range": 0.5,
        "line_thickness": 2
    },
    "map": {
        "json_path": os.path.join(os.path.dirname(__file__), "maps/knuffingen.json"),
        "pixel_per_meter": 266
    }
}
env = gym.make("tinycarlo-v2", config=config, render_mode="human")

k = 10
speed = 0.6
max_steering_angle = 35

observation, info = env.reset(seed=2)

while True:
    cte, heading_error = info["cte"], info["heading_error"]
    # Lateral Control with Stanley Controller
    # cross track steering
    steering_correction = math.atan2(k * cte, speed)
    # Total steering angle
    steering_angle = (heading_error + steering_correction) * 180 / math.pi / max_steering_angle

    action = {"car_control": [speed, steering_angle], "maneuver": 3}
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()