import gymnasium as gym
import tinycarlo
import os
import math

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./segmented_map/config.yaml")
env = gym.make("tinycarlo-v2", config=config_path, render_mode="human")

k = 10
speed = 0.6
max_steering_angle = 35

observation, info = env.reset(seed=11)

while True:
    cte, heading_error = info["cte"], info["heading_error"]
    # Lateral Control with Stanley Controller
    # cross track steering
    steering_correction = math.atan2(k * cte, speed)
    # Total steering angle
    steering_angle = (heading_error + steering_correction) * 180 / math.pi / max_steering_angle

    action = {"car_control": [speed, steering_angle], "maneuver": 0}
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()