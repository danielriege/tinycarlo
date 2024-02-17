import gymnasium as gym
import tinycarlo
import os
import math

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./segmented_map/config.yaml")
env = gym.make("tinycarlo-v2", config=config_path, render_mode="human")

k = 10
max_speed = 0.6
max_steering_angle = 35

ticks_left_on_wait = 120 # 30 ticks = 1 second

observation, info = env.reset(seed=5)

while True:
    cte, heading_error, waitline_distance = info["cte"], info["heading_error"], info["car_distances"]["wait"]
    
    # Longitudinal Control by only looking at the distance to the next waitline
    speed = min(max_speed, 1.4 * waitline_distance)
    if waitline_distance > 0.5:
        ticks_left_on_wait = 120
    if ticks_left_on_wait == 0:
        speed = max_speed
    if speed < 0.08:
        print("Waiting at the stop line")
        ticks_left_on_wait -= 1
        if ticks_left_on_wait <= 0:
            speed = max_speed
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