import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device, GlobalCounters
from typing import Tuple
from tinycarlo.wrapper import CTELinearRewardWrapper, LanelineSparseRewardWrapper, CTETerminationWrapper

import os
import numpy as np
from tqdm import trange
import time
import cv2
import math

from examples.models.tinycar_net import TinycarCombo

IMAGE_DIM = (160, 64)
ENV_SEED = 1

def pre_obs(obs: np.ndarray, image_dim: Tuple[int, int] = IMAGE_DIM) -> np.ndarray:
    # cropping, resizing, and normalizing the image
    return np.stack([cv2.resize(obs[i,obs.shape[1]//2:,:], image_dim)/255 for i in range(obs.shape[0])], axis=0)

def evaluate(model: TinycarCombo, unwrapped_env: gym.Env, maneuver: int, seed: int = 0, speed = 0.5, steps = 5000, episodes = 5, render_mode=None) -> Tuple[float, float, float, int, float]:
    """
    Tests the model in the environment for a given maneuver.
    Returns total reward, average CTE, and average heading error
    """
    unwrapped_env.unwrapped.render_mode = render_mode

    env = CTELinearRewardWrapper(unwrapped_env, min_cte=0.03, max_reward=1.0)
    env = LanelineSparseRewardWrapper(env, sparse_rewards={"outer": -10.0})
    env = CTETerminationWrapper(env, max_cte=0.1)

    @TinyJit
    def get_steering_angle(x: Tensor, m: Tensor) -> Tensor:
        Tensor.no_grad = True
        out = model(x, m.one_hot(model.m_dim))[0].realize()
        Tensor.no_grad = False
        return out
    
    obs = env.reset(seed=seed)[0]
    total_rew, cte, heading_error, terminations, inf_time = 0.0, [], [], 0, []
    terminated, truncated = False, False
    for i in range(int(steps * episodes)):
        st = time.perf_counter()
        steering_angle = get_steering_angle(x=Tensor(pre_obs(obs.astype(np.float32))).unsqueeze(0), m=Tensor(maneuver).unsqueeze(0)).item()
        inf_time.append(time.perf_counter() - st)
        obs, rew, terminated, truncated, info = env.step({"car_control": [speed, steering_angle], "maneuver": maneuver})
        total_rew += rew
        cte.append(abs(info["cte"]))
        heading_error.append(abs(info["heading_error"]))
        if terminated or truncated:
            terminations += 1
            obs = env.reset()[0]
        if i % steps == 0:
            obs = env.reset()[0]
    return total_rew, sum(cte) / len(cte), sum(heading_error) / len(heading_error), terminations, steps * episodes / sum(inf_time)
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    assert tinycar_combo.load_pretrained() == True

    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env, maneuver=maneuver if maneuver != 2 else 3, steps=10000, episodes=1, render_mode="human")
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | Heading Error: {heading_error:.4f} rad/step | Terminations: {terminations:3d} | perf: {stepss:.2f} steps/s")
    



    
