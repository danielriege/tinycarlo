import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device, GlobalCounters
from typing import Tuple
from tinycarlo.wrapper import CTELinearRewardWrapper, LanelineSparseRewardWrapper

import os
import numpy as np
from tqdm import trange
import time
import cv2
import math

from examples.models.tinycar_net import TinycarCombo

IMAGE_DIM = (200, 80)
ENV_SEED = 1

def pre_obs(obs: np.ndarray, image_dim: Tuple[int, int] = (200,80)) -> np.ndarray:
    # cropping, resizing, and normalizing the image
    return np.stack([cv2.resize(obs[i,obs.shape[1]//2:,:], image_dim)/255 for i in range(obs.shape[0])], axis=0)

def evaluate(model: TinycarCombo, unwrapped_env: gym.Env, maneuver: int, seed: int = 0, speed = 0.5, steps = 5000) -> Tuple[float, float, float]:
    """
    Tests the model in the environment for a given maneuver.
    Returns total reward, average CTE, and average heading error
    """
    unwrapped_env.unwrapped.render_mode = "human"

    env = CTELinearRewardWrapper(unwrapped_env, min_cte=0.03, max_reward=1.0)
    env = LanelineSparseRewardWrapper(env, sparse_rewards={"outer": -10.0})

    @TinyJit
    def get_steering_angle(x: Tensor, m: Tensor) -> Tensor:
        Tensor.no_grad = True
        out = model(x, m.one_hot(model.m_dim))[0].realize()
        Tensor.no_grad = False
        return out
    
    obs = env.reset(seed=seed)[0]
    total_rew, cte, heading_error = 0.0, [], []
    for _ in range(steps):
        steering_angle = get_steering_angle(x=Tensor(pre_obs(obs)).unsqueeze(0), m=Tensor(maneuver).unsqueeze(0)).item()
        obs, rew, _, _, info = env.step({"car_control": [speed, steering_angle], "maneuver": maneuver})
        total_rew += rew
        cte.append(abs(info["cte"]))
        heading_error.append(abs(info["heading_error"]))

    return total_rew, sum(cte) / len(cte), sum(heading_error) / len(heading_error)
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    assert tinycar_combo.load_pretrained() == True

    for maneuver in range(3):
        total_rew, avg_cte, avg_heading_error = evaluate(tinycar_combo, env, maneuver if maneuver != 2 else 3, seed=ENV_SEED)
        print(f"Maneuver {maneuver} - Total Reward: {total_rew}, Average CTE: {avg_cte}, Average Heading Error: {avg_heading_error}")
    



    
