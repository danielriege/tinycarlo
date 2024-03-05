import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device, GlobalCounters
from typing import Tuple

import os
import numpy as np
from tqdm import trange
import time
import cv2
import math

from examples.models.tinycar_net import TinycarCombo


LEARNING_RATE = 1e-4
BATCH_SIZE = 1 #NOTE: for some reason, the training is not stable with larger batch sizes (it converges to the mean)
STEPS = 3000

### Data Collection
STEP_SIZE = 1000 # number of steps to take per episode
BUFFER_SIZE = 30000
BUFFER_SAVEFILE = "/tmp/stanley_training_data.npz" # when changing env params, delete this file to re-collect data
MODEL_SAVEFILE = "/tmp/tinycar_combo.safetensors"

ENV_SEED = 2
IMAGE_DIM = (200, 80)
MANEUVER_DIM = 3
ACTION_DIM = 1
SPEED = 0.5
K = 5

def pre_obs(obs: np.ndarray) -> np.ndarray:
    # cropping, resizing, and normalizing the image
    return np.stack([cv2.resize(obs[i,obs.shape[1]//2:,:], IMAGE_DIM)/255 for i in range(obs.shape[0])], axis=0)

def evaluate(model: object, env: gym.Env, maneuver: int) -> Tuple[float, float, float]:
    env.unwrapped.render_mode = "human"

    @TinyJit
    def get_steering_angle(x: Tensor, m: Tensor) -> Tensor:
        Tensor.no_grad = True
        out = model(x, m.one_hot(MANEUVER_DIM))[0].realize()
        Tensor.no_grad = False
        return out
    
    obs = env.reset()[0]
    total_rew, cte, heading_error = 0.0, [], []
    for _ in range(3000):
        steering_angle = get_steering_angle(x=Tensor(pre_obs(obs)).unsqueeze(0), m=Tensor(maneuver).unsqueeze(0)).item()
        obs, rew, _, _, info = env.step({"car_control": [SPEED, steering_angle], "maneuver": maneuver if maneuver != 2 else 3})
        total_rew += rew
        cte.append(abs(info["cte"]))
        heading_error.append(abs(info["heading_error"]))
    return total_rew, sum(cte) / len(cte), sum(heading_error) / len(heading_error)
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape, MANEUVER_DIM, ACTION_DIM)
    opt = nn.optim.Adam(nn.state.get_parameters(tinycar_combo), lr=LEARNING_RATE)
    print(f"using Device: {Device.DEFAULT} | model params: {sum([p.numel() for p in nn.state.get_parameters(tinycar_combo)])}")
    # check if training data exists in /tmp
    if os.path.exists(BUFFER_SAVEFILE):
        print(f"Loading training data from disk: {BUFFER_SAVEFILE}")
        data = np.load(BUFFER_SAVEFILE)
        Xn, Mn, Yn = data["Xn"], data["Mn"], data["Yn"]
        steps = Xn.shape[0]
    else:
        print("Collecting training data:")
        Xn, Mn, Yn = np.zeros((BUFFER_SIZE, *obs.shape), dtype=np.float32), np.zeros(BUFFER_SIZE, dtype=np.float32), np.zeros(BUFFER_SIZE, dtype=np.float32)
        steps = 0 # buffer index
        cte_n, heading_error_n = [], []
        for episode_number in (t:=trange(BUFFER_SIZE//STEP_SIZE)):
            _, info = env.reset()
            maneuver = np.random.randint(0,3)
            for _ in range(STEP_SIZE):
                cte, heading_error = info["cte"], info["heading_error"]
                # Lateral Control with Stanley Controller
                steering_correction = math.atan2(K * cte, SPEED)
                steering_angle = (heading_error + steering_correction) * 180 / math.pi / env.unwrapped.config["car"]["max_steering_angle"]
                action = {"car_control": [SPEED, steering_angle], "maneuver": maneuver if maneuver != 2 else 3}
                Xn[steps] = obs
                obs, _, terminated, truncated, info = env.step(action)
                obs = pre_obs(obs)
                Mn[steps] = maneuver
                Yn[steps] = steering_angle
                cte_n.append(abs(cte))
                heading_error_n.append(abs(heading_error))
                steps += 1
                if terminated or truncated:
                    break
            t.set_description(f"sz: {steps:5d} | cte mean: {sum(cte_n) / len(cte_n):.6f} | heading error mean: {sum(heading_error_n)/len(heading_error_n):.6f} ")
        # save to disk
        Xn, Mn, Yn = Xn[:steps], Mn[:steps], Yn[:steps]
        np.savez_compressed(BUFFER_SAVEFILE, Xn=Xn, Mn=Mn, Yn=Yn)
    start_mem_used = GlobalCounters.mem_used
    X, M, Y = Tensor(Xn), Tensor(Mn), Tensor(Yn)
    del Xn, Mn, Yn
    print(f"Training data memory used: {(GlobalCounters.mem_used-start_mem_used)/1e9:.2f} GB")
    print("Training:")

    @TinyJit
    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(BATCH_SIZE, high=steps)
            out = tinycar_combo(X[samples], M[samples].one_hot(MANEUVER_DIM))
            loss = (out - Y[samples]).pow(2).sum() 
            loss.backward()
            opt.step()
            return loss
    
    loss, loss_mean = 0, float("inf")
    for step in (t:=trange(STEPS)):
        loss += train_step().item()
        if step % 100: loss_mean = loss / 100; loss = 0
        t.set_description(f"loss: {loss_mean:.7f}")
    
    print(f"Saving model to: {MODEL_SAVEFILE}")
    state_dict = nn.state.get_state_dict(tinycar_combo)
    nn.state.safe_save(state_dict, MODEL_SAVEFILE)
    print("Evaluating:")
    rew, cte, heading_error = evaluate(tinycar_combo, env, 0)
    print(f"Total reward: {rew} | CTE mean: {cte} | Heading Error mean: {heading_error}")


