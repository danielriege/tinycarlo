import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device, GlobalCounters
from typing import List
from tinycarlo.helper import getenv

import os, sys, math
import numpy as np
from tqdm import trange

from examples.models.tinycar_net import TinycarCombo
from examples.benchmark_tinycar_net import pre_obs, evaluate

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
STEPS = 1000

### Data Collection
STEP_SIZE = 2000 # number of steps to take per episode
BUFFER_SIZE = 60_000
BUFFER_SAVEFILE = "/tmp/stanley_training_data.npz" if len(sys.argv) != 2 else sys.argv[1]
MODEL_SAVEFILE = "/tmp/tinycar_combo.safetensors" if len(sys.argv) != 3 else sys.argv[2]
PLOT = getenv("PLOT")

ENV_SEED = 2
SPEED = 0.5
K = 5

def create_loss_graph(loss: List[float]):
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.xlabel("Steps * 10")
    plt.ylabel("Loss")
    plt.savefig("/tmp/stanley_loss.png")

def sample_episode(Xn, Mn, Yn, old_steps, env, maneuver = 0, seed = 0):
    # random camera params
    pitch = np.random.randint(0, 25)
    fov = np.random.randint(80, 120)
    env.unwrapped.camera.orientation[0] = pitch
    env.unwrapped.camera.fov = fov
    env.unwrapped.camera.update_params()

    obs, info = env.reset(seed=seed)
    steps = 0
    for i in range(STEP_SIZE*5):
        cte, heading_error = info["cte"], info["heading_error"]
        # Lateral Control with Stanley Controller
        steering_correction = math.atan2(K * cte, SPEED)
        steering_angle = (heading_error + steering_correction) * 180 / math.pi / env.unwrapped.config["car"]["max_steering_angle"]
        action = {"car_control": [SPEED, steering_angle], "maneuver": maneuver}
        env.unwrapped.no_observation = False if (i+1)%5 == 0 else True
        next_obs, _, terminated, truncated, info = env.step(action)
        if i%5 == 0: # collect every 5th step
            Xn[old_steps+steps] = pre_obs(obs)
            Mn[old_steps+steps] = maneuver
            Yn[old_steps+steps] = steering_angle
            steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    env.unwrapped.no_observation = False
    return steps
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    opt = nn.optim.Adam(nn.state.get_parameters(tinycar_combo), lr=LEARNING_RATE)
    print(f"using Device: {Device.DEFAULT} | encoder params: {sum([p.numel() for p in nn.state.get_parameters(tinycar_combo.encoder)]):,} | actor params: {sum([p.numel() for p in nn.state.get_parameters(tinycar_combo.actor)]):,}")
    # check if training data exists in /tmp
    if os.path.exists(BUFFER_SAVEFILE):
        print(f"Loading training data from disk: {BUFFER_SAVEFILE}")
        data = np.load(BUFFER_SAVEFILE)
        Xn, Mn, Yn = data["Xn"].astype(np.float32), data["Mn"].astype(np.float32), data["Yn"].astype(np.float32)
        steps = Xn.shape[0]
    else:
        print("Collecting training data:")
        Xn, Mn, Yn = np.zeros((BUFFER_SIZE, *obs.shape), dtype=np.float32), np.zeros(BUFFER_SIZE, dtype=np.float32), np.zeros((BUFFER_SIZE, tinycar_combo.a_dim), dtype=np.float32)
        steps = 0 # buffer index
        maneuver, seed = 0, 0
        for episode_number in (t:=trange(BUFFER_SIZE//STEP_SIZE)):
            steps += sample_episode(Xn, Mn, Yn, steps, env, maneuver if maneuver != 2 else 3, seed)
            maneuver = (maneuver + 1) % 3
            if episode_number % 3 == 0: seed += 1
            t.set_description(f"sz: {steps:5d}")
        # save to disk
        Xn, Mn, Yn = Xn[:steps], Mn[:steps], Yn[:steps]
        np.savez_compressed(BUFFER_SAVEFILE, Xn=Xn, Mn=Mn, Yn=Yn)
    print(f"Training data memory used: {sum([x.size * x.itemsize for x in [Xn, Mn, Yn]])/1e9:.2f} GB | type: {Xn.dtype} | shape: {Xn.shape}")
    print("Training:")

    @TinyJit
    def train_step(x: Tensor, m: Tensor, y: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            out = tinycar_combo(x, m.one_hot(tinycar_combo.m_dim))
            loss = (out - y).pow(2).mean() 
            loss.backward()
            opt.step()
            return loss
    
    losses, loss, loss_mean = [], 0, float("inf")
    for step in (t:=trange(STEPS)):
        samples = np.random.randint(0, steps, BATCH_SIZE)
        loss += train_step(Tensor(Xn[samples]), Tensor(Mn[samples]), Tensor(Yn[samples])).item()
        if step % 10 == 0:
            loss_mean = loss / 10
            loss = 0
            losses.append(loss_mean)
        t.set_description(f"loss: {loss_mean:.7f}")

    if PLOT: create_loss_graph(losses[1:])
    
    print(f"Saving model to: {MODEL_SAVEFILE}")
    state_dict = nn.state.get_state_dict(tinycar_combo)
    nn.state.safe_save(state_dict, MODEL_SAVEFILE)
    print("Evaluating:")
    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env, maneuver=maneuver if maneuver != 2 else 3, render_mode="human")
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | H-Error: {heading_error:.4f} rad/step | Terms: {terminations:3d} | perf: {stepss:.2f} steps/s")


