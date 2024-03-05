import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device

import os
import numpy as np
from tqdm import trange
import time
import cv2

from tinycarlo.wrapper.reward import CTESparseRewardWrapper
from tinycarlo.wrapper.termination import CTETerminationWrapper, LanelineCrossingTerminationWrapper
from examples.models.pilotnet import PilotNetActor, PilotNetCritic

# *** hyperparameters ***

BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 2000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 40
DISCOUNT_FACTOR = 0.99

# *** environment parameters ***
ENV_SEED = 2
MANEUVER = 0

def pre_obs(obs: np.ndarray) -> np.ndarray:
    height = obs.shape[1]
    obs = obs[:,height//2:,:]
    obs = obs / 255.0
    return obs

def evaluate(actor: PilotNetActor, env):
    obs, terminated, truncated = pre_obs(env.reset()[0]), False, False
    total_rew = 0.0
    while not terminated and not truncated:
        act = actor(Tensor(obs)).numpy()
        obs, rew, terminated, truncated, _ = env.step({"car_control": [0.6, act[1]], "maneuver": MANEUVER})
        total_rew += rew
    return total_rew
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)
    env = LanelineCrossingTerminationWrapper(env, "outer")

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape
    action_dim = (2,)
    state_dim = obs.shape

    actor = PilotNetActor(state_dim)
    actor_target = PilotNetActor(state_dim)
    opt_actor = nn.optim.Adam(nn.state.get_parameters(actor), lr=LEARNING_RATE)
    critic = PilotNetCritic(state_dim, action_dim)
    critic_target = PilotNetCritic(state_dim, action_dim)
    opt_critic = nn.optim.Adam(nn.state.get_parameters(critic), lr=LEARNING_RATE)

    print(f"Device: {Device.DEFAULT}")

    @TinyJit
    def train_step():
        pass

    @TinyJit
    def get_action(obs: Tensor) -> Tensor:
        Tensor.no_grad = True
        ret = actor(obs.unsqueeze(0))[0].realize()
        Tensor.no_grad = False
        return ret
    
    st, steps = time.perf_counter(), 0
    Xn, An, Rn, X1n = np.zeros((REPLAY_BUFFER_SIZE, *state_dim)), np.zeros((REPLAY_BUFFER_SIZE, *action_dim)), np.zeros(REPLAY_BUFFER_SIZE), np.zeros((REPLAY_BUFFER_SIZE, *state_dim))
    rp_idx, rp_sz = 0, 0 # replay buffer index
    for episode_number in (t:=trange(EPISODES)):
        get_action.reset() # NOTE: if you don't reset the jit here it captures the wrong model on the first run through

        obs:np.ndarray = pre_obs(env.reset()[0])
        rews, terminated, truncated = [], False, False
        while not terminated and not truncated:
            act = get_action(Tensor(obs)).numpy()
            Xn[rp_idx] = obs
            An[rp_idx] = act
            obs_next, rew, terminated, truncated, _ = env.step({"car_control": [0.6, act[1]], "maneuver": MANEUVER})
            obs_next = pre_obs(obs_next)
            X1n[rp_idx] = obs_next
            rews.append(rew)
            rp_idx = (rp_idx + 1) % REPLAY_BUFFER_SIZE
            rp_sz = min(rp_sz + 1, REPLAY_BUFFER_SIZE)
            obs = obs_next
            steps += 1
            if steps >= BATCH_SIZE:
                # update the actor and critic networks
                X, A, R, X1 = Tensor(Xn), Tensor(An), Tensor(Rn), Tensor(X1n)

        t.set_description(f"sz: {rp_sz:5d} | steps/s: {steps/(time.perf_counter()-st):.2f}s | rew: {sum(rews):6.2f} ")

    test_rew = evaluate(actor, env)
    print(f"test reward: {test_rew}")