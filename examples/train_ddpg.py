import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device
from typing import Tuple

import os
import numpy as np
from tqdm import trange
import time
import cv2

from tinycarlo.wrapper.reward import CTESparseRewardWrapper
from tinycarlo.wrapper.termination import CTETerminationWrapper, LanelineCrossingTerminationWrapper
from examples.models.tinycar_net import TinycarActor, TinycarCritic, TinycarEncoder

# *** hyperparameters ***

BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 2000
LEARNING_RATE_ACTOR = 1e-2
LEARNING_RATE_CRITIC = 1e-2
EPISODES = 100
DISCOUNT_FACTOR = 0.99
TAU = 0.005 # soft update parameter

# *** environment parameters ***
ENV_SEED = 2
MANEUVER = 0
SPEED = 0.5
IMAGE_DIM = (200, 80)
MANEUVER_DIM = 3
ACTION_DIM = 1
COMBO_WEIGHTS_PATH = "/tmp/tinycar_combo.safetensors"

# *** noise parameters ***
NOISE_THETA = 0.15

def pre_obs(obs: np.ndarray) -> np.ndarray:
    # cropping, resizing, and normalizing the image
    return np.stack([cv2.resize(obs[i,obs.shape[1]//2:,:], IMAGE_DIM)/255 for i in range(obs.shape[0])], axis=0)

def evaluate(actor: TinycarActor, env):
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
    encoder = TinycarEncoder(obs.shape)

    actor = TinycarActor(MANEUVER_DIM, ACTION_DIM)
    actor_target = TinycarActor(MANEUVER_DIM, ACTION_DIM)
    opt_actor = nn.optim.Adam(nn.state.get_parameters(actor), lr=LEARNING_RATE)
    critic = TinycarCritic(MANEUVER_DIM, ACTION_DIM)
    critic_target = TinycarCritic(MANEUVER_DIM, ACTION_DIM)
    opt_critic = nn.optim.Adam(nn.state.get_parameters(critic), lr=LEARNING_RATE)

    state_dict_combo = nn.state.safe_load(COMBO_WEIGHTS_PATH)
    nn.state.load_state_dict(encoder, state_dict_combo)
    nn.state.load_state_dict(actor, state_dict_combo)
    # load the same weights into the target networks initially
    state_dict_actor = nn.state.get_state_dict(actor)
    nn.state.load_state_dict(actor_target, state_dict_actor)
    state_dict_critic = nn.state.get_state_dict(critic)
    nn.state.load_state_dict(critic_target, state_dict_critic)

    noise = Tensor.zeros(ACTION_DIM)

    print(f"using Device: {Device.DEFAULT}")

    @TinyJit
    def train_step():
        pass

    @TinyJit
    def get_action(obs: Tensor) -> Tensor:
        global noise
        Tensor.no_grad = True
        noise += NOISE_THETA * (0.0 - noise) + 0.3 * Tensor.randn(ACTION_DIM) # Ornstein-Uhlenbeck process
        ret = (actor(obs.unsqueeze(0))[0] + noise).realize()
        Tensor.no_grad = False
        return ret
    
    st, steps = time.perf_counter(), 0
    Xn, An, Rn, X1n = np.zeros((REPLAY_BUFFER_SIZE, *obs.shape), dtype=np.float32), np.zeros((REPLAY_BUFFER_SIZE, ACTION_DIM), dtype=np.float32), np.zeros(REPLAY_BUFFER_SIZE, dtype=np.float32), np.zeros((REPLAY_BUFFER_SIZE, *obs.shape), dtype=np.float32)
    rp_idx, rp_sz = 0, 0 # replay buffer index
    for episode_number in (t:=trange(EPISODES)):
        get_action.reset() # NOTE: if you don't reset the jit here it captures the wrong model on the first run through

        obs = pre_obs(env.reset()[0])
        rews, terminated, truncated = [], False, False
        while not terminated and not truncated:
            act = get_action(Tensor(obs)).item()
            Xn[rp_idx] = obs
            An[rp_idx] = act
            obs_next, rew, terminated, truncated, _ = env.step({"car_control": [SPEED, act], "maneuver": MANEUVER})
            obs_next = pre_obs(obs_next)
            X1n[rp_idx] = obs_next
            Rn[rp_idx] = rew
            rews.append(rew)
            rp_idx = (rp_idx + 1) % REPLAY_BUFFER_SIZE
            rp_sz = min(rp_sz + 1, REPLAY_BUFFER_SIZE)
            obs = obs_next
            steps += 1
            if steps >= BATCH_SIZE:
                # update the actor and critic networks
                X, A, R, X1 = Tensor(Xn[:steps]), Tensor(An[:steps]), Tensor(Rn[:steps]), Tensor(X1n[:steps])
                train_step()
                # update target networks with soft update
                state_dict_actor = nn.state.get_state_dict(actor)
                state_dict_actor_target = nn.state.get_state_dict(actor_target)
                for v, v_target in zip(state_dict_actor.values(), state_dict_actor_target.values()):
                    v_target *= (1-TAU) + TAU * v
                state_dict_critic = nn.state.get_state_dict(critic)
                state_dict_critic_target = nn.state.get_state_dict(critic_target)
                for v, v_target in zip(state_dict_critic.values(), state_dict_critic_target.values()):
                    v_target *= (1-TAU) + TAU * v

        t.set_description(f"sz: {rp_sz:5d} | steps/s: {steps/(time.perf_counter()-st):.2f}s | rew: {sum(rews):6.2f} ")

    test_rew = evaluate(actor, env)
    print(f"test reward: {test_rew}")