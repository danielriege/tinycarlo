import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device
from typing import Tuple

import os
import numpy as np
from tqdm import trange
import time
import cv2

from tinycarlo.wrapper.reward import CTELinearRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper
from examples.models.tinycar_net import TinycarActor, TinycarCritic, TinycarEncoder

# *** hyperparameters ***

BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
EPISODES = 100
DISCOUNT_FACTOR = 0.9
TAU = 0.01 # soft update parameter

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

class replaybuffer:
    def __init__(self, size: int, batch_size: int, obs_shape: Tuple[int,int,int], maneuver_dim: int, action_dim: int) -> None:
        self.size, self.batch_size, self.obs_shape, self.maneuver_dim, self.action_dim = size, batch_size, obs_shape, maneuver_dim, action_dim
        self.X, self.M, self.A, self.R, self.X1 = np.zeros((size, *obs_shape), dtype=np.float32), np.zeros((size,), dtype=np.float32), np.zeros((size,self.action_dim), dtype=np.float32), np.zeros((size,1), dtype=np.float32), np.zeros((size, *obs_shape), dtype=np.float32)
        self.rp_idx, self.rp_sz = 0, 0
    
    def add(self, x, m, a, r, x1) -> None:
        self.X[self.rp_idx] = x
        self.M[self.rp_idx] = m
        self.A[self.rp_idx] = a
        self.R[self.rp_idx] = r
        self.X1[self.rp_idx] = x1
        self.rp_idx = (self.rp_idx + 1) % self.size
        self.rp_sz = min(self.rp_sz + 1, self.size)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert self.rp_sz >= self.batch_size
        x, m, a, r, x1, m1 = self[np.random.randint(0, self.rp_sz, self.batch_size)]
        return Tensor(x), Tensor(m).one_hot(self.maneuver_dim), Tensor(a), Tensor(r), Tensor(x1), Tensor(m1).one_hot(self.maneuver_dim)

    def __getitem__(self, indices):
        return self.X[indices], self.M[indices], self.A[indices], self.R[indices], self.X1[indices], self.M[indices]

    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path, render_mode="human")
    env = LanelineCrossingTerminationWrapper(env, "outer")
    env = CTETerminationWrapper(env, max_cte=0.2)
    env = CTELinearRewardWrapper(env, min_cte=0.03, max_reward=1.0)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape
    encoder = TinycarEncoder(obs.shape)

    actor = TinycarActor(MANEUVER_DIM, ACTION_DIM)
    actor_target = TinycarActor(MANEUVER_DIM, ACTION_DIM)
    critic = TinycarCritic(MANEUVER_DIM, ACTION_DIM)
    critic_target = TinycarCritic(MANEUVER_DIM, ACTION_DIM)

    print("loading pretrained weights for actor and encoder")
    state_dict_combo = nn.state.safe_load(COMBO_WEIGHTS_PATH)
    nn.state.load_state_dict(encoder, state_dict_combo)
    nn.state.load_state_dict(actor, state_dict_combo)
    # load the same weights into the target networks initially
    for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
        v_target.assign(v.detach())
    for v, v_target in zip(nn.state.get_parameters(critic), nn.state.get_parameters(critic_target)):
        v_target.assign(v.detach())

    opt_actor = nn.optim.Adam(nn.state.get_parameters(actor), lr=LEARNING_RATE_ACTOR)
    opt_critic = nn.optim.Adam(nn.state.get_parameters(critic), lr=LEARNING_RATE_CRITIC)

    replay_buffer = replaybuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, obs.shape, MANEUVER_DIM, ACTION_DIM)

    noise = Tensor.zeros(ACTION_DIM)

    print(f"using Device: {Device.DEFAULT} | actor params {sum([p.numel() for p in nn.state.get_parameters(actor)])} | critic params {sum([p.numel() for p in nn.state.get_parameters(critic)])}")

    @TinyJit
    def train_step(x: Tensor, m: Tensor, a: Tensor, r: Tensor, x1: Tensor, m1: Tensor) -> Tuple[Tensor, Tensor]:
        # get the target action and Q value
        feature_vec_target = encoder(x1)
        target_action = actor_target(feature_vec_target, m1)
        target_q = critic_target(feature_vec_target, m1, target_action.detach()) * DISCOUNT_FACTOR + r
        feature_vec = encoder(x)

        with Tensor.train():
            opt_critic.zero_grad()
            loss = (target_q - critic(feature_vec, m, a)).pow(2).mean()
            loss.backward()
            opt_critic.step()
            # update the actor
            opt_actor.zero_grad()
            actor_loss = -critic(feature_vec, m, actor(feature_vec, m)).mean()
            actor_loss.backward()
            opt_actor.step()
            return loss, actor_loss

    @TinyJit
    def get_action(obs: Tensor, maneuver: Tensor) -> Tensor:
        global noise
        Tensor.no_grad = True
        feature_vec = encoder(obs.unsqueeze(0))
        noise += NOISE_THETA * (0.0 - noise) + 0.2 * Tensor.randn(ACTION_DIM) # Ornstein-Uhlenbeck process
        ret = (actor(feature_vec, maneuver.unsqueeze(0))[0] + noise).realize()
        Tensor.no_grad = False
        return ret
    
    st, steps = time.perf_counter(), 0
    for episode_number in (t:=trange(EPISODES)):
        obs = pre_obs(env.reset()[0])
        rews, terminated, truncated = [], False, False
        maneuver = np.random.randint(0,3)
        critic_losses, actor_losses = [0], [0]
        while not terminated and not truncated:
            act = get_action(Tensor(obs), Tensor(maneuver).one_hot(MANEUVER_DIM)).item()
            obs_next, rew, terminated, truncated, _ = env.step({"car_control": [SPEED, act], "maneuver": maneuver if maneuver != 2 else 3})
            obs_next = pre_obs(obs_next)
            replay_buffer.add(obs, maneuver, act, rew, obs_next)
            rews.append(rew)
            obs = obs_next
            steps += 1
            if steps >= BATCH_SIZE:
                # update the actor and critic networks
                critic_loss, actor_loss = train_step(*replay_buffer.sample())
                #print(critic_loss.item(), actor_loss.item(), test.numpy(), test2.numpy())
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())
                # update target networks with soft update
                for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
                    v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
                for v, v_target in zip(nn.state.get_parameters(critic), nn.state.get_parameters(critic_target)):
                    v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())

        t.set_description(f"sz: {replay_buffer.rp_sz:5d} | steps/s: {steps/(time.perf_counter()-st):.2f} | rew: {sum(rews):6.2f} | critic loss: {sum(critic_losses)/len(critic_losses):.6f} | actor loss: {sum(actor_losses)/len(actor_losses):.6f}")

    test_rew = evaluate(actor, env)
    print(f"test reward: {test_rew}")