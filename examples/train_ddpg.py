import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device
from typing import Tuple, List

import os
import numpy as np
from tqdm import trange
import time
import cv2

from tinycarlo.wrapper.reward import CTELinearRewardWrapper, LanelineSparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper
from examples.models.tinycar_net import TinycarActor, TinycarCritic, TinycarCombo, TinycarEncoder
from examples.benchmark_tinycar_net import pre_obs, evaluate
from tinycarlo.helper import getenv

# *** hyperparameters ***

BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 70000
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 2e-3
EPISODES = 300
DISCOUNT_FACTOR = 0.99
TAU = 0.005 # soft update parameter
MAX_STEPS = 1000

# *** environment parameters ***
SPEED = 0.5

# *** noise parameters ***
NOISE_THETA = 0.15

MODEL_SAVEFILE = "/tmp/tinycar_combo.safetensors"
PLOT = getenv("PLOT")

def create_critic_loss_graph(c_loss: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mv(c_loss))
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("/tmp/critic_loss.png")

def create_action_loss_graph(a_loss: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(mv(a_loss))
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("/tmp/actor_loss.png")

def create_ep_rew_graph(ep_rews: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ep_rews)
    plt.xlabel("Episodes")
    plt.ylabel("Episodic Reward")
    plt.savefig("/tmp/ep_rew.png")

def avg_w(x: List[float], w: int = 10) -> List[float]:
    if len(x) < w:
        return float("inf")
    return sum(x[-w:]) / w

def mv(x: List[float], w: int = 50) -> List[float]:
    return [sum(x[i:i+w])/w for i in range(len(x)-w)]

class replaybuffer:
    def __init__(self, size: int, batch_size: int, obs_shape: Tuple[int,int,int], maneuver_dim: int, action_dim: int) -> None:
        self.size, self.batch_size, self.obs_shape, self.maneuver_dim, self.action_dim = size, batch_size, obs_shape, maneuver_dim, action_dim
        self.X, self.M, self.A, self.R, self.X1 = np.zeros((size, *obs_shape), dtype=np.float32), np.zeros((size,), dtype=np.float32), np.zeros((size,self.action_dim), dtype=np.float32), np.zeros((size,1), dtype=np.float32), np.zeros((size, *obs_shape), dtype=np.float32)
        self.rp_sz = 0
    
    def add(self, x, m, a, r, x1) -> None:
        rp_idx = self.rp_sz if self.rp_sz < self.size else np.random.randint(0, self.size)
        self.X[rp_idx], self.M[rp_idx], self.A[rp_idx], self.R[rp_idx], self.X1[rp_idx] = x, m, a, r, x1
        self.rp_sz = min(self.rp_sz + 1, self.size)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert self.rp_sz >= self.batch_size
        x, m, a, r, x1, m1 = self[np.random.randint(0, self.rp_sz, self.batch_size)]
        return Tensor(x), Tensor(m).one_hot(self.maneuver_dim), Tensor(a), Tensor(r), Tensor(x1), Tensor(m1).one_hot(self.maneuver_dim)

    def __getitem__(self, indices):
        return self.X[indices], self.M[indices], self.A[indices], self.R[indices], self.X1[indices], self.M[indices]

    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    env = CTELinearRewardWrapper(env, min_cte=0.04, max_reward=1.0)
    #env = LanelineSparseRewardWrapper(env, sparse_rewards={"outer": -10.0})
    env = CTETerminationWrapper(env, max_cte=0.2)

    obs = pre_obs(env.reset()[0]) # seed the environment and get obs shape
    tinycar_combo = TinycarCombo(obs.shape)
    tinycar_combo.load_pretrained()
    encoder = tinycar_combo.encoder
    actor = tinycar_combo.actor # pre trained actor
    actor_target = TinycarActor()
    critic = TinycarCritic()
    critic_target = TinycarCritic()
    action_dim, maneuver_dim = tinycar_combo.a_dim, tinycar_combo.m_dim

    # load the same weights into the target networks initially
    for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
        v_target.assign(v.detach())
    for v, v_target in zip(nn.state.get_parameters(critic), nn.state.get_parameters(critic_target)):
        v_target.assign(v.detach())

    opt_actor = nn.optim.Adam(nn.state.get_parameters(actor), lr=LEARNING_RATE_ACTOR)
    opt_critic = nn.optim.Adam(nn.state.get_parameters(critic), lr=LEARNING_RATE_CRITIC)

    replay_buffer = replaybuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, obs.shape, maneuver_dim, action_dim)

    noise = Tensor.zeros(action_dim)
    exploration_rate = 1.0

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
        
        # update target networks with soft update
        for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
            v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        for v, v_target in zip(nn.state.get_parameters(critic), nn.state.get_parameters(critic_target)):
            v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        return loss, actor_loss

    @TinyJit
    def get_action(obs: Tensor, maneuver: Tensor) -> Tensor:
        global noise, exploration_rate
        Tensor.no_grad = True
        feature_vec = encoder(obs.unsqueeze(0))
        noise += NOISE_THETA * (0.0 - noise) + 0.2 * Tensor.randn(action_dim) # Ornstein-Uhlenbeck process
        ret = (actor(feature_vec, maneuver.unsqueeze(0))[0] + exploration_rate * noise).clip(-1,1).realize()
        Tensor.no_grad = False
        return ret
    
    st, steps, ep_steps = time.perf_counter(), 0, 0
    ep_rews, ep_rew = [], 0.0
    critic_losses, actor_losses = [], []
    for episode_number in (t:=trange(EPISODES)):
        if episode_number % 10 == 0:
            env.unwrapped.render_mode = "human"
        obs = pre_obs(env.reset()[0])
        terminated, truncated = False, False
        maneuver = np.random.randint(0,3)
        #exploration_rate = 1-(episode_number / EPISODES)
        while not terminated and not truncated and ep_steps < MAX_STEPS:
            act = get_action(Tensor(obs), Tensor(maneuver).one_hot(maneuver_dim)).item()
            obs_next, rew, terminated, truncated, _ = env.step({"car_control": [SPEED, act], "maneuver": maneuver if maneuver != 2 else 3})
            obs_next = pre_obs(obs_next)
            replay_buffer.add(obs, maneuver, act, rew, obs_next)
            ep_rew += rew
            obs = obs_next
            steps += 1
            ep_steps += 1
            if steps >= BATCH_SIZE:
                # update the actor and critic networks
                critic_loss, actor_loss = train_step(*replay_buffer.sample())
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())

                t.set_description(f"sz: {replay_buffer.rp_sz:5d} | steps/s: {steps/(time.perf_counter()-st):.2f} | rew: {ep_rew:5.2f} | rew/ep {sum(ep_rews)/(episode_number+1):2.2f}| critic loss: {avg_w(critic_losses):.3f} | actor loss: {avg_w(actor_losses):.3f}")
        ep_rews.append(ep_rew)
        ep_rew, ep_steps = 0.0, 0
        env.unwrapped.render_mode = None

    if PLOT: create_action_loss_graph(actor_losses)
    if PLOT: create_critic_loss_graph(critic_losses)
    if PLOT: create_ep_rew_graph(ep_rews)

    print(f"Saving model to: {MODEL_SAVEFILE}")
    state_dict = nn.state.get_state_dict(tinycar_combo)
    nn.state.safe_save(state_dict, MODEL_SAVEFILE)
    print("Evaluating:")
    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver if maneuver != 2 else 3, render_mode="human")
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | H-Error: {heading_error:.4f} rad/step | Terms: {terminations:3d} | perf: {stepss:.2f} steps/s")