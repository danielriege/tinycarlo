import gymnasium as gym
import tinycarlo
from tinygrad import Tensor, TinyJit, nn, Device
from typing import Tuple, List

import os
import numpy as np
from tqdm import trange
import time

from tinycarlo.wrapper.reward import CTELinearRewardWrapper, LanelineSparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper
from examples.models.tinycar_net import TinycarActor, TinycarCritic, TinycarCombo, TinycarEncoder
from examples.benchmark_tinycar_net import pre_obs, evaluate
from tinycarlo.helper import getenv

# *** hyperparameters ***
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 30000
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 2e-4
EPISODES = 300
DISCOUNT_FACTOR = 0.99
TAU = 0.005  # soft update parameter
TARGET_POLICY_NOISE = 0.2  # Noise added to target policy
TARGET_POLICY_NOISE_CLIP = 0.5  # Noise clipping range
POLICY_DELAY = 2  # Delayed policy updates
MAX_STEPS = 1000

# *** environment parameters ***
SPEED = 0.5

MODEL_SAVEFILE = "/tmp/tinycar_combo_td3.safetensors"
PLOT = getenv("PLOT")

def create_critic_loss_graph(c1_loss: List[float], c2_loss: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(c1_loss)
    plt.plot(c2_loss)
    plt.legend(["Critic 1 Loss", "Critic 2 Loss"])
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("/tmp/critic_loss.png")

def create_action_loss_graph(a_loss: List[float]):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(a_loss)
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

class replaybuffer:
    def __init__(self, size: int, batch_size: int, obs_shape: Tuple[int, int, int], maneuver_dim: int, action_dim: int) -> None:
        self.size, self.batch_size, self.obs_shape, self.maneuver_dim, self.action_dim = size, batch_size, obs_shape, maneuver_dim, action_dim
        self.X, self.M, self.A, self.R, self.X1 = np.zeros((size, *obs_shape), dtype=np.float32), np.zeros( (size,), dtype=np.float32), np.zeros((size, self.action_dim), dtype=np.float32), np.zeros((size, 1), dtype=np.float32), np.zeros( (size, *obs_shape), dtype=np.float32)
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

    env = CTELinearRewardWrapper(env, min_cte=0.03, max_reward=1.0)
    env = LanelineSparseRewardWrapper(env, sparse_rewards={"solid": -5.0, "area": -5.0})
    env = LanelineCrossingTerminationWrapper(env, ["outer"])
    env = CTETerminationWrapper(env, max_cte=0.1)

    obs = pre_obs(env.reset()[0])  # seed the environment and get obs shape
    tinycar_combo = TinycarCombo(obs.shape)
    tinycar_combo.load_pretrained()
    encoder = tinycar_combo.encoder
    actor = tinycar_combo.actor  # pre trained actor
    actor_target = TinycarActor()
    critic1 = TinycarCritic()
    critic2 = TinycarCritic()
    critic_target1 = TinycarCritic()
    critic_target2 = TinycarCritic()
    action_dim, maneuver_dim = tinycar_combo.a_dim, tinycar_combo.m_dim

    # load the same weights into the target networks initially
    for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
        v_target.assign(v.detach())
    for v, v_target in zip(nn.state.get_parameters(critic1), nn.state.get_parameters(critic_target1)):
        v_target.assign(v.detach())
    for v, v_target in zip(nn.state.get_parameters(critic2), nn.state.get_parameters(critic_target2)):
        v_target.assign(v.detach())

    opt_actor = nn.optim.Adam(nn.state.get_parameters(actor), lr=LEARNING_RATE_ACTOR)
    opt_critic1 = nn.optim.Adam(nn.state.get_parameters(critic1), lr=LEARNING_RATE_CRITIC)
    opt_critic2 = nn.optim.Adam(nn.state.get_parameters(critic2), lr=LEARNING_RATE_CRITIC)

    replay_buffer = replaybuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, obs.shape, maneuver_dim, action_dim)

    exploration_rate = 1.0

    print( f"using Device: {Device.DEFAULT} | actor params {sum([p.numel() for p in nn.state.get_parameters(actor)])} | critic params {sum([p.numel() for p in nn.state.get_parameters(critic1)])}")

    @TinyJit
    def train_step(x: Tensor, m: Tensor, a: Tensor, r: Tensor, x1: Tensor, m1: Tensor, step: int) -> Tuple[Tensor, Tensor, Tensor]:
        # Sample noise and clip it to a range
        noise = (Tensor.randn(a.shape) * TARGET_POLICY_NOISE).clip(-TARGET_POLICY_NOISE_CLIP, TARGET_POLICY_NOISE_CLIP)

        # Compute the target Q value
        target_feature_vec = encoder(x1)
        target_action = (actor_target(target_feature_vec, m1) + noise).clip(-1, 1)
        target_q1 = critic_target1(target_feature_vec, m1, target_action)
        target_q2 = critic_target2(target_feature_vec, m1, target_action)
        target_q = target_q1.minimum(target_q2) * DISCOUNT_FACTOR + r
        feature_vec = encoder(x)

        # Update critic networks
        with Tensor.train():
            opt_critic1.zero_grad()
            opt_critic2.zero_grad()
            critic1_loss = (target_q - critic1(feature_vec, m, a)).pow(2).mean()
            critic2_loss = (target_q - critic2(feature_vec, m, a)).pow(2).mean()
            critic1_loss.backward()
            critic2_loss.backward()
            opt_critic1.step()
            opt_critic2.step()

        # Delayed policy updates
        if step % POLICY_DELAY == 0:
            # Update actor network
            opt_actor.zero_grad()
            actor_loss = -critic1(feature_vec, m, actor(feature_vec, m)).mean()
            actor_loss.backward()
            opt_actor.step()

            for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
                v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
            for v, v_target in zip(nn.state.get_parameters(critic1), nn.state.get_parameters(critic_target1)):
                v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
            for v, v_target in zip(nn.state.get_parameters(critic2), nn.state.get_parameters(critic_target2)):
                v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        else:
            actor_loss = None

        return critic1_loss, critic2_loss, actor_loss

    @TinyJit
    def get_action(obs: Tensor, maneuver: Tensor) -> Tensor:
        global exploration_rate
        Tensor.no_grad = True
        feature_vec = encoder(obs.unsqueeze(0))
        action = actor(feature_vec, maneuver.unsqueeze(0))[0] + Tensor.randn(action_dim) * exploration_rate  # Add exploration noise
        Tensor.no_grad = False
        return action.clip(-1, 1)  # Ensure actions are within valid range

    st, steps, ep_steps = time.perf_counter(), 0, 0
    total_rew, ep_rew, ep_rews = 0.0, 0.0, []
    c1_loss, c2_loss, a_loss = [],[],[]
    for episode_number in (t := trange(EPISODES)):
        obs = pre_obs(env.reset()[0])
        terminated, truncated = False, False
        maneuver = np.random.randint(0, 3)
        critic1_losses, critic2_losses, actor_losses = [], [], []
        exploration_rate = 1 - (episode_number / EPISODES)
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
                critic1_loss, critic2_loss, actor_loss = train_step(*replay_buffer.sample(), steps)
                critic1_losses.append(critic1_loss.item())
                critic2_losses.append(critic2_loss.item())
                if actor_loss is not None:
                    actor_losses.append(actor_loss.item())

                t.set_description(
                    f"sz: {replay_buffer.rp_sz:5d} | steps/s: {steps / (time.perf_counter() - st):.2f} | rew: {ep_rew:.2f} | rew/ep {total_rew / (episode_number + 1):2.2f}| critic1 loss: {sum(critic1_losses) / len(critic1_losses):.3f} | critic2 loss: {sum(critic2_losses) / len(critic2_losses):.3f}")
        total_rew += ep_rew
        ep_rews.append(ep_rew)
        ep_rew, ep_steps = 0.0, 0
        c1_loss.extend(critic1_losses)
        c2_loss.extend(critic2_losses)
        a_loss.extend(actor_losses)

    print(f"Saving model to: {MODEL_SAVEFILE}")
    state_dict = nn.state.get_state_dict(tinycar_combo)
    nn.state.safe_save(state_dict, MODEL_SAVEFILE)

    if PLOT: create_critic_loss_graph(c1_loss, c2_loss)
    if PLOT: create_action_loss_graph(a_loss)
    if PLOT: create_ep_rew_graph(ep_rews)

    print("Evaluating:")
    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver if maneuver != 2 else 3,render_mode="human", steps=2000, episodes=5)
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | H-Error: {heading_error:.4f} rad/step | Terms: {terminations:3d} | perf: {stepss:.2f} steps/s")
