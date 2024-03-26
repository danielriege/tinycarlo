import gymnasium as gym
import tinycarlo
from typing import Tuple, List
import torch
import torch.nn.functional as F

import os
import numpy as np
from tqdm import trange
import time

from tinycarlo.wrapper.reward import CTELinearRewardWrapper, LanelineSparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper
from examples.models.tinycar_net import TinycarActor, TinycarCritic, TinycarCombo, TinycarEncoder
from examples.benchmark_tinycar_net import pre_obs, evaluate
from tinycarlo.helper import getenv
from examples.rl_utils import avg_w, create_action_loss_graph, create_critic_loss_graph, create_ep_rew_graph, Replaybuffer

# *** hyperparameters ***
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100_000
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 2e-4
EPISODES = 100
DISCOUNT_FACTOR = 0.99
TAU = 0.001  # soft update parameter
POLICY_DELAY = 2  # Delayed policy updates
MAX_STEPS = 1000

# *** environment parameters ***
SPEED = 0.5

NOISE_THETA = 0.2
NOISE_MEAN = 0.0
NOISE_SIGMA = 0.2

MODEL_SAVEFILE = "/tmp/tinycar_combo_td3.pt"
PLOT = getenv("PLOT")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    env = CTELinearRewardWrapper(env, min_cte=0.03, max_reward=1.0)
    #env = LanelineSparseRewardWrapper(env, sparse_rewards={"solid": -2.0, "area": -2.0, "outer": -10.0})
    #env = LanelineCrossingTerminationWrapper(env, ["outer"])
    env = CTETerminationWrapper(env, max_cte=0.15)

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

    encoder.to(device)
    actor.to(device)
    actor_target.to(device)
    critic1.to(device)
    critic2.to(device)
    critic_target1.to(device)
    critic_target2.to(device)

    action_dim, maneuver_dim = tinycar_combo.a_dim, tinycar_combo.m_dim

    # load the same weights into the target networks initially
    for v, v_target in zip(actor.parameters(), actor_target.parameters()):
        v_target.data.copy_(v.data)
    for v, v_target in zip(critic1.parameters(), critic_target1.parameters()):
        v_target.data.copy_(v.data)
    for v, v_target in zip(critic2.parameters(), critic_target2.parameters()):
        v_target.data.copy_(v.data)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=LEARNING_RATE_CRITIC)
    opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=LEARNING_RATE_CRITIC)

    replay_buffer = Replaybuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, (TinycarEncoder.FEATURE_VEC_SIZE,), maneuver_dim, action_dim)

    noise = torch.zeros(action_dim).to(device)
    exploration_rate = 1.0

    print( f"using Device: {device} | actor params {sum([p.numel() for p in actor.parameters()])} | critic params {sum([p.numel() for p in critic1.parameters()])}")

    def train_step_critic(x, m, a, r, x1, m1):
        global noise
        with torch.no_grad():
           # target_action = (actor_target(x1, m1) + noise).clamp(-1, 1)
            target_action = actor_target(x1, m1)
            target_q1 = critic_target1(x1, m1, target_action)
            target_q2 = critic_target2(x1, m1, target_action)
            target_q = torch.min(target_q1,target_q2) * DISCOUNT_FACTOR + r

        # Update critic networks
        opt_critic1.zero_grad()
        opt_critic2.zero_grad()
        critic1_loss = F.mse_loss(target_q, critic1(x, m, a))
        critic2_loss = F.mse_loss(target_q, critic2(x, m, a))
        critic1_loss.backward()
        critic2_loss.backward()
        opt_critic1.step()
        opt_critic2.step()

        return critic1_loss, critic2_loss
    
    def train_step_action(x: torch.Tensor,m: torch.Tensor) -> torch.Tensor:
        opt_actor.zero_grad()
        actor_loss = -critic1(x, m, actor(x, m)).mean()
        actor_loss.backward()
        opt_actor.step()

        for v, v_target in zip(actor.parameters(), actor_target.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        for v, v_target in zip(critic1.parameters(), critic_target1.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        for v, v_target in zip(critic2.parameters(), critic_target2.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        return actor_loss

    def get_action(feature_vec: torch.Tensor, maneuver: torch.Tensor) -> torch.Tensor:
        global noise, exploration_rate
        with torch.no_grad():
            noise += NOISE_THETA * (NOISE_MEAN - noise) + NOISE_SIGMA * torch.randn(action_dim).to(device) # Ornstein-Uhlenbeck process
            action = (actor(feature_vec, maneuver.unsqueeze(0))[0] + noise * exploration_rate).clamp(-1, 1).cpu()
        return action
    
    def get_feature_vec(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature_vec = encoder(obs.unsqueeze(0))[0]
        return feature_vec

    st, steps = time.perf_counter(), 0
    rews = []
    c1_loss, c2_loss, a_loss = [],[],[]
    for episode_number in (t := trange(EPISODES)):
        feature_vec = get_feature_vec(torch.from_numpy(pre_obs(env.reset()[0])).to(device)) # initial feature vec
        noise = torch.zeros(action_dim).to(device) # reset the noise
        maneuver = np.random.randint(0, 3)
        #exploration_rate = 1 - (episode_number / EPISODES)
        ep_rew = 0
        for ep_step in range(MAX_STEPS):
            m = F.one_hot(torch.tensor(maneuver), tinycar_combo.m_dim).float().to(device)
            act = get_action(feature_vec, m).item()
            obs, rew, terminated, truncated, _ = env.step({"car_control": [SPEED, act], "maneuver": maneuver if maneuver != 2 else 3})
            feature_vec_next = get_feature_vec(torch.from_numpy(pre_obs(obs)).to(device))
            replay_buffer.add(feature_vec.cpu().numpy(), maneuver, act, rew, feature_vec_next.cpu().numpy())
            feature_vec = feature_vec_next
            ep_rew += rew
            steps += 1
            if steps >= BATCH_SIZE:
                sample = replay_buffer.sample()
                critic1_loss, critic2_loss = train_step_critic(*sample)
                c1_loss.append(critic1_loss.item())
                c2_loss.append(critic2_loss.item())
                if steps % POLICY_DELAY == 0:
                    actor_loss = train_step_action(*sample[:2])
                    a_loss.append(actor_loss.item())
            t.set_description(f"sz: {replay_buffer.rp_sz:5d} | steps/s: {steps / (time.perf_counter() - st):.2f} | rew/ep {avg_w(rews, 10):3.2f}| c1 loss: {avg_w(c1_loss):3.3f} | c2 loss: {avg_w(c2_loss):3.3f} | actor loss: {avg_w(a_loss):3.3f}")
            if terminated or truncated:
                break
        rews.append(ep_rew)

    print(f"Saving model to: {MODEL_SAVEFILE}")
    torch.save(tinycar_combo.state_dict(), MODEL_SAVEFILE)

    if PLOT: create_critic_loss_graph(c1_loss, c2_loss)
    if PLOT: create_action_loss_graph(a_loss)
    if PLOT: create_ep_rew_graph(rews)

    print("Evaluating:")
    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver if maneuver != 2 else 3,render_mode="human", steps=2000, episodes=5)
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | H-Error: {heading_error:.4f} rad/step | Terms: {terminations:3d} | perf: {stepss:.2f} steps/s")
