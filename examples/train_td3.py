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
from examples.rl_utils import avg_w, create_action_loss_graph, create_critic_loss_graph, create_ep_rew_graph, Replaybuffer

# *** hyperparameters ***
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100_000
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 2e-4
EPISODES = 300
DISCOUNT_FACTOR = 0.99
TAU = 0.001  # soft update parameter
POLICY_DELAY = 2  # Delayed policy updates
MAX_STEPS = 1000

# *** environment parameters ***
SPEED = 0.5

NOISE_THETA = 0.5
NOISE_MEAN = 0.0
NOISE_SIGMA = 0.2

MODEL_SAVEFILE = "/tmp/tinycar_combo_td3.safetensors"
PLOT = getenv("PLOT")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    env = CTELinearRewardWrapper(env, min_cte=0.03, max_reward=1.0)
   # env = LanelineSparseRewardWrapper(env, sparse_rewards={"solid": -2.0, "area": -2.0, "outer": -10.0})
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

    replay_buffer = Replaybuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, (TinycarEncoder.FEATURE_VEC_SIZE,), maneuver_dim, action_dim)

    noise = Tensor.zeros(action_dim)
    exploration_rate = 1.0

    print( f"using Device: {Device.DEFAULT} | actor params {sum([p.numel() for p in nn.state.get_parameters(actor)])} | critic params {sum([p.numel() for p in nn.state.get_parameters(critic1)])}")

    @TinyJit
    def train_step_critic(x: Tensor, m: Tensor, a: Tensor, r: Tensor, x1: Tensor, m1: Tensor) -> Tuple[Tensor, Tensor]:
        global noise
        target_action = (actor_target(x1, m1) + noise).clip(-1, 1)
        target_q1 = critic_target1(x1, m1, target_action)
        target_q2 = critic_target2(x1, m1, target_action)
        target_q = target_q1.minimum(target_q2) * DISCOUNT_FACTOR + r

        # Update critic networks
        with Tensor.train():
            opt_critic1.zero_grad()
            opt_critic2.zero_grad()
            critic1_loss = (target_q - critic1(x, m, a)).pow(2).mean()
            critic2_loss = (target_q - critic2(x, m, a)).pow(2).mean()
            critic1_loss.backward()
            critic2_loss.backward()
            opt_critic1.step()
            opt_critic2.step()

        return critic1_loss, critic2_loss
    
    @TinyJit
    def train_step_action(x: Tensor, m: Tensor) -> Tensor:
        with Tensor.train():
            opt_actor.zero_grad()
            actor_loss = -critic1(x, m, actor(x, m)).mean()
            actor_loss.backward()
            opt_actor.step()

        for v, v_target in zip(nn.state.get_parameters(actor), nn.state.get_parameters(actor_target)):
            v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        for v, v_target in zip(nn.state.get_parameters(critic1), nn.state.get_parameters(critic_target1)):
            v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        for v, v_target in zip(nn.state.get_parameters(critic2), nn.state.get_parameters(critic_target2)):
            v_target.assign(TAU * v.detach() + (1 - TAU) * v_target.detach())
        return actor_loss

    @TinyJit
    def get_action(feature_vec: Tensor, maneuver: Tensor) -> Tensor:
        global noise, exploration_rate
        Tensor.no_grad = True
        noise += NOISE_THETA * (NOISE_MEAN - noise) + NOISE_SIGMA * Tensor.randn(action_dim) # Ornstein-Uhlenbeck process
        action = (actor(feature_vec, maneuver.unsqueeze(0))[0] + noise * exploration_rate).clip(-1,1).realize() 
        Tensor.no_grad = False
        return action
    
    @TinyJit
    def get_feature_vec(obs: Tensor) -> Tensor:
        Tensor.no_grad = True
        feature_vec = encoder(obs.unsqueeze(0))[0].realize()
        Tensor.no_grad = False
        return feature_vec

    st, steps = time.perf_counter(), 0
    rews = []
    c1_loss, c2_loss, a_loss = [],[],[]
    for episode_number in (t := trange(EPISODES)):
        feature_vec = get_feature_vec(Tensor(pre_obs(env.reset()[0]))) # initial feature vec
        noise = Tensor.zeros(action_dim) # reset the noise
        maneuver = np.random.randint(0, 3)
        #exploration_rate = 1 - (episode_number / EPISODES)
        ep_rew = 0
        for ep_step in range(MAX_STEPS):
            act = get_action(feature_vec, Tensor(maneuver).one_hot(maneuver_dim)).item()
            obs, rew, terminated, truncated, _ = env.step({"car_control": [SPEED, act], "maneuver": maneuver if maneuver != 2 else 3})
            feature_vec_next = get_feature_vec(Tensor(pre_obs(obs)))
            replay_buffer.add(feature_vec.numpy(), maneuver, act, rew, feature_vec_next.numpy())
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
    state_dict = nn.state.get_state_dict(tinycar_combo)
    nn.state.safe_save(state_dict, MODEL_SAVEFILE)

    if PLOT: create_critic_loss_graph(c1_loss, c2_loss)
    if PLOT: create_action_loss_graph(a_loss)
    if PLOT: create_ep_rew_graph(rews)

    print("Evaluating:")
    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver if maneuver != 2 else 3,render_mode="human", steps=2000, episodes=5)
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | H-Error: {heading_error:.4f} rad/step | Terms: {terminations:3d} | perf: {stepss:.2f} steps/s")
