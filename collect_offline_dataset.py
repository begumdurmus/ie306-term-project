"""
Offline RL Dataset Collection (Ch. 20)
Collects trajectories from 3 trained policies:
- Role A: Dueling DQN
- Role B: BC+A2C
- Role C: Dyna-Q

Usage:
    python collect_offline_dataset.py --config configs/eval_standard.yaml
"""

import sys
import os

# path setup
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "drone_dispatch_env"))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, BASE)

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, DroneDispatchEnv
from reinforce_agent import obs_to_vector
from train_dqn import DuelingQNetwork, DQNPolicy, OBS_DIM
from dyna_q_policy import DynaQPolicy


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh()
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class BCA2CPolicy:
    def __init__(self, obs_dim, n_actions, path):
        self.net = ActorCritic(obs_dim, n_actions)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.n_actions = n_actions

    def act(self, obs):
        mask = obs["action_mask"].astype(bool)
        x = torch.tensor(obs_to_vector(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits.squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        return int((logits + inf_mask).argmax().item())


def unified_flatten(obs):
    return obs_to_vector(obs)


def collect_episodes(policy, env, n_episodes, seed_offset=0):
    obs_list, act_list, rew_list, nobs_list, term_list, tout_list = [], [], [], [], [], []
    ep_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(unified_flatten(obs))
            act_list.append(action)
            rew_list.append(reward)
            nobs_list.append(unified_flatten(next_obs))
            term_list.append(terminated)
            tout_list.append(truncated)

            ep_return += reward
            obs = next_obs

        ep_returns.append(ep_return)

    print(f"  Collected {len(act_list)} transitions, mean_return={np.mean(ep_returns):.1f}")
    return obs_list, act_list, rew_list, nobs_list, term_list, tout_list, ep_returns


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--episodes_per_policy", type=int, default=100)
    parser.add_argument("--output", default="offline_dataset.npz")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    env = DroneDispatchEnv(cfg)

    obs_tmp, _ = env.reset(seed=0)
    obs_dim = obs_to_vector(obs_tmp).shape[0]
    n_actions = env.action_space.n

    all_obs, all_act, all_rew, all_nobs, all_term, all_tout, all_ret = \
        [], [], [], [], [], [], []

    # Role A: Dueling DQN
    print("Collecting Role A (Dueling DQN)...")
    dqn_net = DuelingQNetwork(OBS_DIM, n_actions, 512)
    dqn_net.load_state_dict(torch.load("weights/dqn_dueling_tuned_best.pt", map_location="cpu"))
    dqn_policy = DQNPolicy(dqn_net, torch.device("cpu"))
    o, a, r, no, t, to, ret = collect_episodes(dqn_policy, env, args.episodes_per_policy, seed_offset=0)
    all_obs+=o; all_act+=a; all_rew+=r; all_nobs+=no; all_term+=t; all_tout+=to; all_ret+=ret

    # Role B: BC+A2C
    print("Collecting Role B (BC+A2C)...")
    bc_policy = BCA2CPolicy(obs_dim, n_actions, "weights/pure_bc.pt")
    o, a, r, no, t, to, ret = collect_episodes(bc_policy, env, args.episodes_per_policy, seed_offset=200)
    all_obs+=o; all_act+=a; all_rew+=r; all_nobs+=no; all_term+=t; all_tout+=to; all_ret+=ret

    # Role C: Dyna-Q
    print("Collecting Role C (Dyna-Q)...")
    dyna_path = "weights/dyna_q_seed0.npz"
    if os.path.exists(dyna_path):
        dyna_policy = DynaQPolicy(cfg, dyna_path)
        o, a, r, no, t, to, ret = collect_episodes(dyna_policy, env, args.episodes_per_policy, seed_offset=400)
        all_obs+=o; all_act+=a; all_rew+=r; all_nobs+=no; all_term+=t; all_tout+=to; all_ret+=ret
    else:
        print("  [SKIP] Dyna-Q weights not found")

    print(f"\nTotal transitions: {len(all_act)}")
    np.savez_compressed(
        args.output,
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_act, dtype=np.int64),
        rewards=np.array(all_rew, dtype=np.float32),
        next_observations=np.array(all_nobs, dtype=np.float32),
        terminals=np.array(all_term, dtype=bool),
        timeouts=np.array(all_tout, dtype=bool),
        episode_returns=np.array(all_ret, dtype=np.float32),
    )
    print(f"Dataset saved → {args.output}")


if __name__ == "__main__":
    main()
