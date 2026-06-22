"""
run_all.py — B Rolü sonuç tablosu
Kullanım: python run_all.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'drone_dispatch_env'))

import torch
import torch.nn as nn
import numpy as np
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env import DroneDispatchEnv, DroneControlEnv
from drone_dispatch_env.baselines import GreedyNearest

sys.path.insert(0, 'code')
from reinforce_agent import obs_to_vector, REINFORCEAgent
from a2c_agent import A2CAgent
from ddpg_agent import DDPGAgent

cfg = Config.from_yaml("drone_dispatch_env/configs/eval_standard.yaml")
seeds = [0, 1, 2]

env = DroneDispatchEnv()
obs, _ = env.reset(seed=0)
obs_dim = obs_to_vector(obs).shape[0]
n_actions = env.action_space.n

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class BCA2CAgent:
    def __init__(self):
        self.net = ActorCritic(obs_dim, n_actions)
        self.net.load_state_dict(torch.load("weights/pure_bc.pt", map_location="cpu"))
        self.net.eval()
    def act(self, obs):
        mask = obs["action_mask"].astype(bool)
        x = torch.tensor(obs_to_vector(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits.squeeze(0)
        inf_mask = torch.zeros(n_actions)
        inf_mask[~mask] = -float("inf")
        return int((logits + inf_mask).argmax().item())

print("=" * 60)
print(f"{'Policy':<25} | {'cost_per_order':>14} | {'success_rate':>12}")
print("=" * 60)

greedy = GreedyNearest(cfg)
r = evaluate(greedy, cfg, seeds)
print(f"{'greedy_nearest':<25} | {r['mean']['cost_per_order']:>14.4f} | {r['mean']['success_rate']:>12.4f}")

rf = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)
rf.load("weights/reinforce.pt")
r = evaluate(rf, cfg, seeds)
print(f"{'REINFORCE':<25} | {r['mean']['cost_per_order']:>14.4f} | {r['mean']['success_rate']:>12.4f}")

a2c = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)
a2c.load("weights/a2c.pt")
r = evaluate(a2c, cfg, seeds)
print(f"{'A2C':<25} | {r['mean']['cost_per_order']:>14.4f} | {r['mean']['success_rate']:>12.4f}")

bc = BCA2CAgent()
r = evaluate(bc, cfg, seeds)
print(f"{'BC+A2C (best)':<25} | {r['mean']['cost_per_order']:>14.4f} | {r['mean']['success_rate']:>12.4f}")

print("=" * 60)
print("\nDDPG (DroneControl-v0):")
print("=" * 60)

env2 = DroneControlEnv()
ddpg = DDPGAgent(obs_dim=7, action_dim=2)
ddpg.load("weights/ddpg.pt")

returns = []
for seed in seeds:
    obs, _ = env2.reset(seed=seed)
    done = False
    ep_return = 0
    while not done:
        action = ddpg.act(obs, explore=False)
        obs, r2, term, trunc, _ = env2.step(action)
        done = term or trunc
        ep_return += r2
    returns.append(ep_return)

print(f"{'DDPG':<25} | mean_return: {np.mean(returns):.2f} +- {np.std(returns):.2f}")
print("=" * 60)
