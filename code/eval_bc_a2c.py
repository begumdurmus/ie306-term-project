import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import torch
import torch.nn as nn
from torch.distributions import Categorical
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env import DroneDispatchEnv
from reinforce_agent import obs_to_vector

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor  = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class BCA2CAgent:
    def __init__(self, path, obs_dim, n_actions):
        self.net = ActorCritic(obs_dim, n_actions)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.n_actions = n_actions

    def act(self, obs):
        mask = obs["action_mask"].astype(bool)
        vec = obs_to_vector(obs)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits.squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        # stochastic sampling — eğitimle tutarlı
        dist = Categorical(logits=logits + inf_mask)
        return int(dist.sample().item())

env = DroneDispatchEnv()
obs, _ = env.reset(seed=0)
obs_dim = obs_to_vector(obs).shape[0]
n_actions = env.action_space.n
cfg = Config.from_yaml("drone_dispatch_env/configs/eval_standard.yaml")
seeds = [0, 1, 2]

agent = BCA2CAgent("weights/bc_a2c.pt", obs_dim, n_actions)
r = evaluate(agent, cfg, seeds)

print("="*55)
print(f"{'Policy':<25} | cost_per_order | success_rate")
print("="*55)
print(f"{'BC+A2C':<25} | {r['mean']['cost_per_order']:.4f}         | {r['mean']['success_rate']:.4f}")
print(f"{'greedy_nearest':<25} | 4.5700         | 0.8549")
print("="*55)
if r['mean']['cost_per_order'] < 4.57:
    print("GREEDYyi GEÇTİK!")
else:
    print("Henüz geçemedik")
