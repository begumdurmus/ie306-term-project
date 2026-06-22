import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import torch
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env import DroneDispatchEnv
from a2c_agent import A2CAgent
from reinforce_agent import obs_to_vector, REINFORCEAgent

env = DroneDispatchEnv()
obs, _ = env.reset(seed=0)
obs_dim = obs_to_vector(obs).shape[0]
n_actions = env.action_space.n
cfg = Config.from_yaml("drone_dispatch_env/configs/eval_standard.yaml")
seeds = [0, 1, 2]

# A2C — deterministik eval
class DeterministicA2C:
    def __init__(self, path):
        agent = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(path)
        self._agent = agent
    def act(self, obs):
        return self._agent.act_deterministic(obs)

print("="*55)
print(f"{'Policy':<22} | cost_per_order | success_rate")
print("="*55)

d_a2c = DeterministicA2C("weights/a2c.pt")
r = evaluate(d_a2c, cfg, seeds)
print(f"{'A2C (deterministic)':<22} | {r['mean']['cost_per_order']:.4f}         | {r['mean']['success_rate']:.4f}")

rf = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)
rf.load("weights/reinforce.pt")
r2 = evaluate(rf, cfg, seeds)
print(f"{'REINFORCE':<22} | {r2['mean']['cost_per_order']:.4f}         | {r2['mean']['success_rate']:.4f}")

print("="*55)
print("greedy_nearest:         | 4.5700         | 0.8549")
print("="*55)
