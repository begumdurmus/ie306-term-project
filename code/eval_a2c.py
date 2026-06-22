import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import json
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from a2c_agent import A2CAgent
from reinforce_agent import obs_to_vector, REINFORCEAgent
from drone_dispatch_env import DroneDispatchEnv

# obs_dim ve n_actions öğren
env = DroneDispatchEnv()
obs, _ = env.reset(seed=0)
obs_dim = obs_to_vector(obs).shape[0]
n_actions = env.action_space.n

cfg = Config.from_yaml("drone_dispatch_env/configs/eval_standard.yaml")
seeds = [0, 1, 2]

print("="*55)
print(f"{'Policy':<20} | cost_per_order | success_rate")
print("="*55)

# A2C
a2c = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)
a2c.load("weights/a2c.pt")
r = evaluate(a2c, cfg, seeds)
print(f"{'A2C':<20} | {r['mean']['cost_per_order']:.4f}         | {r['mean']['success_rate']:.4f}")

# REINFORCE
rf = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)
rf.load("weights/reinforce.pt")
r2 = evaluate(rf, cfg, seeds)
print(f"{'REINFORCE':<20} | {r2['mean']['cost_per_order']:.4f}         | {r2['mean']['success_rate']:.4f}")

print("="*55)
print("greedy_nearest baseline: cost_per_order=4.57")
print("="*55)
