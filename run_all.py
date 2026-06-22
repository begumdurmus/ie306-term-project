import sys, os
sys.path.insert(0, 'drone_dispatch_env')

from drone_dispatch_env import DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate
from a2c_agent import A2CAgent
from reinforce_agent import obs_to_vector, REINFORCEAgent
import torch

SEEDS = [0, 1, 2]
CONFIG = "drone_dispatch_env/configs/eval_standard.yaml"

def eval_agent(agent, seeds, label):
    import subprocess, json
    # run_eval ile değerlendir
    results = []
    env = DroneDispatchEnv()
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        ep_return = 0
        while not done:
            action = agent.act(obs)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_return += r
        results.append(ep_return)
    mean = sum(results) / len(results)
    print(f"{label:20s} | mean_return={mean:.2f}")
    return mean

# A2C
env = DroneDispatchEnv()
obs, info = env.reset(seed=0)
obs_dim = obs_to_vector(obs).shape[0]
n_actions = env.action_space.n

a2c = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)
a2c.load("weights/a2c.pt")

reinforce = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)
reinforce.load("weights/reinforce.pt")

print("="*50)
print("Baseline karşılaştırması (seeds=0,1,2)")
print("="*50)
eval_agent(a2c, SEEDS, "A2C")
eval_agent(reinforce, SEEDS, "REINFORCE")
print("="*50)
print("Gerçek cost_per_order için:")
print("bash reproduce.sh drone_dispatch_env/configs/eval_standard.yaml '0,1,2' greedy_nearest")
