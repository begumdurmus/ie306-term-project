"""Rol C - Planning: Ablasyon - planlama adimi (n) taramasi.
Her n icin 3-seed ortalama cost_per_order'i raporlar."""
import numpy as np
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, GreedyNearest
from dyna_q_policy import DynaQPolicy

cfg = Config()
eval_seeds = [0, 1, 2]
train_seeds = [0, 1, 2]
tags = ["n0", "n5", "n10", "n50"]

# referans: greedy
g = evaluate(GreedyNearest(cfg), cfg, seeds=eval_seeds)["mean"]
print(f"greedy_nearest cost/order = {g['cost_per_order']:.3f}\n")

print(f"{'planning_steps':>15} | {'cost/order (mean+/-std)':>26} | {'deplete':>8}")
print("-" * 58)
for tag in tags:
    costs, deps = [], []
    for s in train_seeds:
        m = evaluate(DynaQPolicy(cfg, f"../weights/{tag}_seed{s}.npz"), cfg, seeds=eval_seeds)["mean"]
        costs.append(m["cost_per_order"])
        deps.append(m["depletion_events"])
    n = tag[1:]  # "n0" -> "0"
    print(f"{n:>15} | {np.mean(costs):>11.3f} +/- {np.std(costs):<11.3f} | {np.mean(deps):>8.1f}")