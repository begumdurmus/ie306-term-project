"""Rol C - Planning: 3 seed Dyna-Q'yu baseline'lara karsi degerlendir."""
import numpy as np
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, GreedyNearest, RandomPolicy, MILPRolling
from dyna_q_policy import DynaQPolicy

cfg = Config()
seeds = [0, 1, 2]

# baseline'lar
print("=== BASELINE'LAR ===")
for name, pol in [("random", RandomPolicy(cfg)),
                  ("greedy_nearest", GreedyNearest(cfg)),
                  ("milp_rolling", MILPRolling(cfg))]:
    m = evaluate(pol, cfg, seeds=seeds)["mean"]
    print(f"{name:16s} cost/order={m['cost_per_order']:7.3f}  "
          f"deplete={m['depletion_events']:.1f}  drops={m['n_dropped']:.1f}  "
          f"deliv={m['n_delivered']:.1f}")

# Dyna-Q: her seed ayri + ortalama
print("\n=== DYNA-Q (3 seed) ===")
costs = []
for s in seeds:
    pol = DynaQPolicy(cfg, f"../weights/dyna_q_seed{s}.npz")
    m = evaluate(pol, cfg, seeds=seeds)["mean"]
    costs.append(m["cost_per_order"])
    print(f"  seed{s}: cost/order={m['cost_per_order']:7.3f}  "
          f"deplete={m['depletion_events']:.1f}  drops={m['n_dropped']:.1f}  "
          f"deliv={m['n_delivered']:.1f}")

print(f"\ndyna-q ORTALAMA cost/order = {np.mean(costs):.3f} +/- {np.std(costs):.3f}")