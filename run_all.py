"""
run_all.py — Role A evaluation script
Loads trained DQN models and compares against baselines.

Usage:
    python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2
"""

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_dispatch_env"))
import argparse
import torch
import numpy as np
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling
# Rol C - Dyna-Q (Planning)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from dyna_q_policy import DynaQPolicy
# import from train_dqn.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_dqn import DuelingQNetwork, QNetwork, DQNPolicy, OBS_DIM


def load_dqn(path, network_type="dueling", n_actions=169, hidden=256):
    device = torch.device("cpu")
    if network_type == "dueling":
        q_net = DuelingQNetwork(OBS_DIM, n_actions, hidden)
    else:
        q_net = QNetwork(OBS_DIM, n_actions, hidden)
    q_net.load_state_dict(torch.load(path, map_location=device))
    return DQNPolicy(q_net, device)


def print_table(results):
    print("\n" + "="*65)
    print(f"{'Method':<25} {'cost_per_order':>14} {'success_rate':>12} {'ontime_rate':>11}")
    print("="*65)
    for name, m in results:
        print(f"{name:<25} {m['cost_per_order']:>14.4f} {m['success_rate']:>12.4f} {m['ontime_rate']:>11.4f}")
    print("="*65)
    print("Lower cost_per_order = better. Baseline (greedy_nearest) = 4.5700")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds",  default="0,1,2")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]

    results = []

    # Baselines
    print("Evaluating baselines...")
    r = evaluate(RandomPolicy(cfg), cfg, seeds)
    results.append(("Random", r["mean"]))

    r = evaluate(GreedyNearest(cfg), cfg, seeds)
    results.append(("GreedyNearest (baseline)", r["mean"]))

    # DQN models
    weight_dir = "weights"
    models = [
        ("DQN Plain",   "dqn_plain_best.pt",   "plain"),
        ("Double DQN",  "dqn_double_best.pt",  "double"),
        ("Dueling DQN", "dqn_dueling_best.pt", "dueling"),
        ("Dueling DQN (tuned)", "dqn_dueling_tuned_best.pt", "dueling"),
    ]

    for name, fname, ntype in models:
        path = os.path.join(weight_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {name} - weight not found: {path}")
            continue
        print(f"Evaluating {name}...")
        try:
            policy = load_dqn(path, network_type=ntype)
            r = evaluate(policy, cfg, seeds)
            results.append((name, r["mean"]))
        except Exception as e:
            print(f"  [SKIP] {name} - load/eval error: {e}")
            continue
    # Rol C - Dyna-Q (Planning): 3 seed ortalamasi
    print("Evaluating Dyna-Q (Planning)...")
    dyna_costs = []
    for s in [0, 1, 2]:
        wpath = os.path.join(weight_dir, f"dyna_q_seed{s}.npz")
        if os.path.exists(wpath):
            r = evaluate(DynaQPolicy(cfg, wpath), cfg, seeds)
            dyna_costs.append(r["mean"])
    if dyna_costs:
        # 3 seed'in ortalamasini al
        avg = {k: float(np.mean([m[k] for m in dyna_costs])) for k in dyna_costs[0]}
        results.append(("Dyna-Q (Planning, 3-seed avg)", avg))
    else:
        print("  [SKIP] Dyna-Q weights not found")
    print_table(results)


if __name__ == "__main__":
    main()
