"""
run_all.py — Role A + B + C evaluation script
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_dispatch_env"))
import argparse
import torch
import torch.nn as nn
import numpy as np
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling
from drone_dispatch_env import DroneDispatchEnv, DroneControlEnv

# Rol C - Dyna-Q
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from dyna_q_policy import DynaQPolicy

# Rol A - DQN
sys.path.insert(0, os.path.dirname(__file__))
from train_dqn import DuelingQNetwork, QNetwork, DQNPolicy, OBS_DIM

# Rol B - Policy-based
from reinforce_agent import obs_to_vector, REINFORCEAgent
from a2c_agent import A2CAgent
from ddpg_agent import DDPGAgent

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
    def __init__(self, obs_dim, n_actions):
        self.net = ActorCritic(obs_dim, n_actions)
        self.net.load_state_dict(torch.load("weights/pure_bc.pt", map_location="cpu"))
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

    # === ROL A - DQN ===
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
            print(f"  [SKIP] {name} - weight not found")
            continue
        print(f"Evaluating {name}...")
        try:
            policy = load_dqn(path, network_type=ntype)
            r = evaluate(policy, cfg, seeds)
            results.append((name, r["mean"]))
        except Exception as e:
            print(f"  [SKIP] {name} - error: {e}")

    # === ROL B - Policy-based ===
    print("Evaluating Rol B (Policy-based)...")
    env_tmp = DroneDispatchEnv()
    obs_tmp, _ = env_tmp.reset(seed=0)
    obs_dim = obs_to_vector(obs_tmp).shape[0]
    n_actions = env_tmp.action_space.n

    try:
        rf = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)
        rf.load("weights/reinforce.pt")
        r = evaluate(rf, cfg, seeds)
        results.append(("REINFORCE (B)", r["mean"]))
    except Exception as e:
        print(f"  [SKIP] REINFORCE: {e}")

    try:
        a2c = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)
        a2c.load("weights/a2c.pt")
        r = evaluate(a2c, cfg, seeds)
        results.append(("A2C (B)", r["mean"]))
    except Exception as e:
        print(f"  [SKIP] A2C: {e}")

    try:
        bc = BCA2CAgent(obs_dim, n_actions)
        r = evaluate(bc, cfg, seeds)
        results.append(("BC+A2C best (B)", r["mean"]))
    except Exception as e:
        print(f"  [SKIP] BC+A2C: {e}")

    # === ROL C - Dyna-Q ===
    print("Evaluating Dyna-Q (C)...")
    dyna_costs = []
    for s in [0, 1, 2]:
        wpath = os.path.join(weight_dir, f"dyna_q_seed{s}.npz")
        if os.path.exists(wpath):
            try:
                r = evaluate(DynaQPolicy(cfg, wpath), cfg, seeds)
                dyna_costs.append(r["mean"])
            except Exception as e:
                print(f"  [SKIP] Dyna-Q seed{s}: {e}")
    if dyna_costs:
        avg = {k: float(np.mean([m[k] for m in dyna_costs])) for k in dyna_costs[0]}
        results.append(("Dyna-Q (C)", avg))

    print_table(results)

if __name__ == "__main__":
    main()
