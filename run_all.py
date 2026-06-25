"""
run_all.py — Role A + B + C + Offline RL evaluation script
Usage:
    python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_dispatch_env"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import numpy as np
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest
from drone_dispatch_env import DroneDispatchEnv

from dyna_q_policy import DynaQPolicy
from train_dqn import DuelingQNetwork, QNetwork, DQNPolicy, OBS_DIM
from reinforce_agent import obs_to_vector, REINFORCEAgent
from a2c_agent import A2CAgent
from train_offline import OfflineDQNPolicy


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
        q_net = DuelingQNetwork(341, n_actions, hidden)
    else:
        q_net = QNetwork(341, n_actions, hidden)
    q_net.load_state_dict(torch.load(path, map_location=device))
    return DQNPolicy(q_net, device)


def print_table(results):
    print("\n" + "="*70)
    print(f"{'Method':<30} {'cost_per_order':>14} {'success_rate':>12} {'ontime_rate':>11}")
    print("="*70)
    for name, m in results:
        print(f"{name:<30} {m['cost_per_order']:>14.4f} {m['success_rate']:>12.4f} {m['ontime_rate']:>11.4f}")
    print("="*70)
    print("Lower cost_per_order = better. Baseline (greedy_nearest) = 4.5700")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds",  default="0,1,2")
    args = parser.parse_args()

    cfg   = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []

    # === Baselines ===
    print("Evaluating baselines...")
    r = evaluate(RandomPolicy(cfg), cfg, seeds)
    results.append(("Random", r["mean"]))
    r = evaluate(GreedyNearest(cfg), cfg, seeds)
    results.append(("GreedyNearest (baseline)", r["mean"]))

    weight_dir = "weights"

    # === ROL A - DQN ===
    print("Evaluating Role A (DQN)...")
    models = [
        ("DQN Plain",           "dqn_plain_best.pt",         "plain",   256),
        ("Double DQN",          "dqn_double_best.pt",        "double",  256),
        ("Dueling DQN",         "dqn_dueling_best.pt",       "dueling", 256),
        ("Dueling DQN (tuned)", "dqn_dueling_tuned_best.pt", "dueling", 512),
    ]
    for name, fname, ntype, hidden in models:
        path = os.path.join(weight_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {name} - weight not found")
            continue
        try:
            policy = load_dqn(path, network_type=ntype, hidden=hidden)
            r = evaluate(policy, cfg, seeds)
            results.append((name, r["mean"]))
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # === ROL B - Policy-based ===
    print("Evaluating Role B (Policy-based)...")
    env_tmp = DroneDispatchEnv()
    obs_tmp, _ = env_tmp.reset(seed=0)
    obs_dim   = obs_to_vector(obs_tmp).shape[0]
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
    print("Evaluating Role C (Dyna-Q)...")
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

    # === Offline RL ===
    print("Evaluating Offline RL...")
    obs_dim_offline = 581

    try:
        q_net = DuelingQNetwork(obs_dim_offline, n_actions, 256)
        q_net.load_state_dict(torch.load(
            os.path.join(weight_dir, "offline_naive_dqn.pt"), map_location="cpu"))
        policy = OfflineDQNPolicy(q_net, torch.device("cpu"), obs_dim_offline)
        r = evaluate(policy, cfg, seeds)
        results.append(("Naive Offline DQN", r["mean"]))
    except Exception as e:
        print(f"  [SKIP] Naive Offline DQN: {e}")

    try:
        q_net = DuelingQNetwork(obs_dim_offline, n_actions, 256)
        q_net.load_state_dict(torch.load(
            os.path.join(weight_dir, "offline_cql.pt"), map_location="cpu"))
        policy = OfflineDQNPolicy(q_net, torch.device("cpu"), obs_dim_offline)
        r = evaluate(policy, cfg, seeds)
        results.append(("CQL (Offline RL)", r["mean"]))
    except Exception as e:
        print(f"  [SKIP] CQL: {e}")

    print_table(results)
# === JOINT: Multi-agent IDQN (ayri ortam, ayri metrik) ===
    print("\nEvaluating Multi-agent IDQN (separate env: DroneDispatchMA-v0)...")
    try:
        import numpy as _np
        import gymnasium as gym
        import drone_dispatch_env
        ma_env = gym.make("DroneDispatchMA-v0")
        ma_obs, _ = ma_env.reset(seed=0)
        ma_agents = list(ma_obs.keys())
        ma_odim = ma_obs[ma_agents[0]].shape[0]
        ma_nact = ma_env.action_space[ma_agents[0]].n
        from ma_idqn_agent import IDQNAgent
        ma_agent = IDQNAgent(ma_odim, ma_nact, seed=0)
        ma_agent.load(os.path.join(weight_dir, "ma_idqn.pt"))

        def _ma_run(choose, n_ep=10):
            tot = []
            for ep in range(n_ep):
                o, _ = ma_env.reset(seed=10000 + ep)
                R = 0.0; done = False
                while not done:
                    acts = {ag: choose(o[ag]) for ag in ma_agents}
                    o, rew, term, trunc, _ = ma_env.step(acts)
                    done = all(term.values()) or all(trunc.values())
                    R += sum(rew.values())
                tot.append(R)
            return _np.mean(tot), _np.std(tot)

        rng = _np.random.default_rng(0)
        rnd_m, rnd_s = _ma_run(lambda x: int(rng.integers(ma_nact)))
        trn_m, trn_s = _ma_run(lambda x: ma_agent.select(x, 0.0))
        print("  Metric: total reward per episode (MA env has no cost_per_order)")
        print(f"  Random baseline : {rnd_m:8.1f} +/- {rnd_s:.1f}")
        print(f"  Trained IDQN    : {trn_m:8.1f} +/- {trn_s:.1f}")
        print(f"  Improvement     : {(trn_m-rnd_m)/abs(rnd_m)*100:.0f}%")
    except Exception as e:
        print(f"  [SKIP] Multi-agent IDQN: {e}")

if __name__ == "__main__":
    main()