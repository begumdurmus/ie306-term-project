"""
Offline RL (Ch. 20)
Step 1: Naive offline DQN — demonstrates Q overestimation / OOD failure
Step 2: CQL (Conservative Q-Learning) — fixes overestimation

Usage:
    python train_offline.py --mode naive --config configs/eval_standard.yaml
    python train_offline.py --mode cql   --config configs/eval_standard.yaml
"""

import sys
import os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "drone_dispatch_env"))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, BASE)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate
from reinforce_agent import obs_to_vector
from train_dqn import DuelingQNetwork, QNetwork, OBS_DIM


# ── Offline Dataset Loader ────────────────────────────────────────────────────
def load_dataset(path):
    d = np.load(path)
    print(f"Dataset loaded: {len(d['actions'])} transitions")
    print(f"  obs shape: {d['observations'].shape}")
    print(f"  episode_returns: mean={d['episode_returns'].mean():.1f}, "
          f"min={d['episode_returns'].min():.1f}, max={d['episode_returns'].max():.1f}")
    return d


# ── Policy wrapper ────────────────────────────────────────────────────────────
class OfflineDQNPolicy:
    def __init__(self, q_net, device, obs_dim):
        self.q_net = q_net
        self.device = device
        self.obs_dim = obs_dim
        self.q_net.eval()

    def act(self, obs):
        x = torch.tensor(obs_to_vector(obs), dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # pad or truncate to obs_dim
        if x.shape[1] < self.obs_dim:
            x = torch.nn.functional.pad(x, (0, self.obs_dim - x.shape[1]))
        elif x.shape[1] > self.obs_dim:
            x = x[:, :self.obs_dim]
        with torch.no_grad():
            q = self.q_net(x).squeeze(0)
        mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=self.device)
        q[~mask] = -float("inf")
        return int(q.argmax().item())


# ── Naive Offline DQN ─────────────────────────────────────────────────────────
def train_naive_offline(data, cfg, device, n_steps=50000, batch_size=256, lr=3e-4):
    """
    Naive offline DQN: train on static dataset without any conservatism.
    This WILL overestimate Q values on OOD actions — that's the point.
    """
    print("\n=== NAIVE OFFLINE DQN ===")
    obs_dim = data["observations"].shape[1]
    n_actions = int(data["actions"].max()) + 1
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")

    q_net    = DuelingQNetwork(obs_dim, n_actions, 256).to(device)
    q_target = DuelingQNetwork(obs_dim, n_actions, 256).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    obs_t   = torch.tensor(data["observations"],      dtype=torch.float32)
    act_t   = torch.tensor(data["actions"],           dtype=torch.long)
    rew_t   = torch.tensor(data["rewards"],           dtype=torch.float32)
    nobs_t  = torch.tensor(data["next_observations"], dtype=torch.float32)
    done_t  = torch.tensor(data["terminals"] | data["timeouts"], dtype=torch.float32)

    N = len(act_t)
    q_values_log = []  # track Q values to show overestimation

    os.makedirs("logs", exist_ok=True)
    log_file = open("logs/offline_naive.csv", "w")
    log_file.write("step,loss,mean_q,max_q\n")

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,))
        ob  = obs_t[idx].to(device)
        ac  = act_t[idx].to(device)
        re  = rew_t[idx].to(device)
        nob = nobs_t[idx].to(device)
        dn  = done_t[idx].to(device)

        with torch.no_grad():
            next_actions = q_net(nob).argmax(dim=1)
            next_q = q_target(nob).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = re + 0.99 * next_q * (1 - dn)

        current_q = q_net(ob).gather(1, ac.unsqueeze(1)).squeeze(1)
        loss = nn.functional.huber_loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if step % 2000 == 0:
            q_target.load_state_dict(q_net.state_dict())

        if step % 5000 == 0:
            with torch.no_grad():
                sample_q = q_net(obs_t[:1000].to(device))
                mean_q = sample_q.max(dim=1)[0].mean().item()
                max_q  = sample_q.max().item()
            q_values_log.append((step, mean_q, max_q))
            log_file.write(f"{step},{loss.item():.4f},{mean_q:.2f},{max_q:.2f}\n")
            log_file.flush()
            print(f"  step={step:>6}  loss={loss.item():.4f}  "
                  f"mean_Q={mean_q:.2f}  max_Q={max_q:.2f}")

    log_file.close()

    # evaluate
    print("\nEvaluating naive offline DQN...")
    policy = OfflineDQNPolicy(q_net, device, obs_dim)
    result = evaluate(policy, cfg, seeds=[0, 1, 2])
    cost = result["mean"]["cost_per_order"]
    print(f"Naive Offline DQN cost_per_order={cost:.4f}")
    print(f"Note: max Q={max([x[2] for x in q_values_log]):.1f} — "
          f"overestimation expected due to OOD actions!")

    os.makedirs("weights", exist_ok=True)
    torch.save(q_net.state_dict(), "weights/offline_naive_dqn.pt")
    return q_net, q_values_log


# ── CQL (Conservative Q-Learning) ────────────────────────────────────────────
def train_cql(data, cfg, device, n_steps=50000, batch_size=256, lr=3e-4, alpha=1.0):
    """
    CQL: adds a conservatism penalty that pushes Q values DOWN on OOD actions.
    Loss = Bellman loss + alpha * (logsumexp(Q) - Q(s,a_data))
    """
    print(f"\n=== CQL (alpha={alpha}) ===")
    obs_dim = data["observations"].shape[1]
    n_actions = int(data["actions"].max()) + 1

    q_net    = DuelingQNetwork(obs_dim, n_actions, 256).to(device)
    q_target = DuelingQNetwork(obs_dim, n_actions, 256).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    obs_t   = torch.tensor(data["observations"],      dtype=torch.float32)
    act_t   = torch.tensor(data["actions"],           dtype=torch.long)
    rew_t   = torch.tensor(data["rewards"],           dtype=torch.float32)
    nobs_t  = torch.tensor(data["next_observations"], dtype=torch.float32)
    done_t  = torch.tensor(data["terminals"] | data["timeouts"], dtype=torch.float32)

    N = len(act_t)
    q_values_log = []

    os.makedirs("logs", exist_ok=True)
    log_file = open("logs/offline_cql.csv", "w")
    log_file.write("step,bellman_loss,cql_loss,total_loss,mean_q,max_q\n")

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,))
        ob  = obs_t[idx].to(device)
        ac  = act_t[idx].to(device)
        re  = rew_t[idx].to(device)
        nob = nobs_t[idx].to(device)
        dn  = done_t[idx].to(device)

        # Bellman loss
        with torch.no_grad():
            next_actions = q_net(nob).argmax(dim=1)
            next_q = q_target(nob).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = re + 0.99 * next_q * (1 - dn)

        q_all = q_net(ob)  # [B, n_actions]
        current_q = q_all.gather(1, ac.unsqueeze(1)).squeeze(1)
        bellman_loss = nn.functional.huber_loss(current_q, target_q)

        # CQL penalty: logsumexp(Q) - Q(s, a_data)
        # This penalizes high Q values on actions NOT in the dataset
        cql_loss = (torch.logsumexp(q_all, dim=1) - current_q).mean()

        total_loss = bellman_loss + alpha * cql_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if step % 2000 == 0:
            q_target.load_state_dict(q_net.state_dict())

        if step % 5000 == 0:
            with torch.no_grad():
                sample_q = q_net(obs_t[:1000].to(device))
                mean_q = sample_q.max(dim=1)[0].mean().item()
                max_q  = sample_q.max().item()
            q_values_log.append((step, mean_q, max_q))
            log_file.write(f"{step},{bellman_loss.item():.4f},{cql_loss.item():.4f},"
                           f"{total_loss.item():.4f},{mean_q:.2f},{max_q:.2f}\n")
            log_file.flush()
            print(f"  step={step:>6}  bellman={bellman_loss.item():.4f}  "
                  f"cql={cql_loss.item():.4f}  mean_Q={mean_q:.2f}  max_Q={max_q:.2f}")

    log_file.close()

    # evaluate
    print("\nEvaluating CQL...")
    policy = OfflineDQNPolicy(q_net, device, obs_dim)
    result = evaluate(policy, cfg, seeds=[0, 1, 2])
    cost = result["mean"]["cost_per_order"]
    print(f"CQL cost_per_order={cost:.4f}")
    print(f"Note: max Q={max([x[2] for x in q_values_log]):.1f} — "
          f"should be LOWER than naive DQN (conservatism working!)")

    os.makedirs("weights", exist_ok=True)
    torch.save(q_net.state_dict(), "weights/offline_cql.pt")
    return q_net, q_values_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="both", choices=["naive", "cql", "both"])
    parser.add_argument("--config",  default="configs/eval_standard.yaml")
    parser.add_argument("--dataset", default="offline_dataset.npz")
    parser.add_argument("--steps",   type=int, default=50000)
    parser.add_argument("--alpha",   type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg  = Config.from_yaml(args.config)
    data = load_dataset(args.dataset)

    if args.mode in ("naive", "both"):
        train_naive_offline(data, cfg, device, n_steps=args.steps)

    if args.mode in ("cql", "both"):
        train_cql(data, cfg, device, n_steps=args.steps, alpha=args.alpha)


if __name__ == "__main__":
    main()
