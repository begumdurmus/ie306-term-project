"""
Role A — DQN Dispatcher (final)
Supports: plain DQN, Double DQN, Dueling DQN
Config controls which variant runs via 'network' key.

Usage:
    python train_dqn.py --config configs/dqn_plain.yaml   --seed 0
    python train_dqn.py --config configs/dqn_double.yaml  --seed 0
    python train_dqn.py --config configs/dqn_dueling.yaml --seed 0
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import evaluate, Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_obs(obs, H=20, W=20, sla=60):
    N, K = 8, 20
    drones = obs["drones"].copy()
    drones[:, 0] /= H
    drones[:, 1] /= W

    orders = obs["orders"].copy()
    orders[:, 0] /= H
    orders[:, 1] /= W
    orders[:, 2] /= H
    orders[:, 3] /= W
    orders[:, 4] = np.clip((sla - orders[:, 4]) / sla, 0.0, 1.0)

    drone_pos = obs["drones"][:, :2]
    order_ox  = obs["orders"][:, 0:2]
    dist_matrix = np.zeros((N, K), dtype=np.float32)
    max_dist = H + W
    for d in range(N):
        for k in range(K):
            if obs["orders"][k, 0] == 0 and obs["orders"][k, 1] == 0:
                dist_matrix[d, k] = 1.0
            else:
                dist_matrix[d, k] = (abs(drone_pos[d,0]-order_ox[k,0]) +
                                     abs(drone_pos[d,1]-order_ox[k,1])) / max_dist

    return np.concatenate([
        drones.flatten(), orders.flatten(),
        dist_matrix.flatten(), obs["time"].flatten()
    ]).astype(np.float32)


OBS_DIM = 8*10 + 20*5 + 8*20 + 1  # 341


# ── Networks ──────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Plain DQN network."""
    def __init__(self, obs_size, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN network."""
    def __init__(self, obs_size, n_actions, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)


def make_network(network_type, obs_size, n_actions, hidden):
    if network_type == "dueling":
        return DuelingQNetwork(obs_size, n_actions, hidden)
    else:  # plain or double
        return QNetwork(obs_size, n_actions, hidden)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),     dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.long),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def select_action(q_values, action_mask, epsilon):
    valid = np.where(action_mask)[0]
    if random.random() < epsilon:
        return int(random.choice(valid))
    q = q_values.clone()
    q[~torch.tensor(action_mask, dtype=torch.bool)] = -float("inf")
    return int(q.argmax().item())


class DQNPolicy:
    def __init__(self, q_net, device):
        self.q_net = q_net
        self.device = device
        self.q_net.eval()

    def act(self, obs):
        x = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(x).squeeze(0)
        mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=self.device)
        q[~mask] = -float("inf")
        return int(q.argmax().item())


# ── Training ──────────────────────────────────────────────────────────────────
def train(cfg, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_type = cfg.get("network", "dueling")
    print(f"[DQN] variant={network_type}  seed={seed}  device={device}")

    env = gym.make("DroneDispatch-v0")
    n_actions = env.action_space.n

    q_net    = make_network(network_type, OBS_DIM, n_actions, cfg["hidden"]).to(device)
    q_target = make_network(network_type, OBS_DIM, n_actions, cfg["hidden"]).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["lr"])
    buffer    = ReplayBuffer(cfg["buffer_size"])

    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/dqn_{network_type}_seed{seed}.csv", "w")
    log_file.write("step,episode,ep_return,epsilon\n")

    epsilon         = cfg["eps_start"]
    eps_end         = cfg["eps_end"]
    eps_decay_steps = cfg["eps_decay_steps"]
    total_steps     = cfg["total_steps"]
    batch_size      = cfg["batch_size"]
    target_freq     = cfg["target_update_freq"]
    train_freq      = cfg["train_freq"]
    warmup          = cfg["warmup_steps"]
    use_double      = network_type in ("double", "dueling")

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    episode   = 0
    best_cost = float("inf")

    for step in range(1, total_steps + 1):
        obs_flat = flatten_obs(obs)
        with torch.no_grad():
            q_vals = q_net(torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
        action = select_action(q_vals, obs["action_mask"], epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs_flat, action, reward, flatten_obs(next_obs), float(done))
        ep_return += reward
        obs = next_obs

        if done:
            episode += 1
            log_file.write(f"{step},{episode},{ep_return:.4f},{epsilon:.4f}\n")
            log_file.flush()
            if episode % 50 == 0:
                print(f"  step={step:>7}  ep={episode:>4}  return={ep_return:>8.1f}  eps={epsilon:.3f}")
            ep_return = 0.0
            obs, _ = env.reset()

        # linear epsilon decay
        if step < eps_decay_steps:
            epsilon = cfg["eps_start"] - (cfg["eps_start"] - eps_end) * step / eps_decay_steps
        else:
            epsilon = eps_end

        if step >= warmup and step % train_freq == 0 and len(buffer) >= batch_size:
            obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(batch_size)
            obs_b, act_b, rew_b, nobs_b, done_b = [x.to(device) for x in [obs_b, act_b, rew_b, nobs_b, done_b]]

            with torch.no_grad():
                if use_double:
                    # Double DQN: online selects, target evaluates
                    next_actions = q_net(nobs_b).argmax(dim=1)
                    next_q = q_target(nobs_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Plain DQN: target selects and evaluates
                    next_q = q_target(nobs_b).max(dim=1)[0]
                target_q = rew_b + cfg["gamma"] * next_q * (1 - done_b)

            current_q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.huber_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % target_freq == 0:
            q_target.load_state_dict(q_net.state_dict())

        if step % cfg.get("eval_freq", 50000) == 0:
            policy = DQNPolicy(q_net, device)
            result = evaluate(policy, Config(), seeds=[0, 1, 2])
            cost   = result["mean"]["cost_per_order"]
            print(f"  [EVAL] step={step}  cost_per_order={cost:.4f}  (baseline=4.570)")
            if cost < best_cost:
                best_cost = cost
                os.makedirs("weights", exist_ok=True)
                torch.save(q_net.state_dict(), f"weights/dqn_{network_type}_best.pt")
                print(f"  [SAVE] new best → weights/dqn_{network_type}_best.pt")

    log_file.close()
    env.close()
    os.makedirs("weights", exist_ok=True)
    torch.save(q_net.state_dict(), f"weights/dqn_{network_type}_seed{seed}.pt")
    print(f"\n[DONE] variant={network_type}  best cost_per_order={best_cost:.4f}")
    return q_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_dueling.yaml")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed)
