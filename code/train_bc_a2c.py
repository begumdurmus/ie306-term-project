import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from drone_dispatch_env import DroneDispatchEnv
from drone_dispatch_env.config import Config
from drone_dispatch_env.baselines import GreedyNearest
from reinforce_agent import obs_to_vector

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor  = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

def collect_greedy_data(n_episodes=200, seed=0):
    """Greedy'nin yaptığı aksiyonları topla."""
    cfg = Config.from_yaml("drone_dispatch_env/configs/eval_standard.yaml")
    env = DroneDispatchEnv()
    greedy = GreedyNearest(cfg)
    data = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed+ep)
        done = False
        while not done:
            action = greedy.act(obs)
            vec = obs_to_vector(obs)
            data.append((vec, action))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
    print(f"Greedy verisi toplandı: {len(data)} adım")
    return data

def pretrain_bc(net, data, n_epochs=10, lr=1e-3):
    """Behavior cloning: greedy'yi taklit et."""
    optimizer = optim.Adam(net.parameters(), lr=lr)
    vecs    = torch.tensor([d[0] for d in data], dtype=torch.float32)
    actions = torch.tensor([d[1] for d in data], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(vecs, actions)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in loader:
            logits, _ = net(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"BC Epoch {epoch+1}/{n_epochs} | loss={total_loss/len(loader):.4f}")

def train_a2c(net, seed=0, n_episodes=1000,
              save_path="weights/bc_a2c.pt",
              log_path="logs/bc_a2c_seed{}.csv"):
    log_path = log_path.format(seed)
    env = DroneDispatchEnv()
    obs, _ = env.reset(seed=seed)
    n_actions = env.action_space.n
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    best_return = -float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "loss"])

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed+ep)
            done = False
            ep_return = 0.0
            log_probs, values, rewards, entropies = [], [], [], []

            while not done:
                mask = obs["action_mask"].astype(bool)
                vec = obs_to_vector(obs)
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                logits, value = net(x)
                logits = logits.squeeze(0)
                inf_mask = torch.zeros(n_actions)
                inf_mask[~mask] = -float("inf")
                dist = Categorical(logits=logits + inf_mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                values.append(value.squeeze(0))
                entropies.append(dist.entropy())

                obs, reward, term, trunc, _ = env.step(int(action.item()))
                done = term or trunc
                rewards.append(reward)
                ep_return += reward

            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            values_t = torch.stack(values).squeeze(-1)
            adv = returns - values_t.detach()
            if adv.std() > 1e-8:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            loss = (-(torch.stack(log_probs) * adv).mean()
                    + 0.5 * nn.functional.mse_loss(values_t, returns)
                    - 0.01 * torch.stack(entropies).mean())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            writer.writerow([ep, round(ep_return,3), round(float(loss.detach()),4)])
            if ep % 100 == 0:
                print(f"Ep {ep:4d} | return={ep_return:.1f}")

            if ep_return > best_return:
                best_return = ep_return
                torch.save(net.state_dict(), save_path)

    print(f"Bitti. En iyi return: {best_return:.2f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bc_episodes", type=int, default=200)
    p.add_argument("--a2c_episodes", type=int, default=1000)
    a = p.parse_args()

    env = DroneDispatchEnv()
    obs, _ = env.reset(seed=0)
    obs_dim = obs_to_vector(obs).shape[0]
    n_actions = env.action_space.n
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    net = ActorCritic(obs_dim, n_actions)

    print("=== Adım 1: Greedy veri toplama ===")
    data = collect_greedy_data(n_episodes=a.bc_episodes, seed=a.seed)

    print("=== Adım 2: Behavior Cloning ===")
    pretrain_bc(net, data, n_epochs=50)

    print("=== Adım 3: A2C Fine-tuning ===")
    train_a2c(net, seed=a.seed, n_episodes=a.a2c_episodes)
