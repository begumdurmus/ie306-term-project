import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from drone_dispatch_env import DroneDispatchEnv
from reinforce_agent import obs_to_vector

class SmallActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

def train(seed=0, n_episodes=2000, save_path="weights/a2c_v2.pt", log_path="logs/a2c_v2_seed{}.csv"):
    log_path = log_path.format(seed)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneDispatchEnv()
    obs, _ = env.reset(seed=seed)
    obs_dim = obs_to_vector(obs).shape[0]
    n_actions = env.action_space.n

    net = SmallActorCritic(obs_dim, n_actions)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # daha düşük lr

    print(f"A2C_v2 başlıyor: seed={seed}")
    best_return = -float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "loss"])

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
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

                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                rewards.append(reward)
                ep_return += reward

            # returns
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
            if ep % 200 == 0:
                print(f"Ep {ep:4d} | return={ep_return:.1f}")

            if ep_return > best_return:
                best_return = ep_return
                torch.save(net.state_dict(), save_path)

    print(f"Bitti. En iyi return: {best_return:.2f} | Model: {save_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=2000)
    a = p.parse_args()
    train(seed=a.seed, n_episodes=a.episodes)
