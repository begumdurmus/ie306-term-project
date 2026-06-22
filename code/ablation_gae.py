import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from drone_dispatch_env import DroneDispatchEnv
from reinforce_agent import obs_to_vector

class ActorCritic(nn.Module):
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

def train_with_gae(lam, seed=0, n_episodes=500):
    """GAE lambda sweep için A2C eğitimi."""
    env = DroneDispatchEnv()
    obs, _ = env.reset(seed=seed)
    obs_dim = obs_to_vector(obs).shape[0]
    n_actions = env.action_space.n

    net = ActorCritic(obs_dim, n_actions)
    opt = optim.Adam(net.parameters(), lr=1e-4)

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/ablation_gae_lam{str(lam).replace('.','')}_seed{seed}.csv"

    best_return = -float("inf")
    returns_history = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "gae_lambda"])

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            log_probs, values, rewards, entropies = [], [], [], []

            while not done:
                mask = obs["action_mask"].astype(bool)
                x = torch.tensor(obs_to_vector(obs), dtype=torch.float32).unsqueeze(0)
                logits, value = net(x)
                logits = logits.squeeze(0)
                inf_mask = torch.zeros(n_actions)
                inf_mask[~mask] = -float("inf")
                dist = Categorical(logits=logits + inf_mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                values.append(value.squeeze(0))
                entropies.append(dist.entropy())
                obs, r, term, trunc, _ = env.step(int(action.item()))
                done = term or trunc
                rewards.append(r)
                ep_return += r

            # GAE hesapla
            values_t = torch.stack(values).squeeze(-1)
            T = len(rewards)
            advantages = torch.zeros(T)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values_t[t+1].item() if t+1 < T else 0.0
                delta = rewards[t] + 0.99 * next_val - values_t[t].item()
                gae = delta + 0.99 * lam * gae
                advantages[t] = gae

            returns = advantages + values_t.detach()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            loss = (-(torch.stack(log_probs) * advantages).mean()
                    + 0.5 * nn.functional.mse_loss(values_t, returns)
                    - 0.01 * torch.stack(entropies).mean())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

            writer.writerow([ep, round(ep_return, 3), lam])
            returns_history.append(ep_return)

            if ep_return > best_return:
                best_return = ep_return

    mean_last100 = sum(returns_history[-100:]) / 100
    print(f"λ={lam} | best={best_return:.1f} | mean_last100={mean_last100:.1f}")
    return mean_last100

if __name__ == "__main__":
    print("GAE Lambda Ablation başlıyor...")
    print("="*50)
    results = {}
    for lam in [0.0, 0.5, 0.8, 0.95, 1.0]:
        r = train_with_gae(lam=lam, seed=0, n_episodes=300)
        results[lam] = r

    print("\n=== Ablation Sonuçları ===")
    for lam, r in results.items():
        print(f"λ={lam:.2f} | mean_return_last100={r:.1f}")
