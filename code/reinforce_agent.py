import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def obs_to_vector(obs):
    """Gözlemi düz bir numpy vektörüne çevirir."""
    drones = obs["drones"].flatten()      # drone pozisyon/durum bilgisi
    orders = obs["orders"].flatten()      # sipariş bilgisi
    grid  = obs["grid"].flatten()         # grid bilgisi
    time  = obs["time"].flatten()         # zaman
    return np.concatenate([drones, orders, grid, time]).astype(np.float32)

class PolicyNet(nn.Module):
    """Basit ileri beslemeli politika ağı."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) ajanı.
    Policy-based B rolü — ilk algoritma.
    """
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99):
        self.gamma    = gamma
        self.n_actions = n_actions
        self.policy   = PolicyNet(obs_dim, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # episode boyunca tutulanlar
        self.log_probs = []
        self.rewards   = []

    # ---- Policy protocol (agent_interface.py) ----
    def act(self, obs):
        mask = obs["action_mask"].astype(bool)
        vec  = obs_to_vector(obs)
        x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

        logits = self.policy(x).squeeze(0)

        # geçersiz aksiyonları -inf yap
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        logits = logits + inf_mask

        dist   = Categorical(logits=logits)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def action_probs(self, obs):
        mask = obs["action_mask"].astype(bool)
        vec  = obs_to_vector(obs)
        x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x).squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        probs = torch.softmax(logits + inf_mask, dim=0).numpy()
        return probs

    def state_values(self, obs):
        return None   # REINFORCE'ta value tahmini yok

    # ---- Eğitim ----
    def store_reward(self, r):
        self.rewards.append(r)

    def update(self):
        """Episode bittikten sonra parametreleri güncelle."""
        G, returns = 0.0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        # normalize et — eğitimi kararlı yapar
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for log_p, G in zip(self.log_probs, returns):
            loss += -log_p * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # episode verilerini temizle
        self.log_probs.clear()
        self.rewards.clear()

        return float(loss.detach())

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location="cpu"))
        self.policy.eval()
