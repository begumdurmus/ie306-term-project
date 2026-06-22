import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from reinforce_agent import obs_to_vector

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

class A2CAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_actions    = n_actions
        self.net       = ActorCriticNet(obs_dim, n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []

    def act(self, obs):
        mask = obs["action_mask"].astype(bool)
        vec  = obs_to_vector(obs)
        x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        logits, value = self.net(x)
        logits = logits.squeeze(0)
        value  = value.squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        logits = logits + inf_mask
        dist    = Categorical(logits=logits)
        action  = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())
        return int(action.item())

    def action_probs(self, obs):
        mask = obs["action_mask"].astype(bool)
        vec  = obs_to_vector(obs)
        x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits.squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        return torch.softmax(logits + inf_mask, dim=0).numpy()

    def state_values(self, obs):
        vec = obs_to_vector(obs)
        x   = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.net(x)
        return float(value.item())

    def store_reward(self, r):
        self.rewards.append(r)

    def update(self):
        G, returns = 0.0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values  = torch.stack(self.values).squeeze(-1)
        advantages = returns - values.detach()
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        return float(loss.detach())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    def act_deterministic(self, obs):
        """Eval için deterministik aksiyon — en yüksek olasılıklı aksiyonu seç."""
        mask = obs["action_mask"].astype(bool)
        vec  = obs_to_vector(obs)
        x    = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
        logits = logits.squeeze(0)
        inf_mask = torch.zeros(self.n_actions)
        inf_mask[~mask] = -float("inf")
        return int((logits + inf_mask).argmax().item())
