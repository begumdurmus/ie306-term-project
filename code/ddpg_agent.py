import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # output: speed [0,1], heading [-1,1]
        self.speed_act  = nn.Sigmoid()
        self.heading_act = nn.Tanh()

    def forward(self, x):
        out = self.net(x)
        speed   = self.speed_act(out[:, 0:1])
        heading = self.heading_act(out[:, 1:2])
        return torch.cat([speed, heading], dim=1)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=1))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s),  dtype=torch.float32),
                torch.tensor(np.array(a),  dtype=torch.float32),
                torch.tensor(np.array(r),  dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(np.array(d),  dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)

class DDPGAgent:
    def __init__(self, obs_dim=7, action_dim=2, lr=3e-4, gamma=0.99,
                 tau=0.005, noise_std=0.1):
        self.gamma     = gamma
        self.tau       = tau
        self.noise_std = noise_std
        self.action_dim = action_dim

        self.actor        = Actor(obs_dim, action_dim)
        self.actor_target = Actor(obs_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer()

    def act(self, obs, explore=False):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(x).squeeze(0).numpy()
        if explore:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action[0] = np.clip(action[0], 0, 1)    # speed
            action[1] = np.clip(action[1], -1, 1)   # heading
        return action

    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return 0.0, 0.0

        s, a, r, s2, d = self.buffer.sample(batch_size)

        with torch.no_grad():
            a2     = self.actor_target(s2)
            q_next = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * q_next

        q_val    = self.critic(s, a)
        c_loss   = nn.functional.mse_loss(q_val, q_target)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        a_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        # soft update
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return float(c_loss.detach()), float(a_loss.detach())

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor.eval()
