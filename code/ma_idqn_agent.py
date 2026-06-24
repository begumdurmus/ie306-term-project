"""Multi-agent IDQN (parametre paylasimli) - takim ortak bileseni (Bolum 21)."""
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class QNetwork(nn.Module):
    """Basit MLP: gozlem (obs_dim) -> her aksiyonun Q degeri (n_actions)."""
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


class ReplayBuffer:
    """Deneyim havuzu: gecisleri biriktirir, rastgele batch dondurur."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
class IDQNAgent:
    """Parametre paylasimli IDQN: tek Q-ag, tum ajanlar onu kullanir."""
    def __init__(self, obs_dim, n_actions, hidden=128, lr=1e-3,
                 gamma=0.99, target_update=500, seed=0):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update = target_update
        self.q = QNetwork(obs_dim, n_actions, hidden)          # ana ag
        self.q_target = QNetwork(obs_dim, n_actions, hidden)   # hedef ag
        self.q_target.load_state_dict(self.q.state_dict())     # ayni baslat
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.learn_steps = 0
        self.rng = np.random.default_rng(seed)

    def select(self, obs, eps):
        """Epsilon-greedy: tek bir ajanin gozlemi -> aksiyon (0-3)."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.q(x).squeeze(0)
            return int(torch.argmax(q).item())

    def learn(self, batch_size=64):
        """Replay buffer'dan batch cek, Q-agi DQN kaybiyla guncelle."""
        if len(self.buffer) < batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(batch_size)
        s  = torch.tensor(s);  a = torch.tensor(a)
        r  = torch.tensor(r);  s2 = torch.tensor(s2)
        done = torch.tensor(done)

        # mevcut Q(s,a)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        # hedef: r + gamma * max Q_target(s')  (bitti ise sadece r)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1)[0]
            target = r + self.gamma * q_next * (1.0 - done)
        loss = ((q_sa - target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # ara sira hedef agi guncelle
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location="cpu"))
        self.q_target.load_state_dict(self.q.state_dict())