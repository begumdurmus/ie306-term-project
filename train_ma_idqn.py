"""Multi-agent IDQN egitim dongusu (parametre paylasimli) - Bolum 21."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "drone_dispatch_env"))

import numpy as np
import gymnasium as gym
import drone_dispatch_env
from ma_idqn_agent import IDQNAgent


def train(episodes=300, seed=0, eps_start=1.0, eps_end=0.05,
          out_weights="weights/ma_idqn.pt", out_log="logs/ma_idqn.csv"):
    env = gym.make("DroneDispatchMA-v0")
    obs, _ = env.reset(seed=seed)
    agents = list(obs.keys())                 # drone_0 ... drone_7
    obs_dim = obs[agents[0]].shape[0]         # 59
    n_actions = env.action_space[agents[0]].n # 4

    # TEK paylasilan ajan (parametre paylasimi)
    agent = IDQNAgent(obs_dim, n_actions, seed=seed)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    rows = []

    for ep in range(episodes):
        frac = min(1.0, ep / (0.8 * episodes))
        eps = eps_start + frac * (eps_end - eps_start)
        obs, _ = env.reset(seed=seed * 100000 + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # her ajan kendi gozlemine gore aksiyon secsin (paylasilan ag)
            actions = {ag: agent.select(obs[ag], eps) for ag in agents}
            next_obs, rewards, term, trunc, _ = env.step(actions)
            done = (term if isinstance(term, bool) else all(term.values())) \
                or (trunc if isinstance(trunc, bool) else all(trunc.values()))

            # her ajanin gecisini ortak buffer'a at
            for ag in agents:
                d = done
                agent.buffer.push(obs[ag], actions[ag], rewards[ag],
                                  next_obs[ag], d)
                ep_reward += rewards[ag]

            agent.learn(batch_size=64)
            obs = next_obs

        rows.append((ep, ep_reward, round(eps, 4)))
        if ep % 20 == 0 or ep == episodes - 1:
            print(f"ep {ep:4d}/{episodes}  toplam_odul={ep_reward:8.1f}  eps={eps:.3f}")

    import csv
    with open(out_log, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "reward", "eps"]); w.writerows(rows)
    agent.save(out_weights)
    print(f"\nBitti. Agirlik -> {out_weights}, log -> {out_log}")
    return agent


if __name__ == "__main__":
    train(episodes=30, seed=0)   # once kisa test