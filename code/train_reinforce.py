import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import numpy as np
import csv
import torch
from drone_dispatch_env import DroneDispatchEnv
from reinforce_agent import REINFORCEAgent, obs_to_vector

def train(seed=0, n_episodes=500, save_path="weights/reinforce.pt", log_path="logs/reinforce_seed{}.csv"):
    log_path = log_path.format(seed)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneDispatchEnv()
    obs, info = env.reset(seed=seed)

    # obs boyutunu öğren
    obs_dim = obs_to_vector(obs).shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(obs_dim=obs_dim, n_actions=n_actions)

    print(f"obs_dim={obs_dim}, n_actions={n_actions}, seed={seed}")

    best_cost = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "cost_per_order", "loss"])

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0

            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_reward(reward)
                ep_return += reward

            loss = agent.update()
            stats = info.get("stats", {})
            n_delivered = max(stats.get("delivered", 1), 1)
            cost = -ep_return / n_delivered

            writer.writerow([ep, round(ep_return, 3), round(cost, 4), round(loss, 4)])

            if ep % 50 == 0:
                print(f"Ep {ep:4d} | return={ep_return:.1f} | cost_per_order={cost:.3f} | loss={loss:.4f}")

            if cost < best_cost:
                best_cost = cost
                agent.save(save_path)

    print(f"\nEğitim bitti. En iyi cost_per_order: {best_cost:.4f}")
    print(f"Model kaydedildi: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    train(seed=args.seed, n_episodes=args.episodes)
