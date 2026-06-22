import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import csv
import numpy as np
from drone_dispatch_env import DroneControlEnv
from ddpg_agent import DDPGAgent

def train(seed=0, n_episodes=500, save_path="weights/ddpg.pt", log_path="logs/ddpg_seed{}.csv"):
    log_path = log_path.format(seed)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneControlEnv()
    agent = DDPGAgent(obs_dim=7, action_dim=2)

    np.random.seed(seed)
    print(f"DDPG başlıyor: seed={seed}")

    best_return = -float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "critic_loss", "actor_loss"])

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            c_losses, a_losses = [], []

            while not done:
                action = agent.act(obs, explore=True)
                obs2, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                agent.buffer.push(obs, action, reward, obs2, float(done))
                obs = obs2
                ep_return += reward

                c_loss, a_loss = agent.update(batch_size=256)
                if c_loss > 0:
                    c_losses.append(c_loss)
                    a_losses.append(a_loss)

            mean_c = np.mean(c_losses) if c_losses else 0
            mean_a = np.mean(a_losses) if a_losses else 0

            writer.writerow([ep, round(ep_return,3), round(mean_c,4), round(mean_a,4)])

            if ep % 50 == 0:
                print(f"Ep {ep:4d} | return={ep_return:.1f} | buffer={len(agent.buffer)}")

            if ep_return > best_return:
                best_return = ep_return
                agent.save(save_path)

    print(f"\nEğitim bitti. En iyi return: {best_return:.2f}")
    print(f"Model: {save_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=500)
    a = p.parse_args()
    train(seed=a.seed, n_episodes=a.episodes)
