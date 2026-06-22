import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_dispatch_env'))

import csv
from drone_dispatch_env import DroneDispatchEnv
from a2c_agent import A2CAgent
from reinforce_agent import obs_to_vector

def train(seed=0, n_episodes=1000, save_path="weights/a2c.pt", log_path="logs/a2c_seed{}.csv"):
    log_path = log_path.format(seed)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneDispatchEnv()
    obs, info = env.reset(seed=seed)
    obs_dim = obs_to_vector(obs).shape[0]
    n_actions = env.action_space.n
    agent = A2CAgent(obs_dim=obs_dim, n_actions=n_actions)

    print(f"A2C başlıyor: obs_dim={obs_dim}, n_actions={n_actions}, seed={seed}")

    best_return = -float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "loss"])

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
            writer.writerow([ep, round(ep_return, 3), round(loss, 4)])

            if ep % 100 == 0:
                print(f"Ep {ep:4d} | return={ep_return:.1f} | loss={loss:.4f}")

            if ep_return > best_return:
                best_return = ep_return
                agent.save(save_path)

    print(f"\nEğitim bitti. En iyi return: {best_return:.2f}")
    print(f"Model: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    train(seed=args.seed, n_episodes=args.episodes)
