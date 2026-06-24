"""Multi-agent IDQN degerlendirme: egitilmis vs rastgele baseline.
MA env global metrik dondurmuyor, bu yuzden episode basina toplam odul kiyaslanir."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drone_dispatch_env"))
import numpy as np
import gymnasium as gym
import drone_dispatch_env
from ma_idqn_agent import IDQNAgent


def run_policy(env, agents, choose_fn, n_episodes=10, base_seed=10000):
    """Bir politikayi n episode kostur, episode basina toplam odulu dondur."""
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            actions = {ag: choose_fn(obs[ag]) for ag in agents}
            obs, rewards, term, trunc, _ = env.step(actions)
            done = all(term.values()) or all(trunc.values())
            ep_reward += sum(rewards.values())
        totals.append(ep_reward)
    return np.array(totals)


env = gym.make("DroneDispatchMA-v0")
obs, _ = env.reset(seed=0)
agents = list(obs.keys())
obs_dim = obs[agents[0]].shape[0]
n_actions = env.action_space[agents[0]].n

# 1) Egitilmis IDQN (paylasilan ag), eps=0 (deterministik)
agent = IDQNAgent(obs_dim, n_actions, seed=0)
agent.load("../weights/ma_idqn.pt")
trained = run_policy(env, agents, lambda o: agent.select(o, 0.0))

# 2) Rastgele baseline
rng = np.random.default_rng(0)
random_pol = run_policy(env, agents, lambda o: int(rng.integers(n_actions)))

print("=== Multi-agent IDQN degerlendirme (10 episode) ===")
print(f"Rastgele baseline : odul = {random_pol.mean():8.1f} +/- {random_pol.std():.1f}")
print(f"Egitilmis IDQN    : odul = {trained.mean():8.1f} +/- {trained.std():.1f}")
improve = (trained.mean() - random_pol.mean()) / abs(random_pol.mean()) * 100
print(f"Iyilesme          : %{improve:.0f}")