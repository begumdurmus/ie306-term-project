"""Rol C - Planning: Dyna-Q egitim dongusu."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))   # code/ klasorunu bul
# ic ice 'drone_dispatch_env' klasoru kurulu paketi golgeliyor; gercek paketi zorla bul
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "drone_dispatch_env"))
import numpy as np
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config

from dispatch_features import decision_context, nearest_hub_distance, ASSIGN, CHARGE, HOLD
from dyna_q_agent import DynaQAgent


def macro_to_env_action(macro, assign_action, charge_action, noop_action):
    """Secilen makro-aksiyonu (0/1/2) env'in gercek aksiyon numarasina cevir."""
    if macro == ASSIGN:
        return assign_action
    if macro == CHARGE:
        return charge_action
    return noop_action


def run_episode(agent, env, cfg, seed, eps):
    """Tek bir episode kostur; ajan ogrensin. Toplam odulu dondur."""
    obs, _ = env.reset(seed=seed)
    hub_field = nearest_hub_distance(obs["grid"])      # bu episode icin hub mesafeleri
    total_reward = 0.0
    done = False

    while not done:
        state, mask, a_act, c_act, n_act, info = decision_context(obs, cfg, hub_field)

        if state is None:                              # gercek karar yok -> sadece bekle
            obs, r, term, trunc, _ = env.step(n_act)
            total_reward += r
            done = term or trunc
            continue

        macro = agent.select(state, mask, eps)         # ajan karar versin
        env_action = macro_to_env_action(macro, a_act, c_act, n_act)
        obs, r, term, trunc, _ = env.step(env_action)  # env'de uygula
        total_reward += r
        done = term or trunc

        if done:
            agent.observe(state, macro, r, None, True)
        else:
            next_state, _, _, _, _, _ = decision_context(obs, cfg, hub_field)
            agent.observe(state, macro, r, next_state, False)
        agent.plan()                                   # hayali pratik

    return total_reward


import csv


def train(episodes=30, planning_steps=10, seed=0,
          eps_start=1.0, eps_end=0.05,
          out_weights="weights/dyna_q_seed0.npz",
          out_log="logs/dyna_q_seed0.csv"):
    cfg = Config()
    env = gym.make("DroneDispatch-v0")
    agent = DynaQAgent(planning_steps=planning_steps, seed=seed)

    os.makedirs(os.path.dirname(out_weights), exist_ok=True)
    os.makedirs(os.path.dirname(out_log), exist_ok=True)

    rows = []
    for ep in range(episodes):
        # epsilon: ilk %80'de eps_start -> eps_end dogrusal azalir, sonra sabit
        frac = min(1.0, ep / (0.8 * episodes))
        eps = eps_start + frac * (eps_end - eps_start)

        R = run_episode(agent, env, cfg, seed=seed * 100000 + ep, eps=eps)
        rows.append((ep, R, round(eps, 4), len(agent.Q)))

        if ep % 20 == 0 or ep == episodes - 1:
            print(f"ep {ep:4d}/{episodes}  odul={R:8.1f}  eps={eps:.3f}  |Q|={len(agent.Q)}")

    # logu kaydet
    with open(out_log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "eps", "n_states"])
        w.writerows(rows)

    # agirliklari kaydet
    agent.save(out_weights)
    print(f"\nBitti. Agirliklar -> {out_weights}, log -> {out_log}")
    return agent


if __name__ == "__main__":
    train(episodes=400, planning_steps=10, seed=0)