"""Rol C - Planning: Ablasyon cubuk grafigi (planlama adimi -> cost_per_order)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, GreedyNearest
from dyna_q_policy import DynaQPolicy

cfg = Config()
eval_seeds = [0, 1, 2]
train_seeds = [0, 1, 2]
tags = ["n0", "n5", "n10", "n50"]
labels = ["0", "5", "10", "50"]

means, stds = [], []
for tag in tags:
    costs = []
    for s in train_seeds:
        m = evaluate(DynaQPolicy(cfg, f"../weights/{tag}_seed{s}.npz"), cfg, seeds=eval_seeds)["mean"]
        costs.append(m["cost_per_order"])
    means.append(np.mean(costs))
    stds.append(np.std(costs))

greedy = evaluate(GreedyNearest(cfg), cfg, seeds=eval_seeds)["mean"]["cost_per_order"]

plt.figure(figsize=(8, 6))
x = np.arange(len(tags))
bars = plt.bar(x, means, yerr=stds, capsize=6, color="steelblue", alpha=0.85)
plt.axhline(greedy, color="red", linestyle="--", linewidth=2, label=f"greedy_nearest ({greedy:.2f})")

# cubuklarin uzerine deger yaz
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.05, f"{m:.3f}", ha="center", fontsize=10)

plt.xticks(x, labels)
plt.xlabel("Planlama adimi (n)")
plt.ylabel("cost_per_order (3 seed ortalama +/- std)")
plt.title("Ablasyon: Dyna planlama adiminin etkisi\n(dusuk = iyi; tum n'ler greedy'yi geciyor)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../logs/ablation_planning.png", dpi=130)
print("Kaydedildi: logs/ablation_planning.png")