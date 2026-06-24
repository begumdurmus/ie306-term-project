"""Rol C - Planning: Ogrenme egrileri (planlama adimi ablasyonu).
Her n icin 3-seed ortalama +/- std egriyi (yumusatilmis) cizer."""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # ekran olmadan dosyaya kaydet
import matplotlib.pyplot as plt

LOG_DIR = "../logs"
SEEDS = [0, 1, 2]
TAGS = {"n0": "n=0 (planlama yok)", "n5": "n=5", "n10": "n=10", "n50": "n=50"}
WINDOW = 20                    # hareketli ortalama penceresi


def moving_average(x, w):
    """Basit hareketli ortalama (egriyi yumusatir)."""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_seed_rewards(tag, seed):
    """Bir koşunun episode odullerini yukle (basligi atla)."""
    path = f"{LOG_DIR}/{tag}_seed{seed}.csv"
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 1]          # reward sutunu (indeks 1)


plt.figure(figsize=(9, 6))

for tag, label in TAGS.items():
    # her seed icin yumusatilmis egri
    curves = [moving_average(load_seed_rewards(tag, s), WINDOW) for s in SEEDS]
    L = min(len(c) for c in curves)          # ayni uzunluga kirp
    curves = np.array([c[:L] for c in curves])
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    x = np.arange(L)
    plt.plot(x, mean, label=label, linewidth=2)
    plt.fill_between(x, mean - std, mean + std, alpha=0.15)   # +/- std bandi

plt.xlabel("Episode (yumusatilmis)")
plt.ylabel("Episode toplam odulu")
plt.title("Dyna-Q ogrenme egrisi: planlama adimi (n) etkisi\n(3 seed ortalama +/- std)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../logs/learning_curve_ablation.png", dpi=130)
print("Kaydedildi: logs/learning_curve_ablation.png")