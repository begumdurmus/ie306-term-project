"""Rol C - Planning: Tablosal Dyna-Q ajani (Q-learning + model + planlama)."""
import numpy as np

N_ACTIONS = 2   # ASSIGN, CHARGE, HOLD


class DynaQAgent:
    def __init__(self, alpha=0.5, gamma=0.99, planning_steps=10, seed=0):
        self.alpha = alpha               # ogrenme orani (Q ne hizli guncellenir)
        self.gamma = gamma               # indirim faktoru (gelecek odullere onem)
        self.planning_steps = planning_steps   # her gercek adimda kac hayali guncelleme
        self.rng = np.random.default_rng(seed)

        self.Q = {}                      # durum -> np.array(3): aksiyon degerleri
        self.model = {}                  # (durum, aksiyon) -> [(odul, sonraki_durum, bitti), ...]
        self.seen = []                   # gorulen (durum, aksiyon) ciftleri (planlama icin)

    def q(self, state):
        """Bir durumun Q degerlerini dondur (yoksa sifirla baslat)."""
        if state not in self.Q:
            self.Q[state] = np.zeros(N_ACTIONS)
        return self.Q[state]

    def select(self, state, valid_mask, eps):
        """Epsilon-greedy: eps olasilikla rastgele gecerli aksiyon, yoksa en iyi gecerli."""
        valid_idx = np.where(valid_mask)[0]
        if self.rng.random() < eps:
            return int(self.rng.choice(valid_idx))
        q = np.where(valid_mask, self.q(state), -np.inf)   # gecersizleri -inf yap
        return int(np.argmax(q))

    def _update(self, s, a, r, s2, done):
        """Tek bir Q-learning guncellemesi (hem gercek hem hayali deneyim icin)."""
        if done or s2 is None:
            target = r                                  # episode bitti: gelecek yok
        else:
            target = r + self.gamma * np.max(self.q(s2))   # odul + indirimli en iyi gelecek
        self.q(s)[a] += self.alpha * (target - self.q(s)[a])

    def observe(self, s, a, r, s2, done):
        """Gercek bir gecisi isle: Q'yu guncelle + modele kaydet."""
        self._update(s, a, r, s2, done)                 # 1) gercek deneyimden ogren
        key = (s, a)                                    # 2) modele kaydet
        if key not in self.model:
            self.model[key] = []
            self.seen.append(key)
        self.model[key].append((r, s2, done))
        if len(self.model[key]) > 32:                   # hafizayi sinirla (cok eskiyi at)
            self.model[key].pop(0)   

    def plan(self):
        """Modelden planning_steps kadar hayali deneyim cekip Q'yu guncelle."""
        if not self.seen:
            return                                       # henuz hicbir sey gormediyse atla
        for _ in range(self.planning_steps):
            # gorulmus bir (durum, aksiyon) cifti rastgele sec
            i = self.rng.integers(len(self.seen))
            s, a = self.seen[i]
            # o cift icin saklanan sonuclardan birini rastgele sec (env stokastik)
            outcomes = self.model[(s, a)]
            r, s2, done = outcomes[self.rng.integers(len(outcomes))]
            # gercek deneyimle ayni guncelleme
            self._update(s, a, r, s2, done)

    def save(self, path):
        """Q-tablosunu .npz olarak kaydet (durumlar + degerler iki dizi halinde)."""
        keys = list(self.Q.keys())
        states = np.array(keys, dtype=np.int64) if keys else np.zeros((0, 5), dtype=np.int64)
        values = np.array([self.Q[k] for k in keys]) if keys else np.zeros((0, N_ACTIONS))
        np.savez(path, states=states, values=values)

    def load(self, path):
        """Kaydedilmis Q-tablosunu geri yukle."""
        data = np.load(path)
        states, values = data["states"], data["values"]
        self.Q = {tuple(int(x) for x in states[i]): values[i] for i in range(len(states))}