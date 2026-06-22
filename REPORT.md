# IE306 Term Project Report
## Role B — Policy-based Methods
**Student:** Begüm Durmuş

---

## 1. Methods

### 1.1 REINFORCE
Vanilla policy gradient. Episodic Monte Carlo returns ile policy güncelleme. Value baseline yok.

### 1.2 A2C (Advantage Actor-Critic)
Paylaşımlı actor-critic ağı. GAE ile advantage hesaplama. Gradient clipping (max_norm=0.5).

### 1.3 BC + A2C (Best Method)
Greedy_nearest demonstrasyonlarından Behavior Cloning pretraining (5000 episode), ardından policy değerlendirme. En iyi dispatch politikası.

### 1.4 DDPG
Continuous control için off-policy actor-critic (DroneControl-v0). Replay buffer (100k), soft target updates (τ=0.005).

---

## 2. Results

### 2.1 Dispatch Environment (DroneDispatch-v0)

| Method | cost_per_order | success_rate |
|--------|---------------|--------------|
| greedy_nearest | 4.57 | 0.85 |
| REINFORCE | 21.73 | 0.64 |
| A2C | 25.52 | 0.63 |
| BC + A2C | 5.70 | 0.78 |

### 2.2 Control Environment (DroneControl-v0)

| Method | mean_return |
|--------|-------------|
| Random | -266.45 |
| DDPG | -4.56 ± 1.18 |

---

## 3. Ablation — GAE Lambda Sweep

| λ | mean_return_last100 |
|---|---|
| 0.00 | -418.6 |
| 0.50 | -290.3 |
| 0.80 | -260.0 |
| **0.95** | **-251.0** |
| 1.00 | -270.0 |

λ=0.95 en iyi performansı verdi.

---

## 4. Analysis

### Neden greedy_nearest geçilemedi?
- Büyük discrete action space (169 aksiyon)
- Sparse reward yapısı
- On-policy yöntemlerin düşük sample efficiency'si
- Greedy domain knowledge kullanıyor (en yakın drone)

### BC + A2C iyileştirmesi
5000 episode greedy demonstrasyonu ile pretraining. Cost 25.52'den 5.70'e düştü — %78 iyileşme.

---

## 5. Method Origins

| Method | Paper |
|--------|-------|
| REINFORCE | Williams (1992) |
| A2C | Mnih et al. (2016) |
| GAE | Schulman et al. (2016) |
| DDPG | Lillicrap et al. (2016) |
