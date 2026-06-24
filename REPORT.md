# IE306 Term Project Report
---

## Role A — Value-based Methods (DQN Family)
**Student:** Furkan Çalışkan

---

## A.1 Methods

### A.1.1 Plain DQN
Mnih et al. (2015) tarafından önerilen temel Deep Q-Network. 2-layer MLP, replay buffer ve target network kullanıyor.

**Implementasyon detayları:**
- Observation: drones (8×10), orders (20×5), drone-order Manhattan distance matrix (8×20), time (1) → 341 boyutlu vektör
- Grid observation kaldırıldı (400 sayı, çok gürültülü, öğrenmeyi zorlaştırıyor)
- age → time_left = (60 - age) / 60 (deadline urgency için)
- Action mask: geçersiz aksiyonlar -inf ile maskelendi
- Replay buffer: 200k kapasite
- Linear epsilon decay: 1.0 → 0.05 (300k step)

### A.1.2 Double DQN
Van Hasselt et al. (2016). Plain DQN'deki Q-value overestimation sorununu çözmek için online network aksiyon seçer, target network değerlendirir.

**Plain DQN'den fark:**
```python
# Plain DQN: target net hem seçer hem değerlendirir
next_q = q_target(nobs).max(dim=1)[0]

# Double DQN: online net seçer, target net değerlendirir
next_actions = q_net(nobs).argmax(dim=1)
next_q = q_target(nobs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
```

### A.1.3 Dueling DQN
Wang et al. (2016). Q(s,a) = V(s) + A(s,a) - mean(A) şeklinde ayrıştırılıyor. State value ve advantage ayrı ayrı tahmin ediliyor, daha iyi generalization sağlıyor.

**Hiperparametre tuning:**
- v1: hidden=256, lr=0.0003 → cost=11.14
- v2 (tuned): hidden=512, lr=0.0001 → cost=9.09 (en iyi)

---

## A.2 Results

### A.2.1 Baseline Karşılaştırması (seeds: 0,1,2)

| Method | cost_per_order | success_rate | ontime_rate |
|--------|---------------|--------------|-------------|
| Random | 18.78 | 0.653 | 0.890 |
| greedy_nearest (baseline) | **4.57** | 0.855 | 0.903 |
| Plain DQN | 22.46 | 0.453 | 0.972 |
| Double DQN | 9.79 | 0.625 | 0.931 |
| Dueling DQN (tuned) | 9.09 | 0.614 | 0.876 |

**En iyi sonuç: Dueling DQN (tuned) → 9.09**

Baseline (4.57) yenilemiyor. Bunun nedenleri Section A.4'te açıklanıyor.

---

## A.3 Ablation — DQN Variant Karşılaştırması

| Variant | cost_per_order | Yorum |
|---------|---------------|-------|
| Plain DQN | 22.46 | En kötü — overestimation problemi |
| Double DQN | 9.79 | %56 iyileşme — overestimation çözüldü |
| Dueling DQN | 11.14 | Plain'den iyi ama Double'dan kötü |
| Dueling DQN (tuned) | 9.09 | En iyi — lr ve hidden size optimize edildi |

**Bulgu:** Double DQN en tutarlı iyileşmeyi sağladı. Dueling tek başına Double'dan kötü çıktı — bu seed'e ve hiperparametreye bağlı olabilir. Tuned Dueling (hidden=512, lr=0.0001) en iyi sonucu verdi.

---

## A.4 Analysis

### Neden greedy_nearest yenilemiyor?

**1. Büyük ve sparse action space:**
169 aksiyon (8 drone × 20 sipariş slot + 8 şarj + 1 no-op). Her aksiyonun değerini öğrenmek çok fazla örnek gerektiriyor. CPU'da 500k step yaklaşık 50 dakika sürüyor; yeterli exploration yapılamıyor.

**2. Greedy çok güçlü bir baseline:**
greedy_nearest domain knowledge kullanıyor: routed distance hesaplayarak en yakın drone-sipariş çiftini atıyor. Bu basit ama son derece etkili. RL'in bunu sıfırdan öğrenmesi için milyonlarca adım gerekiyor.

**3. Observation'da routed distance yok:**
Greedy nearest BFS ile gerçek routed distance kullanıyor. Bizim DQN sadece Manhattan distance görüyor — no-fly zone'lar etrafında yanlış tahminler yapıyor.

**4. Unstable training:**
Cost değerleri eğitim boyunca dalgalandı (örneğin 50k: 78, 100k: 11, 150k: 17). Bu, replay buffer'ın henüz yeterince dolmadığını ve epsilon'ın çok hızlı düştüğünü gösteriyor.

**5. CPU sınırlılığı:**
GPU olmadan 1M step ~90 dakika sürüyor. Yeterli hyperparameter search yapılamadı.

### Neden Double DQN Plain'den çok daha iyi?

Plain DQN Q değerlerini sistematik olarak overestimate ediyor. 169 aksiyonlu bu ortamda bu overestimation çok belirgin — ajan yanlış aksiyonları iyi sanıp tekrar tekrar seçiyor. Double DQN bunu düzeltiyor.

### Neden imitation learning çalışmadı?

Greedy nearest'in kararlarını supervised learning ile öğretmeyi denedik (v5). Ancak cross-entropy loss 1.6'da takıldı — greedy 169 aksiyondan stochastic seçim yapıyor gibi görünüyor (farklı koşullarda farklı aksiyonlar), network bunu öğrenemedi.

---

## A.5 Method Origins

| Method | Paper | Neden Seçildi |
|--------|-------|---------------|
| DQN | Mnih et al. (2015) | Temel value-based baseline |
| Double DQN | Van Hasselt et al. (2016) | Overestimation düzeltmek için |
| Dueling DQN | Wang et al. (2016) | State value/advantage ayrıştırması |

---

## A.6 Engineering Log

| Deney | Sonuç | Neden |
|-------|-------|-------|
| v1: Grid dahil, exp decay | cost=89 (50k) | Grid gürültülü, epsilon çok hızlı düştü |
| v2: Grid kaldırıldı | cost=19.3 (100k) | İyileşme ama unstable |
| v3: time_left eklendi | cost=13.4 (100k) | Deadline urgency eklendi |
| v4: Distance matrix eklendi | cost=11.5 (100k) | Drone-order mesafesi eklendi |
| v5: Imitation learning | cost=13.8 | Greedy taklit edilemedi |
| Final: Dueling tuned | cost=9.09 | lr=0.0001, hidden=512 |


## Role B — Policy-based Methods
**Student:** Begüm Durmuş

---

## 1. Methods

### 1.1 REINFORCE
Vanilla policy gradient (Williams, 1992). 2-layer MLP policy network (hidden=128, ReLU). Episodic Monte Carlo returns ile güncelleme. Return normalization uygulandı (variance azaltmak için). Value baseline yok — bu yüksek variance'a yol açıyor ve REINFORCE'un temel zayıflığı.

**Implementasyon detayları:**
- obs_to_vector() ile 581 boyutlu observation vektörü
- action_mask ile geçersiz aksiyonlar -inf yapıldı
- Categorical distribution ile stochastic sampling
- Episode sonunda tek güncelleme (on-policy)

### 1.2 A2C (Advantage Actor-Critic)
Paylaşımlı backbone + ayrı actor/critic kafaları (hidden=256, Tanh). REINFORCE'a göre temel fark: Critic ağı V(s) tahmin ediyor ve advantage = G - V(s) hesaplanıyor. Bu variance'ı ciddi düşürüyor.

**Implementasyon detayları:**
- GAE (λ=0.95) ile advantage hesaplama
- Entropy bonus (coef=0.01) — exploration için, erken convergence'ı önlüyor
- Value loss coefficient: 0.5
- Gradient clipping (max_norm=0.5) — eğitim kararlılığı için
- Adam optimizer, lr=1e-4

**Denenen varyantlar:**
- Vanilla A2C (lr=3e-4, hidden=256): cost=25.52
- Küçük ağ (hidden=128, lr=1e-4): cost=24.62
- Reward shaping (teslimat bonusu +5, düşürme cezası -3): eğitimi daha da bozdu
- 3000 episode: overtraining, 26'ya çıktı

### 1.3 BC + A2C (En İyi Yöntem)
Greedy'nin başarısız olduğu durumları değil, başarılı olduğu durumları öğrenmek için Behavior Cloning kullanıldı. İki aşama:

**Aşama 1 — Behavior Cloning:**
greedy_nearest'in 5000 episode kararı toplandı (~748k adım). Cross-entropy loss ile supervised learning. 100 epoch, batch_size=1024, lr=5e-4. En düşük loss'ta model kaydedildi.

**Aşama 2 — A2C Fine-tuning:**
BC modelini başlangıç noktası olarak kullanıp A2C ile iyileştirme denendi. Ancak A2C fine-tuning modeli bozdu (25'e çıktı). Bu yüzden pure BC modeli kullanıldı.

### 1.4 DDPG
DroneControl-v0 (continuous sub-env) için off-policy actor-critic (Lillicrap et al., 2016).

**Implementasyon detayları:**
- Actor: speed [0,1] için Sigmoid, heading [-1,1] için Tanh
- Critic: Q(s,a) tahmini
- Replay buffer: 100k kapasite
- Soft target updates: τ=0.005
- Gaussian exploration noise: σ=0.1
- 2000 episode eğitim, 3 seed

---

## 2. Results

### 2.1 Dispatch Environment (DroneDispatch-v0)

| Method | cost_per_order | n_delivered | n_dropped |
|--------|---------------|-------------|-----------|
| greedy_nearest (baseline) | 4.57 | 118 | 20 |
| REINFORCE | 21.73 | 71 | 67 |
| A2C | 25.52 | 68 | 70 |
| BC + A2C (best) | **5.70** | **110** | **28** |

### 2.2 Control Environment (DroneControl-v0)

| Method | mean_return |
|--------|-------------|
| Random baseline | -266.45 |
| DDPG | -4.56 ± 1.18 |

DDPG random baseline'ı geçti (98% iyileşme).

---

## 3. Ablation — GAE Lambda Sweep

A2C'de advantage hesaplama parametresi λ'nın etkisi (seed=0, 300 episode):

| λ | mean_return_last100 | Yorum |
|---|---|---|
| 0.00 | -418.6 | TD(0) — yüksek bias |
| 0.50 | -290.3 | Orta denge |
| 0.80 | -260.0 | İyileşiyor |
| **0.95** | **-251.0** | **En iyi** |
| 1.00 | -270.0 | Monte Carlo — yüksek variance |

**Bulgu:** λ=0.95 literatürle tutarlı olarak en iyi sonucu verdi. λ=0 (saf TD) yüksek bias nedeniyle en kötü; λ=1.0 (saf Monte Carlo) yüksek variance nedeniyle λ=0.95'ten kötü.

---

## 4. Analysis

### Neden REINFORCE ve A2C greedy_nearest'i geçemedi?

**1. On-policy sample inefficiency:**
REINFORCE ve A2C her veriyi bir kez kullanıp atıyor. Replay buffer yok. Bu ortamda greedy'yi geçmek için 10-50 milyon adım gerekiyor; biz 1000-3000 episode (~1.5M adım) yaptık. DQN gibi off-policy yöntemler aynı veriyi defalarca kullandığı için çok daha verimli.

**2. Büyük discrete action space:**
169 aksiyon (8 drone × 13 sipariş + 8 şarj + 1 no-op). Her aksiyonun ne zaman iyi olduğunu öğrenmek çok fazla örnek gerektiriyor. Policy gradient bu tür büyük action space'lerde yavaş converge ediyor.

**3. Sparse reward ve credit assignment problemi:**
Teslimat ödülü (+10) teslimat yapılana kadar gelmiyor — bu yüzlerce adım sürebilir. Ajan hangi kararın teslimatı sağladığını anlamakta zorlanıyor (credit assignment problemi). Her adımda sadece küçük negatif enerji reward'ı var.

**4. Greedy çok güçlü bir baseline:**
greedy_nearest domain knowledge kullanıyor: en yakın idle drone'u en yakın siparişe atıyor, düşük SoC'lu drone'ları şarja gönderiyor. Bu basit ama çok etkili bir kural. RL ajanının bunu sıfırdan öğrenmesi için büyük miktarda veri gerekiyor.

**5. Reward shaping başarısız oldu:**
Teslimat yaklaştıkça küçük bonus vermeyi denedik. Ama bu ajan davranışını değiştirmedi, aksine eğitimi daha da dengesiz hale getirdi.

### Neden A2C REINFORCE'dan daha kötü çıktı?

Teoride A2C daha iyi olmalı, ama bu ortamda:
- Critic ağı sparse reward ile V(s) tahminini öğrenemedi
- Yanlış advantage tahminleri policy'yi bozdu
- Daha küçük learning rate ve daha uzun eğitim ile belki converge ederdi

### BC neden işe yaradı?

Greedy demonstrasyonlarından öğrenmek sıfırdan RL'den çok daha verimli:
- 748k adım greedy verisi → ağ "greedy gibi davran" öğrendi
- Cost 25.52'den 5.70'e düştü: **%78 iyileşme**
- n_delivered 68'den 110'a çıktı
- n_dropped 70'den 28'e düştü

**Neden pure BC, BC+A2C'den daha iyi?**
A2C fine-tuning BC modelini bozdu çünkü sparse reward ortamında A2C yanlış gradyanlar üretiyor. Pure BC modeli greedy'yi daha iyi taklit etti.

### DDPG neden random'ı geçti?

DroneControl-v0 çok daha basit bir ortam: 7 boyutlu obs, 2 boyutlu continuous action. Replay buffer sayesinde off-policy öğrenme çok daha verimli. 2000 episode yeterli oldu.

---

## 5. Method Origins

| Method | Paper | Neden Seçildi |
|--------|-------|---------------|
| REINFORCE | Williams (1992) | Temel policy gradient baseline |
| A2C | Mnih et al. (2016) | Variance azaltma için critic |
| GAE | Schulman et al. (2016) | Daha iyi advantage tahmini |
| Behavior Cloning | Pomerleau (1989) | Güçlü initialization için |
| DDPG | Lillicrap et al. (2016) | Continuous action space |

---

## 6. Engineering Log

| Deney | Sonuç | Neden |
|-------|-------|-------|
| REINFORCE 500 ep | cost=21.73 | Az veri, yüksek variance |
| A2C 1000 ep | cost=25.52 | Critic yanlış öğrendi |
| A2C 3000 ep | cost=26+ | Overtraining |
| Reward shaping | Kötüleşti | Yanlış sinyal |
| BC 2000 ep, 30 epoch | cost=6.86 | İlk BC denemesi |
| BC 5000 ep, 100 epoch | cost=5.70 | En iyi sonuç |
| BC 10000 ep, 200 epoch | cost=8.06 | Overfitting |
| BC+A2C fine-tune | cost=25+ | A2C modeli bozdu |

