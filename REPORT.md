# IE306 Term Project Report
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



---
---

# ROL C — Planning / Model-Based Acceleration (Dyna-Q)

## C.1 Methods

### C.1.1 Tabular Dyna-Q
Model-tabanli hizlandirma yontemi (Sutton, 1990). Uc bilesen: (1) gercek
deneyimle Q-learning guncellemesi, (2) ogrenilen bir cevre modeli, (3) her
gercek adimdan sonra modelden cekilen n adet "hayali" deneyimle ek Q
guncellemesi (planlama). Planlama, az sayidaki gercek deneyimden cok daha fazla
ogrenme cikararak ornek-verimliligini artirir â€” rolun ozu budur.

Saf numpy ile tablosal olarak yazildi (sinir agi yok). Bu, A ve B rollerinin
derin ag tabanli yontemlerinden kasitli olarak farkli: daha yorumlanabilir,
hafif ve tamamen tekrar-uretilebilir.

### C.1.2 Durum soyutlamasi (state abstraction)
169 boyutlu ham aksiyon uzayinda ogrenmek yerine, problemi operasyonel olarak
ayristirdik. Her karar aninda bir "odak drone" secilir (greedy gibi, atanmayi
bekleyen siparise en yakin bos drone) ve kontrolcu sadece su iki makro-aksiyon
arasinda secim yapar:
- **ASSIGN:** odak drone'a en yakin siparisi ata
- **CHARGE:** odak drone'u sarja gonder

Odak drone'un durumu 5 ozellige indirgenip ayrik kovalara (bucket) bolunur:
SoC, en yakin hub mesafesi, **bitirme marji** (soc - tahmini is enerjisi),
siparis aciliyeti, talep baskisi. ~2000 durum Ã— 2 aksiyon â€” tablosal ogrenmeye
uygun, hizli yakinsayan kompakt bir tablo.

Tum rotali (no-fly-farkinda) mesafeler, gozlemdeki grid uzerinden kendi
yazdigimiz BFS ile hesaplanir â€” donmus `Policy` kontratina sadik kalinir
(env ic organlarina dokunulmaz).

### C.1.3 Depletion-farkinda maske (kilit tasarim)
greedy enerji-kor: sadece anlik SoC'a bakar, bir drone'u bitiremeyecegi uzun
bir ise yollayip yolda tuketebilir (-50 ceza). Bizim kontrolcumuz **ileriye
bakar**: `marj = soc - is_enerjisi` negatifse, o is drone'u tuketir. Bu durumda
ASSIGN maskeden cikarilir ve CHARGE'a zorlanir. greedy'nin yapisal olarak
goremedigi bu on-gorulu enerji yonetimi, greedy'yi gecmenin temel sebebidir.

---

## C.2 Results

Standart eval config, 3 seed (0,1,2) ortalamasi. `run_all.py` ile uretildi.

| Method | cost_per_order | depletion | n_dropped | n_delivered |
|--------|---------------|-----------|-----------|-------------|
| random | 18.78 | 8.0 | 21.7 | 39.7 |
| greedy_nearest (baseline) | 4.57 | 4.0 | 20.0 | 118.3 |
| milp_rolling | 4.72 | 3.3 | 23.0 | 118.0 |
| **Dyna-Q (Planning)** | **0.78 Â± 0.04** | **0.0** | **4.7** | **138.7** |

Dyna-Q, greedy'yi ~6 kat (4.57 â†’ 0.78) ve MILP'i 6 kat geride birakti. Uc seedde
de **sifir depletion** elde edildi (std=0.04 â€” yontem kararli, tek sanslik kosu
degil). Kazanim, depletion-farkinda maskenin tum drone'lari hayatta tutmasindan
kaynaklanir: filo tam kapasite caliÅŸinca hem teslimat artar (138 > 118) hem
dusen siparis azalir (4.7 < 20).

---

## C.3 Ablation â€” Planning Steps (n) Sweep

Dyna planlama adiminin (n) etkisi. Her n icin 3 seed, 400 episode.

| n (planning_steps) | cost_per_order | Yorum |
|---|---|---|
| 0 | 0.964 Â± 0.063 | Planlama yok = saf Q-learning (model-free) |
| 5 | 0.807 Â± 0.072 | Planlama faydasi basliyor |
| **10** | **0.781 Â± 0.037** | **En iyi (tatli nokta)** |
| 50 | 0.846 Â± 0.053 | Marjinal fazlalik, hafif geriye |

**Bulgu:** Planlama performansi belirgin iyilestiriyor (0.96 â†’ 0.78). Ancak fayda
n=10 civarinda doyuma ulasiyor; n=50'de hafif geriliyor â€” ders kitabi Dyna
davranisiyla tutarli (planlama ornek-verimliligini artirir ama doyum noktasi
vardir). Bu yuzden ana model n=10 ile egitildi. **Onemli ayrim:** greedy'yi
gecmenin sebebi planlama degil, mimari tasarim (depletion-farkinda maske) â€”
cunku n=0 bile greedy'yi rahatca geciyor. Planlama bunun uzerine ek iyilestirme
saglar.

(Bkz. `logs/ablation_planning.png` ve `logs/learning_curve_ablation.png`.)

---

## C.4 Method Origins

| Method | Paper | Neden Secildi |
|--------|-------|---------------|
| Dyna-Q | Sutton (1990), "Integrated Architectures for Learning, Planning and Reacting" | Model-tabanli hizlandirma; teslimat kalemleri (ogrenme egrisi, agirlik, n-sweep ablasyonu) ile birebir ortusur ve eval'de sadece obs gerektirir (donmus act() kontratina uygun) |
| Q-learning | Watkins (1989) | Dyna'nin direkt-RL bileseni |

---

## C.5 Engineering Log â€” "Ne Bozuldu, Nasil Teshis Ettik"

| Deney / Sorun | Belirti | Teshis | Cozum |
|-------|-------|--------|-------|
| Ilk 3-aksiyonlu tasarim (ASSIGN/CHARGE/HOLD) | cost=21.7, drops=72 | Aksiyon sayimi: ajan 344 kez CHARGE/HOLD, 71 kez ASSIGN seciyor | Mimari sadelestirme: HOLD kaldirildi |
| 2-aksiyonlu, ham odul | deliv=0, cost=2033 | Ajan hep CHARGE seciyor; odul kredi atamasi bozuk (env'de assign/charge zamani ilerletmez) | CHARGE'a anlik odul cezasi + CHARGE_ZONE |
| Politika sarmalayici | deliv=0 (hala) | Aksiyon izleme: ASSIGN secilse bile `act()` noop donduruyor | `return n_act` -> `return a_act` (2'liye gecerken kalan bug) |
| Depletion patlamasi | episode sonu 8 drone'dan 1'i sag | Hayatta kalan drone sayaci: 7 drone yolda tukeniyor | Depletion-farkinda maske: marj negatifse ASSIGN yasak |
| **Nihai** | **cost=0.78, depletion=0** | â€” | **greedy 6 kat geÃ§ildi** |

Bu teshis zinciri (asiri-CHARGE -> mimari sadelestirme -> noop bug -> depletion
korumasi) yontemin nasil adim adim duzeltildigini gosterir. Her sorun, ajanin
gercek davranisi olcuLerek (aksiyon dagilimi, hayatta kalan drone sayisi) teshis
edildi; korlemesine parametre denenmedi.
