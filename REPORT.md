# IE306 Term Project Report
## Role B — Policy-based Methods
**Student:** Begüm Durmuş

---

## 1. Methods

### 1.1 REINFORCE
Vanilla policy gradient (Williams, 1992). Episodic Monte Carlo returns used to update a 2-layer MLP policy. No value baseline.

### 1.2 A2C (Advantage Actor-Critic)
Shared-backbone actor-critic with GAE (Schulman et al., 2016). Actor minimizes -log_prob * advantage; critic minimizes MSE to discounted returns. Gradient clipping (max_norm=0.5).

### 1.3 BC + A2C
Behavior cloning pretraining on greedy_nearest demonstrations (2000 episodes), followed by A2C fine-tuning. Best performing dispatch policy.

### 1.4 DDPG
Off-policy actor-critic for continuous control (Lillicrap et al., 2016) on DroneControl-v0. Replay buffer (100k), soft target updates (τ=0.005), Gaussian exploration noise.

---

## 2. Results

### 2.1 Dispatch Environment (DroneDispatch-v0)

| Method | cost_per_order | success_rate |
|--------|---------------|--------------|
| random | — | — |
| greedy_nearest | 4.57 | 0.85 |
| REINFORCE | 18.72 | 0.64 |
| A2C | 24.62 | 0.65 |
| BC + A2C | 6.86 | 0.73 |

### 2.2 Control Environment (DroneControl-v0)

| Method | mean_return |
|--------|-------------|
| Random | -266.45 |
| DDPG (DroneControl-v0) | mean_return: -4.56 +- 1.18 |

---

## 3. Ablation — GAE Lambda Sweep

| λ | mean_return_last100 |
|---|---|
| 0.00 | -418.6 |
| 0.50 | -290.3 |
| 0.80 | -260.0 |
| **0.95** | **-251.0** |
| 1.00 | -270.0 |

λ=0.95 achieved the best performance, consistent with literature recommendations.

---

## 4. Analysis

### Why didn't REINFORCE/A2C beat greedy_nearest?
- Large discrete action space (169 actions)
- Sparse reward signal
- On-policy methods require many samples to converge
- greedy_nearest uses domain knowledge (nearest drone assignment)

### BC + A2C improvement
Pretraining on greedy demonstrations provided a strong initialization. Cost reduced from 24.62 to 6.86 — a 72% improvement over vanilla A2C.

---

## 5. Engineering Log

- REINFORCE: converged but high variance, cost ~18
- A2C: more stable but still above greedy, cost ~24
- Reward shaping: tried but destabilized training
- BC pretraining: significant improvement, cost 6.86
- DDPG: continuous env, training in progress

---

## 6. Method Origins

| Method | Paper | Why chosen |
|--------|-------|-----------|
| REINFORCE | Williams (1992) | Baseline policy gradient |
| A2C | Mnih et al. (2016) | Variance reduction via critic |
| GAE | Schulman et al. (2016) | Better advantage estimation |
| DDPG | Lillicrap et al. (2016) | Continuous action space |
