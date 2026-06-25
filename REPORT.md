# IE306 Term Project Report
## Reinforcement Learning for City-Scale Drone Delivery

---

## Role A — Value-based Methods (DQN Family)
**Student:** Furkan Çalışkan

---

## A.1 Methods

### A.1.1 Plain DQN
Basic Deep Q-Network proposed by Mnih et al. (2015). Uses a 2-layer MLP, replay buffer, and target network.

**Implementation details:**
- Observation: drones (8×10), orders (20×5), drone-order Manhattan distance matrix (8×20), time (1) → 341-dimensional vector
- Grid observation removed (400 noisy features that hindered learning)
- age → time_left = (60 - age) / 60 (for deadline urgency)
- Action mask: invalid actions masked with -inf
- Replay buffer: 200k capacity
- Linear epsilon decay: 1.0 → 0.05 (300k steps)

### A.1.2 Double DQN
Van Hasselt et al. (2016). Fixes Q-value overestimation in Plain DQN: the online network selects actions, the target network evaluates them.

**Difference from Plain DQN:**
```python
# Plain DQN: target net both selects and evaluates
next_q = q_target(nobs).max(dim=1)[0]

# Double DQN: online net selects, target net evaluates
next_actions = q_net(nobs).argmax(dim=1)
next_q = q_target(nobs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
```

### A.1.3 Dueling DQN
Wang et al. (2016). Q(s,a) = V(s) + A(s,a) - mean(A). State value and advantage are estimated separately, enabling better generalization.

**Hyperparameter tuning:**
- v1: hidden=256, lr=0.0003 → cost=11.14
- v2 (tuned): hidden=512, lr=0.0001 → cost=9.09 (best)

---

## A.2 Results

### A.2.1 Baseline Comparison (seeds: 0,1,2)

| Method | cost_per_order | success_rate | ontime_rate |
|--------|---------------|--------------|-------------|
| Random | 18.78 | 0.653 | 0.890 |
| greedy_nearest (baseline) | **4.57** | 0.855 | 0.903 |
| Plain DQN | 15.15 | 0.526 | 0.987 |
| Double DQN | 14.04 | 0.563 | 0.629 |
| Dueling DQN | 13.19 | 0.590 | 0.868 |
| Dueling DQN (tuned) | 9.09 | 0.665 | 0.949 |

**Best result: Dueling DQN (tuned) → 9.09**

The greedy_nearest baseline (4.57) was not beaten. Reasons are discussed in Section A.4.

---

## A.3 Ablation — DQN Variant Comparison

| Variant | cost_per_order | Note |
|---------|---------------|------|
| Plain DQN | 15.15 | Worst — overestimation problem |
| Double DQN | 14.04 | 7% improvement — overestimation fixed |
| Dueling DQN | 13.19 | Better than Plain, worse than tuned |
| Dueling DQN (tuned) | 9.09 | Best — lr and hidden size optimized |

**Finding:** Double DQN provided the most consistent improvement. Tuned Dueling DQN (hidden=512, lr=0.0001) achieved the best final result.

---

## A.4 Analysis

### Why greedy_nearest was not beaten

**1. Large and sparse action space:**
169 actions (8 drones × 20 order slots + 8 charge + 1 no-op). Learning the value of each action requires enormous amounts of experience. On CPU, 500k steps takes ~50 minutes; insufficient exploration.

**2. Greedy is a very strong baseline:**
greedy_nearest uses domain knowledge: it assigns the nearest idle drone to the nearest order using routed distance. Simple but highly effective. RL needs millions of steps to learn this from scratch.

**3. No routed distance in observation:**
Greedy nearest uses BFS-based routed distances. Our DQN only sees Manhattan distances — leading to incorrect predictions around no-fly zones.

**4. Unstable training:**
Cost values fluctuated throughout training (e.g. 50k: 78, 100k: 11, 150k: 17), indicating the replay buffer had not yet filled sufficiently and epsilon decayed too fast.

**5. CPU limitation:**
Without GPU, 1M steps takes ~90 minutes. Insufficient hyperparameter search was possible.

### Why Double DQN is much better than Plain DQN

Plain DQN systematically overestimates Q values. In this 169-action environment the overestimation is severe — the agent incorrectly selects overvalued actions repeatedly. Double DQN corrects this.

---

## A.5 Method Origins

| Method | Paper | Why Chosen |
|--------|-------|------------|
| DQN | Mnih et al. (2015) | Foundational value-based baseline |
| Double DQN | Van Hasselt et al. (2016) | Fix overestimation |
| Dueling DQN | Wang et al. (2016) | State value/advantage decomposition |

---

## A.6 Engineering Log

| Experiment | Result | Reason |
|------------|--------|--------|
| v1: Grid included, exp decay | cost=89 (50k) | Grid noisy, epsilon decayed too fast |
| v2: Grid removed | cost=19.3 (100k) | Improved but unstable |
| v3: time_left added | cost=13.4 (100k) | Deadline urgency added |
| v4: Distance matrix added | cost=11.5 (100k) | Drone-order distance added |
| v5: Imitation learning | cost=13.8 | Could not imitate greedy |
| Final: Dueling tuned | cost=9.09 | lr=0.0001, hidden=512 |

---

## Role B — Policy-based Methods
**Student:** Begüm Durmuş

---

## B.1 Methods

### B.1.1 REINFORCE
Vanilla policy gradient (Williams, 1992). 2-layer MLP policy network (hidden=128, ReLU). Episodic Monte Carlo returns for updates. Return normalization applied to reduce variance. No value baseline — this leads to high variance and is the fundamental weakness of REINFORCE.

**Implementation details:**
- obs_to_vector() produces a 581-dimensional observation vector
- action_mask applied: invalid actions set to -inf
- Categorical distribution with stochastic sampling
- Single update per episode end (on-policy)

### B.1.2 A2C (Advantage Actor-Critic)
Shared backbone + separate actor/critic heads (hidden=256, Tanh). Key difference from REINFORCE: a Critic network estimates V(s) and advantage = G - V(s), significantly reducing variance.

**Implementation details:**
- GAE (λ=0.95) for advantage estimation
- Entropy bonus (coef=0.01) for exploration, prevents premature convergence
- Value loss coefficient: 0.5
- Gradient clipping (max_norm=0.5) for training stability
- Adam optimizer, lr=1e-4

**Variants tried:**
- Vanilla A2C (lr=3e-4, hidden=256): cost=22.95
- Small network (hidden=128, lr=1e-4): cost=24.62
- Reward shaping (delivery bonus +5, drop penalty -3): further destabilized training
- 3000 episodes: overtraining, cost rose above 26

### B.1.3 BC + A2C (Best Method)
Behavior Cloning was used to learn from greedy's successful decisions as a strong initialization. Two phases:

**Phase 1 — Behavior Cloning:**
5000 episodes of greedy_nearest decisions collected (~748k steps). Supervised learning with cross-entropy loss. 100 epochs, batch_size=1024, lr=5e-4. Best checkpoint saved at lowest validation loss.

**Phase 2 — A2C Fine-tuning:**
Attempted to improve the BC model using A2C fine-tuning. However, A2C fine-tuning degraded the model (cost rose to 25+). Therefore the pure BC model was used as the final submission.

### B.1.4 DDPG
Off-policy actor-critic for the DroneControl-v0 continuous sub-environment (Lillicrap et al., 2016).

**Implementation details:**
- Actor: Sigmoid for speed [0,1], Tanh for heading [-1,1]
- Critic: Q(s,a) estimation
- Replay buffer: 100k capacity
- Soft target updates: τ=0.005
- Gaussian exploration noise: σ=0.1
- 2000 episodes of training, 3 seeds

---

## B.2 Results

### B.2.1 Dispatch Environment (DroneDispatch-v0)

| Method | cost_per_order | success_rate | ontime_rate |
|--------|---------------|--------------|-------------|
| greedy_nearest (baseline) | 4.57 | 0.855 | 0.903 |
| REINFORCE | 17.39 | 0.738 | 0.969 |
| A2C | 22.95 | 0.686 | 0.962 |
| BC + A2C (best) | **8.06** | **0.724** | **0.893** |

Note: During training-time evaluation (single seed, 500 episodes), BC+A2C achieved cost=5.70. The held-out evaluation via run_all.py (3 seeds, eval_standard config) yields 8.06, reflecting generalization across seeds.

### B.2.2 Control Environment (DroneControl-v0)

| Method | mean_return |
|--------|-------------|
| Random baseline | -266.45 |
| DDPG | -4.56 ± 1.18 |

DDPG beat the random baseline by 98%.

---

## B.3 Ablation — GAE Lambda Sweep

Effect of the advantage estimation parameter λ in A2C (seed=0, 300 episodes):

| λ | mean_return_last100 | Note |
|---|---|---|
| 0.00 | -418.6 | TD(0) — high bias |
| 0.50 | -290.3 | Moderate balance |
| 0.80 | -260.0 | Improving |
| **0.95** | **-251.0** | **Best** |
| 1.00 | -270.0 | Monte Carlo — high variance |

**Finding:** λ=0.95 achieved the best result, consistent with the literature. λ=0 (pure TD) was worst due to high bias; λ=1.0 (pure Monte Carlo) was worse than λ=0.95 due to high variance.

---

## B.4 Analysis

### Why REINFORCE and A2C did not beat greedy_nearest

**1. On-policy sample inefficiency:**
REINFORCE and A2C use each data sample only once. No replay buffer. Beating greedy in this environment requires 10–50 million steps; we trained for 1000–3000 episodes (~1.5M steps). Off-policy methods like DQN reuse data many times and are far more efficient.

**2. Large discrete action space:**
169 actions (8 drones × 13 order slots + 8 charge + 1 no-op). Learning when each action is good requires enormous samples. Policy gradient methods converge slowly on large action spaces.

**3. Sparse reward and credit assignment problem:**
Delivery reward (+10) does not arrive until delivery is complete — which can take hundreds of steps. The agent struggles to determine which decision caused the delivery (credit assignment problem). Only small negative energy rewards are available at each step.

**4. Greedy is a very strong baseline:**
greedy_nearest uses domain knowledge: assigns the nearest idle drone to the nearest order, sends low-SoC drones to charge. Simple but highly effective. RL needs large amounts of data to learn this from scratch.

**5. Reward shaping failed:**
We tried giving small bonuses as delivery approached. This did not improve agent behavior; instead it further destabilized training.

### Why A2C performed worse than REINFORCE

In theory A2C should be better, but in this environment:
- The Critic network could not learn V(s) reliably with sparse rewards
- Incorrect advantage estimates degraded the policy
- A smaller learning rate and longer training might allow convergence

### Why BC worked

Learning from greedy demonstrations is far more efficient than learning from scratch:
- 748k steps of greedy data → network learned to "act like greedy"
- Cost dropped from 22.95 to 8.06
- n_delivered rose from 68 to ~110
- n_dropped fell from 70 to ~28

**Why pure BC outperforms BC+A2C fine-tuning:**
A2C fine-tuning degraded the BC model because A2C produces incorrect gradients in a sparse reward environment. The pure BC model better imitates greedy.

### Why DDPG beat random

DroneControl-v0 is a much simpler environment: 7-dimensional observation, 2-dimensional continuous action. Off-policy learning with a replay buffer is very efficient. 2000 episodes were sufficient.

---

## B.5 Method Origins

| Method | Paper | Why Chosen |
|--------|-------|------------|
| REINFORCE | Williams (1992) | Foundational policy gradient baseline |
| A2C | Mnih et al. (2016) | Critic for variance reduction |
| GAE | Schulman et al. (2016) | Better advantage estimation |
| Behavior Cloning | Pomerleau (1989) | Strong initialization from expert |
| DDPG | Lillicrap et al. (2016) | Continuous action space |

---

## B.6 Engineering Log

| Experiment | Result | Reason |
|------------|--------|--------|
| REINFORCE 500 ep | cost=17.39 | Insufficient data, high variance |
| A2C 1000 ep | cost=22.95 | Critic learned poorly |
| A2C 3000 ep | cost=26+ | Overtraining |
| Reward shaping | Worsened | Wrong signal |
| BC 2000 ep, 30 epoch | cost=6.86 | First BC attempt |
| BC 5000 ep, 100 epoch | cost=5.70 (training eval) | Best training result |
| BC 10000 ep, 200 epoch | cost=8.06+ | Overfitting |
| BC+A2C fine-tune | cost=25+ | A2C degraded the model |

---

## Role C — Planning / Model-Based Acceleration (Dyna-Q)
**Student:** Abdulsamet Kavas

---

## C.1 Methods

### C.1.1 Tabular Dyna-Q
Model-based acceleration method (Sutton, 1990). Three components: (1) a Q-learning update from real experience, (2) a learned environment model, and (3) after each real step, n additional "imaginary" updates sampled from the model (planning). Planning extracts far more learning from a small amount of real experience, improving sample efficiency — this is the core of the role.

Implemented in pure NumPy as a tabular method (no neural network). This is a deliberate contrast to the deep-network methods of Roles A and B: more interpretable, lightweight, and fully reproducible.

### C.1.2 State Abstraction
Instead of learning over the raw 169-dimensional action space, the problem is decomposed operationally. At each decision point a "focal drone" is selected (like greedy: the idle drone nearest to an unassigned order), and the controller chooses only between two macro-actions:
- **ASSIGN:** assign the nearest order to the focal drone
- **CHARGE:** send the focal drone to charge

The focal drone's situation is reduced to 5 features, each discretized into buckets: SoC, nearest-hub distance, finish-margin (soc − estimated job energy), order urgency, and demand pressure. ~2000 states × 2 actions — a compact table well-suited to tabular learning that converges quickly.

All routed (no-fly-aware) distances are computed via BFS over the grid in the observation — staying faithful to the frozen Policy contract (no access to the environment's internals).

### C.1.3 Depletion-aware Masking (Key Design)
Greedy is energy-blind: it looks only at instantaneous SoC and may send a drone on a long job it cannot finish, depleting it mid-flight (−50 penalty). Our controller **looks ahead**: if `margin = soc − job_energy` is negative, the job will deplete the drone. In that case ASSIGN is removed from the mask and CHARGE is forced. This forward-looking energy management — which greedy structurally cannot do — is the primary reason we beat greedy.

---

## C.2 Results

Standard eval config, averaged over 3 seeds (0,1,2). Produced via run_all.py.

| Method | cost_per_order | success_rate | ontime_rate |
|--------|---------------|--------------|-------------|
| Random | 18.78 | 0.653 | 0.890 |
| greedy_nearest (baseline) | 4.57 | 0.855 | 0.903 |
| **Dyna-Q (Planning)** | **0.78 ± 0.04** | **0.967** | **0.947** |

Dyna-Q beats greedy by ~6x (4.57 → 0.78). All three seeds achieve **zero depletion** (std = 0.04 — stable, not a single lucky run). The gain comes from the depletion-aware mask keeping all drones alive: with the full fleet operational, deliveries rise and dropped orders fall dramatically.

---

## C.3 Ablation — Planning Steps (n) Sweep

Effect of the Dyna planning step count (n). 3 seeds, 400 episodes per setting.

| n (planning_steps) | cost_per_order | Note |
|---|---|---|
| 0 | 0.964 ± 0.063 | No planning = pure Q-learning (model-free) |
| 5 | 0.807 ± 0.072 | Planning benefit begins |
| **10** | **0.781 ± 0.037** | **Best (sweet spot)** |
| 50 | 0.846 ± 0.053 | Marginal excess, slightly worse |

**Finding:** Planning clearly improves performance (0.96 → 0.78). The benefit saturates around n=10 and slightly regresses at n=50 — consistent with textbook Dyna behavior. **Important distinction:** the reason we beat greedy is not planning per se, but the architectural design (depletion-aware mask) — since even n=0 comfortably beats greedy. Planning provides additional improvement on top of that.

---

## C.4 Method Origins

| Method | Paper | Why Chosen |
|--------|-------|------------|
| Dyna-Q | Sutton (1990) | Model-based acceleration; matches deliverables exactly |
| Q-learning | Watkins (1989) | Direct RL component of Dyna |

---

## C.5 Engineering Log

| Experiment / Issue | Symptom | Diagnosis | Fix |
|---|---|---|---|
| Initial 3-action design (ASSIGN/CHARGE/HOLD) | cost=21.7, drops=72 | Agent chose CHARGE/HOLD 344x, ASSIGN only 71x | Removed HOLD action |
| 2-action, raw reward | delivered=0, cost=2033 | Agent always charges; reward credit-assignment broken | Instant penalty on CHARGE + CHARGE_ZONE |
| Policy wrapper bug | delivered=0 (still) | Even when ASSIGN chosen, act() returns noop | Fixed: return n_act → return a_act |
| Depletion blowup | 1 of 8 drones alive at episode end | 7 drones deplete mid-flight | Depletion-aware mask: forbid ASSIGN when margin negative |
| **Final** | **cost=0.78, depletion=0** | — | **Beat greedy by 6x** |

---

## Joint Component 1 — Offline RL (Ch. 20)

### ORL.1 Dataset

Mixed-quality dataset pooled from all three trained policies:
- **Role A (Dueling DQN):** 19,986 transitions, mean_return=469.3
- **Role B (BC+A2C):** 13,866 transitions, mean_return=823.8
- **Role C (Dyna-Q):** 17,794 transitions, mean_return=1848.8
- **Total:** 51,646 transitions

Observation format: Role B's obs_to_vector() (581-dim) used as unified format. Dataset saved as offline_dataset.npz.

---

### ORL.2 Naive Offline DQN — Failure Analysis

Standard Double DQN trained on the static dataset without any conservatism constraint.

**Q-value divergence (overestimation):**

| Step | mean_Q | max_Q |
|------|--------|-------|
| 5,000 | 20.23 | 53.92 |
| 15,000 | 87.87 | 138.92 |
| 30,000 | 201.49 | 294.95 |
| 50,000 | 372.15 | 528.69 |

**Result:** cost_per_order = **17.12**

**Why it fails:** The static dataset does not cover all (state, action) pairs. For out-of-distribution (OOD) actions never seen in the dataset, the Q-network has no ground truth and freely extrapolates upward. The Bellman backup then propagates these inflated values, causing a feedback loop of ever-growing Q estimates. The agent ends up selecting OOD actions with unrealistically high Q values, performing poorly in the actual environment.

---

### ORL.3 CQL (Conservative Q-Learning) — Fix

CQL adds a conservatism penalty to the standard Bellman loss:

```
L_CQL = L_Bellman + α * (logsumexp(Q(s,·)) - Q(s, a_data))
```

This penalizes high Q values on actions not supported by the dataset, keeping Q estimates conservative on OOD actions.

**Q-value comparison (step 50,000):**

| Method | mean_Q | max_Q | cost_per_order |
|--------|--------|-------|---------------|
| Naive Offline DQN | 372.15 | 528.69 | 17.12 |
| CQL (α=1.0) | 61.66 | 216.25 | **11.80** |

**Result:** CQL reduces max_Q by 59% and cost_per_order by 31% compared to naive offline DQN.

**Why CQL works:** The logsumexp term explicitly penalizes the network for assigning high Q values to any action, while the subtraction of Q(s, a_data) ensures in-distribution actions are not penalized. This pushes the policy to stay within the support of the dataset.

---

### ORL.4 Summary

| Method | cost_per_order |
|--------|---------------|
| Behavioral Cloning (BC+A2C) | 8.06 |
| CQL (Offline RL) | 11.80 |
| Naive Offline DQN | 17.12 |
| greedy_nearest (baseline) | 4.57 |

CQL beats naive offline DQN as expected, but does not beat the behavioral cloning baseline. This is consistent with the literature: mixed dataset quality (ranging from poor random policy to strong Dyna-Q) limits CQL's performance.

---

## Joint Component 2 — Multi-Agent RL (Ch. 21)

### MA.1 Method — Parameter-Sharing IDQN

Instead of the centralized dispatcher (a single decision-maker over 169 actions), a **decentralized** structure is built: in the DroneDispatchMA-v0 environment, each drone is its own agent. Each agent observes its 59-dimensional **local** observation and selects one of 4 actions: accept (take an order), move (go to target), charge, or stay (wait).

**IDQN (Independent DQN) + parameter sharing:** A single Q-network (MLP 59→128→128→4) is used by all eight agents. It is "independent" because each agent decides on its own, without coordinating with the others; it uses "parameter sharing" because instead of 8 separate networks there is one network — all agents' experiences are pooled into a shared replay buffer and train the single network. This improves sample efficiency (the network sees 8x the data) and preserves consistency across agents. Standard DQN components are used: replay buffer, target network, and epsilon-greedy exploration.

### MA.2 Results

300 episodes of training, 10 episodes of evaluation. Since the MA environment does not return a global cost_per_order, total reward per episode (summed over the 8 agents) is used for comparison.

| Policy | Total reward per episode |
|--------|-------------------------|
| Random baseline | 442.2 ± 144.1 |
| **Trained IDQN** | **1490.0 ± 160.2** |

The trained IDQN beats the random baseline by **237%** (≈3.4x). This demonstrates that the decentralized structure successfully learns with parameter sharing.

### MA.3 Centralized vs. Decentralized Comparison

The two approaches run in different environments (centralized: DroneDispatch-v0, single decision-maker; MA: DroneDispatchMA-v0, 8 agents), so a direct numerical comparison is not possible. A conceptual comparison:

- **Centralized (Role C Dyna-Q):** A single decision-maker with full fleet visibility → globally consistent, near-optimal decisions (cost 0.78). But it is a central bottleneck; hard to scale and deploy in a distributed real-world setting.
- **Decentralized (IDQN):** Each drone acts independently → scalable, distributed, fault-tolerant. But coordination is harder and non-stationarity makes convergence difficult.

### MA.4 Non-stationarity Discussion

The fundamental challenge of multi-agent learning is **non-stationarity**. Each agent treats the other agents as part of its environment. But since the other agents are also learning and changing their policies simultaneously, the environment each agent faces is **constantly changing** — it is not stationary. This causes each agent to chase a **moving target**: a strategy that is good today may be bad tomorrow once the others change. The result is unstable learning and difficult convergence.

This phenomenon was directly observed in our results: training rewards showed strong fluctuation across episodes (e.g. 270 → 948 → 244) rather than a stable convergence curve. This volatility is a direct symptom of agents continuously changing each other's effective environment.

**Role of parameter sharing:** Sharing a single network across the 8 agents partially mitigates non-stationarity by reducing inter-agent divergence and learning a common policy — but does not eliminate it. Full convergence would require methods such as centralized-training/decentralized-execution (CTDE) (e.g. QMIX, MADDPG); this is a natural extension of the work.

---

## Overall Results Summary

| Method | Role | cost_per_order | Beats Baseline? |
|--------|------|---------------|-----------------|
| Random | — | 18.78 | — |
| greedy_nearest | Baseline | **4.57** | — |
| Plain DQN | A | 15.15 | No |
| Double DQN | A | 14.04 | No |
| Dueling DQN | A | 13.19 | No |
| Dueling DQN (tuned) | A | 9.09 | No |
| REINFORCE | B | 17.39 | No |
| A2C | B | 22.95 | No |
| BC+A2C | B | 8.06 | No |
| **Dyna-Q** | **C** | **0.78** | **Yes ✓** |
| Naive Offline DQN | Joint | 17.12 | No |
| CQL | Joint | 11.80 | No |
| IDQN (MA) | Joint | +237% vs random | — |
