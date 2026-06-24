# AI Use Declaration
# AI Tool Usage Declaration

## Furkan Çalışkan (Role A)

### Claude (Anthropic)
- DQN, Double DQN, Dueling DQN training code architecture and implementation
- Observation engineering (time_left, drone-order distance matrix)
- Hyperparameter tuning guidance
- Git workflow and debugging
- General RL concepts explanation

### Google Gemini
- Project comprehension and task breakdown for Role A requirements
- Environment setup and debugging (.venv, requirements.txt, pytest errors)
- Baseline execution guidance (reproduce.sh, greedy_nearest)
- Initial DQN boilerplate code generation (stable_baselines3)

## What I Did
- Understood, reviewed, and ran all generated code
- Made design decisions (hyperparameters, architecture, which features to use)
- Analyzed results and diagnosed training behavior
- Will defend all implementation decisions in oral exam

## Begüm Durmuş (Role B)

## Tools Used

* **Claude (Anthropic)** — code scaffolding for REINFORCE, A2C, DDPG agents; debugging; project setup guidance

## What AI Did

* Generated initial boilerplate for agent classes and training loops
* Helped interpret simulator API (agent\_interface.py, env\_dispatch.py)
* Assisted with environment setup and Git configuration

## What I Did

* Understood, reviewed, and ran all generated code
* Made design decisions (hyperparameters, architecture choices)
* Analyzed results and diagnosed training behavior
* Will defend all implementation decisions in oral exam







Abdulsamet Kavas (Role C — Planning / Dyna-Q)



Tool used: Claude (Anthropic) — interactive coding assistant.



What I did:

\- Chose the method family for my role: I selected tabular Dyna-Q (over an MCTS/rollout planner) after reasoning that it fits the deliverables (learning curves, saved weights, a planning-steps ablation) and the frozen `act(obs)` evaluation contract.

\- Studied the underlying ideas (Dyna-Q from Sutton \& Barto, Q-learning) and the simulator's API contract to decide how to model the problem — in particular the state abstraction (SoC, hub distance, finish-margin, urgency, demand pressure) and the ASSIGN-vs-CHARGE decomposition.

\- Drove the design decisions: the depletion-aware masking rule, the charge-zone / charge-penalty / safety-margin thresholds, and the planning-step setting (n=10, chosen from my ablation).

\- Analyzed results and iterated on the method: when the agent failed (over-charging bias, then zero deliveries, then fleet depletion), I diagnosed each failure from measured agent behavior (action distribution, surviving-drone count) and proposed the next fix, rather than tuning blindly.

\- Ran, tested and validated every component on my machine.



What AI did:

\- Provided step-by-step guidance and produced code drafts for each component (BFS distances, state features, the Dyna-Q agent, the training loop) based on the design choices above.

\- Helped interpret the simulator API, set up the environment, and configure Git.



All implementation decisions are mine to explain and defend in the oral exam.

