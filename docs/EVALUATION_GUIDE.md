# Evaluation and viva guide

## Three-minute demonstration

1. Open `demo/index.html`. Explain that the page uses a saved Q table trained in Python, not a rule pretending to be RL.
2. Show a high-utility request with permission and sufficient budget. Read the largest feasible Q value and its visit count.
3. Lower the budget for the same request. Explain that the budget must also serve future requests. Some actions become impossible and others become unattractive before they are impossible.
4. Remove full permission. Full is blocked. An ask may still obtain one-request consent; coarse sharing has only its assumed limited permission.
5. Select the "Ask first" preset. Then raise fatigue and inspect how its consent probability and Q values change. Do not claim every slice changes action: learned policies and numerical estimates vary.
6. Run a full session. Show actual consent rejections, reward changes, and the budget audit. Click New session to reproduce the same browser seed. The browser uses its own explicitly seeded RNG, so it need not match Python's NumPy trace.
7. Show measured results and explain the myopic ablation. Its zero discount tests whether considering the next request matters.

For a terminal demonstration run `python -m privacyguard demo`, then `python -m privacyguard explain --state 6 6 0 2 1 1`.

## One-minute opening

PrivacyGuard solves a sequential data-sharing problem. An assistant receives twelve requests with changing task utility, sensitivity, and permission. It can deny, provide a coarse response, ask for consent, or share fully. Sharing spends a finite exposure budget; repeated asking causes consent fatigue. Our Q-learning agent learns future action values from simulated experience. A separate guard enforces permission and budget limits. We compare it with SARSA, an immediate-reward learner, observation ablations, fixed strategies, and an analytical ceiling. All reported numbers come from eight-seed held-out experiments, and the saved agent runs in an offline, auditable demo.

## Questions to prepare

**Why RL rather than a classifier?** There are no labelled correct actions. Actions change remaining budget and fatigue, which change future rewards. The target is expected total session return.

**Why not DQN or PPO?** A six-variable discrete state has only 7,128 nonterminal combinations. A table is fast, explainable, easy to audit, and sufficient for this mini-project. A neural model would add complexity without resolving a limitation of this finite benchmark.

**What makes the process Markov?** Under our assumptions, remaining requests, budget, fatigue, and current context determine the next-state distribution and reward. Requests are independently drawn. Real users need a richer model; we do not claim this is universal.

**Why include requests remaining?** The same unit of budget is more valuable when many future requests remain. Including time turns a finite-horizon problem into a stationary table lookup on an augmented state.

**Does consent make privacy cost zero?** No. Consent changes authorisation, not information exposure. An accepted consent action consumes exactly the same full exposure as direct full sharing.

**Is the budget differential privacy?** No. It is an additive, interpretable modelling resource. No formal DP mechanism or epsilon composition is claimed.

**Can exploration violate permission?** No. Both epsilon-greedy exploration and greedy evaluation select only feasible actions. The environment also rejects invalid actions.

**Is the guard itself the policy?** No. It eliminates impossible/unauthorised actions. RL ranks the remaining choices. The same guard is applied to every reference strategy and ablation.

**Is your exact dynamic program the submitted solution?** No. Q-learning and SARSA learn from sampled transitions with zero initial values. The separate exact solver provides a ceiling and checks the simulator after training. There is no expert pretraining or copying its actions.

**Why gamma=1?** Episodes terminate after at most twelve requests. The total undiscounted return is finite. We value future requests equally while their opportunity cost is learned through remaining budget.

**Do you prove convergence?** No. Classical tabular convergence requires conditions beyond a finite practical run. We use a learning-rate floor and report measured performance plus the exact ceiling.

**Why eight seeds and paired tapes?** Independent training runs expose variation from exploration. Identical exogenous request/consent uniforms reduce noise when comparing policies. Intervals use eight independent seed means, not 40,000 sessions as if they were separate training runs.

**Why exploring starts?** Normal starts under-sample rare budget/fatigue/time combinations. Twenty percent of training episodes start in randomly chosen nonterminal resource states. The remaining eighty percent and all evaluations use the normal initial state.

**Is your privacy-cost model validated?** It is explicit and synthetic. Utility bins, exposure units, prompt costs, and consent probabilities are assumptions. Stress tests vary several assumptions. Empirical calibration is a future step, not an existing result.

**What is distinctive?** A simple explainable RL agent unifies permission-aware disclosure, a consumable exposure resource, stochastic consent, and action-induced fatigue. The exceptional-effort component is the complete audit path: invariants, exact ceiling, paired multiseed experiments, ablations, stress tests, saved models, and an offline explanation demo. Do not claim global research novelty.

**What if an ablation performs similarly?** Report it. A small/noisy fatigue effect means its marginal benefit is limited under these parameters. The core sequential claim is tested separately through the myopic learner and greedy reference.

**Why does greedy perform worse?** It picks the greatest expected immediate reward without considering the value of budget and fatigue at later requests. Compare measured returns rather than assuming it must always lose in every environment.

**What did each member contribute?** Fill this answer from your actual work. Do not invent contributions. Both members should trace `env.py`, the TD update, and the result tables before presenting.

## Suggested real division of remaining work

Patel Jaimil can verify the simulator/reward model and explain the privacy assumptions; Nitin can reproduce the training and verify the evaluation statistics. This is a suggested study split, not a record of completed work. Both should test the demo, review the report, and commit their actual edits.
