# Model and environment card

## Intended task

A personal assistant receives requests for information. It must choose how much to disclose while preserving a session-level information budget and limiting consent fatigue. Examples for explanation are precise vs coarse location, exact vs banded spending, or a complete vs aggregated activity summary. The simulator represents their utility/exposure bins; it does not access or process those data types.

## Observation and Markov property

State `s=(h,b,f,u,d,p)` has six discrete components. `h` is remaining requests (0–12), `b` is remaining exposure units (0–10), `f` is recent-prompt fatigue (0–2), `u` and `d` are request utility and sensitivity indices (0–2), and `p` is standing full-sharing permission (0 or 1). `h=0` terminates the episode.

The next context `(u',d',p')` is independent of all previous requests, with independent marginal probabilities `(0.45,0.35,0.20)`, `(0.45,0.35,0.20)`, and `(0.40,0.60)`. Given the current state and action, realised exposure, fatigue, reward, and the next-context distribution require no earlier history. This is the Markov assumption. Real users can have memory and correlated requests beyond these variables; those are outside this simulator.

## Complete action mechanics

| Action | Delivered utility | Exposure | Attention cost | Permission |
|---|---:|---:|---:|---|
| Deny | 0 | 0 | 0 | Always feasible |
| Coarse | 0.60 U | ceil(0.35 E) | 0 | Limited permission assumed; enough budget required |
| Ask, accepted | U | E | 0.25 + 0.30 f | One-request consent; enough budget for acceptance required |
| Ask, rejected | 0 | 0 | 0.25 + 0.30 f | No disclosure |
| Full | U | E | 0 | Standing full permission and enough budget required |

`U=(1,3,7)[u]`, `E=(1,2,4)[d]`. Asking is feasible only when the full exposure could be afforded; this avoids requesting consent that cannot be honoured. It is technically allowed with standing permission too, but is dominated by a direct full share for that immediate request. Learning is free to recognise this.

Consent probability is `P(accept|d,f)=clip(0.92-0.12d-0.25f,0.10,0.95)`. Asking updates `f'=min(2,f+1)` regardless of acceptance; all other actions update `f'=max(0,f-1)`. Let realised exposure be `e`. Then `b'=b-e` and `h'=h-1`.

## Reward rationale and calibration limits

`r=v-λe-c_prompt-0.15 U I[v=0]`, with `λ=0.45` and `c_prompt=(0.25+0.30f)I[ask]`.

Delivered utility rewards useful task completion. Exposure charges data disclosure even when consented. Prompt cost makes attention finite. The missed-utility penalty discourages the trivial deny-everything policy. The hard guard is separate from reward: violating permission is never an available trade-off. Reward weights and probabilities were declared as an interpretable synthetic design, not fitted to maximise a reported method's score. All baselines use the identical reward and transition model.

Illustration at high utility, high sensitivity, rested user: full share yields `7-0.45×4=5.20`; coarse yields `4.2-0.45×2=3.30`; deny yields `-1.05`. An accepted ask yields `4.95`; a rejected ask yields `-1.30`. Exposure is four units after both an accepted ask and a direct full share.

## Learning

All Q values start at zero. Q-learning updates one sampled transition with `Q(s,a) += α[r + γ max Q(s',a') - Q(s,a)]`, maximising only feasible actions. The terminal future value is zero. `γ=1` is well-defined because the horizon is finite. SARSA instead uses the next sampled epsilon-greedy action. The myopic learner uses `γ=0`.

The exploration probability decays linearly from 1 to 0.05 over the first 80% of training. Per-action learning rate is `max(0.04, 0.70/(1+N(s,a)/80)^0.60)`, using the post-increment visit count. The positive floor and finite training mean no asymptotic convergence claim is made. Eighty percent of training episodes start normally; twenty percent use uniformly sampled `(h,b,f)` to improve coverage of rare states. The environment then proceeds normally until its sampled horizon ends. All evaluation sessions start at `(h=12,b=10,f=0)`.

The budget-blind ablation aliases all budgets in its Q table; the fatigue-blind ablation aliases all fatigue levels. Both still use the same external action guard. Consequently the budget-blind agent can infer some information from which actions are feasible. It is an observation-ablation control with a guard, not complete removal of all budget information. State aliasing also means those ablations are not fully observed MDP learners.

## Evaluation design

Eight independently seeded trained agents per method are evaluated on separate held-out exogenous event tapes, 5,000 sessions per seed. Every policy in a seed uses identical request contexts and uniform random numbers for possible consent responses. Fatigue changes consent probability, so identical uniforms need not give identical outcomes. Training seeds 0–7, validation seed 50,000, held-out test seeds 90,000–90,007, and stress-test seeds 120,000–120,007 are disjoint.

Final checkpoints are evaluated, not the checkpoint with the highest validation or test score. Seed 0 is exported as the demo agent in advance. Means and Student-t 95% intervals use the eight seed means as independent units; the 40,000 individual sessions are not treated as independent training runs. Paired gains are first averaged within each seed and then given a t interval over the eight gains.

The random reference is a fixed random feasible action lookup table drawn with seed 2026; it is not a fresh random action at every visit. The exact model-informed optimal policy is computed only after training as an analytical ceiling. It never supplies actions, Q targets, or training data.

Stress conditions are predeclared in `experiments.py`: sensitive-heavy requests, high-value-heavy requests, lower consent willingness, and doubled privacy weight. Trained Q is frozen. The greedy comparator is recomputed from the stress environment's immediate reward model, so it knows those one-step parameters. These tests measure robustness within a small synthetic family, not real-world generalisation. Doubling the weight is not a retrained utility/privacy frontier.

## Boundaries

Exposure units are not probabilities of harm, legal compliance, money, or differential-privacy epsilon. Coarse utility and exposure are assumed; no anonymisation guarantee is proved. Repeated disclosure composition, external attackers, long-term user preference inference, emergencies, and user heterogeneity are not modelled. Full permission is supplied as an observation, not inferred. The demo never asks an actual person for permission or shares their data. A real product would require empirical preference calibration, verified data transforms, permission enforcement at the data layer, and independent review.
