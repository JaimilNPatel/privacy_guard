# Measured results

All numbers below were generated from the saved experiment CSVs.

**Q-learning improves session return by 35.26% over greedy immediate sharing and reaches 95.98% of the exact model optimum.**

| Policy | Return ± 95% t CI | Utility | Exposure | Prompts |
|---|---:|---:|---:|---:|
| Q-learning | 18.232 ± 0.118 | 23.761 | 9.483 | 1.181 |
| SARSA | 17.912 ± 0.143 | 23.448 | 9.610 | 1.011 |
| Myopic Q | 12.900 ± 0.115 | 19.274 | 9.979 | 0.465 |
| No budget obs. | 12.086 ± 0.395 | 19.262 | 9.962 | 2.212 |
| No fatigue obs. | 18.178 ± 0.092 | 23.707 | 9.496 | 1.110 |
| Greedy | 13.479 ± 0.106 | 20.170 | 9.975 | 1.457 |
| Always coarse | 8.735 ± 0.064 | 14.767 | 9.998 | 0.000 |
| Exact DP | 19.017 ± 0.100 | 24.522 | 9.293 | 1.353 |

Exact expected optimum: 18.996224. Its sampled test return differs due to Monte Carlo noise. The learned-return gap estimated against that ceiling is 0.764.

## Paired comparisons

Positive values favour full Q-learning. Intervals use eight independent seed means.

| Opponent | Paired return gain ± 95% t CI |
|---|---:|---:|
| Greedy immediate | 4.753 ± 0.076 |
| Myopic Q (gamma=0) | 5.332 ± 0.115 |
| Q without budget | 6.146 ± 0.323 |
| Q without fatigue | 0.054 ± 0.055 |
| SARSA | 0.320 ± 0.098 |

The fatigue-observation advantage is small and its paired interval includes zero. This experiment does not establish a reliable benefit from observing fatigue under the nominal parameters. Budget observation and future planning have much larger measured effects.

## Stress tests

Q-learning is frozen; greedy knows the shifted immediate reward parameters. These are simulation stress tests, not evidence of generalisation to real users.

| Condition | Frozen Q return | Greedy return |
|---|---:|---:|
| Nominal | 18.237 ± 0.109 | 13.421 ± 0.080 |
| Sensitive requests | 13.476 ± 0.083 | 9.085 ± 0.054 |
| High-value requests | 29.041 ± 0.138 | 20.019 ± 0.131 |
| Consent reluctance | 17.488 ± 0.121 | 12.979 ± 0.071 |
| Higher exposure cost | 13.973 ± 0.113 | 12.796 ± 0.067 |

## Protocol

Eight seeds × five variants × 120,000 training episodes = 4,800,000 simulated training episodes (including shortened exploring-start episodes). Each policy has 5,000 held-out sessions per seed. Final checkpoints, no test-based selection. Saved demo agent: seed 0.

Training/evaluation wall time on the execution machine: 203.0 seconds, excluding plotting. All seed-level metrics, action fractions, checkpoint hashes, and validation curves are retained in results/.

## Interpretation

The agent allocates scarce exposure toward valuable requests rather than maximising immediate task benefit. It delivers higher measured utility while using less exposure than the greedy reference. This is an internal benchmark result under the declared synthetic cost model. It does not validate the cost weights, prove privacy, or guarantee a course grade.
