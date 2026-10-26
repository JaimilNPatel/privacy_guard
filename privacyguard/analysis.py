"""Analytical checks and vectorised evaluation, never used to train an agent."""
import numpy as np
from .env import ASK, COARSE, DENY, FULL


def exact_values(kernel):
    """Finite-horizon backward induction, a model-informed diagnostic ceiling."""
    k = kernel
    q = np.full((k.nstates, 4), -1e9)
    v = np.zeros(k.nstates)
    block = (k.config.budget + 1) * 3 * 18
    for h in range(1, k.config.horizon + 1):
        rows = np.arange(h * block, (h + 1) * block)
        previous = v.reshape(k.shape[:3] + (18,))
        future = (previous * k.context_probs).sum(axis=-1).reshape(-1)
        for a in range(4):
            p = k.paccept[rows]
            n0 = k.next_base[rows, a, 0] // 18
            n1 = k.next_base[rows, a, 1] // 18
            q[rows, a] = ((1 - p) * (k.reward[rows, a, 0] + future[n0])
                          + p * (k.reward[rows, a, 1] + future[n1]))
        q[rows] = np.where(k.mask[rows], q[rows], -1e9)
        v[rows] = q[rows].max(axis=1)
    q[:block, DENY] = 0
    return q, v


def evaluate(kernel, policy, contexts, uniforms, details=False):
    """Policy maps every true state to an action. Reject any guard violation."""
    k, n = kernel, len(contexts)
    state = k.encode((k.config.horizon, k.config.budget, 0, 0, 0, 0)) + contexts[:, 0]
    ret, utility, exposure, prompts, accepted = [np.zeros(n) for _ in range(5)]
    counts = np.zeros(4, dtype=int)
    trace = []
    for t in range(k.config.horizon):
        action = policy[state]
        if not np.all(k.mask[state, action]):
            raise AssertionError("Evaluation policy violates action guard")
        outcome = (uniforms[:, t] < k.paccept[state]).astype(int)
        r = k.reward[state, action, outcome]
        ret += r
        utility += k.utility[state, action, outcome]
        exposure += k.exposure[state, action, outcome]
        prompts += action == ASK
        accepted += (action == ASK) & (outcome == 1)
        counts += np.bincount(action, minlength=4)
        if details:
            trace.append({"step": t + 1, "state": list(k.decode(state[0])),
                          "action": int(action[0]), "accepted": bool(action[0] == ASK and outcome[0]),
                          "reward": float(r[0]), "total_return": float(ret[0])})
        state = k.next_base[state, action, outcome] + contexts[:, t + 1]
    result = {"return": ret, "utility": utility, "exposure": exposure, "prompts": prompts,
              "accepted": accepted, "action_counts": counts,
              "budget_left": k.config.budget - exposure, "trace": trace}
    return result


def baseline_policies(k):
    """Simple reference strategies, not the submitted RL solution."""
    greedy = k.expected_reward.argmax(axis=1)
    coarse = np.where(k.mask[:, COARSE], COARSE, DENY)
    full = np.where(k.mask[:, FULL], FULL, np.where(k.mask[:, ASK], ASK, DENY))
    ask = np.where(k.mask[:, ASK], ASK, DENY)
    rng = np.random.default_rng(2026)
    random = np.array([rng.choice(allowed) for allowed in k.allowed])
    return {"Greedy immediate": greedy, "Always coarse": coarse, "Full or consent": full,
            "Always consent": ask, "Always deny": np.zeros(k.nstates, dtype=int),
            "Random feasible": random}


def policy_from_q(k, q, mode="q"):
    if mode == "no_budget":
        lookup = np.arange(k.nstates) % (3 * 18)
        lookup += (np.arange(k.nstates) // ((k.config.budget + 1) * 3 * 18)) * 3 * 18
        values = q[lookup]
    elif mode == "no_fatigue":
        sc = k.state_components
        lookup = (((sc[:, 0] * (k.config.budget + 1) + sc[:, 1]) * 3 + sc[:, 3]) * 3 + sc[:, 4]) * 2 + sc[:, 5]
        values = q[lookup]
    else:
        values = q
    return np.where(k.mask, values, -1e9).argmax(axis=1)


def start_value(k, v):
    base = k.encode((k.config.horizon, k.config.budget, 0, 0, 0, 0))
    return float(v[base:base + 18] @ k.context_probs)
