"""Sample-based tabular Q-learning and SARSA, written for this project."""
import time
import numpy as np
from .env import Config, Kernel
from .analysis import evaluate, policy_from_q


def observation_map(k, mode):
    ids = np.arange(k.nstates)
    if mode == "no_budget":
        return ids % 54 + ids // ((k.config.budget + 1) * 54) * 54
    if mode == "no_fatigue":
        h, b, f, u, s, p = k.state_components.T.astype(np.int64)
        return (((h * (k.config.budget + 1) + b) * 3 + u) * 3 + s) * 2 + p
    return ids


def train(config=Config(), seed=0, episodes=120000, mode="q", checkpoints=12):
    """20% exploring starts improve rare-state coverage; 80% normal sessions.

    Every update uses one simulated transition. No dynamic-programming target,
    expert action, replay dataset, or held-out test trajectory is used.
    """
    if mode not in ("q", "sarsa", "myopic", "no_budget", "no_fatigue"):
        raise ValueError(f"Unknown mode {mode}")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    started = time.perf_counter()
    k = Kernel(config)
    rng = np.random.default_rng(seed)
    mapping = observation_map(k, mode)
    q = np.zeros((int(mapping.max()) + 1, 4))
    visits = np.zeros_like(q, dtype=np.int32)
    gamma = 0.0 if mode == "myopic" else 1.0
    validation = k.tape(384, 50000)
    curve = []
    interval = max(1, episodes // checkpoints)
    # Generate randomness in blocks, then retain ordinary sequential TD updates.
    for start in range(0, episodes, 2048):
        size = min(2048, episodes - start)
        contexts = rng.choice(18, size=(size, config.horizon + 1), p=k.context_probs)
        acceptance = rng.random((size, config.horizon))
        exploration = rng.random((size, config.horizon + 1))
        random_action = rng.integers(0, 24, size=(size, config.horizon + 1))
        starts = rng.random(size) < .20
        hs = rng.integers(1, config.horizon + 1, size=size)
        bs = rng.integers(0, config.budget + 1, size=size)
        fs = rng.integers(0, 3, size=size)
        for i in range(size):
            episode = start + i
            epsilon = max(.05, 1 - .95 * episode / max(1, .80 * episodes))
            h, b, f = (int(hs[i]), int(bs[i]), int(fs[i])) if starts[i] else (config.horizon, config.budget, 0)
            state = k.encode((h, b, f, 0, 0, 0)) + contexts[i, 0]
            obs = mapping[state]
            allowed = k.allowed[state]
            action = int(allowed[random_action[i, 0] % len(allowed)]) if exploration[i, 0] < epsilon else int(allowed[np.argmax(q[obs, allowed])])
            for t in range(h):
                outcome = int(acceptance[i, t] < k.paccept[state])
                reward = k.reward[state, action, outcome]
                next_state = k.next_base[state, action, outcome] + contexts[i, t + 1]
                next_obs = mapping[next_state]
                next_allowed = k.allowed[next_state]
                next_action = int(next_allowed[random_action[i, t + 1] % len(next_allowed)]) if exploration[i, t + 1] < epsilon else int(next_allowed[np.argmax(q[next_obs, next_allowed])])
                future = 0.0
                if t + 1 < h:
                    future = q[next_obs, next_action] if mode == "sarsa" else np.max(q[next_obs, next_allowed])
                visits[obs, action] += 1
                alpha = max(.04, .70 / (1 + visits[obs, action] / 80) ** .60)
                q[obs, action] += alpha * (reward + gamma * future - q[obs, action])
                state, obs, action = next_state, next_obs, next_action
            if (episode + 1) % interval == 0 or episode + 1 == episodes:
                val = evaluate(k, policy_from_q(k, q, mode), *validation)
                curve.append({"episode": episode + 1, "validation_return": float(val["return"].mean())})
    return {"q": q, "visits": visits, "curve": curve, "seconds": time.perf_counter() - started,
            "seed": seed, "episodes": episodes, "mode": mode}
