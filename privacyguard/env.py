"""Original finite MDP. No personal data or external RL code is used.

The table is a simulator implementation, not a learned or optimal policy.
The learner receives sampled transitions only. The analytical comparator is
implemented separately in analysis.py and is never used as a training target.
"""
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np

DENY, COARSE, ASK, FULL = range(4)
ACTIONS = ("Deny", "Coarse", "Ask consent", "Full")
UTILITY = np.array([1.0, 3.0, 7.0])
EXPOSURE = np.array([1, 2, 4])


@dataclass(frozen=True)
class Config:
    horizon: int = 12
    budget: int = 10
    privacy_weight: float = 0.45
    coarse_fraction: float = 0.60
    prompt_base: float = 0.25
    prompt_fatigue: float = 0.30
    missed_weight: float = 0.15
    accept_base: float = 0.92
    accept_sensitivity: float = 0.12
    accept_fatigue: float = 0.25
    utility_probs: tuple = (0.45, 0.35, 0.20)
    sensitivity_probs: tuple = (0.45, 0.35, 0.20)
    scope_probs: tuple = (0.40, 0.60)

    def __post_init__(self):
        if self.horizon < 1 or self.budget < 0:
            raise ValueError("Horizon must be positive and budget nonnegative")
        for name in ("utility_probs", "sensitivity_probs", "scope_probs"):
            values = np.asarray(getattr(self, name), dtype=float)
            if np.any(values < 0) or not np.isclose(values.sum(), 1):
                raise ValueError(f"Invalid probability vector: {name}")
        if not 0 <= self.coarse_fraction <= 1 or self.privacy_weight < 0:
            raise ValueError("Invalid utility fraction or privacy weight")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class Kernel:
    """Precomputed stochastic transition kernel, indexed by six observations.

    State: (requests remaining, budget, fatigue, utility bin, sensitivity bin,
    standing-full-permission bit). Fatigue is in {0,1,2}. Outcome 1 means
    acceptance for ASK; both outcomes are identical for other actions.
    """

    def __init__(self, config=Config()):
        self.config = config
        self.shape = (config.horizon + 1, config.budget + 1, 3, 3, 3, 2)
        self.nstates = int(np.prod(self.shape))
        self.contexts = list(product(range(3), range(3), range(2)))
        self.context_probs = np.array([
            config.utility_probs[u] * config.sensitivity_probs[s] * config.scope_probs[p]
            for u, s, p in self.contexts
        ])
        self.mask = np.zeros((self.nstates, 4), dtype=bool)
        self.reward = np.zeros((self.nstates, 4, 2))
        self.utility = np.zeros_like(self.reward)
        self.exposure = np.zeros_like(self.reward, dtype=np.int16)
        self.prompt = np.zeros_like(self.reward)
        self.next_base = np.zeros_like(self.reward, dtype=np.int32)
        self.paccept = np.zeros(self.nstates)
        self.state_components = np.array(list(np.ndindex(self.shape)), dtype=np.int16)
        c = config
        for state, (h, b, f, u, s, permission) in enumerate(self.state_components):
            if h == 0:
                self.mask[state, DENY] = True
                continue
            full_cost = int(EXPOSURE[s])
            coarse_cost = int(np.ceil(0.35 * full_cost))
            self.mask[state] = (True, b >= coarse_cost, b >= full_cost,
                                permission == 1 and b >= full_cost)
            self.paccept[state] = np.clip(c.accept_base - c.accept_sensitivity * s
                                         - c.accept_fatigue * f, 0.10, 0.95)
            for a in range(4):
                for accepted in range(2):
                    share_full = a == FULL or (a == ASK and accepted == 1)
                    cost = full_cost if share_full else coarse_cost if a == COARSE else 0
                    delivered = UTILITY[u] if share_full else c.coarse_fraction * UTILITY[u] if a == COARSE else 0
                    prompt = c.prompt_base + c.prompt_fatigue * f if a == ASK else 0
                    missed = c.missed_weight * UTILITY[u] if delivered == 0 else 0
                    self.utility[state, a, accepted] = delivered
                    self.exposure[state, a, accepted] = cost
                    self.prompt[state, a, accepted] = prompt
                    self.reward[state, a, accepted] = delivered - c.privacy_weight * cost - prompt - missed
                    nf = min(2, f + 1) if a == ASK else max(0, f - 1)
                    # Infeasible actions never execute. Clamp only keeps table indices valid.
                    self.next_base[state, a, accepted] = self.encode((h - 1, max(0, b - cost), nf, 0, 0, 0))
        self.allowed = [np.flatnonzero(row) for row in self.mask]
        self.expected_reward = ((1 - self.paccept[:, None]) * self.reward[:, :, 0]
                                + self.paccept[:, None] * self.reward[:, :, 1])
        self.expected_reward[~self.mask] = -1e9

    def encode(self, state):
        h, b, f, u, s, p = map(int, state)
        return (((((h * self.shape[1] + b) * 3 + f) * 3 + u) * 3 + s) * 2 + p)

    def decode(self, state):
        return tuple(map(int, self.state_components[state]))

    def tape(self, episodes, seed):
        """Exogenous events do not depend on chosen actions (paired comparisons)."""
        rng = np.random.default_rng(seed)
        contexts = rng.choice(18, size=(episodes, self.config.horizon + 1), p=self.context_probs)
        uniforms = rng.random((episodes, self.config.horizon))
        return contexts, uniforms


class PrivacyEnv:
    """Small reset/step interface; invalid actions raise rather than silently map."""

    def __init__(self, config=Config()):
        self.kernel = Kernel(config)
        self.rng = np.random.default_rng()
        self.state = None
        self.done = True

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        k = self.kernel
        ctx = int(self.rng.choice(18, p=k.context_probs))
        self.state = k.encode((k.config.horizon, k.config.budget, 0, 0, 0, 0)) + ctx
        self.done = False
        return k.decode(self.state), {"action_mask": k.mask[self.state].copy()}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a terminal environment")
        k, state = self.kernel, self.state
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4 or not k.mask[state, action]:
            raise ValueError("Action blocked by permission/budget guard")
        accepted = int(self.rng.random() < k.paccept[state])
        reward = float(k.reward[state, action, accepted])
        self.state = int(k.next_base[state, action, accepted]) + int(self.rng.choice(18, p=k.context_probs))
        self.done = k.decode(self.state)[0] == 0
        info = {"action_mask": k.mask[self.state].copy(), "utility": float(k.utility[state, action, accepted]),
                "exposure": int(k.exposure[state, action, accepted]),
                "prompt_cost": float(k.prompt[state, action, accepted]),
                "accepted": bool(action == ASK and accepted), "action": ACTIONS[action]}
        return k.decode(self.state), reward, self.done, False, info
