"""Checks for safety, model semantics, stochastic evaluation, and DP consistency."""
from dataclasses import replace
import numpy as np
import pytest
from privacyguard.env import ASK, COARSE, DENY, FULL, Config, Kernel, PrivacyEnv
from privacyguard.analysis import evaluate, exact_values, start_value
from privacyguard.train import train, observation_map


@pytest.fixture(scope="module")
def k():
    return Kernel()


def test_state_encoding_is_bijective(k):
    for i in range(k.nstates):
        assert k.encode(k.decode(i)) == i


def test_every_feasible_transition_respects_budget_and_permission(k):
    for state, (h, b, f, u, s, permission) in enumerate(k.state_components):
        if not h:
            continue
        assert k.mask[state, DENY]
        if not permission:
            assert not k.mask[state, FULL]
        for a in k.allowed[state]:
            for outcome in range(2):
                nh, nb, nf, *_ = k.decode(k.next_base[state, a, outcome])
                assert nh == h - 1 and 0 <= nb <= b and 0 <= nf <= 2
                assert b - nb == k.exposure[state, a, outcome]


def test_consent_does_not_erase_exposure(k):
    state = k.encode((8, 10, 0, 2, 2, 1))
    assert k.exposure[state, ASK, 1] == k.exposure[state, FULL, 1] == 4
    assert k.utility[state, ASK, 0] == k.exposure[state, ASK, 0] == 0
    assert k.reward[state, ASK, 0] < k.reward[state, DENY, 0]
    assert k.reward[state, FULL, 1] - k.reward[state, ASK, 1] == pytest.approx(.25)


def test_fatigue_is_action_dependent(k):
    s = k.encode((8, 10, 1, 2, 1, 0))
    assert k.decode(k.next_base[s, ASK, 0])[2] == 2
    assert k.decode(k.next_base[s, COARSE, 0])[2] == 0
    assert k.paccept[k.encode((8, 10, 2, 2, 1, 0))] < k.paccept[s]


def test_one_step_dp_matches_expected_immediate_rewards():
    k = Kernel(Config(horizon=1, budget=4))
    q, v = exact_values(k)
    nonterminal = k.state_components[:, 0] > 0
    assert np.allclose(q[nonterminal], k.expected_reward[nonterminal])
    assert np.allclose(v[nonterminal], k.expected_reward[nonterminal].max(axis=1))


def test_dp_ceiling_matches_independent_monte_carlo():
    k = Kernel(Config(horizon=3, budget=3))
    q, v = exact_values(k)
    observed = evaluate(k, q.argmax(axis=1), *k.tape(50000, 711))["return"]
    assert abs(observed.mean() - start_value(k, v)) < 5 * observed.std() / np.sqrt(len(observed))


def test_trajectory_reward_accounting_and_determinism(k):
    policy = k.expected_reward.argmax(axis=1)
    a = evaluate(k, policy, *k.tape(20, 765))
    b = evaluate(k, policy, *k.tape(20, 765))
    assert np.array_equal(a["return"], b["return"])
    assert np.all(a["exposure"] <= k.config.budget)
    assert np.all(a["budget_left"] >= 0)


def test_reset_step_and_terminal_guard():
    env = PrivacyEnv(Config(horizon=1, budget=0))
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(FULL)
    _, _, done, _, info = env.step(DENY)
    assert done and info["exposure"] == 0
    with pytest.raises(RuntimeError):
        env.step(DENY)


def test_ablation_really_hides_observation(k):
    a = k.encode((6, 2, 0, 1, 1, 0))
    b = k.encode((6, 8, 0, 1, 1, 0))
    assert observation_map(k, "no_budget")[a] == observation_map(k, "no_budget")[b]
    c = k.encode((6, 2, 2, 1, 1, 0))
    assert observation_map(k, "no_fatigue")[a] == observation_map(k, "no_fatigue")[c]
    assert observation_map(k, "q")[a] != observation_map(k, "q")[b]


def test_sample_learning_reproducible_and_beats_initial_deny():
    cfg = Config(horizon=2, budget=2)
    a = train(cfg, seed=17, episodes=1200, checkpoints=1)
    b = train(cfg, seed=17, episodes=1200, checkpoints=1)
    assert np.array_equal(a["q"], b["q"])
    k = Kernel(cfg)
    from privacyguard.analysis import policy_from_q
    tape = k.tape(2000, 123456)
    assert evaluate(k, policy_from_q(k, a["q"]), *tape)["return"].mean() > evaluate(k, np.zeros(k.nstates, dtype=int), *tape)["return"].mean() + 1


def test_invalid_config():
    with pytest.raises(ValueError):
        replace(Config(), utility_probs=(.1, .2, .3))
