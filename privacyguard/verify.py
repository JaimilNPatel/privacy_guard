"""Audit saved artifacts by recomputing every nominal held-out seed mean."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from .env import ACTIONS, Config, Kernel
from .analysis import baseline_policies, evaluate, exact_values, policy_from_q, start_value
from .experiments import LABELS, ROOT, ci
from .plots import read


def main():
    folder = ROOT / "results"
    manifest = json.loads((folder / "manifest.json").read_text())
    cfg = Config.load(folder / "config.json")
    k = Kernel(cfg)
    qstar, vstar = exact_values(k)
    assert abs(start_value(k, vstar) - manifest["exact_optimal_expected_start_return"]) < 1e-10
    records = read(folder / "per_seed.csv")
    summary = read(folder / "summary.csv")
    recorded_hashes = json.loads((folder / "model_sha256.json").read_text())
    transition_count = 0
    for relative, digest in recorded_hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
        with np.load(ROOT / relative, allow_pickle=False) as agent:
            assert np.all(np.isfinite(agent["q"]))
            transition_count += int(agent["visits"].sum())
    checks = 0
    for seed in range(manifest["arguments"]["seeds"]):
        tape = k.tape(manifest["arguments"]["eval_episodes"], 90000 + seed)
        policies = baseline_policies(k)
        policies["Exact DP ceiling"] = qstar.argmax(axis=1)
        for mode, label in LABELS.items():
            with np.load(folder / "models" / f"{mode}_seed{seed}.npz", allow_pickle=False) as agent:
                policies[label] = policy_from_q(k, agent["q"], mode)
        for label, policy in policies.items():
            metric = evaluate(k, policy, *tape)
            row = next(r for r in records if r["policy"] == label and int(r["seed"]) == seed)
            for field in ("return", "utility", "exposure", "prompts", "budget_left"):
                assert abs(float(metric[field].mean()) - float(row[field])) < 1e-10, (seed, label, field)
                checks += 1
            assert metric["exposure"].max() <= cfg.budget
    for row in summary:
        subset = [r for r in records if r["policy"] == row["policy"]]
        for field in ("return", "utility", "exposure", "prompts", "budget_left"):
            values = [float(r[field]) for r in subset]
            assert abs(np.mean(values) - float(row[field])) < 1e-10
            assert abs(ci(values) - float(row[field + "_ci95"])) < 1e-10
            checks += 2
    exported = ROOT / "agents" / "q_learning_seed0.npz"
    assert exported.read_bytes() == (folder / "models" / "q_seed0.npz").read_bytes()
    demo_text = (ROOT / "demo" / "model.js").read_text()
    browser = json.loads(demo_text[len("window.PRIVACYGUARD = "):].rstrip(";\n"))
    with np.load(exported, allow_pickle=False) as agent:
        python_policy = policy_from_q(k, agent["q"])
        browser_q = np.asarray(browser["q"])
        assert np.all(np.where(k.mask, browser_q, -1e9).argmax(axis=1) == python_policy)
        assert np.array_equal(browser["visits"], agent["visits"])
    paired = read(folder / "paired_summary.csv")
    for row in paired:
        values = [float(r["paired_gain"]) for r in read(folder / "paired_gains.csv") if r["opponent"] == row["opponent"]]
        assert abs(float(row["gain"]) - np.mean(values)) < 1e-10
        assert abs(float(row["ci95"]) - ci(values)) < 1e-10
    source_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((ROOT / "privacyguard").glob("*.py"))}
    verification = {"status": "passed", "recomputed_metric_checks": checks,
                    "checkpoint_hashes_checked": len(recorded_hashes),
                    "total_sampled_training_transitions": transition_count,
                    "all_exported_actions_match_python": True,
                    "nominal_policy_seed_evaluations_reproduced": len(records),
                    "permission_and_budget_guard_violations": 0,
                    "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                    "source_sha256_at_verification": source_hashes,
                    "remaining_external_step": "Publish actual GitHub repository and replace report profile note"}
    (folder / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
