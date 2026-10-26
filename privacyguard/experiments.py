"""One-command, seeded experiments. All published numbers originate here."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from .env import Config, Kernel
from .train import train
from .analysis import baseline_policies, evaluate, exact_values, policy_from_q, start_value

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"q": "Q-learning", "sarsa": "SARSA", "myopic": "Myopic Q (gamma=0)",
          "no_budget": "Q without budget", "no_fatigue": "Q without fatigue"}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker(job):
    seed, episodes, mode, config, directory = job
    result = train(config, seed, episodes, mode)
    path = Path(directory) / f"{mode}_seed{seed}.npz"
    np.savez_compressed(path, q=result.pop("q"), visits=result.pop("visits"),
                        seed=seed, episodes=episodes, mode=mode,
                        config=json.dumps(asdict(config)))
    result["path"] = str(path.name)
    return result


def ci(values):
    """Student-t 95% CI over independent training/evaluation seed means."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return float(critical * values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=120000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Smoke run in results_quick; does not overwrite final results")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible saved training runs")
    args = parser.parse_args()
    if args.quick:
        args.episodes, args.seeds, args.eval_episodes = 2000, 2, 128
    if min(args.episodes, args.seeds, args.eval_episodes, args.workers) < 1:
        parser.error("All counts must be positive")
    result_dir = ROOT / ("results_quick" if args.quick else "results")
    model_dir = result_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.save(result_dir / "config.json")
    started = time.perf_counter()
    modes = list(LABELS)
    jobs = [(seed, args.episodes, mode, cfg, str(model_dir)) for mode in modes for seed in range(args.seeds)]
    training = []
    pending = []
    for job in jobs:
        cache = result_dir / f"training_{job[2]}_{job[0]}.json"
        model = model_dir / f"{job[2]}_seed{job[0]}.npz"
        if args.resume and cache.exists() and model.exists():
            record = json.loads(cache.read_text())
            with np.load(model, allow_pickle=False) as checkpoint:
                compatible = (int(checkpoint["episodes"]) == args.episodes
                              and json.loads(str(checkpoint["config"])) == asdict_json(cfg))
            if compatible:
                training.append(record)
                continue
        pending.append(job)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in pending]
        for future in as_completed(futures):
            record = future.result()
            training.append(record)
            dump(result_dir / f"training_{record['mode']}_{record['seed']}.json", record)
            print(f"Trained {record['mode']} seed {record['seed']}: {record['seconds']:.1f}s", flush=True)
    k = Kernel(cfg)
    optimal_q, optimal_v = exact_values(k)
    baselines = baseline_policies(k)
    per_seed, paired, action_rows, curves = [], [], [], []
    training.sort(key=lambda row: (row["mode"], row["seed"]))
    for seed in range(args.seeds):
        tape = k.tape(args.eval_episodes, 90000 + seed)
        policies = dict(baselines)
        policies["Exact DP ceiling"] = optimal_q.argmax(axis=1)
        for mode in modes:
            with np.load(model_dir / f"{mode}_seed{seed}.npz", allow_pickle=False) as checkpoint:
                policies[LABELS[mode]] = policy_from_q(k, checkpoint["q"], mode)
        evaluated = {}
        for label, policy in policies.items():
            metrics = evaluate(k, policy, *tape)
            evaluated[label] = metrics
            row = {"policy": label, "seed": seed}
            row.update({name: float(metrics[name].mean()) for name in ("return", "utility", "exposure", "prompts", "budget_left")})
            per_seed.append(row)
            for a, count in enumerate(metrics["action_counts"]):
                action_rows.append({"policy": label, "seed": seed, "action": a,
                                    "fraction": float(count / (args.eval_episodes * cfg.horizon))})
        for opponent in ("Greedy immediate", LABELS["myopic"], LABELS["no_budget"], LABELS["no_fatigue"], "SARSA"):
            delta = evaluated["Q-learning"]["return"] - evaluated[opponent]["return"]
            paired.append({"opponent": opponent, "seed": seed, "paired_gain": float(delta.mean())})
    for record in training:
        curves.extend({"policy": LABELS[record["mode"]], "seed": record["seed"], **point} for point in record["curve"])
    summary = []
    for label in sorted({row["policy"] for row in per_seed}):
        subset = [row for row in per_seed if row["policy"] == label]
        row = {"policy": label}
        for metric in ("return", "utility", "exposure", "prompts", "budget_left"):
            values = [item[metric] for item in subset]
            row[metric] = float(np.mean(values))
            row[metric + "_ci95"] = ci(values)
        summary.append(row)
    paired_summary = []
    for label in sorted({row["opponent"] for row in paired}):
        values = [row["paired_gain"] for row in paired if row["opponent"] == label]
        paired_summary.append({"opponent": label, "gain": float(np.mean(values)), "ci95": ci(values)})
    # Frozen trained policies under predeclared, separate stress conditions.
    robustness = []
    shifts = {
        "Nominal": cfg,
        "Sensitive requests": replace(cfg, sensitivity_probs=(.20, .30, .50)),
        "High-value requests": replace(cfg, utility_probs=(.20, .30, .50)),
        "Consent reluctance": replace(cfg, accept_base=.72, accept_fatigue=.30),
        "Higher exposure cost": replace(cfg, privacy_weight=.90),
    }
    for shift, shifted_cfg in shifts.items():
        sk = Kernel(shifted_cfg)
        for seed in range(args.seeds):
            tape = sk.tape(args.eval_episodes, 120000 + seed)
            with np.load(model_dir / f"q_seed{seed}.npz", allow_pickle=False) as checkpoint:
                frozen = policy_from_q(sk, checkpoint["q"])
            # Greedy knows shifted immediate rewards; learned Q is frozen.
            for label, policy in {"Q-learning": frozen, "Greedy immediate": sk.expected_reward.argmax(axis=1)}.items():
                metrics = evaluate(sk, policy, *tape)
                robustness.append({"condition": shift, "policy": label, "seed": seed,
                                   "return": float(metrics["return"].mean())})
    csv_write(result_dir / "per_seed.csv", per_seed)
    csv_write(result_dir / "summary.csv", summary)
    csv_write(result_dir / "paired_gains.csv", paired)
    csv_write(result_dir / "paired_summary.csv", paired_summary)
    csv_write(result_dir / "learning_curves.csv", curves)
    csv_write(result_dir / "action_fractions.csv", action_rows)
    csv_write(result_dir / "robustness.csv", robustness)
    # First seed is chosen in advance, not the strongest test seed.
    if not args.quick:
        agent_dir = ROOT / "agents"
        agent_dir.mkdir(exist_ok=True)
        (agent_dir / "q_learning_seed0.npz").write_bytes((model_dir / "q_seed0.npz").read_bytes())
        with np.load(agent_dir / "q_learning_seed0.npz", allow_pickle=False) as checkpoint:
            selected_policy = policy_from_q(k, checkpoint["q"])
        trace = evaluate(k, selected_policy, *k.tape(1, 314159), details=True)["trace"]
        dump(result_dir / "demo_trace.json", trace)
        export_demo(k, checkpoint_path=agent_dir / "q_learning_seed0.npz", summary=summary, directory=ROOT / "demo")
    manifest = {"config": asdict(cfg), "arguments": vars(args), "python": sys.version,
                "numpy": np.__version__, "platform": platform.platform(),
                "test_seeds": [90000 + seed for seed in range(args.seeds)],
                "validation_seed": 50000, "stress_test_seeds": [120000 + seed for seed in range(args.seeds)],
                "states_including_terminal": k.nstates, "nonterminal_states": k.nstates - (cfg.budget + 1) * 54,
                "exact_optimal_expected_start_return": start_value(k, optimal_v),
                "total_wall_seconds": time.perf_counter() - started,
                "training": training, "summary": summary, "paired_summary": paired_summary}
    dump(result_dir / "manifest.json", manifest)
    hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(model_dir.glob("*.npz"))}
    dump(result_dir / "model_sha256.json", hashes)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved experiments in {result_dir}; wall time {manifest['total_wall_seconds']:.1f}s", flush=True)
    from .plots import make_plots
    make_plots(result_dir)


def asdict_json(config):
    return json.loads(json.dumps(asdict(config)))


def export_demo(k, checkpoint_path, summary, directory):
    directory.mkdir(exist_ok=True)
    with np.load(checkpoint_path, allow_pickle=False) as agent:
        q = agent["q"]
        visits = agent["visits"]
    values = np.where(k.mask, q, -1e9)
    data = {"shape": list(k.shape), "config": asdict(k.config),
            "q": values.tolist(), "visits": visits.tolist(),
            "summary": summary}
    # JSON is embedded in a JS assignment so the demo works from file:// offline.
    (directory / "model.js").write_text("window.PRIVACYGUARD = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")


if __name__ == "__main__":
    main()
