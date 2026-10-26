"""Evaluator-facing commands; no retraining required to inspect the saved agent."""
import argparse
import json
from pathlib import Path
import numpy as np
from .env import ACTIONS, Config, Kernel
from .analysis import baseline_policies, evaluate, policy_from_q


def main():
    parser = argparse.ArgumentParser(description="Evaluate or demonstrate the saved PrivacyGuard RL agent")
    parser.add_argument("command", choices=["demo", "evaluate", "explain"])
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--state", type=int, nargs=6, default=[6, 6, 0, 2, 1, 1], metavar=("H", "B", "F", "U", "S", "P"))
    parser.add_argument("--agent", type=Path, default=Path(__file__).resolve().parents[1] / "agents/q_learning_seed0.npz")
    args = parser.parse_args()
    with np.load(args.agent, allow_pickle=False) as checkpoint:
        q, visits = checkpoint["q"], checkpoint["visits"]
        config = Config(**json.loads(str(checkpoint["config"])))
    k = Kernel(config)
    policy = policy_from_q(k, q)
    if args.command == "explain":
        if any(value < 0 or value >= limit for value, limit in zip(args.state, k.shape)) or args.state[0] == 0:
            parser.error("State outside nonterminal observation bounds")
        state = k.encode(args.state)
        print("State (remaining, budget, fatigue, utility bin, sensitivity bin, full permission):", args.state)
        print("Chosen action:", ACTIONS[policy[state]])
        for a, name in enumerate(ACTIONS):
            print(f"  {name:12s}: Q={q[state,a]:7.3f}, visits={visits[state,a]:5d}" if k.mask[state,a] else f"  {name:12s}: BLOCKED by permission/budget guard")
        print("Q is the learned expected session return; it is not a probability or a causal explanation.")
        return
    tape = k.tape(1 if args.command == "demo" else args.episodes, args.seed)
    result = evaluate(k, policy, *tape, details=args.command == "demo")
    if args.command == "demo":
        for row in result["trace"]:
            print(f"{row['step']:2d} state={row['state']}  {ACTIONS[row['action']]:12s}  reward={row['reward']:6.2f}  cumulative={row['total_return']:6.2f}")
    else:
        for label, pol in {"Saved Q-learning": policy, **baseline_policies(k)}.items():
            metrics = evaluate(k, pol, *tape)
            print(f"{label:20s} return={metrics['return'].mean():7.3f} utility={metrics['utility'].mean():7.3f} exposure={metrics['exposure'].mean():5.2f} prompts={metrics['prompts'].mean():4.2f}")
    print("Guard check passed: exposure never exceeded budget and no unapproved full action occurred.")


if __name__ == "__main__":
    main()
