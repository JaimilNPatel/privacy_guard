"""Publication-ready figures, directly from experiment CSV files."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .env import Kernel
from .analysis import policy_from_q

COLORS = {"Q-learning": "#087f8c", "SARSA": "#b36b00", "Myopic Q (gamma=0)": "#9b4d96",
          "Q without budget": "#c44e52", "Q without fatigue": "#7f7f7f",
          "Greedy immediate": "#c44e52", "Always coarse": "#7a9d42", "Exact DP ceiling": "#202e46"}


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_plots(directory):
    from .experiments import ci
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 200})
    summary = {r["policy"]: r for r in read(directory / "summary.csv")}
    curves = read(directory / "learning_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), layout="constrained")
    for label in ("Q-learning", "SARSA", "Myopic Q (gamma=0)"):
        rows = [r for r in curves if r["policy"] == label]
        episodes = sorted({int(r["episode"]) for r in rows})
        data = np.array([[float(r["validation_return"]) for r in rows if int(r["episode"]) == e] for e in episodes])
        avg = data.mean(axis=1)
        err = np.array([ci(row) for row in data])
        axes[0].plot(np.array(episodes) / 1000, avg, label=label.replace(" (gamma=0)", ""), color=COLORS[label], lw=1.5)
        axes[0].fill_between(np.array(episodes) / 1000, avg - err, avg + err, color=COLORS[label], alpha=.13)
    import json
    ceiling = json.loads((directory / "manifest.json").read_text())["exact_optimal_expected_start_return"]
    axes[0].axhline(ceiling, color="#202e46", ls="--", lw=.9, label="Exact DP ceiling")
    axes[0].set(xlabel="Training episodes (thousands)", ylabel="Validation return", title="(a) Sampled learning")
    axes[0].legend(fontsize=6.5, loc="lower right")
    labels = ["Q-learning", "SARSA", "Myopic Q (gamma=0)", "Q without budget", "Q without fatigue", "Greedy immediate", "Always coarse"]
    y = np.arange(len(labels))
    axes[1].barh(y, [float(summary[n]["return"]) for n in labels],
                 xerr=[float(summary[n]["return_ci95"]) for n in labels],
                 color=[COLORS[n] for n in labels], capsize=2)
    axes[1].set_yticks(y, ["Q-learning", "SARSA", "Myopic Q", "No budget obs.", "No fatigue obs.", "Greedy", "Coarse"])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Test session return (95% t CI)", title="(b) Held-out results")
    fig.savefig(figures / "main_results.pdf")
    fig.savefig(figures / "main_results.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.15), layout="constrained")
    model = directory / "models" / "q_seed0.npz"
    k = Kernel()
    with np.load(model, allow_pickle=False) as a:
        policy = policy_from_q(k, a["q"])
    # Fix sensitivity=medium, fatigue=0, standing permission=1, six requests left.
    matrix = np.array([[policy[k.encode((6, b, 0, u, 1, 1))] for b in range(11)] for u in range(3)])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#b4bdc9", "#7ec6bd", "#f3be63", "#668bc3"])
    axes[0].imshow(matrix, vmin=0, vmax=3, cmap=cmap, aspect="auto", origin="lower")
    for u in range(3):
        for b in range(11):
            axes[0].text(b, u, ("D", "C", "A", "F")[matrix[u, b]], ha="center", va="center", fontsize=8)
    axes[0].set(xticks=range(11), yticks=range(3), yticklabels=["Low", "Medium", "High"],
                xlabel="Remaining exposure budget", ylabel="Request utility", title="(c) Seed-0 learned decisions")
    axes[0].grid(False)
    stress = read(directory / "robustness.csv")
    conditions = list(dict.fromkeys(row["condition"] for row in stress))
    x = np.arange(len(conditions))
    for offset, label in ((-.18, "Q-learning"), (.18, "Greedy immediate")):
        vals = [[float(r["return"]) for r in stress if r["condition"] == condition and r["policy"] == label] for condition in conditions]
        axes[1].bar(x + offset, [np.mean(v) for v in vals], width=.34, label=label,
                    color=COLORS[label], yerr=[ci(v) for v in vals], capsize=2)
    axes[1].set(xticks=x, xticklabels=["Nominal", "Sensitive", "High value", "Reluctant", "Cost x2"],
                ylabel="Test session return", title="(d) Frozen policy under stress")
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(figures / "policy_stress.pdf")
    fig.savefig(figures / "policy_stress.png")
    plt.close(fig)
    # More readable standalone charts for the evaluation demo and study notes.
    fig, ax = plt.subplots(figsize=(8.3, 3.6), layout="constrained")
    marker_index = 0
    for label, row in summary.items():
        if label in COLORS:
            ax.errorbar(float(row["exposure"]), float(row["utility"]),
                        xerr=float(row["exposure_ci95"]), yerr=float(row["utility_ci95"]),
                        fmt=["o", "s", "D", "^", "v", "P", "X", "*"][marker_index % 8],
                        color=COLORS[label], capsize=2, label=label.replace(" (gamma=0)", ""))
            marker_index += 1
    ax.set(xlabel="Exposure per session (lower is better)", ylabel="Delivered utility (higher is better)", title="The utility / exposure trade-off")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig.savefig(figures / "utility_exposure.png")
    fig.savefig(figures / "utility_exposure.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results")
    make_plots(parser.parse_args().directory)
