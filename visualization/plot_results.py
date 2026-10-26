# visualization/plot_results.py
"""
Generate 4 plots for OfficeBot RL results.

Plots saved to results/plots/:
  training_curves_all.png   -- eval reward vs timesteps, one line per algo
  comparison_bar_chart.png  -- final mean reward per algo with error bars
  task_completion_rate.png  -- bar chart per algo
  collision_rate.png        -- bar chart per algo
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "results/plots"
ALGOS      = ["ppo", "dqn", "a2c"]
COLORS     = {"ppo": "#1f77b4", "dqn": "#ff7f0e", "a2c": "#2ca02c"}


def load_training_curves(seeds=(42,)):
    """Load evaluations.npz -> {algo: {timesteps, mean, std, n}}."""
    curves = {}
    for algo in ALGOS:
        per_seed, ts = [], None
        for seed in seeds:
            p = os.path.join("logs", "metrics", algo, f"seed{seed}", "evaluations.npz")
            if not os.path.exists(p):
                continue
            data = np.load(p)
            ts = data["timesteps"]
            per_seed.append(data["results"].mean(axis=1))
        if per_seed:
            arr = np.array(per_seed)
            curves[algo] = {"timesteps": ts, "mean": arr.mean(0),
                            "std": arr.std(0), "n": len(per_seed)}
    return curves


def load_eval_results():
    """Scan logs/metrics/*/*/eval_results.json -> {algo: [metrics,...]}."""
    found = {}
    for path in sorted(glob.glob(
            os.path.join("logs", "metrics", "*", "seed*", "eval_results.json"))):
        algo = path.replace("\\", "/").split("/")[2]
        with open(path) as f:
            found.setdefault(algo, []).append(json.load(f))
    return found


def _bar_data(found, key):
    """Return (labels, means, errs) for a metric key across algos."""
    labels, means, errs = [], [], []
    for algo in sorted(found):
        vals = [m[key] for m in found[algo] if key in m]
        if not vals:
            continue
        labels.append(algo.upper())
        means.append(float(np.mean(vals)))
        if len(vals) > 1:
            errs.append(float(np.std(vals)))
        else:
            # single seed: use within-episode std for reward, else 0
            errs.append(found[algo][0].get("std_episode_reward", 0.0)
                        if key == "mean_episode_reward" else 0.0)
    return labels, means, errs


def plot_training_curves_all(curves):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, d in curves.items():
        c = COLORS.get(algo, "gray")
        ax.plot(d["timesteps"], d["mean"],
                label=f"{algo.upper()} (n={d['n']})", linewidth=2, color=c)
        ax.fill_between(d["timesteps"],
                        d["mean"] - d["std"], d["mean"] + d["std"],
                        alpha=0.2, color=c)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.set_title("Training Curves -- All Algorithms", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "training_curves_all.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def plot_bar(labels, means, errs, title, ylabel, filename, pct=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    x      = np.arange(len(labels))
    colors = [COLORS.get(l.lower(), "#999999") for l in labels]
    ax.bar(x, means, yerr=errs, capsize=7, alpha=0.82, color=colors,
           error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    if pct:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. training_curves_all.png
    print("Loading training curves ...")
    curves = load_training_curves(seeds=[42])
    if curves:
        plot_training_curves_all(curves)
    else:
        print("  No evaluations.npz found; writing placeholder.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No training data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("Training Curves -- All Algorithms")
        plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_all.png"), dpi=150)
        plt.close()

    # Load eval metrics for bar charts
    print("Loading evaluation metrics ...")
    found = load_eval_results()
    if not found:
        print("No eval_results.json found. Run evaluate.py first.")
        return 1

    # 2. comparison_bar_chart.png
    labels, means, errs = _bar_data(found, "mean_episode_reward")
    plot_bar(labels, means, errs,
             "Mean Episode Reward by Algorithm",
             "Mean Episode Reward",
             "comparison_bar_chart.png", pct=False)

    # 3. task_completion_rate.png
    labels, means, errs = _bar_data(found, "task_completion_rate")
    plot_bar(labels, means, errs,
             "Task Completion Rate by Algorithm",
             "Task Completion Rate",
             "task_completion_rate.png", pct=True)

    # 4. collision_rate.png
    labels, means, errs = _bar_data(found, "collision_rate")
    plot_bar(labels, means, errs,
             "Collision Rate by Algorithm",
             "Collisions per Step",
             "collision_rate.png", pct=False)

    print(f"\nAll 4 plots saved to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    exit(main())
