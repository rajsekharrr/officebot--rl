# evaluation/compare.py
"""
Aggregate eval_results.json across all available algos/seeds.

Saves:
    results/comparison_table.csv
    results/experiment_summary.json
"""

import csv
import glob
import json
import os

import numpy as np
from tabulate import tabulate

METRIC_KEYS = [
    "mean_episode_reward",
    "std_episode_reward",
    "success_rate",
    "task_completion_rate",
    "mean_episode_length",
    "mean_collision_count",
    "collision_rate",
    "mean_battery_remaining",
    "battery_efficiency",
    "mean_completed_tasks",
    "mean_failed_tasks",
]

METRIC_LABELS = {
    "mean_episode_reward":    "Mean Episode Reward",
    "std_episode_reward":     "Reward Std Dev",
    "success_rate":           "Success Rate",
    "task_completion_rate":   "Task Completion Rate",
    "mean_episode_length":    "Mean Episode Length",
    "mean_collision_count":   "Mean Collisions",
    "collision_rate":         "Collision Rate",
    "mean_battery_remaining": "Mean Battery Remaining",
    "battery_efficiency":     "Battery Efficiency",
    "mean_completed_tasks":   "Mean Completed Tasks",
    "mean_failed_tasks":      "Mean Failed Tasks",
}


def discover_results():
    """Scan logs/metrics/*/*/eval_results.json -> {algo: [metrics, ...]}."""
    found = {}
    pattern = os.path.join("logs", "metrics", "*", "seed*", "eval_results.json")
    for path in sorted(glob.glob(pattern)):
        algo = path.replace("\\", "/").split("/")[2]
        with open(path) as f:
            found.setdefault(algo, []).append(json.load(f))
    return found


def aggregate(found):
    """Compute mean +/- std across seeds for every metric and algo."""
    result = {}
    for algo, records in found.items():
        result[algo] = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in records if key in r]
            if vals:
                result[algo][key] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }
    return result


def save_csv(aggregated, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    algos = sorted(aggregated)
    cols  = ["Metric"] + [a.upper() for a in algos]
    rows  = []
    for key in METRIC_KEYS:
        row = {"Metric": METRIC_LABELS.get(key, key)}
        for algo in algos:
            s = aggregated[algo].get(key, )
            row[algo.upper()] = (f"{s.get('mean', float('nan')):.4f}"
                                 f" +/- {s.get('std', float('nan')):.4f}")
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {path}")


def save_json(aggregated, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"algorithms": sorted(aggregated), "metrics": aggregated}, f, indent=2)
    print(f"Saved -> {path}")


def main():
    print("\n" + "=" * 70)
    print("OfficeBot RL -- Algorithm Comparison")
    print("=" * 70 + "\n")

    found = discover_results()
    if not found:
        print("No eval_results.json files found. Run evaluate.py first.")
        return 1

    for algo, recs in sorted(found.items()):
        seeds = ", ".join(f"seed{r['seed']}" for r in recs)
        print(f"  {algo.upper()}: {len(recs)} seed(s) -> {seeds}")

    aggregated = aggregate(found)
    algos   = sorted(aggregated)
    headers = ["Metric"] + [a.upper() for a in algos]
    table   = []
    for key in METRIC_KEYS:
        row = [METRIC_LABELS.get(key, key)]
        for algo in algos:
            s = aggregated[algo].get(key, {})
            row.append(f"{s.get('mean', float('nan')):.4f}"
                       f" +/- {s.get('std', float('nan')):.4f}")
        table.append(row)

    print("\n" + tabulate(table, headers=headers, tablefmt="grid") + "\n")
    save_csv(aggregated, "results/comparison_table.csv")
    save_json(aggregated, "results/experiment_summary.json")
    print("\nComparison complete.\n")
    return 0


if __name__ == "__main__":
    exit(main())
