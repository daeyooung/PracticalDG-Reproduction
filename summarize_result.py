#!/usr/bin/env python
# summarize_results.py
# Collect target acc from done.txt and max h-score from eval_h_score.txt
# Compute mean values per class_space and visualize them.

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_done_txt(path):
    """Extract target acc from done.txt"""
    target_acc = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "target acc:" in line:
                try:
                    target_acc = float(line.split("target acc:")[1])
                except ValueError:
                    pass
                break
    return target_acc


def parse_eval_hscore_txt(path):
    """Extract max_h-score from eval_h_score.txt"""
    max_h = None
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "max_h-score:" in line:
                try:
                    max_h = float(line.split("max_h-score:")[1])
                except ValueError:
                    pass
                break
    return max_h


def collect_results(base_dir):
    """
    Scan base_dir and find:
      - */class_spaceX/*/done.txt
      - */class_spaceX/*/eval/eval_h_score.txt
    Group results by class_space.
    """
    records = defaultdict(list)

    for root, dirs, files in os.walk(base_dir):
        if "done.txt" not in files:
            continue

        m = re.search(r"class_space(\d+)", root)
        if not m:
            continue
        class_space = int(m.group(1))

        done_path = os.path.join(root, "done.txt")
        eval_hscore_path = os.path.join(root, "eval", "eval_h_score.txt")

        target_acc = parse_done_txt(done_path)
        max_h = parse_eval_hscore_txt(eval_hscore_path)

        if target_acc is None or max_h is None:
            print(f"[WARN] skipped: class_space={class_space}, root={root}")
            continue

        records[class_space].append((target_acc, max_h))
        print(f"[INFO] class_space={class_space}, "
              f"target_acc={target_acc:.4f}, max_h={max_h:.4f}")

    return records


def compute_means(records):
    """
    Compute mean target acc and mean h-score per class_space.
    """
    means = {}
    for cs, vals in records.items():
        arr = np.array(vals)
        target_mean = float(arr[:, 0].mean())
        hscore_mean = float(arr[:, 1].mean())
        means[cs] = {
            "target_mean": target_mean,
            "hscore_mean": hscore_mean,
            "n": len(vals),
        }
    return means


def plot_results(means, save_path=None, show=True):
    """
    Draw bar plot for mean target acc and mean h-score.
    """
    if not means:
        print("[ERROR] no results to plot.")
        return

    class_spaces = sorted(means.keys())
    target_means = [means[cs]["target_mean"] for cs in class_spaces]
    hscore_means = [means[cs]["hscore_mean"] for cs in class_spaces]

    x = np.arange(len(class_spaces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, target_means, width, label="Target Acc")
    ax.bar(x + width / 2, hscore_means, width, label="Max H-score")

    ax.set_xticks(x)
    ax.set_xticklabels([str(cs) for cs in class_spaces])
    ax.set_xlabel("Class Space")
    ax.set_ylabel("Score")
    ax.set_title("Reproduction Performance per Class Space")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] saved figure to: {save_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output/ablation/hyper",
        help="Directory where run.sh outputs are stored",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./output/ablation/hyper/class_space_summary.png",
        help="Path to save figure",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable plt.show()",
    )
    args = parser.parse_args()

    records = collect_results(args.base_dir)
    means = compute_means(records)

    print("\n=== Summary per Class Space ===")
    for cs in sorted(means.keys()):
        m = means[cs]
        print(
            f"class_space {cs}: "
            f"target_acc_mean={m['target_mean']:.4f}, "
            f"hscore_mean={m['hscore_mean']:.4f}, "
            f"n_runs={m['n']}"
        )

    save_path = args.save_path if args.save_path else None
    plot_results(means, save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()