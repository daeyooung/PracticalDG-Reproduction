#!/usr/bin/env python
# summarize_hyper_by_classspace.py
# For each hyper-parameter value (alpha or beta),
# collect target acc and H-score from all class_space folders,
# average first within each class_space and then over class_spaces,
# and plot line charts like the paper figure.

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

def collect_values_for_hyper(hyper_dir):
    """
    Walk one hyper directory (e.g. alpha/0.1) and collect all
    (acc, h_score) pairs from done.txt and eval_h_score.txt.
    No class_space averaging, just flat list.
    """
    values = []  # list of (acc, hscore)

    for root, dirs, files in os.walk(hyper_dir):
        if "done.txt" not in files:
            continue

        done_path = os.path.join(root, "done.txt")
        eval_hscore_path = os.path.join(root, "eval", "eval_h_score.txt")

        acc = parse_done_txt(done_path)
        hscore = parse_eval_hscore_txt(eval_hscore_path)

        if acc is None or hscore is None:
            continue

        values.append((acc, hscore))

    return values

def parse_eval_hscore_txt(path):
    """Extract H-score from eval_h_score.txt (logged as max_h-score)"""
    h_score = None
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "max_h-score:" in line:
                try:
                    h_score = float(line.split("max_h-score:")[1])
                except ValueError:
                    pass
                break
    return h_score


def parse_class_space(root):
    """Parse class_space id from directory path"""
    m = re.search(r"class_space(\d+)", root)
    if not m:
        return None
    return int(m.group(1))


def collect_scores_for_hyper(hyper_dir):
    """
    Collect (acc, h_score) values for one hyper value folder.

    Returns:
      scores_per_cs[class_space] -> list of (acc, h_score)
    """
    scores_per_cs = defaultdict(list)

    for root, dirs, files in os.walk(hyper_dir):
        if "done.txt" not in files:
            continue

        cs = parse_class_space(root)
        if cs is None:
            continue

        done_path = os.path.join(root, "done.txt")
        eval_hscore_path = os.path.join(root, "eval", "eval_h_score.txt")

        acc = parse_done_txt(done_path)
        hscore = parse_eval_hscore_txt(eval_hscore_path)

        if acc is None or hscore is None:
            continue

        scores_per_cs[cs].append((acc, hscore))

    return scores_per_cs


def aggregate_over_classspaces(scores_per_cs):
    """
    First average within each class_space, then average over class_spaces.

    Returns:
      acc_mean_over_cs, hscore_mean_over_cs
    """
    if not scores_per_cs:
        return None, None

    acc_means_cs = []
    hscore_means_cs = []

    for cs, vals in scores_per_cs.items():
        arr = np.array(vals, dtype=float)  # (N, 2)
        acc_mean_cs = float(arr[:, 0].mean())
        hscore_mean_cs = float(arr[:, 1].mean())
        acc_means_cs.append(acc_mean_cs)
        hscore_means_cs.append(hscore_mean_cs)

    acc_mean = float(np.mean(acc_means_cs))
    hscore_mean = float(np.mean(hscore_means_cs))
    return acc_mean, hscore_mean


def save_csv(hyper_vals, acc_means, hscore_means, csv_path, hyper_name):
    """Save aggregated results to CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(f"{hyper_name},acc_mean,hscore_mean\n")
        for h, a, hs in zip(hyper_vals, acc_means, hscore_means):
            f.write(f"{h},{a:.6f},{hs:.6f}\n")
    print(f"[INFO] saved CSV to: {csv_path}")


def plot_line(labels, y, xlabel, ylabel, title, save_path=None, show=True):
    """Draw simple line plot with labels on x-axis"""
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

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
        "--root_dir",
        type=str,
        required=True,
        help=(
            "Directory that contains hyper-value folders "
            "(e.g. ./output/ablation/alpha or ./output/ablation/beta)"
        ),
    )
    parser.add_argument(
        "--hyper_name",
        type=str,
        default="alpha",
        help="Name of hyper-parameter (e.g. alpha or beta)",
    )
    parser.add_argument(
        "--acc_fig_path",
        type=str,
        default="./output/hyper_acc_by_classspace.png",
        help="Path to save Acc line plot",
    )
    parser.add_argument(
        "--hscore_fig_path",
        type=str,
        default="./output/hyper_hscore_by_classspace.png",
        help="Path to save H-score line plot",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./output/hyper_summary_by_classspace.csv",
        help="Path to save CSV summary",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable plt.show()",
    )
    args = parser.parse_args()

    hyper_vals = []      # float for sorting
    hyper_names = []     # string for x-axis labels
    acc_means = []
    hscore_means = []

    for name in os.listdir(args.root_dir):
        subdir = os.path.join(args.root_dir, name)
        if not os.path.isdir(subdir):
            continue

        try:
            h_val = float(name)   # for sorting
        except ValueError:
            print(f"[WARN] skip non-numeric folder: {subdir}")
            continue

        vals = collect_values_for_hyper(subdir)
        if not vals:
            print(f"[WARN] no valid (acc, H-score) in: {subdir}")
            continue

        arr = np.array(vals, dtype=float)
        acc_mean = float(arr[:, 0].mean())
        hscore_mean = float(arr[:, 1].mean())

        hyper_vals.append(h_val)
        hyper_names.append(name)      # <- use folder name as label
        acc_means.append(acc_mean)
        hscore_means.append(hscore_mean)

        print(
            f"[INFO] {args.hyper_name}={name}: "
            f"acc_mean={acc_mean:.4f}, H-score_mean={hscore_mean:.4f}, "
            f"n={len(vals)}"
        )

    if not hyper_vals:
        print("[ERROR] no valid hyper folders found.")
        return

    idx = np.argsort(hyper_vals)
    hyper_vals = np.array(hyper_vals)[idx]
    hyper_names = np.array(hyper_names, dtype=str)[idx]   # <- sort names
    acc_means = np.array(acc_means)[idx]
    hscore_means = np.array(hscore_means)[idx]

    if args.csv_path:
        save_csv(hyper_vals, acc_means, hscore_means,
                 args.csv_path, args.hyper_name)

    # Acc vs alpha
    plot_line(
        hyper_names,
        acc_means,
        xlabel=args.hyper_name,
        ylabel="Acc (mean)",
        title=f"{args.hyper_name} vs Acc",
        save_path=args.acc_fig_path,
        show=not args.no_show,
    )

    # H-score vs alpha
    plot_line(
        hyper_names,
        hscore_means,
        xlabel=args.hyper_name,
        ylabel="H-score (mean)",
        title=f"{args.hyper_name} vs H-score",
        save_path=args.hscore_fig_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()