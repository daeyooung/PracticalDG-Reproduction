#!/usr/bin/env python
# summarize_val_vs_test.py
# Compare H-score from eval_h_score_val.txt and eval_h_score.txt
# for each office-home_class_space{0,1,2,3},
# restricted to SCIPD / test_envs_0 / seed_0.

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_hscore_file(path):
    """Extract H-score from file (line with 'max_h-score:')"""
    if not os.path.isfile(path):
        return None
    hscore = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "max_h-score:" in line:
                try:
                    hscore = float(line.split("max_h-score:")[1])
                except ValueError:
                    pass
                break
    return hscore


def parse_class_space(root):
    """Parse class_space id from directory path"""
    m = re.search(r"class_space(\d+)", root)
    if not m:
        return None
    return int(m.group(1))


def collect_scores(base_dir):
    """
    Walk base_dir and find eval_h_score_val.txt and eval_h_score.txt
    under each class_space directory,
    but only inside SCIPD/test_envs_0/seed_0.

    Returns:
      val_scores[class_space]  -> list of H-scores
      test_scores[class_space] -> list of H-scores
    """
    val_scores = defaultdict(list)
    test_scores = defaultdict(list)

    for root, dirs, files in os.walk(base_dir):
        # only use SCIPD / test_envs_0 / seed_0
        if "SCIPD" not in root:
            continue
        if "test_envs_0" not in root or "seed_0" not in root:
            continue

        if "eval_h_score_val.txt" not in files and "eval_h_score.txt" not in files:
            continue

        cs = parse_class_space(root)
        if cs is None:
            continue

        val_path = os.path.join(root, "eval_h_score_val.txt")
        test_path = os.path.join(root, "eval_h_score.txt")

        if os.path.isfile(val_path):
            hv = parse_hscore_file(val_path)
            if hv is not None:
                val_scores[cs].append(hv)
                print(f"[INFO] class_space {cs} val from {val_path}: {hv:.4f}")

        if os.path.isfile(test_path):
            ht = parse_hscore_file(test_path)
            if ht is not None:
                test_scores[cs].append(ht)
                print(f"[INFO] class_space {cs} test from {test_path}: {ht:.4f}")

    return val_scores, test_scores


def compute_means(score_dict):
    """Compute mean and count for each class_space"""
    means = {}
    for cs, vals in score_dict.items():
        arr = np.array(vals, dtype=float)
        means[cs] = {
            "mean": float(arr.mean()),
            "n": int(arr.size),
        }
    return means


def plot_bar_compare(val_means, test_means, save_path=None, show=True):
    """
    Draw bar plot:
      x-axis: H in {0, 1/6, 1/3, 1} as categorical positions
      bars: val H-score vs test H-score
    """
    class_spaces = sorted(set(val_means.keys()) | set(test_means.keys()))
    if not class_spaces:
        print("[ERROR] no class_space found.")
        return

    val_list = []
    test_list = []
    for cs in class_spaces:
        v = val_means.get(cs, {"mean": np.nan})["mean"]
        t = test_means.get(cs, {"mean": np.nan})["mean"]
        val_list.append(v)
        test_list.append(t)

    # positions: 0,1,2,3 ...
    x = np.arange(len(class_spaces))
    width = 0.35

    # H labels corresponding to class_space 0,1,2,3,...
    H_LABEL_MAP = {
        0: "0.000",
        1: "0.167",   # 1/6
        2: "0.333",   # 1/3
        3: "1.000",
    }
    xlabels = [H_LABEL_MAP.get(cs, str(cs)) for cs in class_spaces]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_val = ax.bar(x - width / 2, val_list, width, label="Val H-score")
    bars_test = ax.bar(x + width / 2, test_list, width, label="Test H-score")

    for bars in [bars_val, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height,                              
                f"{height:.3f}",                     
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Hybridness")
    ax.set_ylabel("H-score")
    ax.set_title("Val vs Test H-score per H")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

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
        required=True,
        help=(
            "Root directory that contains office-home_class_space* folders. "
            "Example: ./output/ablation/daml_loader/lr_1e-3_dataaug_yes/office-home"
        ),
    )
    parser.add_argument(
        "--fig_path",
        type=str,
        default="./output/val_vs_test_hscore_SCIPD_env0_seed0.png",
        help="Path to save comparison figure",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable plt.show()",
    )
    args = parser.parse_args()

    val_scores, test_scores = collect_scores(args.base_dir)
    val_means = compute_means(val_scores)
    test_means = compute_means(test_scores)

    print("\n=== Summary (H-score per class_space, SCIPD, test_envs_0/seed_0) ===")
    for cs in sorted(set(val_means.keys()) | set(test_means.keys())):
        v = val_means.get(cs, {"mean": np.nan, "n": 0})
        t = test_means.get(cs, {"mean": np.nan, "n": 0})
        print(
            f"class_space {cs}: "
            f"val_mean={v['mean']:.4f} (n={v['n']}), "
            f"test_mean={t['mean']:.4f} (n={t['n']})"
        )

    plot_bar_compare(val_means, test_means,
                     save_path=args.fig_path,
                     show=not args.no_show)


if __name__ == "__main__":
    main()