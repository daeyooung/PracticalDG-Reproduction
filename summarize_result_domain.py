#!/usr/bin/env python
# summarize_results.py
# Collect target acc from done.txt and max h-score from eval_h_score.txt
# Aggregate by class_space and test_env, then visualize and save results.

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


def parse_class_space(root):
    """Parse class_space id from directory path"""
    m = re.search(r"class_space(\d+)", root)
    if not m:
        return None
    return int(m.group(1))


def parse_test_env(root):
    """Parse test_env label from directory path"""
    # Example patterns: .../test_env3/..., .../test_envs3/..., .../test_envs0_1/...
    m = re.search(r"test_envs?([0-9_]+)", root)
    if not m:
        return "unknown"
    return m.group(1)  # keep as string (e.g. "0", "3", "0_1")


def collect_results(base_dir):
    """
    Walk base_dir and find:
      - */class_spaceX/*/done.txt
      - */class_spaceX/*/eval/eval_h_score.txt
    Group results by (class_space, test_env).
    """
    # records[class_space][test_env] -> list of (target_acc, max_h)
    records = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(base_dir):
        if "done.txt" not in files:
            continue

        class_space = parse_class_space(root)
        if class_space is None:
            continue

        test_env = parse_test_env(root)

        done_path = os.path.join(root, "done.txt")
        eval_hscore_path = os.path.join(root, "eval", "eval_h_score.txt")

        target_acc = parse_done_txt(done_path)
        max_h = parse_eval_hscore_txt(eval_hscore_path)

        if target_acc is None or max_h is None:
            print(f"[WARN] skipped: class_space={class_space}, "
                  f"test_env={test_env}, root={root}")
            continue

        records[class_space][test_env].append((target_acc, max_h))
        print(f"[INFO] class_space={class_space}, test_env={test_env}, "
              f"target_acc={target_acc:.4f}, max_h={max_h:.4f}")

    return records


def compute_overall_means(records):
    """
    Compute mean over all test_envs for each class_space.
    Returns: {class_space: {'target_mean', 'hscore_mean', 'n'}}
    """
    overall = {}
    for cs, env_dict in records.items():
        all_vals = []
        for vals in env_dict.values():
            all_vals.extend(vals)
        if not all_vals:
            continue
        arr = np.array(all_vals)
        target_mean = float(arr[:, 0].mean())
        hscore_mean = float(arr[:, 1].mean())
        overall[cs] = {
            "target_mean": target_mean,
            "hscore_mean": hscore_mean,
            "n": len(all_vals),
        }
    return overall


def compute_means_by_env(records):
    """
    Compute mean per (class_space, test_env).
    Returns: {class_space: {test_env: {'target_mean', 'hscore_mean', 'n'}}}
    """
    means_env = {}
    for cs, env_dict in records.items():
        means_env[cs] = {}
        for env, vals in env_dict.items():
            if not vals:
                continue
            arr = np.array(vals)
            target_mean = float(arr[:, 0].mean())
            hscore_mean = float(arr[:, 1].mean())
            means_env[cs][env] = {
                "target_mean": target_mean,
                "hscore_mean": hscore_mean,
                "n": len(vals),
            }
    return means_env


def save_csv(means_env, csv_path):
    """
    Save aggregated results to CSV:
      class_space, test_env, n_runs, target_mean, hscore_mean
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("class_space,test_env,n_runs,target_mean,hscore_mean\n")
        for cs in sorted(means_env.keys()):
            for env in sorted(means_env[cs].keys(), key=str):
                info = means_env[cs][env]
                f.write(
                    f"{cs},{env},{info['n']},"
                    f"{info['target_mean']:.6f},{info['hscore_mean']:.6f}\n"
                )
    print(f"[INFO] saved CSV to: {csv_path}")


def plot_bar_overall(means_overall, save_path=None, show=True):
    """
    Draw bar plot for overall mean target acc and mean h-score per class_space.
    """
    if not means_overall:
        print("[ERROR] no results for overall bar plot.")
        return

    class_spaces = sorted(means_overall.keys())
    target_means = [means_overall[cs]["target_mean"] for cs in class_spaces]
    hscore_means = [means_overall[cs]["hscore_mean"] for cs in class_spaces]

    x = np.arange(len(class_spaces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, target_means, width, label="Target Acc")
    ax.bar(x + width / 2, hscore_means, width, label="Max H-score")

    ax.set_xticks(x)
    ax.set_xticklabels([str(cs) for cs in class_spaces])
    ax.set_xlabel("Class Space")
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance per Class Space")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] saved bar figure to: {save_path}")
    if show:
        plt.show()


def plot_line_by_test_env(means_env, metric="target_mean",
                          save_path=None, show=True):
    """
    Draw line plot per test_env.
    X-axis: class_space
    Y-axis: metric (target_mean or hscore_mean)
    Color: test_env
    """
    # Collect set of all test_env labels
    all_envs = set()
    for cs in means_env.keys():
        all_envs.update(means_env[cs].keys())
    all_envs = sorted(all_envs, key=str)

    if not all_envs:
        print("[ERROR] no results for line plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for env in all_envs:
        xs = []
        ys = []
        for cs in sorted(means_env.keys()):
            if env not in means_env[cs]:
                continue
            xs.append(cs)
            ys.append(means_env[cs][env][metric])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=f"test_env {env}")

    ylabel = "Target Acc (mean)" if metric == "target_mean" else "Max H-score (mean)"
    title = f"{ylabel} per Class Space (by test_env)"

    ax.set_xlabel("Class Space")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] saved line figure to: {save_path}")
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
        "--bar_path",
        type=str,
        default="./output/ablation/hyper/class_space_bar.png",
        help="Path to save bar plot",
    )
    parser.add_argument(
        "--line_target_path",
        type=str,
        default="./output/ablation/hyper/class_space_line_target.png",
        help="Path to save line plot for target acc",
    )
    parser.add_argument(
        "--line_hscore_path",
        type=str,
        default="./output/ablation/hyper/class_space_line_hscore.png",
        help="Path to save line plot for max h-score",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./output/ablation/hyper/class_space_test_env_summary.csv",
        help="Path to save aggregated CSV",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable plt.show()",
    )
    args = parser.parse_args()

    records = collect_results(args.base_dir)
    means_overall = compute_overall_means(records)
    means_env = compute_means_by_env(records)

    print("\n=== Overall Summary per Class Space (all test_envs) ===")
    for cs in sorted(means_overall.keys()):
        m = means_overall[cs]
        print(
            f"class_space {cs}: "
            f"target_acc_mean={m['target_mean']:.4f}, "
            f"hscore_mean={m['hscore_mean']:.4f}, "
            f"n_runs={m['n']}"
        )

    print("\n=== Summary per Class Space and test_env ===")
    for cs in sorted(means_env.keys()):
        for env in sorted(means_env[cs].keys(), key=str):
            m = means_env[cs][env]
            print(
                f"class_space {cs}, test_env {env}: "
                f"target_acc_mean={m['target_mean']:.4f}, "
                f"hscore_mean={m['hscore_mean']:.4f}, "
                f"n_runs={m['n']}"
            )

    if args.csv_path:
        save_csv(means_env, args.csv_path)

    bar_path = args.bar_path if args.bar_path else None
    line_target_path = args.line_target_path if args.line_target_path else None
    line_hscore_path = args.line_hscore_path if args.line_hscore_path else None

    plot_bar_overall(means_overall, save_path=bar_path, show=not args.no_show)
    plot_line_by_test_env(
        means_env,
        metric="target_mean",
        save_path=line_target_path,
        show=not args.no_show,
    )
    plot_line_by_test_env(
        means_env,
        metric="hscore_mean",
        save_path=line_hscore_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()