#!/usr/bin/env python
# summarize_loss.py
# For each loss subfolder, collect target acc and H-score
# from done.txt and eval/eval_h_score.txt, then save mean values to CSV.

import os
import argparse
import numpy as np


def parse_done(path):
    """Read target acc from done.txt or done.text"""
    if not os.path.isfile(path):
        return None
    acc = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "target acc:" in line:
                try:
                    acc = float(line.split("target acc:")[1])
                except ValueError:
                    pass
                break
    return acc


def parse_hscore(path):
    """Read H-score from eval_h_score.txt (line with max_h-score)"""
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


def collect_for_one_loss(loss_dir):
    """
    Walk one loss directory and collect all (acc, hscore) pairs.
    """
    values = []  # list of (acc, hscore)

    for root, dirs, files in os.walk(loss_dir):
        # support both done.txt and done.text
        done_file = None
        if "done.txt" in files:
            done_file = "done.txt"
        elif "done.text" in files:
            done_file = "done.text"
        else:
            continue

        done_path = os.path.join(root, done_file)
        eval_path = os.path.join(root, "eval", "eval_h_score.txt")

        acc = parse_done(done_path)
        hscore = parse_hscore(eval_path)

        if acc is None or hscore is None:
            continue

        values.append((acc, hscore))

    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Directory that contains loss folders (e.g. ./output/ablation/loss)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./output/loss_summary.csv",
        help="Path to save CSV file",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    rows = []  # (loss_name, n, acc_mean, hscore_mean)

    for name in os.listdir(args.root_dir):
        loss_dir = os.path.join(args.root_dir, name)
        if not os.path.isdir(loss_dir):
            continue

        values = collect_for_one_loss(loss_dir)
        if not values:
            print(f"[WARN] no valid results in: {loss_dir}")
            continue

        arr = np.array(values, dtype=float)  # (N, 2)
        acc_mean = float(arr[:, 0].mean())
        hscore_mean = float(arr[:, 1].mean())
        n = int(arr.shape[0])

        rows.append((name, n, acc_mean, hscore_mean))
        print(
            f"[INFO] loss={name}: n={n}, "
            f"acc_mean={acc_mean:.4f}, hscore_mean={hscore_mean:.4f}"
        )

    if not rows:
        print("[ERROR] no loss folders found with valid results.")
        return

    # write CSV
    with open(args.csv_path, "w") as f:
        f.write("loss_name,n_runs,acc_mean,hscore_mean\n")
        for loss_name, n, acc_mean, hscore_mean in rows:
            f.write(f"{loss_name},{n},{acc_mean:.6f},{hscore_mean:.6f}\n")

    print(f"[INFO] saved CSV to: {args.csv_path}")


if __name__ == "__main__":
    main()