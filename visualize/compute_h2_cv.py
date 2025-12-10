#!/usr/bin/env python
# summarize_hscore_cv_4methods.py
# For CLIPBase, IP, SCI, SI:
# 1) group H-score by class_space
# 2) compute mean H-score per class_space
# 3) compute std across these means
# 4) compute cv = std / overall_mean

import os
import re
import argparse
import numpy as np

import os

def detect_method(root, methods):
    """Detect method name from path by exact folder name match."""
    parts = os.path.normpath(root).split(os.sep)
    for m in methods:
        if m in parts:      # folder name exactly equals method
            return m
    return None

def parse_hscore(path):
    """Read H-score from eval_h_score.txt (line with max_h-score)."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "max_h-score:" in line:
                try:
                    return float(line.split("max_h-score:")[1])
                except ValueError:
                    return None
    return None


def extract_classspace(path):
    """Extract class_space id from folder path."""
    m = re.search(r"class_space(\d+)", path)
    if not m:
        return None
    return int(m.group(1))


def collect_hscores_by_classspace(root_dir, methods):
    """
    Collect H-scores grouped by (method, class_space).

    Returns:
      hscores[method][class_space] = list of H-scores
    """
    hscores = {m: {} for m in methods}

    for root, dirs, files in os.walk(root_dir):
        if "eval_h_score.txt" not in files:
            continue

        # use exact folder name detection
        method = detect_method(root, methods)
        if method is None:
            continue

        cs = extract_classspace(root)
        if cs is None:
            continue

        eval_path = os.path.join(root, "eval_h_score.txt")
        h = parse_hscore(eval_path)
        if h is None:
            continue

        if cs not in hscores[method]:
            hscores[method][cs] = []
        hscores[method][cs].append(h)

    return hscores



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory that contains class_space folders.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="./hscore_cv_summary_4methods.csv",
        help="Path to save summary CSV (per method).",
    )
    parser.add_argument(
        "--detail_csv",
        type=str,
        default="./hscore_per_classspace_4methods.csv",
        help="Path to save per-class_space mean H-score CSV.",
    )
    args = parser.parse_args()

    methods = ["CLIPBase", "IP", "SCI", "SI"]

    hscores = collect_hscores_by_classspace(args.root_dir, methods)

    # save per-class_space mean for each method
    os.makedirs(os.path.dirname(args.detail_csv), exist_ok=True)
    with open(args.detail_csv, "w") as f_det:
        f_det.write("method,class_space,n_samples,mean_hscore\n")
        for method in methods:
            cs_dict = hscores.get(method, {})
            for cs in sorted(cs_dict.keys()):
                vals = np.array(cs_dict[cs], dtype=float)
                mean_cs = float(vals.mean())
                f_det.write(f"{method},{cs},{vals.size},{mean_cs:.6f}\n")

    print(f"[INFO] saved per-class_space means to: {args.detail_csv}")

    # compute std across class_space means and cv = std / overall_mean
    rows = []
    for method in methods:
        cs_dict = hscores.get(method, {})
        if not cs_dict:
            print(f"[WARN] no H-scores for {method}")
            continue

        classspace_means = []
        for cs, vals in cs_dict.items():
            arr = np.array(vals, dtype=float)
            classspace_means.append(float(arr.mean()))

        classspace_means = np.array(classspace_means, dtype=float)
        overall_mean = float(classspace_means.mean())
        std_across_cs = float(classspace_means.std(ddof=0))
        cv = std_across_cs / overall_mean if overall_mean != 0 else float("nan")

        rows.append((method,
                     len(classspace_means),
                     overall_mean,
                     std_across_cs,
                     cv))

        print(f"[INFO] {method}")
        print(f"  num_class_spaces : {len(classspace_means)}")
        print(f"  overall_mean     : {overall_mean:.6f}")
        print(f"  std_across_means : {std_across_cs:.6f}")
        print(f"  cv(std/mean)     : {cv:.6f}")
        print()

    # save summary CSV per method
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w") as f_sum:
        f_sum.write("method,num_class_spaces,overall_mean,std_across_means,cv\n")
        for method, n_cs, mean_all, std_cs, cv in rows:
            f_sum.write(
                f"{method},{n_cs},{mean_all:.6f},{std_cs:.6f},{cv:.6f}\n"
            )

    print(f"[INFO] saved summary stats to: {args.summary_csv}")


if __name__ == "__main__":
    main()


