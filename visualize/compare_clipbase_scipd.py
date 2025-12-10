#!/usr/bin/env python
# compare_clipbase_scipd.py
# Compare CLIPBase vs SCIPD across class_space and domains.

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_done(path):
    """Read target acc from done.txt"""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            if "target acc:" in line:
                try:
                    return float(line.split("target acc:")[1])
                except:
                    return None
    return None


def parse_hscore(path):
    """Read max_h-score from eval_h_score.txt"""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        for line in f:
            if "max_h-score:" in line:
                try:
                    return float(line.split("max_h-score:")[1])
                except:
                    return None
    return None


def extract_classspace(path):
    """Extract class_space ID from folder name"""
    m = re.search(r"class_space(\d+)", path)
    return int(m.group(1)) if m else None


def collect_scores(base_dir):
    """
    Collect all results under:
    class_spaceX / {CLIPBase, SCIPD} / test_envs_Y / seed_Z / eval/*.txt
    """
    results = {}  # results[class_space][method][domain] = list of (acc, hscore)

    for root, dirs, files in os.walk(base_dir):
        if "done.txt" not in files:
            continue

        cs = extract_classspace(root)
        if cs is None:
            continue

        method = None
        if "CLIPBase" in root:
            method = "CLIPBase"
        elif "SCIPD" in root:
            method = "SCIPD"
        else:
            continue  # ignore other folders

        # test_envs number
        m = re.search(r"test_envs_(\d+)", root)
        if not m:
            continue
        domain = int(m.group(1))

        done_path = os.path.join(root, "done.txt")
        eval_path = os.path.join(root, "eval", "eval_h_score.txt")

        acc = parse_done(done_path)
        hscore = parse_hscore(eval_path)
        if acc is None or hscore is None:
            continue

        results.setdefault(cs, {})
        results[cs].setdefault(method, {})
        results[cs][method].setdefault(domain, [])
        results[cs][method][domain].append((acc, hscore))

    return results


def compute_means(results):
    """
    Convert raw lists to mean tables:
    mean_table[domain][method][class_space] = (acc_mean, hmean)
    """
    mean_table = {}

    for cs in results:
        for method in results[cs]:
            for domain in results[cs][method]:
                arr = np.array(results[cs][method][domain], float)
                acc_mean = float(arr[:, 0].mean())
                h_mean = float(arr[:, 1].mean())

                mean_table.setdefault(domain, {})
                mean_table[domain].setdefault(method, {})
                mean_table[domain][method][cs] = (acc_mean, h_mean)

    return mean_table


def plot_domain(domain, table, save_path):
    """Draw line plot for CLIPBase vs SCIPD for this domain (H as 4 equal bins)."""

    if "CLIPBase" not in table or "SCIPD" not in table:
        print(f"[WARN] skip domain {domain} (missing methods)")
        return

    # only class_spaces existing in both
    cs_clip = set(table["CLIPBase"].keys())
    cs_scipd = set(table["SCIPD"].keys())
    cs_values = sorted(cs_clip & cs_scipd)

    if not cs_values:
        print(f"[WARN] domain {domain}: no common class_space found")
        return

    # x positions: 0,1,2,3,...  (equally spaced)
    x = np.arange(len(cs_values))

    # y-values
    clip_h = [table["CLIPBase"][cs][1] for cs in cs_values]
    scipd_h = [table["SCIPD"][cs][1] for cs in cs_values]

    # labels for H on x-axis
    H_LABEL_MAP = {0: "0", 1: "1/6", 2: "1/3", 3: "1"}
    xlabels = [H_LABEL_MAP.get(cs, str(cs)) for cs in cs_values]

    plt.figure()
    plt.plot(x, clip_h, marker="o", label="CLIPBase")
    plt.plot(x, scipd_h, marker="o", label="SCIPD")

    plt.xticks(x, xlabels) 
    plt.xlabel("Hybridness")
    plt.ylabel("H-score")
    plt.title(f"Domain test_envs_{domain}")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved plot: {save_path}")
    plt.close()




def save_csv(mean_table, save_path):
    """Save CSV summary comparing CLIPBase vs SCIPD by classspace"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("class_space,CLIPBase_acc,SCIPD_acc,CLIPBase_h,SCIPD_h\n")

        # collect unique classspaces
        cs_list = set()
        for dom in mean_table:
            if "CLIPBase" in mean_table[dom]:
                cs_list |= mean_table[dom]["CLIPBase"].keys()
            if "SCIPD" in mean_table[dom]:
                cs_list |= mean_table[dom]["SCIPD"].keys()

        for cs in sorted(cs_list):
            # average across domains
            clip_acc, clip_h, n1 = 0, 0, 0
            scipd_acc, scipd_h, n2 = 0, 0, 0

            for dom in mean_table:
                if "CLIPBase" in mean_table[dom] and cs in mean_table[dom]["CLIPBase"]:
                    a, h = mean_table[dom]["CLIPBase"][cs]
                    clip_acc += a; clip_h += h; n1 += 1

                if "SCIPD" in mean_table[dom] and cs in mean_table[dom]["SCIPD"]:
                    a, h = mean_table[dom]["SCIPD"][cs]
                    scipd_acc += a; scipd_h += h; n2 += 1

            if n1 > 0: 
                clip_acc /= n1; clip_h /= n1
            if n2 > 0:
                scipd_acc /= n2; scipd_h /= n2

            f.write(f"{cs},{clip_acc:.4f},{scipd_acc:.4f},{clip_h:.4f},{scipd_h:.4f}\n")

    print(f"[INFO] Saved CSV: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--plot_dir", default="./plots")
    parser.add_argument("--csv_path", default="./compare_clip_scipd.csv")
    args = parser.parse_args()

    results = collect_scores(args.root_dir)
    mean_table = compute_means(results)

    # Draw domain plots
    for domain in sorted(mean_table.keys()):
        plot_domain(
            domain,
            mean_table[domain],
            save_path=f"{args.plot_dir}/domain_{domain}.png"
        )

    # Save CSV (averaged across domains)
    save_csv(mean_table, args.csv_path)


if __name__ == "__main__":
    main()