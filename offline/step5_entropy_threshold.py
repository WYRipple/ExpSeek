# step5_entropy_threshold.py
# Input : {eval_dir}/eval_round*.jsonl  (to extract correct trajectory entropy)
#         {exp_kb_dir}/pair-EXP.jsonl   (to extract incorrect trajectory entropy)
# Output: {exp_kb_dir}/entropy_threshold.png
#         {exp_kb_dir}/entropy_threshold.pdf

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression
from scipy.stats import gaussian_kde
from collections import defaultdict


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def detect_rollout_count(eval_dir):
    count = 0
    while os.path.exists(os.path.join(eval_dir, f"eval_round{count + 1}.jsonl")):
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No eval_round*.jsonl found in: {eval_dir}")
    return count


# ── Entropy collection ────────────────────────────────────────────────────────

def extract_entropy_avg_list(messages):
    return [
        msg.get("token_entropy_avg", 0.0)
        for msg in messages
        if msg.get("role") == "assistant"
    ]


def collect_entropy(eval_dir, pair_exp_path):
    """
    Collect entropy values split by:
      - process steps (all steps except last)
      - final steps   (last step only)
    And split by correct / incorrect trajectory.

    Incorrect comes from pair-EXP.jsonl (index 0 = incorrect).
    Correct   comes from eval_round*.jsonl (all-correct questions
              + correct rollouts in mixed questions).
    """
    # ── Incorrect entropy from pair-EXP ──────────────────────────────────────
    process_false, final_false = [], []

    for sample in load_jsonl(pair_exp_path):
        entropy_list = sample["token_entropy_avg_list"][0]   # index 0 = incorrect
        if not entropy_list:
            continue
        step_num = len(entropy_list)
        for i in range(step_num - 1):
            process_false.append(entropy_list[i])
        final_false.append(entropy_list[-1])

    # ── Correct entropy from eval_round*.jsonl ────────────────────────────────
    process_true, final_true = [], []

    total_K = detect_rollout_count(eval_dir)

    # Aggregate by question to find correct rollouts
    merged = defaultdict(lambda: {
        "eval_result_list": [],
        "entropy_lists": [],
    })

    for k in range(1, total_K + 1):
        path = os.path.join(eval_dir, f"eval_round{k}.jsonl")
        if not os.path.exists(path):
            continue
        for item in load_jsonl(path):
            q = item.get("question", "").strip()
            if not q:
                continue
            merged[q]["eval_result_list"].append(item.get("eval_result", "Incorrect"))
            merged[q]["entropy_lists"].append(
                extract_entropy_avg_list(item.get("messages", []))
            )

    for q, data in merged.items():
        for idx, result in enumerate(data["eval_result_list"]):
            if result == "Correct":
                entropy_list = data["entropy_lists"][idx]
                if not entropy_list:
                    continue
                step_num = len(entropy_list)
                for i in range(step_num - 1):
                    process_true.append(entropy_list[i])
                final_true.append(entropy_list[-1])

    print(f"[Step5] Process steps — correct: {len(process_true)}, "
          f"incorrect: {len(process_false)}")
    print(f"[Step5] Final steps   — correct: {len(final_true)}, "
          f"incorrect: {len(final_false)}")

    return (np.array(process_true),  np.array(process_false),
            np.array(final_true),    np.array(final_false))


# ── Bootstrap analysis ────────────────────────────────────────────────────────

def sigmoid(x, w, b):
    z = np.clip(w * x + b, -500, 500)
    return 1 / (1 + np.exp(-z))


def run_bootstrap_analysis(true_arr, false_arr, ax1, ax2,
                           stage_name, n_bootstraps=1000):
    print(f"[Step5] Analyzing: {stage_name}")

    if len(true_arr) < 2 or len(false_arr) < 2:
        print(f"[Step5] Not enough data for {stage_name}, skipping.")
        return None, None, None

    np.random.seed(42)
    boot_thresholds = []
    boot_params     = []

    for _ in range(n_bootstraps):
        s_true  = np.random.choice(true_arr,  size=len(true_arr),  replace=True)
        s_false = np.random.choice(false_arr, size=len(false_arr), replace=True)

        X = np.concatenate([s_true, s_false]).reshape(-1, 1)
        y = np.concatenate([np.zeros(len(s_true)), np.ones(len(s_false))])

        clf = LogisticRegression(C=1e5, solver="lbfgs")
        try:
            clf.fit(X, y)
        except Exception:
            continue

        w = clf.coef_[0][0]
        b = clf.intercept_[0]
        if abs(w) > 1e-4:
            boot_thresholds.append(-b / w)
            boot_params.append((w, b))

    boot_thresholds = np.array(boot_thresholds)
    lower  = np.percentile(boot_thresholds, 2.5)
    upper  = np.percentile(boot_thresholds, 97.5)
    median = np.median(boot_thresholds)

    print(f"[Step5] {stage_name} — median: {median:.4f}, "
          f"95% CI: [{lower:.4f}, {upper:.4f}]")

    # Plotting range
    x_min = min(true_arr.min(), false_arr.min()) - 0.1
    x_max = max(true_arr.max(), false_arr.max()) + 0.1
    x_range = np.linspace(x_min, x_max, 500)

    # Background zones
    ax1.axvspan(x_min,  lower, color="green", alpha=0.1, label="Safe Zone")
    ax1.axvspan(lower,  upper, color="gold",  alpha=0.2, label="Buffer Zone")
    ax1.axvspan(upper,  x_max, color="red",   alpha=0.1, label="Intervention Zone")

    # Sigmoid cloud
    for (w_i, b_i) in boot_params[:500]:
        ax1.plot(x_range, sigmoid(x_range, w_i, b_i),
                 color="steelblue", alpha=0.01, linewidth=1)

    # Median sigmoid curve
    all_X = np.concatenate([true_arr, false_arr]).reshape(-1, 1)
    all_y = np.concatenate([np.zeros(len(true_arr)), np.ones(len(false_arr))])
    clf_main = LogisticRegression(C=1e5).fit(all_X, all_y)
    y_main = clf_main.predict_proba(x_range.reshape(-1, 1))[:, 1]
    ax1.plot(x_range, y_main, color="blue", linewidth=2,
             linestyle="--", label="Median Probability Curve")

    # KDE distributions
    ax1_kde = ax1.twinx()
    kde_true  = gaussian_kde(true_arr)
    kde_false = gaussian_kde(false_arr)
    ax1_kde.fill_between(x_range, kde_true(x_range),
                         color="forestgreen", alpha=0.3, label="Correct Dist.")
    ax1_kde.fill_between(x_range, kde_false(x_range),
                         color="firebrick",   alpha=0.3, label="Incorrect Dist.")
    ax1_kde.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Styling ax1
    ax1.set_title(stage_name, fontsize=17, pad=15)
    ax1.set_ylabel("Probability of Intervention", color="blue", fontsize=15)
    ax1_kde.set_ylabel("Sample Density", color="black", fontsize=15)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(labelsize=12)
    ax1_kde.tick_params(labelsize=12)
    ax1.axvline(lower, color="black", linestyle=":", alpha=0.6)
    ax1.axvline(upper, color="black", linestyle=":", alpha=0.6)

    lines,  labels  = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_kde.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc="center right", fontsize=12)

    # Threshold histogram
    ax2.hist(boot_thresholds, bins=30, color="gray", alpha=0.7, edgecolor="white")
    ax2.axvline(lower,  color="green", linestyle="--", linewidth=2, label="Lower 2.5%")
    ax2.axvline(median, color="blue",  linestyle="-",  linewidth=2, label="Median")
    ax2.axvline(upper,  color="red",   linestyle="--", linewidth=2, label="Upper 97.5%")
    ax2.set_xlabel("Entropy Value", fontsize=15)
    ax2.set_ylabel("Frequency",     fontsize=15)
    ax2.set_title("Simulated Thresholds", fontsize=15, pad=10)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=12, labelspacing=0.3)
    ax2.grid(True, alpha=0.3)

    return lower, median, upper


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step5: Entropy threshold analysis and visualization."
    )
    parser.add_argument("--eval_dir",    type=str, required=True,
                        help="Directory containing eval_round*.jsonl.")
    parser.add_argument("--exp_kb_dir",  type=str, required=True,
                        help="Experience KB directory (contains pair-EXP.jsonl).")
    args = parser.parse_args()

    pair_exp_path = os.path.join(args.exp_kb_dir, "pair-EXP.jsonl")
    png_path      = os.path.join(args.exp_kb_dir, "entropy_threshold.png")
    pdf_path      = os.path.join(args.exp_kb_dir, "entropy_threshold.pdf")

    if not os.path.exists(pair_exp_path):
        raise FileNotFoundError(
            f"pair-EXP.jsonl not found in {args.exp_kb_dir}. "
            "Please run step2 first."
        )

    if os.path.exists(png_path) and os.path.exists(pdf_path):
        print("[Step5] entropy_threshold.png/pdf already exist, skipping.")
        return

    process_true, process_false, final_true, final_false = collect_entropy(
        args.eval_dir, pair_exp_path
    )

    # Figure layout
    fig = plt.figure(figsize=(10, 14))
    gs  = fig.add_gridspec(4, 1, height_ratios=[4, 1.2, 4, 1.2], hspace=0.35)

    ax1_process = fig.add_subplot(gs[0])
    ax2_process = fig.add_subplot(gs[1])
    ax1_final   = fig.add_subplot(gs[2])
    ax2_final   = fig.add_subplot(gs[3])

    p_lower, p_median, p_upper = run_bootstrap_analysis(
        process_true, process_false,
        ax1_process, ax2_process,
        stage_name="(a) Process Steps",
    )
    f_lower, f_median, f_upper = run_bootstrap_analysis(
        final_true, final_false,
        ax1_final, ax2_final,
        stage_name="(b) Final Steps",
    )

    # Fine-tune subplot spacing
    pos1 = ax1_process.get_position()
    pos2 = ax2_process.get_position()
    pos3 = ax1_final.get_position()
    pos4 = ax2_final.get_position()

    gap_within  = 0.05
    gap_between = 0.08

    ax2_process.set_position([pos2.x0,
                               pos1.y0 - pos2.height - gap_within,
                               pos2.width, pos2.height])
    new_y3 = pos1.y0 - pos2.height - gap_within - pos3.height - gap_between
    ax1_final.set_position([pos3.x0, new_y3, pos3.width, pos3.height])
    ax2_final.set_position([pos4.x0, new_y3 - pos4.height - gap_within,
                             pos4.width, pos4.height])

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path,           bbox_inches="tight")
    plt.close()

    print(f"[Step5] Saved: {png_path}")
    print(f"[Step5] Saved: {pdf_path}")

    # Print threshold summary
    print(f"\n[Step5] ── Threshold Summary ─────────────────────────")
    if p_lower is not None:
        print(f"  Process Steps: lower={p_lower:.4f}, "
              f"median={p_median:.4f}, upper={p_upper:.4f}")
    if f_lower is not None:
        print(f"  Final Steps  : lower={f_lower:.4f}, "
              f"median={f_median:.4f}, upper={f_upper:.4f}")
    print(f"──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
