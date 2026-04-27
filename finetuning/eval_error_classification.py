#!/usr/bin/env python3
"""Compute per-class precision/recall/F1 for PRM error classification and generate plots."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CATEGORIES = [
    "Task Specification Violations",
    "Role Specification Violations",
    "Step Repetition",
    "Termination Condition Unawareness",
    "Problem Misidentification",
    "Tool Selection Errors",
    "Hallucinations",
    "Information Processing Failures",
    "Task Derailment",
    "Goal Deviation",
    "Context Handling Failures",
    "Verification Failures",
]

SHORT_NAMES = [
    "Task Spec Viol.",
    "Role Spec Viol.",
    "Step Repetition",
    "Term. Cond. Unaware",
    "Problem Misident.",
    "Tool Selection Err.",
    "Hallucinations",
    "Info Processing Fail.",
    "Task Derailment",
    "Goal Deviation",
    "Context Handling Fail.",
    "Verification Fail.",
]


def extract_detections(text: str) -> dict[str, int]:
    """Extract DETECTED: Yes/No for each category from a PRM response."""
    results = {}
    for cat in CATEGORIES:
        pattern = re.escape(cat) + r".*?DETECTED:\s*(Yes|No)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            results[cat] = 1 if m.group(1).strip() == "Yes" else 0
    return results


def compute_metrics(pred_dets: dict, gt_dets: dict, valid_idxs: list) -> dict:
    """Compute per-class TP/FP/FN/TN/precision/recall/F1."""
    results = {}
    for cat in CATEGORIES:
        tp = fp = fn = tn = 0
        n_evaluated = 0
        for idx in valid_idxs:
            gt_val = gt_dets[idx].get(cat)
            pred_val = pred_dets[idx].get(cat)
            if gt_val is None or pred_val is None:
                continue
            n_evaluated += 1
            if gt_val == 1 and pred_val == 1:
                tp += 1
            elif gt_val == 0 and pred_val == 1:
                fp += 1
            elif gt_val == 1 and pred_val == 0:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cat] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "support_pos": tp + fn,
            "n_evaluated": n_evaluated,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Error classification metrics and plots")
    parser.add_argument("--model-a", type=str, default="qwen3-8b-base", help="First model label")
    parser.add_argument("--model-b", type=str, default="qwen3-8b-opus-distill-32k-lr5e6", help="Second model label")
    parser.add_argument("--data-path", type=str,
                        default=str(SCRIPT_DIR / "prm_sft_data_opus_distill_full_feedback_history_32k_old" / "prm_sft_train.jsonl"))
    parser.add_argument("--eval-dir", type=str, default=str(SCRIPT_DIR / "eval_results"))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Load ground truth
    gt_texts = []
    with open(args.data_path) as f:
        for line in f:
            gt_texts.append(json.loads(line)["messages"][2]["content"])
    print(f"Ground truth: {len(gt_texts)} samples from {args.data_path}")

    # Load model responses
    def load_responses(label):
        resp = {}
        path = eval_dir / label / "responses.jsonl"
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                resp[item["idx"]] = item["response"]
        print(f"  {label}: {len(resp)} responses")
        return resp

    resp_a = load_responses(args.model_a)
    resp_b = load_responses(args.model_b)

    # Parse detections
    gt_dets, dets_a, dets_b = {}, {}, {}
    valid_idxs = []
    for i, gt_text in enumerate(gt_texts):
        gt_d = extract_detections(gt_text)
        if len(gt_d) < 12:
            continue
        if i not in resp_a or i not in resp_b:
            continue
        gt_dets[i] = gt_d
        dets_a[i] = extract_detections(resp_a[i])
        dets_b[i] = extract_detections(resp_b[i])
        valid_idxs.append(i)

    print(f"Valid samples (GT parseable + both models have responses): {len(valid_idxs)}")

    # Parse rate
    a_full = sum(1 for i in valid_idxs if len(dets_a[i]) == 12)
    b_full = sum(1 for i in valid_idxs if len(dets_b[i]) == 12)
    print(f"  {args.model_a} — all 12 categories parsed: {a_full}/{len(valid_idxs)} ({100*a_full/len(valid_idxs):.1f}%)")
    print(f"  {args.model_b} — all 12 categories parsed: {b_full}/{len(valid_idxs)} ({100*b_full/len(valid_idxs):.1f}%)")

    metrics_a = compute_metrics(dets_a, gt_dets, valid_idxs)
    metrics_b = compute_metrics(dets_b, gt_dets, valid_idxs)

    # Print table
    print(f"\n{'Category':<24} {'Support':>7} | {'--- ' + args.model_a + ' ---':^28} | {'--- ' + args.model_b + ' ---':^28}")
    print(f"{'':24} {'(pos)':>7} | {'Prec':>7} {'Rec':>7} {'F1':>7} | {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 95)
    f1s_a, f1s_b = [], []
    for cat, sn in zip(CATEGORIES, SHORT_NAMES):
        ma, mb = metrics_a[cat], metrics_b[cat]
        f1s_a.append(ma["f1"])
        f1s_b.append(mb["f1"])
        print(f"{sn:<24} {ma['support_pos']:>7} | {ma['precision']:>7.3f} {ma['recall']:>7.3f} {ma['f1']:>7.3f} | {mb['precision']:>7.3f} {mb['recall']:>7.3f} {mb['f1']:>7.3f}")

    macro_a = np.mean(f1s_a)
    macro_b = np.mean(f1s_b)
    print("-" * 95)
    print(f"{'Macro-Average F1':<24} {'':>7} | {'':>7} {'':>7} {macro_a:>7.3f} | {'':>7} {'':>7} {macro_b:>7.3f}")

    # Save CSV
    csv_path = eval_dir / f"error_classification_{args.model_a}_vs_{args.model_b}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Error Category", "Opus Positives",
                     f"{args.model_a} Precision", f"{args.model_a} Recall", f"{args.model_a} F1",
                     f"{args.model_a} TP", f"{args.model_a} FP", f"{args.model_a} FN",
                     f"{args.model_b} Precision", f"{args.model_b} Recall", f"{args.model_b} F1",
                     f"{args.model_b} TP", f"{args.model_b} FP", f"{args.model_b} FN"])
        for cat, sn in zip(CATEGORIES, SHORT_NAMES):
            ma, mb = metrics_a[cat], metrics_b[cat]
            w.writerow([sn, ma["support_pos"],
                         f"{ma['precision']:.3f}", f"{ma['recall']:.3f}", f"{ma['f1']:.3f}", ma["tp"], ma["fp"], ma["fn"],
                         f"{mb['precision']:.3f}", f"{mb['recall']:.3f}", f"{mb['f1']:.3f}", mb["tp"], mb["fp"], mb["fn"]])
        w.writerow([])
        w.writerow(["Macro-Average F1", "", "", "", f"{macro_a:.3f}", "", "", "", "", "", f"{macro_b:.3f}"])
    print(f"\nSaved CSV: {csv_path}")

    # =========================================================================
    # PLOT 1: Grouped bar chart — F1 per category for both models
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CATEGORIES))
    width = 0.35

    bars_a = ax.bar(x - width / 2, f1s_a, width, label=args.model_a, color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars_b = ax.bar(x + width / 2, f1s_b, width, label=args.model_b, color="#ED7D31", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score: Error Category Classification vs Claude Opus Ground Truth", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=macro_a, color="#5B9BD5", linestyle="--", alpha=0.8, linewidth=2)
    ax.axhline(y=macro_b, color="#ED7D31", linestyle="--", alpha=0.8, linewidth=2)
    # Place macro labels on the left side with a background box for legibility
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(0.3, macro_a + 0.02, f"Macro-Avg F1 = {macro_a:.3f}",
            color="#5B9BD5", fontsize=11, fontweight="bold", ha="left", bbox=bbox_props)
    ax.text(0.3, macro_b + 0.02, f"Macro-Avg F1 = {macro_b:.3f}",
            color="#ED7D31", fontsize=11, fontweight="bold", ha="left", bbox=bbox_props)

    # Add support counts on top
    for i, cat in enumerate(CATEGORIES):
        support = metrics_a[cat]["support_pos"]
        ax.text(i, max(f1s_a[i], f1s_b[i]) + 0.03, f"n={support}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    f1_plot_path = eval_dir / f"error_classification_f1_{args.model_a}_vs_{args.model_b}.png"
    fig.savefig(f1_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved F1 plot: {f1_plot_path}")
    plt.close()

    # =========================================================================
    # PLOT 2: Precision vs Recall scatter for both models
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (cat, sn) in enumerate(zip(CATEGORIES, SHORT_NAMES)):
        ma, mb = metrics_a[cat], metrics_b[cat]
        if ma["support_pos"] == 0:
            continue
        ax.scatter(ma["recall"], ma["precision"], color="#5B9BD5", s=80, zorder=3, marker="o",
                   edgecolors="white", linewidth=0.5)
        ax.scatter(mb["recall"], mb["precision"], color="#ED7D31", s=80, zorder=3, marker="s",
                   edgecolors="white", linewidth=0.5)
        # Draw line connecting the two
        ax.plot([ma["recall"], mb["recall"]], [ma["precision"], mb["precision"]],
                color="gray", alpha=0.3, linewidth=1, zorder=1)
        # Label near the SFT point
        ax.annotate(sn, (mb["recall"], mb["precision"]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points", color="#ED7D31")

    ax.scatter([], [], color="#5B9BD5", s=80, marker="o", label=args.model_a)
    ax.scatter([], [], color="#ED7D31", s=80, marker="s", label=args.model_b)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision vs Recall per Error Category", fontsize=13, pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    # F1 iso-curves
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        r_vals = np.linspace(0.01, 1.0, 200)
        p_vals = f1_val * r_vals / (2 * r_vals - f1_val)
        mask = (p_vals > 0) & (p_vals <= 1.0)
        ax.plot(r_vals[mask], p_vals[mask], color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
        # Label the iso-curve
        label_idx = np.argmin(np.abs(r_vals[mask] - 0.95))
        if label_idx < len(r_vals[mask]):
            ax.text(r_vals[mask][label_idx], p_vals[mask][label_idx], f"F1={f1_val}",
                    fontsize=7, color="lightgray", ha="right")

    plt.tight_layout()
    pr_plot_path = eval_dir / f"error_classification_pr_{args.model_a}_vs_{args.model_b}.png"
    fig.savefig(pr_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved PR plot: {pr_plot_path}")
    plt.close()

    # =========================================================================
    # PLOT 3: Heatmap — confusion matrix grid for both models
    # =========================================================================
    # Skip categories with 0 positives (Role Spec Viol.)
    active_cats = [(cat, sn) for cat, sn in zip(CATEGORIES, SHORT_NAMES) if metrics_a[cat]["support_pos"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (model_label, metrics) in zip(axes, [(args.model_a, metrics_a), (args.model_b, metrics_b)]):
        n_cats = len(active_cats)
        # Build a matrix: rows = categories, cols = [TP rate, FP rate, FN rate, TN rate]
        # Better: just show TP, FP, FN, TN as raw counts in a heatmap
        # Actually most useful: show prediction breakdown for each category
        # Rows = categories, 2 cols = [predicted Yes given actual Yes (recall), predicted Yes given actual No (FPR)]
        recall_vals = []
        fpr_vals = []
        for cat, sn in active_cats:
            m = metrics[cat]
            recall_vals.append(m["recall"])
            neg = m["tn"] + m["fp"]
            fpr_vals.append(m["fp"] / neg if neg > 0 else 0)

        data = np.array([recall_vals, fpr_vals]).T  # shape: (n_cats, 2)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Recall\n(TP Rate)", "False Positive\nRate"], fontsize=10)
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels([sn for _, sn in active_cats], fontsize=9)
        ax.set_title(model_label, fontsize=12, pad=10)

        # Annotate cells
        for i in range(n_cats):
            for j in range(2):
                val = data[i, j]
                color = "white" if val < 0.3 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    fig.suptitle("Error Detection: Recall & False Positive Rate per Category\n(Green = good for Recall col, bad for FPR col)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    heatmap_path = eval_dir / f"error_classification_heatmap_{args.model_a}_vs_{args.model_b}.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap: {heatmap_path}")
    plt.close()


if __name__ == "__main__":
    main()
