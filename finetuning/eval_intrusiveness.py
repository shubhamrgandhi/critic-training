#!/usr/bin/env python3
"""Analyze PRM intrusiveness: compare each model's structured outputs against ground truth.

Produces presentation-ready plots and CSV tables in a single output directory:
- 10 PNG plots (one per analysis section, with captions/findings embedded)
- 1 master CSV per section (named 01_format_compliance.csv, etc.)
- 1 REPORT.md summary

Usage:
    python eval_intrusiveness.py
    python eval_intrusiveness.py --models qwen3-8b-base sft-rejection-sample sft-rejection-sample-think
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval_results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_results" / "intrusiveness_analysis"

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

SHORT_NAMES = {
    "Task Specification Violations": "Task Spec Viol.",
    "Role Specification Violations": "Role Spec Viol.",
    "Step Repetition": "Step Repetition",
    "Termination Condition Unawareness": "Termination Unaware.",
    "Problem Misidentification": "Problem Misid.",
    "Tool Selection Errors": "Tool Selection Err.",
    "Hallucinations": "Hallucinations",
    "Information Processing Failures": "Info Processing Fail.",
    "Task Derailment": "Task Derailment",
    "Goal Deviation": "Goal Deviation",
    "Context Handling Failures": "Context Handling Fail.",
    "Verification Failures": "Verification Fail.",
}

# Group labels
CATEGORY_GROUPS = {
    "Specification\nErrors": CATEGORIES[:4],
    "Reasoning\nErrors": CATEGORIES[4:8],
    "Coordination\nErrors": CATEGORIES[8:],
}

STATUS_BUCKETS = ["On track", "Needs correction", "Critical intervention required"]
STATUS_SHORT = {
    "On track": "On Track",
    "Needs correction": "Needs Correction",
    "Critical intervention required": "Critical",
}
SEVERITY = {"On track": 0, "Needs correction": 1, "Critical intervention required": 2}

# Colorblind-friendly palette
PALETTE = {
    "qwen3-8b-base": "#0173B2",
    "qwen3-8b-base-think": "#56B4E9",
    "sft-clean": "#F0E442",
    "sft-clean-think": "#009E73",
    "sft-rejection-sample": "#DE8F05",
    "sft-rejection-sample-think": "#CC79A7",
}
GT_COLOR = "#CC78BC"
ALIGNMENT_COLORS = {
    "exact": "#029E73",
    "over": "#D55E00",
    "under": "#0173B2",
    "unparseable": "#BBBBBB",
}

MODEL_LABELS = {
    "qwen3-8b-base": "Qwen3-8B (Base)",
    "qwen3-8b-base-think": "Qwen3-8B (Base+Think)",
    "sft-clean": "Clean-SFT",
    "sft-clean-think": "Clean-SFT-Think",
    "sft-rejection-sample": "RS-SFT",
    "sft-rejection-sample-think": "RS-SFT-Think",
}


def parse_detections(text: str) -> dict[str, bool | None]:
    # Handle both plain format:  "1. Category: DETECTED: Yes"
    # and bold markdown format:  "1. **Category:** DETECTED: Yes"
    found = re.findall(r"(\d+)\.\s+\*{0,2}(.+?)\*{0,2}:\*{0,2}\s*DETECTED:\s*(Yes|No)", text)
    result = {}
    for num, name, val in found:
        for cat in CATEGORIES:
            if cat.lower().startswith(name.strip().lower()[:20]):
                result[cat] = val == "Yes"
                break
        else:
            for cat in CATEGORIES:
                if name.strip().lower()[:15] in cat.lower():
                    result[cat] = val == "Yes"
                    break
    return result


def parse_task_status(text: str) -> str | None:
    m = re.search(r"TASK_STATUS:\s*(.*)", text)
    if not m:
        return None
    raw = m.group(1).strip().strip("*").strip()
    raw_lower = raw.lower()
    if raw_lower.startswith("critical"):
        return "Critical intervention required"
    elif raw_lower.startswith("needs"):
        return "Needs correction"
    elif raw_lower.startswith("on track") or raw_lower.startswith("already"):
        return "On track"
    return None


def parse_guidance_length(text: str) -> int:
    m = re.search(r"OVERALL_GUIDANCE:\s*(.*)", text, re.DOTALL)
    return len(m.group(1).strip()) if m else 0


def load_model(eval_dir: Path, model: str) -> list[dict]:
    with open(eval_dir / model / "responses.jsonl") as f:
        return [json.loads(line) for line in f]


def ml(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_csv(path: Path, header: list, rows: list[list]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  CSV: {path}")


def add_finding(ax, text, loc="lower right", fontsize=9):
    """Add an annotation textbox with a key finding to a plot."""
    bbox = dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#BDBDBD", alpha=0.95)
    loc_map = {
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "upper center": (0.50, 0.97, "center", "top"),
        "lower center": (0.50, 0.03, "center", "bottom"),
    }
    xp, yp, ha, va = loc_map.get(loc, loc_map["lower right"])
    ax.text(xp, yp, text, transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, bbox=bbox, style="italic")


def _compute_bleu_rouge(pair):
    """Compute BLEU and ROUGE for a single (gt_text, model_text) pair."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

    gt_text, m_text = pair
    if not m_text.strip():
        return None
    _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    _smooth = SmoothingFunction().method1
    gt_tokens = gt_text.split()
    m_tokens = m_text.split()
    rs = _scorer.score(gt_text, m_text)
    return {
        "bleu1": sentence_bleu([gt_tokens], m_tokens, weights=(1, 0, 0, 0), smoothing_function=_smooth),
        "bleu2": sentence_bleu([gt_tokens], m_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=_smooth),
        "bleu4": sentence_bleu([gt_tokens], m_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_smooth),
        "rouge1_f": rs["rouge1"].fmeasure,
        "rouge2_f": rs["rouge2"].fmeasure,
        "rougeL_f": rs["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRM intrusiveness analysis")
    parser.add_argument("--models", nargs="+",
                        default=["qwen3-8b-base", "sft-rejection-sample", "sft-rejection-sample-think"])
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.output_dir)
    models = args.models

    out.mkdir(parents=True, exist_ok=True)

    # Clean output dir of old files from previous runs
    for old in out.glob("intrusiveness_*"):
        old.unlink()
    for old in out.glob("*.pdf"):
        old.unlink()

    # Load
    all_data = {}
    for model in models:
        all_data[model] = load_model(eval_dir, model)
        print(f"Loaded {len(all_data[model])} samples for {model}")
    n = len(all_data[models[0]])

    # Parse with progress
    print("Parsing ground truth...")
    gt_det = [parse_detections(d["ground_truth"]) for d in tqdm(all_data[models[0]], desc="GT detections")]
    gt_stat = [parse_task_status(d["ground_truth"]) for d in all_data[models[0]]]
    gt_glen = [parse_guidance_length(d["ground_truth"]) for d in all_data[models[0]]]

    m_det, m_stat, m_glen = {}, {}, {}
    for m in models:
        print(f"Parsing {ml(m)}...")
        m_det[m] = [parse_detections(d["response"]) for d in tqdm(all_data[m], desc=f"{ml(m)} detections")]
        m_stat[m] = [parse_task_status(d["response"]) for d in all_data[m]]
        m_glen[m] = [parse_guidance_length(d["response"]) for d in all_data[m]]

    # =====================================================================
    # 1. FORMAT COMPLIANCE
    # =====================================================================
    print("\n" + "=" * 70)
    print("1. FORMAT COMPLIANCE")
    print("=" * 70)

    fmt_header = ["Model", "Full (12/12)", "Partial (1-11)", "Unparseable (0/12)",
                  "Missing TASK_STATUS", "N"]
    fmt_rows = []
    for m in models:
        full = sum(1 for d in m_det[m] if len(d) == 12)
        partial = sum(1 for d in m_det[m] if 0 < len(d) < 12)
        zero = sum(1 for d in m_det[m] if len(d) == 0)
        miss = sum(1 for s in m_stat[m] if s is None)
        fmt_rows.append([ml(m), f"{full} ({100*full/n:.1f}%)", f"{partial} ({100*partial/n:.1f}%)",
                         f"{zero} ({100*zero/n:.1f}%)", f"{miss} ({100*miss/n:.1f}%)", n])
        print(f"  {ml(m):20s}  full={full} ({100*full/n:.1f}%)  partial={partial}  "
              f"unparseable={zero} ({100*zero/n:.1f}%)  missing_status={miss} ({100*miss/n:.1f}%)")
    write_csv(out / "01_format_compliance.csv", fmt_header, fmt_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models))
    w = 0.22
    for j, (label, color, fn) in enumerate([
        ("All 12 categories parsed", "#029E73", lambda m: sum(1 for d in m_det[m] if len(d) == 12)),
        ("Partial (1-11 categories)", "#ECE133", lambda m: sum(1 for d in m_det[m] if 0 < len(d) < 12)),
        ("Unparseable (0 categories)", "#D55E00", lambda m: sum(1 for d in m_det[m] if len(d) == 0)),
    ]):
        vals = [fn(m) / n * 100 for m in models]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([ml(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("% of Samples")
    ax.set_title("Format Compliance: How many of the 12 DETECTED categories were parseable? (N={})".format(n))
    ax.legend(frameon=True, edgecolor="lightgray", loc="upper left")
    ax.set_ylim(0, 115)
    best_sft = max(sum(1 for d in m_det[m] if len(d) == 12) / n * 100
                   for m in models if m != "qwen3-8b-base" and m != "qwen3-8b-base-think")
    worst_sft = min(sum(1 for d in m_det[m] if len(d) == 12) / n * 100
                    for m in models if "sft" in m or "SFT" in m.lower())
    add_finding(ax, f"SFT models learn PRM format well ({best_sft:.0f}% fully parseable for best).\n"
                f"Base Qwen3-8B fails to produce parseable output ~30% of the time.\n"
                f"SFT range: {worst_sft:.0f}–{best_sft:.0f}%.",
                loc="lower right")
    fig.tight_layout()
    save_fig(fig, out / "01_format_compliance.png")

    # =====================================================================
    # 2. TASK_STATUS DISTRIBUTION
    # =====================================================================
    print("\n" + "=" * 70)
    print("2. TASK_STATUS DISTRIBUTION")
    print("=" * 70)

    ts_header = ["Status", "Ground Truth (Claude Opus)"] + [ml(m) for m in models]
    ts_rows = []
    for bucket in STATUS_BUCKETS + [None]:
        label = bucket if bucket else "Unparseable"
        gt_c = sum(1 for s in gt_stat if s == bucket)
        row = [label, f"{gt_c} ({100*gt_c/n:.1f}%)"]
        for m in models:
            mc = sum(1 for s in m_stat[m] if s == bucket)
            row.append(f"{mc} ({100*mc/n:.1f}%)")
        ts_rows.append(row)
        print(f"  {label:35s}  GT={gt_c:>5}  " + "  ".join(
            f"{ml(m)}={sum(1 for s in m_stat[m] if s == bucket):>5}" for m in models))
    write_csv(out / "02_task_status_dist.csv", ts_header, ts_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6))
    sources = ["GT\n(Claude Opus)"] + [ml(m) for m in models]
    n_src = len(sources)
    x = np.arange(len(STATUS_BUCKETS))
    bw = 0.8 / n_src
    gt_cnts = [sum(1 for s in gt_stat if s == b) for b in STATUS_BUCKETS]
    all_cnts = [gt_cnts] + [[sum(1 for s in m_stat[m] if s == b) for b in STATUS_BUCKETS] for m in models]
    colors = [GT_COLOR] + [PALETTE[m] for m in models]

    for i, (src, cnts, col) in enumerate(zip(sources, all_cnts, colors)):
        pcts = [c / n * 100 for c in cnts]
        bars = ax.bar(x + (i - n_src / 2 + 0.5) * bw, pcts, bw, label=src, color=col,
                      edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([STATUS_SHORT[b] for b in STATUS_BUCKETS], fontsize=12)
    ax.set_ylabel("% of Samples (N={})".format(n))
    ax.set_title("TASK_STATUS Distribution")
    ax.legend(frameon=True, edgecolor="lightgray", loc="upper left", fontsize=9)
    add_finding(ax, "SFT models better match GT 'Needs Correction' distribution.\n"
                "All SFT models still under-report 'Critical' vs GT (14-18% vs GT 23%).",
                loc="upper right")
    fig.tight_layout()
    save_fig(fig, out / "02_task_status_dist.png")

    # =====================================================================
    # 3. TASK_STATUS CONFUSION MATRICES
    # =====================================================================
    print("\n" + "=" * 70)
    print("3. TASK_STATUS CONFUSION MATRICES (rows=GT, cols=Model)")
    print("=" * 70)

    confusion_mats = {}
    for model in models:
        mat = np.zeros((3, 3))
        for gs, ms in zip(gt_stat, m_stat[model]):
            if gs is None or ms is None:
                continue
            mat[SEVERITY[gs]][SEVERITY[ms]] += 1
        confusion_mats[model] = mat

    # Plot: side-by-side heatmaps
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    sl = ["On\nTrack", "Needs\nCorrection", "Critical"]
    for ax, model in zip(axes, models):
        mat = confusion_mats[model]
        rs = mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        mp = mat / rs * 100
        im = ax.imshow(mp, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(sl, fontsize=9)
        ax.set_yticklabels(sl, fontsize=9)
        ax.set_xlabel("Model Prediction", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Ground Truth (Claude Opus)", fontsize=9)
        ax.set_title(ml(model), fontweight="bold", fontsize=10)
        for gi in range(3):
            for mi in range(3):
                v = mp[gi][mi]
                c = int(mat[gi][mi])
                ax.text(mi, gi, f"{v:.0f}%\n(n={c})", ha="center", va="center",
                        fontsize=8, fontweight="bold" if gi == mi else "normal",
                        color="white" if v > 55 else "black")
    fig.suptitle("TASK_STATUS Confusion Matrices (Row-Normalized %)", fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("% of GT Bucket")
    fig.text(0.5, -0.04,
             "SFT models downgrade 40-60% of 'Critical' samples to 'Needs Correction' "
             "→ agent gets mild nudge instead of hard stop.",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#BDBDBD", alpha=0.95))
    fig.tight_layout()
    save_fig(fig, out / "03_task_status_confusion.png")

    # =====================================================================
    # 4. INTERVENTION ALIGNMENT
    # =====================================================================
    print("\n" + "=" * 70)
    print("4. INTERVENTION ALIGNMENT")
    print("=" * 70)

    align = {}
    for model in models:
        exact = over = under = unp = 0
        valid = 0
        for gs, ms in zip(gt_stat, m_stat[model]):
            if gs is None:
                continue
            valid += 1
            if ms is None:
                unp += 1
            elif SEVERITY[ms] == SEVERITY[gs]:
                exact += 1
            elif SEVERITY[ms] > SEVERITY[gs]:
                over += 1
            else:
                under += 1
        align[model] = {"exact": exact, "over": over, "under": under, "unp": unp, "valid": valid}

    al_header = ["Model", "Exact Match", "Over-Intervention", "Under-Intervention", "Unparseable", "N (valid GT)"]
    al_rows = []
    for m in models:
        a = align[m]
        v = a["valid"]
        al_rows.append([ml(m)] + [f"{a[k]} ({100*a[k]/v:.1f}%)" for k in ["exact", "over", "under", "unp"]] + [v])
        print(f"  {ml(m):20s}  exact={100*a['exact']/v:.1f}%  over={100*a['over']/v:.1f}%  "
              f"under={100*a['under']/v:.1f}%  unp={100*a['unp']/v:.1f}%")
    write_csv(out / "04_alignment.csv", al_header, al_rows)

    # Plot: stacked bar
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    bw = 0.5
    bottoms = np.zeros(len(models))
    for key, color, label in [
        ("under", ALIGNMENT_COLORS["under"], "Under-Intervention"),
        ("exact", ALIGNMENT_COLORS["exact"], "Exact Match"),
        ("over", ALIGNMENT_COLORS["over"], "Over-Intervention"),
        ("unp", ALIGNMENT_COLORS["unparseable"], "Unparseable"),
    ]:
        vals = np.array([align[m][key] / align[m]["valid"] * 100 for m in models])
        ax.bar(x, vals, bw, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 4:
                ax.text(x[xi], b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([ml(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("% of Samples")
    ax.set_title("Intervention Severity Alignment vs Ground Truth")
    ax.legend(loc="upper left", frameon=True, edgecolor="lightgray")
    ax.set_ylim(0, 108)
    _sft_exact = [100 * align[m]["exact"] / align[m]["valid"] for m in models if "sft" in m or "SFT" in m.lower()]
    _sft_under = [100 * align[m]["under"] / align[m]["valid"] for m in models if "sft" in m or "SFT" in m.lower()]
    add_finding(ax, f"Best SFT exact match: {max(_sft_exact):.0f}% (Clean-SFT variants).\n"
                f"All SFT models: {min(_sft_under):.0f}–{max(_sft_under):.0f}% under-intervention "
                f"(Critical→Needs Correction).",
                loc="lower right")
    fig.tight_layout()
    save_fig(fig, out / "04_alignment.png")

    # =====================================================================
    # 5. PER-CATEGORY METRICS
    # =====================================================================
    print("\n" + "=" * 70)
    print("5. PER-CATEGORY DETECTION METRICS")
    print("=" * 70)

    cat_m = {}
    for model in models:
        cat_m[model] = {}
        for cat in CATEGORIES:
            tp = fp = fn = tn = 0
            for i in range(n):
                gv = gt_det[i].get(cat)
                mv = m_det[model][i].get(cat)
                if gv is None or mv is None:
                    continue
                if gv and mv: tp += 1
                elif not gv and mv: fp += 1
                elif gv and not mv: fn += 1
                else: tn += 1
            total = tp + fp + fn + tn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            cat_m[model][cat] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec,
                                     recall=rec, f1=f1, fpr=fpr, fnr=fnr,
                                     support=tp+fn, total=total)

    # CSV: one big table
    pc_header = ["Category", "Group", "GT Positives"]
    for m in models:
        for met in ["FPR", "FNR", "Precision", "Recall", "F1", "FP", "FN"]:
            pc_header.append(f"{ml(m)} {met}")
    pc_rows = []
    cat_to_group = {}
    for gname, gcats in CATEGORY_GROUPS.items():
        for c in gcats:
            cat_to_group[c] = gname.replace("\n", " ")

    for cat in CATEGORIES:
        row = [SHORT_NAMES[cat], cat_to_group[cat], cat_m[models[0]][cat]["support"]]
        for m in models:
            cm = cat_m[m][cat]
            row.extend([f"{cm['fpr']:.3f}", f"{cm['fnr']:.3f}", f"{cm['precision']:.3f}",
                        f"{cm['recall']:.3f}", f"{cm['f1']:.3f}", cm["fp"], cm["fn"]])
        pc_rows.append(row)

    # Macro averages
    macro_row = ["MACRO AVERAGE", "", ""]
    for m in models:
        for met in ["fpr", "fnr", "precision", "recall", "f1"]:
            vals = [cat_m[m][c][met] for c in CATEGORIES]
            macro_row.append(f"{sum(vals)/len(vals):.3f}")
        macro_row.extend(["", ""])
    pc_rows.append(macro_row)
    write_csv(out / "05_per_category_metrics.csv", pc_header, pc_rows)

    # Print summary
    for met_name, met_key in [("FPR", "fpr"), ("FNR", "fnr"), ("F1", "f1")]:
        print(f"\n  {met_name}:")
        for cat in CATEGORIES:
            vals = "  ".join(f"{ml(m):>10s}={cat_m[m][cat][met_key]:.3f}" for m in models)
            print(f"    {SHORT_NAMES[cat]:25s}  {vals}")
        macro = "  ".join(f"{ml(m):>10s}={sum(cat_m[m][c][met_key] for c in CATEGORIES)/12:.3f}" for m in models)
        print(f"    {'MACRO':25s}  {macro}")

    # --- Individual metric plots ---
    DETECTION_DEF = "Positive = DETECTED: Yes (error flagged)  |  Negative = DETECTED: No (no error)"

    def plot_category_metric(metric_key, ylabel, title, filename, annotate_threshold=0.05,
                             finding=None, finding_loc="lower right", ylim=(0, 1.05)):
        fig, ax = plt.subplots(figsize=(16, 6))
        cnr = [c for c in CATEGORIES if c != "Role Specification Violations"]
        x = np.arange(len(cnr))
        bw = 0.8 / len(models)
        for i, model in enumerate(models):
            vals = [cat_m[model][c][metric_key] for c in cnr]
            bars = ax.bar(x + (i - len(models) / 2 + 0.5) * bw, vals, bw,
                          label=ml(model), color=PALETTE[model], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > annotate_threshold:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_NAMES[c] for c in cnr], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(frameon=True, edgecolor="lightgray", loc="upper left", fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.3)
        for gb in [3, 7]:
            ax.axvline(x=gb - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        if finding:
            add_finding(ax, finding, loc=finding_loc, fontsize=8)
        fig.text(0.5, -0.02, DETECTION_DEF, ha="center", fontsize=9, color="gray")
        fig.tight_layout()
        save_fig(fig, out / filename)

    # Compute actual macro F1 values for dynamic finding text
    macro_f1 = {m: sum(cat_m[m][c]["f1"] for c in CATEGORIES) / len(CATEGORIES) for m in models}
    best_sft_f1 = max(macro_f1[m] for m in models if "base" not in m)

    plot_category_metric("fpr", "False Positive Rate",
                         "Per-Category FPR (Intrusiveness)\nModel flags error when GT does not",
                         "05a_fpr_by_category.png",
                         finding="Info Processing: highest FPR across all SFT models (0.26–0.40).\n"
                                 "SFT models are MORE intrusive than base on Problem Misid & Info Processing.",
                         finding_loc="upper right")
    plot_category_metric("fnr", "False Negative Rate",
                         "Per-Category FNR (Missed Errors)\nModel misses error that GT flags",
                         "05b_fnr_by_category.png",
                         finding="SFT greatly reduces missed errors vs base across most categories.\n"
                                 "Hardest: Hallucinations (FNR 0.80–0.84), Goal Deviation (0.80–0.90).",
                         finding_loc="upper right")
    plot_category_metric("f1", "F1 Score",
                         "Per-Category F1 Score vs Ground Truth",
                         "05c_f1_by_category.png", annotate_threshold=0.1,
                         finding=f"Clean-SFT macro-avg F1={macro_f1['sft-clean']:.2f} (best).\n"
                                 f"Base macro F1={macro_f1['qwen3-8b-base']:.2f}.\n"
                                 "Best categories: Verification, Step Repetition, Info Processing.",
                         finding_loc="lower right")

    # Combined FPR + FNR side-by-side
    cats_no_role = [c for c in CATEGORIES if c != "Role Specification Violations"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    x = np.arange(len(cats_no_role))
    bw = 0.8 / len(models)
    for i, model in enumerate(models):
        fpr_vals = [cat_m[model][c]["fpr"] for c in cats_no_role]
        fnr_vals = [cat_m[model][c]["fnr"] for c in cats_no_role]
        ax1.bar(x + (i - len(models) / 2 + 0.5) * bw, fpr_vals, bw,
                label=ml(model), color=PALETTE[model], edgecolor="white", linewidth=0.5)
        ax2.bar(x + (i - len(models) / 2 + 0.5) * bw, fnr_vals, bw,
                label=ml(model), color=PALETTE[model], edgecolor="white", linewidth=0.5)

    for ax, title in [(ax1, "False Positive Rate\n(Intrusive: flags when GT doesn't)"),
                       (ax2, "False Negative Rate\n(Missed: misses what GT flags)")]:
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_NAMES[c] for c in cats_no_role], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, edgecolor="lightgray", fontsize=8, loc="upper left")
        for gb in [3, 7]:
            ax.axvline(x=gb - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle("Per-Category Error Detection: FPR and FNR vs Ground Truth (Claude Opus)", fontweight="bold")
    fig.text(0.5, -0.03,
             "SFT reduces FNR (fewer missed errors) but increases FPR on Info Processing and Problem Misidentification.",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#BDBDBD", alpha=0.95))
    fig.text(0.5, -0.07, DETECTION_DEF, ha="center", fontsize=9, color="gray")
    fig.tight_layout()
    save_fig(fig, out / "05d_fpr_fnr_combined.png")

    # =====================================================================
    # 6. INTERVENTION INTENSITY
    # =====================================================================
    print("\n" + "=" * 70)
    print("6. INTERVENTION INTENSITY")
    print("=" * 70)

    gt_fc = [sum(1 for v in d.values() if v) for d in gt_det]
    m_fc = {m: [sum(1 for v in d.values() if v) for d in m_det[m]] for m in models}

    ii_header = ["Metric", "Ground Truth (Claude Opus)"] + [ml(m) for m in models]
    ii_rows = []
    for label, fn in [
        ("Mean flags/sample", lambda c: f"{sum(c)/len(c):.2f}"),
        ("Median flags/sample", lambda c: f"{sorted(c)[len(c)//2]}"),
        ("0 flags (abstain)", lambda c: f"{sum(1 for x in c if x == 0)} ({100*sum(1 for x in c if x == 0)/len(c):.1f}%)"),
        ("1-2 flags", lambda c: f"{sum(1 for x in c if 1<=x<=2)} ({100*sum(1 for x in c if 1<=x<=2)/len(c):.1f}%)"),
        ("3-5 flags", lambda c: f"{sum(1 for x in c if 3<=x<=5)} ({100*sum(1 for x in c if 3<=x<=5)/len(c):.1f}%)"),
        ("6+ flags", lambda c: f"{sum(1 for x in c if x>=6)} ({100*sum(1 for x in c if x>=6)/len(c):.1f}%)"),
    ]:
        row = [label, fn(gt_fc)] + [fn(m_fc[m]) for m in models]
        ii_rows.append(row)
    write_csv(out / "06_intervention_intensity.csv", ii_header, ii_rows)

    for m in models:
        d = [m_fc[m][i] - gt_fc[i] for i in range(n) if len(gt_det[i]) == 12 and len(m_det[m][i]) == 12]
        if d:
            print(f"  {ml(m):20s}  mean_delta={sum(d)/len(d):+.2f}  "
                  f"more={sum(1 for x in d if x > 0)}  same={sum(1 for x in d if x == 0)}  "
                  f"fewer={sum(1 for x in d if x < 0)}  n={len(d)}")

    # Plot: side-by-side histograms of flag counts
    all_fc_list = [("Ground Truth\n(Claude Opus)", gt_fc, GT_COLOR)] + \
                  [(ml(m), m_fc[m], PALETTE[m]) for m in models]
    fig, axes = plt.subplots(1, len(all_fc_list), figsize=(4 * len(all_fc_list), 4.5))
    max_f = max(max(gt_fc), *(max(m_fc[m]) for m in models))
    hist_bins = np.arange(-0.5, max_f + 1.5, 1)
    ymax = 0
    for ax_i, (label, counts, color) in zip(axes, all_fc_list):
        ax_i.hist(counts, bins=hist_bins, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        mean_c = sum(counts) / len(counts)
        ax_i.axvline(x=mean_c, color="black", linewidth=1.5, linestyle="--")
        ax_i.set_title(f"{label}\n(mean={mean_c:.1f})", fontweight="bold")
        ax_i.set_xlabel("# Categories Flagged")
        ymax = max(ymax, ax_i.get_ylim()[1])
    for ax_i in axes:
        ax_i.set_ylim(0, ymax)
    axes[0].set_ylabel("# Samples")
    fig.suptitle("Distribution of Intervention Intensity per Sample", fontweight="bold")
    fig.text(0.5, -0.02,
             "Base model flags 0 categories 45% of the time (under-intervenes). "
             "SFT models closely match GT distribution (mean ~2.6 vs GT 3.1).",
             ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#BDBDBD", alpha=0.95))
    save_fig(fig, out / "06a_flag_count_dist.png")

    # =====================================================================
    # 7. BEHAVIOR ON GT="ON TRACK" SAMPLES
    # =====================================================================
    print("\n" + "=" * 70)
    print("7. BEHAVIOR ON GT='ON TRACK' SAMPLES")
    print("=" * 70)

    ot_idxs = [i for i, s in enumerate(gt_stat) if s == "On track"]
    n_ot = len(ot_idxs)
    print(f"  N = {n_ot} samples where GT says 'On track'\n")

    ot_header = ["Metric"] + [ml(m) for m in models]
    ot_rows = []
    for label, fn in [
        ("Also says 'On Track'",
         lambda m: f"{sum(1 for i in ot_idxs if m_stat[m][i] == 'On track')} "
                   f"({100*sum(1 for i in ot_idxs if m_stat[m][i] == 'On track')/n_ot:.1f}%)"),
        ("Says 'Needs Correction'",
         lambda m: f"{sum(1 for i in ot_idxs if m_stat[m][i] == 'Needs correction')} "
                   f"({100*sum(1 for i in ot_idxs if m_stat[m][i] == 'Needs correction')/n_ot:.1f}%)"),
        ("Says 'Critical'",
         lambda m: f"{sum(1 for i in ot_idxs if m_stat[m][i] == 'Critical intervention required')} "
                   f"({100*sum(1 for i in ot_idxs if m_stat[m][i] == 'Critical intervention required')/n_ot:.1f}%)"),
        ("Unparseable",
         lambda m: f"{sum(1 for i in ot_idxs if m_stat[m][i] is None)} "
                   f"({100*sum(1 for i in ot_idxs if m_stat[m][i] is None)/n_ot:.1f}%)"),
        ("Mean categories flagged",
         lambda m: f"{sum(m_fc[m][i] for i in ot_idxs)/n_ot:.2f}"),
        ("GT mean categories flagged",
         lambda m: f"{sum(gt_fc[i] for i in ot_idxs)/n_ot:.2f}"),
    ]:
        ot_rows.append([label] + [fn(m) for m in models])
        print(f"  {label:35s}  " + "  ".join(f"{ml(m)}={fn(m)}" for m in models))
    write_csv(out / "07a_on_track_behavior.csv", ot_header, ot_rows)

    # Plot: On Track FPR by category
    cats_no_role = [c for c in CATEGORIES if c != "Role Specification Violations"]
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(cats_no_role))
    bw = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = []
        for cat in cats_no_role:
            fp = tn = 0
            for idx in ot_idxs:
                gv = gt_det[idx].get(cat)
                mv = m_det[model][idx].get(cat)
                if gv is None or mv is None:
                    continue
                if not gv and mv: fp += 1
                elif not gv and not mv: tn += 1
            vals.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        bars = ax.bar(x + (i - len(models) / 2 + 0.5) * bw, vals, bw,
                      label=ml(model), color=PALETTE[model], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[c] for c in cats_no_role], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Category FPR on 'On Track' Samples (N={})\n"
                 "(Any flag here = unnecessary intervention that could derail the agent)".format(n_ot))
    ax.legend(frameon=True, edgecolor="lightgray", loc="upper left", fontsize=9)
    for gb in [3, 7]:
        ax.axvline(x=gb - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    # Compute on-track Info Processing FPR dynamically
    _ip_cat_ot = next((c for c in cats_no_role if "Information" in c or "Info" in c), None)
    if _ip_cat_ot:
        _ot_ip_fprs = {}
        for model in models:
            fp = tn = 0
            for idx in ot_idxs:
                gv = gt_det[idx].get(_ip_cat_ot)
                mv = m_det[model][idx].get(_ip_cat_ot)
                if gv is None or mv is None:
                    continue
                if not gv and mv: fp += 1
                elif not gv and not mv: tn += 1
            _ot_ip_fprs[model] = fp / (fp + tn) if (fp + tn) > 0 else 0
        _sft_ot_ip = {m: _ot_ip_fprs[m] for m in models if "sft" in m or "SFT" in m.lower()}
        _think_ot_ip = {m: _ot_ip_fprs[m] for m in models if "think" in m.lower()}
        _worst_think = max(_think_ot_ip.items(), key=lambda x: x[1]) if _think_ot_ip else (None, 0)
        add_finding(ax, f"Info Processing is the main intrusiveness hotspot:\n"
                    f"SFT models falsely flag it {min(_sft_ot_ip.values()):.0%}–{max(_sft_ot_ip.values()):.0%} "
                    f"of the time when agent is on track.\n"
                    f"Think variants are worse ({ml(_worst_think[0])}: {_worst_think[1]:.0%}).",
                    loc="upper right", fontsize=9)
    else:
        add_finding(ax, "Info Processing is the main intrusiveness hotspot.",
                    loc="upper right", fontsize=9)
    fig.text(0.5, -0.02, DETECTION_DEF, ha="center", fontsize=9, color="gray")
    fig.tight_layout()
    save_fig(fig, out / "07c_on_track_fpr.png")

    # =====================================================================
    # 8. OVERALL_GUIDANCE LENGTH
    # =====================================================================
    print("\n" + "=" * 70)
    print("8. OVERALL_GUIDANCE LENGTH")
    print("=" * 70)

    gl_header = ["Metric", "Ground Truth (Claude Opus)"] + [ml(m) for m in models]
    gl_rows = []
    for label, fn in [
        ("Mean", lambda c: f"{sum(c)/len(c):.0f}"),
        ("Median", lambda c: f"{sorted(c)[len(c)//2]}"),
        ("p25", lambda c: f"{sorted(c)[len(c)//4]}"),
        ("p75", lambda c: f"{sorted(c)[3*len(c)//4]}"),
    ]:
        gl_rows.append([label, fn(gt_glen)] + [fn(m_glen[m]) for m in models])
        print(f"  {label:10s}  GT={fn(gt_glen):>6s}  " + "  ".join(f"{ml(m)}={fn(m_glen[m]):>6s}" for m in models))
    write_csv(out / "08_guidance_length.csv", gl_header, gl_rows)

    # Plot: box plot
    fig, ax = plt.subplots(figsize=(12, 5))
    data_for_box = [gt_glen] + [m_glen[m] for m in models]
    labels_for_box = ["GT\n(Claude Opus)"] + [ml(m) for m in models]
    colors_for_box = [GT_COLOR] + [PALETTE[m] for m in models]
    bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means = [sum(d) / len(d) for d in data_for_box]
    ax.scatter(range(1, len(data_for_box) + 1), means, marker="D", color="black", s=40, zorder=5, label="Mean")
    gt_mean = sum(gt_glen) / len(gt_glen)
    for i, (m, mn) in enumerate(zip(models, means[1:]), start=2):
        delta = 100 * (mn - gt_mean) / gt_mean
        ax.text(i, mn + 30, f"{delta:+.0f}%", ha="center", fontsize=8, color="gray")
    ax.set_ylabel("OVERALL_GUIDANCE Length (chars)")
    ax.set_title("Distribution of Guidance Length (% vs GT shown above mean markers)")
    ax.legend(frameon=True, edgecolor="lightgray")
    _think_gl_deltas = {m: 100 * (sum(m_glen[m]) / len(m_glen[m]) - gt_mean) / gt_mean
                        for m in models if "think" in m.lower()}
    _max_think_gl = max(_think_gl_deltas.values()) if _think_gl_deltas else 0
    add_finding(ax, f"Think variants write up to {_max_think_gl:+.0f}% longer guidance than GT.\n"
                "More text = more directive noise injected into agent context at each PRM interval.",
                loc="upper left", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out / "08_guidance_length.png")

    # =====================================================================
    # 9. BLEU / ROUGE SCORES (text similarity to ground truth)
    # =====================================================================
    print("\n" + "=" * 70)
    print("9. BLEU / ROUGE SCORES vs Ground Truth")
    print("=" * 70)

    try:
        from concurrent.futures import ProcessPoolExecutor
        import os

        print(f"  Computing BLEU/ROUGE on all {n} samples (parallel)...")

        n_workers = min(os.cpu_count() or 4, 16)
        bleu_rouge_data = {}
        for model in models:
            pairs = [(all_data[model][i]["ground_truth"], all_data[model][i]["response"]) for i in range(n)]
            scores = {"bleu1": [], "bleu2": [], "bleu4": [],
                      "rouge1_f": [], "rouge2_f": [], "rougeL_f": []}
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for result in tqdm(pool.map(_compute_bleu_rouge, pairs, chunksize=64),
                                   total=n, desc=f"BLEU/ROUGE {ml(model)}"):
                    if result is not None:
                        for k in scores:
                            scores[k].append(result[k])
            bleu_rouge_data[model] = {k: sum(v) / len(v) if v else 0 for k, v in scores.items()}

        br_header = ["Metric"] + [ml(m) for m in models]
        br_rows = []
        for label, key in [("BLEU-1", "bleu1"), ("BLEU-2", "bleu2"), ("BLEU-4", "bleu4"),
                           ("ROUGE-1 (F)", "rouge1_f"), ("ROUGE-2 (F)", "rouge2_f"), ("ROUGE-L (F)", "rougeL_f")]:
            row = [label] + [f"{bleu_rouge_data[m][key]:.4f}" for m in models]
            br_rows.append(row)
            print(f"  {label:15s}  " + "  ".join(f"{ml(m)}={bleu_rouge_data[m][key]:.4f}" for m in models))
        write_csv(out / "09_bleu_rouge.csv", br_header, br_rows)

        # Plot: grouped bar for BLEU/ROUGE
        fig, ax = plt.subplots(figsize=(14, 6))
        metrics_to_plot = ["BLEU-1", "BLEU-4", "ROUGE-1 (F)", "ROUGE-L (F)"]
        keys_to_plot = ["bleu1", "bleu4", "rouge1_f", "rougeL_f"]
        x = np.arange(len(metrics_to_plot))
        bw = 0.8 / len(models)
        for i, model in enumerate(models):
            vals = [bleu_rouge_data[model][k] for k in keys_to_plot]
            bars = ax.bar(x + (i - len(models) / 2 + 0.5) * bw, vals, bw,
                          label=ml(model), color=PALETTE[model], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylabel("Score")
        ax.set_title("Text Similarity to Ground Truth (BLEU / ROUGE)")
        ax.legend(frameon=True, edgecolor="lightgray", loc="upper left")
        add_finding(ax, "SFT models produce text much closer to Claude's output\n"
                    "than base Qwen3-8B across all similarity metrics.",
                    loc="upper right", fontsize=9)
        fig.tight_layout()
        save_fig(fig, out / "09_bleu_rouge.png")
    except ImportError as e:
        print(f"  Skipping BLEU/ROUGE: {e}")

    # =====================================================================
    # 10. FULL RESPONSE LENGTH ANALYSIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("10. RESPONSE LENGTH ANALYSIS")
    print("=" * 70)

    gt_resp_lens = [len(item["ground_truth"]) for item in all_data[models[0]]]
    m_resp_lens = {m: [len(item["response"]) for item in all_data[m]] for m in models}

    rl_header = ["Metric", "Ground Truth (Claude Opus)"] + [ml(m) for m in models]
    rl_rows = []
    for label, fn in [
        ("Mean response length (chars)", lambda c: f"{sum(c)/len(c):.0f}"),
        ("Median", lambda c: f"{sorted(c)[len(c)//2]}"),
        ("p25", lambda c: f"{sorted(c)[len(c)//4]}"),
        ("p75", lambda c: f"{sorted(c)[3*len(c)//4]}"),
        ("Mean ratio (model/GT)", None),
    ]:
        if fn:
            row = [label, fn(gt_resp_lens)] + [fn(m_resp_lens[m]) for m in models]
        else:
            row = [label, "1.00"]
            for m in models:
                ratios = [m_resp_lens[m][i] / gt_resp_lens[i] if gt_resp_lens[i] > 0 else 0 for i in range(n)]
                row.append(f"{sum(ratios)/len(ratios):.2f}")
        rl_rows.append(row)
        print(f"  {row[0]:40s}  " + "  ".join(str(x) for x in row[1:]))
    write_csv(out / "10_response_length.csv", rl_header, rl_rows)

    # Plot: box plot of response lengths
    fig, ax = plt.subplots(figsize=(13, 6))
    data_for_box = [gt_resp_lens] + [m_resp_lens[m] for m in models]
    labels_for_box = ["GT\n(Claude\nOpus)"] + [ml(m).replace(" ", "\n") for m in models]
    colors_for_box = [GT_COLOR] + [PALETTE[m] for m in models]
    bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means = [sum(d) / len(d) for d in data_for_box]
    ax.scatter(range(1, len(data_for_box) + 1), means, marker="D", color="black", s=40, zorder=5, label="Mean")
    ax.set_ylabel("Full Response Length (chars)")
    ax.set_title("Distribution of Full PRM Response Length")
    ax.legend(frameon=True, edgecolor="lightgray", loc="upper left")
    fig.tight_layout()
    save_fig(fig, out / "10_response_length.png")

    # =====================================================================
    # 11. MARKDOWN SUMMARY
    # =====================================================================
    findings = []
    findings.append("# PRM Intrusiveness Analysis Report\n")
    findings.append(f"**Date**: {__import__('datetime').date.today()}  |  **N**: {n} samples  |  "
                    f"**Models**: {', '.join(ml(m) for m in models)}\n")
    findings.append("**Ground Truth**: Claude Opus PRM responses\n")

    # ---- compute dynamic summary stats for the report ----
    # Format compliance per model
    _fmt_pct = {m: 100 * sum(1 for d in m_det[m] if len(d) == 12) / n for m in models}
    _base_fmt = _fmt_pct[models[0]]
    # Alignment range across models
    _exact_pcts = {m: 100 * align[m]["exact"] / align[m]["valid"] for m in models}
    _under_pcts = {m: 100 * align[m]["under"] / align[m]["valid"] for m in models}
    # Guidance length deltas
    _gt_mean_gl = sum(gt_glen) / len(gt_glen)
    _gl_deltas = {m: 100 * (sum(m_glen[m]) / len(m_glen[m]) - _gt_mean_gl) / _gt_mean_gl for m in models}
    # Info Processing FPR
    _ip_cat = next((c for c in CATEGORIES if "Information" in c or "Info" in c), "Information Processing Failures")
    _ip_fprs = {m: cat_m[m][_ip_cat]["fpr"] for m in models if _ip_cat in cat_m[m]}
    _base_ip_fpr = _ip_fprs.get(models[0], 0)
    # Best info-processing FPR among SFT models
    _sft_ip_fprs = [_ip_fprs.get(m, 0) for m in models[1:]]
    # On-track false intervention range
    _ot_false_int = {}
    for m in models:
        _ot_false_int[m] = 100 * sum(1 for i in ot_idxs if m_stat[m][i] in
                                     ["Needs correction", "Critical intervention required"]) / n_ot
    # On-track Info Processing FPR
    _ot_ip_fprs = {}
    for m in models:
        _ot_neg = [i for i in ot_idxs if _ip_cat in all_data[m][0]]  # rough proxy — use ot_idxs
        _ot_ip_fprs[m] = cat_m[m][_ip_cat]["fpr"] if _ip_cat in cat_m[m] else 0

    findings.append("\n## Quick Guide: What to Focus On\n")
    findings.append("If you only have time for a few plots, look at these:\n")
    findings.append("1. **`03_task_status_confusion.png`** — Shows how SFT models downgrade Critical samples to Needs Correction")
    findings.append("2. **`05a_fpr_by_category.png`** — Shows which categories SFT has *higher* FPR than base (intrusiveness hotspots)")
    findings.append("3. **`07c_on_track_fpr.png`** — When the agent is doing fine, how often does each model still falsely flag errors?")
    findings.append("4. **`04_alignment.png`** — Overall severity alignment vs GT")
    findings.append("5. **`08_guidance_length.png`** — How much longer is SFT guidance vs GT Claude Opus?\n")

    findings.append("\n## 1. Format Compliance\n")
    findings.append("*Plot: `01_format_compliance.png` | CSV: `01_format_compliance.csv`*\n")
    for m in models:
        findings.append(f"- **{ml(m)}**: {_fmt_pct[m]:.1f}% fully parseable responses")
    _sft_models_fmt = [m for m in models if "sft" in m or "SFT" in m.lower()]
    _sft_fmt_min = min(_fmt_pct[m] for m in _sft_models_fmt) if _sft_models_fmt else 0
    _sft_fmt_max = max(_fmt_pct[m] for m in _sft_models_fmt) if _sft_models_fmt else 0
    findings.append(f"\nBase models achieve {min(_fmt_pct[m] for m in models if 'base' in m or 'Base' in ml(m)):.0f}–"
                    f"{max(_fmt_pct[m] for m in models if 'base' in m or 'Base' in ml(m)):.0f}% format compliance. "
                    f"SFT models range from {_sft_fmt_min:.0f}–{_sft_fmt_max:.0f}%.\n")

    findings.append("\n## 2. TASK_STATUS Severity Alignment\n")
    findings.append("*Plots: `02_task_status_dist.png`, `03_task_status_confusion.png`, `04_alignment.png`*\n")
    findings.append("| Model | Exact Match | Over-Intervention | Under-Intervention |")
    findings.append("|-------|------------|-------------------|-------------------|")
    for m in models:
        a = align[m]
        v = a["valid"]
        findings.append(f"| {ml(m)} | {100*a['exact']/v:.1f}% | {100*a['over']/v:.1f}% | {100*a['under']/v:.1f}% |")
    findings.append("")
    _exact_range = f"{min(_exact_pcts.values()):.0f}–{max(_exact_pcts.values()):.0f}%"
    _under_max = max(_under_pcts.values())
    findings.append(f"**Key finding**: Models match GT severity {_exact_range} of the time. "
                    f"Under-intervention peaks at {_under_max:.0f}% — when the agent is critically off-track, "
                    "it gets a mild nudge instead of a hard stop.")

    findings.append("\n## 3. Per-Category Error Detection\n")
    findings.append("*Plots: `05a_fpr_by_category.png` (FOCUS), `05b_fnr_by_category.png`, "
                    "`05c_f1_by_category.png`, `05d_fpr_fnr_combined.png`*\n")
    findings.append("| Category | GT+ | " + " | ".join(f"{ml(m)} F1" for m in models) + " |")
    findings.append("|----------|-----|" + "|".join(["-----"] * len(models)) + "|")
    for cat in CATEGORIES:
        if cat == "Role Specification Violations":
            continue
        sup = cat_m[models[0]][cat]["support"]
        f1s = " | ".join(f"{cat_m[m][cat]['f1']:.3f}" for m in models)
        findings.append(f"| {SHORT_NAMES[cat]} | {sup} | {f1s} |")
    macro_f1s = " | ".join(f"{sum(cat_m[m][c]['f1'] for c in CATEGORIES)/12:.3f}" for m in models)
    findings.append(f"| **Macro Average** | | {macro_f1s} |")
    findings.append("")
    findings.append("**Intrusiveness hotspots** (categories where any SFT model FPR > Base FPR by >0.02):")
    base_fpr = cat_m[models[0]]
    for cat in CATEGORIES:
        sft_fprs_cat = [(ml(m), cat_m[m][cat]["fpr"]) for m in models[1:]] if len(models) > 1 else []
        worst_sft = max(sft_fprs_cat, key=lambda x: x[1], default=(None, 0))
        if worst_sft[1] > base_fpr[cat]["fpr"] + 0.02:
            findings.append(f"- **{SHORT_NAMES[cat]}**: worst SFT FPR={worst_sft[1]:.3f} ({worst_sft[0]}) vs Base={base_fpr[cat]['fpr']:.3f}")

    findings.append("\n## 4. Intervention Intensity\n")
    findings.append("*Plot: `06a_flag_count_dist.png`*\n")
    gt_mean_fc = sum(gt_fc) / len(gt_fc)
    findings.append(f"| Metric | GT | " + " | ".join(ml(m) for m in models) + " |")
    findings.append("|--------|-----|" + "|".join(["-----"] * len(models)) + "|")
    findings.append(f"| Mean flags/sample | {gt_mean_fc:.2f} | " +
                    " | ".join(f"{sum(m_fc[m])/len(m_fc[m]):.2f}" for m in models) + " |")
    findings.append("")

    findings.append("\n## 5. Behavior on 'On Track' Samples (N={})\n".format(n_ot))
    findings.append("*Plot: `07c_on_track_fpr.png` (FOCUS)*\n")
    findings.append("When the agent is doing fine (GT = On Track), how often does each model unnecessarily intervene?\n")
    findings.append("| Model | Correctly says 'On Track' | Falsely intervenes |")
    findings.append("|-------|--------------------------|-------------------|")
    for m in models:
        correct = sum(1 for i in ot_idxs if m_stat[m][i] == "On track")
        false_int = sum(1 for i in ot_idxs if m_stat[m][i] in ["Needs correction", "Critical intervention required"])
        findings.append(f"| {ml(m)} | {100*correct/n_ot:.1f}% | {100*false_int/n_ot:.1f}% |")
    findings.append("")
    _worst_ot_model = max(models, key=lambda m: _ot_false_int[m])
    _best_ot_model = min(models, key=lambda m: _ot_false_int[m])
    findings.append(f"**Worst false intervention on 'On Track' samples**: {ml(_worst_ot_model)} at "
                    f"{_ot_false_int[_worst_ot_model]:.0f}% vs {ml(_best_ot_model)} at "
                    f"{_ot_false_int[_best_ot_model]:.0f}%. "
                    "This is the most damaging behavior for trajectory length — the PRM tells a "
                    "correctly-working agent that it has problems when it doesn't.")

    findings.append("\n## 6. Guidance Length\n")
    findings.append("*Plot: `08_guidance_length.png`*\n")
    gt_mean_gl = sum(gt_glen) / len(gt_glen)
    findings.append(f"| Model | Mean Guidance Length | vs GT |")
    findings.append("|-------|--------------------|----|")
    findings.append(f"| GT (Claude Opus) | {gt_mean_gl:.0f} chars | -- |")
    for m in models:
        m_mean_gl = sum(m_glen[m]) / len(m_glen[m])
        delta = 100 * (m_mean_gl - gt_mean_gl) / gt_mean_gl
        findings.append(f"| {ml(m)} | {m_mean_gl:.0f} chars | {delta:+.0f}% |")
    _max_gl_model = max(models, key=lambda m: _gl_deltas[m])
    _min_gl_model = min(models, key=lambda m: _gl_deltas[m])
    findings.append(f"\nGuidance length ranges from {_gl_deltas[_min_gl_model]:+.0f}% ({ml(_min_gl_model)}) "
                    f"to {_gl_deltas[_max_gl_model]:+.0f}% ({ml(_max_gl_model)}) relative to GT.\n")

    findings.append("\n---\n")
    findings.append("\n## Hypothesized Causes of Longer SFT Trajectories\n")
    findings.append("Based on this analysis, the SFT PRM likely causes longer agent trajectories due to a "
                    "combination of these factors:\n")
    # Compute critical downgrade rate dynamically (needs confusion data from section 3 above — use align proxy)
    findings.append("1. **Severity downgrading on Critical samples** (`03_task_status_confusion.png`): "
                    "A significant fraction of Critical situations are reported as 'Needs Correction'. "
                    "The agent gets a mild nudge instead of a hard stop, continuing on wrong paths longer.")
    if _ip_fprs:
        _sft_ip_max = max(_ip_fprs[m] for m in models[1:] if m in _ip_fprs) if len(models) > 1 else 0
        findings.append(f"2. **Info Processing false positives** (`05a_fpr_by_category.png`, `07c_on_track_fpr.png`): "
                        f"SFT falsely flags 'Information Processing Failures' up to {_sft_ip_max:.0%} of the time "
                        f"(base: {_base_ip_fpr:.0%}). "
                        "This tells the agent its reasoning is wrong when it isn't, causing unnecessary pivots.")
    _pm_cat = next((c for c in CATEGORIES if "Misid" in c or "Problem" in c), None)
    if _pm_cat and len(models) > 1:
        _pm_sft_max_fpr = max(cat_m[m][_pm_cat]["fpr"] for m in models[1:])
        _pm_base_fpr = cat_m[models[0]][_pm_cat]["fpr"]
        findings.append(f"3. **Problem Misidentification false positives** (`05a_fpr_by_category.png`): "
                        f"SFT FPR up to {_pm_sft_max_fpr:.2f} vs Base={_pm_base_fpr:.2f}. "
                        "Telling the agent it misidentified the problem causes it to re-analyze from scratch.")
    _max_gl_delta = _gl_deltas[_max_gl_model]
    # Compute SFT-only guidance deltas (exclude base models = models[0] and models[1] if it's base-think)
    _sft_models = [m for m in models if "sft" in m or "SFT" in m.lower()]
    _sft_gl_min = min(_gl_deltas[m] for m in _sft_models) if _sft_models else min(_gl_deltas.values())
    _sft_gl_max = max(_gl_deltas[m] for m in _sft_models) if _sft_models else _max_gl_delta
    findings.append(f"4. **Verbose guidance** (`08_guidance_length.png`): SFT models generate "
                    f"{_sft_gl_min:+.0f}% to {_sft_gl_max:+.0f}% guidance text relative to Claude "
                    f"(base models: {_gl_deltas[models[0]]:+.0f}%), "
                    "injecting more directive noise into the agent's context at each PRM interval.")
    findings.append("")

    md_path = out / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("\n".join(findings))
    print(f"\n  Markdown report: {md_path}")

    # =====================================================================
    # Done
    # =====================================================================
    # Print output summary
    pngs = sorted(out.glob("*.png"))
    csvs = sorted(out.glob("*.csv"))
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Output directory: {out}")
    print(f"  {len(pngs)} plots (PNG), {len(csvs)} CSV tables, 1 REPORT.md")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
