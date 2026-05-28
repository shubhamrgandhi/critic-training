#!/usr/bin/env python3
"""
Three main plots for the paper / advisor presentation.

Visual recipes (copied from the working appendix versions):
  figure1_headline:        like appendix/headline.png  (stacked bars, base+fallback)
  figure2_pareto:          like appendix/pareto_minimal.png  (4-point scatter)
  figure3_prompt_ablation: like appendix/opus_ablation.png  (grouped k=5/k=10 bars)

Differences from the appendix versions:
  - Opus baseline = concise prompt everywhere (was the user's request: detailed
    prompt is "backseat driving", not the right comparable for a small student).
  - figure 1 shows only 4 hand-picked bars instead of every run.
"""
import csv
import json
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"
OUT = RESULTS / "plots" / "main"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── Shared helpers (copied verbatim from make_plots.py) ───

def lighten(color, amount=0.55):
    rgb = np.array(mc.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def stacked_bar(ax, xs, prm_only, combined, width, color, hatch_lighten=True):
    """Draw bars: solid = prm_only, hatched = (combined - prm_only) on top."""
    light = lighten(color) if hatch_lighten else color
    for xi, p, c in zip(xs, prm_only, combined):
        if np.isnan(c) and np.isnan(p):
            continue
        if np.isnan(p):
            ax.bar(xi, c, width, color=color, edgecolor="black", linewidth=0.5)
            continue
        ax.bar(xi, p, width, color=color, edgecolor="black", linewidth=0.5)
        gain = (c if not np.isnan(c) else p) - p
        if gain > 0.01:
            ax.bar(xi, gain, width, bottom=p, color=light, edgecolor="black",
                   linewidth=0.5, hatch="//")


def annotate_tops(ax, xs, prm_only, combined, submitted_prm, submitted_combined,
                  benchmark_n, fontsize=9, dy=0.012):
    """Bold label above bar = combined resolved/submitted; white in-bar = critic-only."""
    ymax_offset = benchmark_n * dy
    for xi, p, c, sp, sc in zip(xs, prm_only, combined, submitted_prm, submitted_combined):
        top = c if not np.isnan(c) else p
        if np.isnan(top):
            continue
        if sc is None or np.isnan(sc):
            label_top = f"{int(top)}"
        else:
            label_top = f"{int(top)}/{int(sc)}"
        ax.text(xi, top + ymax_offset, label_top, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")
        if not np.isnan(p) and not np.isnan(sp) and p > benchmark_n * 0.05:
            ax.text(xi, p / 2, f"{int(p)}/{int(sp)}", ha="center", va="center",
                    fontsize=fontsize - 1, color="white", fontweight="bold")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {OUT / name}.{{png,pdf}}")
    plt.close(fig)


# ─── CSV-side helpers (we read from the unified CSVs) ───

def find_row(rows, predicate):
    return next((r for r in rows if predicate(r)), None)


def get(row, n, key):
    """Return int(row[f'{key} (/{n})']) or NaN."""
    if row is None:
        return float("nan")
    try:
        v = row[f"{key} (/{n})"]
        return int(v) if v not in ("", "N/A") else float("nan")
    except (KeyError, ValueError, TypeError):
        return float("nan")


def get_float(row, n, key):
    if row is None:
        return float("nan")
    try:
        v = row[f"{key} (/{n})"]
        return float(v) if v not in ("", "N/A") else float("nan")
    except (KeyError, ValueError, TypeError):
        return float("nan")


# ─── For each headline point, get BOTH "no fallback" and "+ base fallback" rows ───

HEADLINE_PREDICATES_NO_FB = {
    "base":      lambda r: r["Critic Model"] == "Base CWM (run 0)",
    "untrained": lambda r: "Qwen3-8B (base)" in r["Critic Model"] and "+ base fallback" not in r["Critic Model"],
    "sft":       lambda r: "r2esb-detailed-k5-mt" in r["Critic Model"] and "+ base fallback" not in r["Critic Model"],
    "opus":      lambda r: r["Category"] == "Opus Critic"
                           and "+ base fallback" not in r["Critic Model"]
                           and "concise-k5" in r["Critic Model"],
}

HEADLINE_PREDICATES_FB = {
    "base":      lambda r: r["Critic Model"] == "Base CWM + run-1 fallback",
    "untrained": lambda r: "Qwen3-8B (base)" in r["Critic Model"] and "+ base fallback" in r["Critic Model"],
    "sft":       lambda r: "r2esb-detailed-k5-mt" in r["Critic Model"] and "+ base fallback" in r["Critic Model"],
    "opus":      lambda r: r["Category"] == "Opus Critic"
                           and "+ base fallback" in r["Critic Model"]
                           and "concise-k5" in r["Critic Model"],
}

POINT_STYLE = {
    "base":      {"label": "CWM base\n(no critic)",                     "color": "#666666", "marker": "s"},
    "untrained": {"label": "CWM + Qwen3-8B critic\n(untrained)",        "color": "#9467bd", "marker": "v"},
    "sft":       {"label": "CWM + SFT Qwen3-8B critic\n(ours)",         "color": "#e6b800", "marker": "D"},
    "opus":      {"label": "CWM + Opus critic\n(concise prompt)",       "color": "#d62728", "marker": "o"},
}


def headline_data(rows, n):
    """Return list of dicts: per point, both PRM-only and combined values."""
    out = []
    for key in ("base", "untrained", "sft", "opus"):
        row_nofb = find_row(rows, HEADLINE_PREDICATES_NO_FB[key])
        row_fb   = find_row(rows, HEADLINE_PREDICATES_FB[key])
        # Base + fallback row name is "Base CWM + run-1 fallback" (different from PRM cases)
        st = POINT_STYLE[key]
        out.append({
            "key": key,
            "label": st["label"],
            "color": st["color"],
            "marker": st["marker"],
            "prm_resolved":     get(row_nofb, n, "Resolved"),
            "prm_submitted":    get(row_nofb, n, "Submitted"),
            "comb_resolved":    get(row_fb, n, "Resolved"),
            "comb_submitted":   get(row_fb, n, "Submitted"),
            "comb_cost":        get_float(row_fb, n, "Avg Total Cost ($)"),
        })
    return out


# =====================================================================
# FIGURE 1 — Headline bar chart, mini50 only
# =====================================================================

def figure1_headline():
    rows = list(csv.DictReader(open(RESULTS / "mini50_all.csv")))
    n = 50
    pts = headline_data(rows, n)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(pts))
    width = 0.6
    for xi, p in zip(x, pts):
        # Stacked bar: solid critic-only + hatched fallback gain
        po = [p["prm_resolved"]]
        co = [p["comb_resolved"]]
        # For the Base CWM case, "PRM-only" doesn't really apply; just use solid bar at run-0 level
        if p["key"] == "base":
            # Base CWM run 0 = 13/47, base + run-1 fallback = 14/49.
            # Treat run 0 as "base only", run-1 fallback as the "+ fallback" gain.
            stacked_bar(ax, [xi], po, co, width, p["color"])
        else:
            stacked_bar(ax, [xi], po, co, width, p["color"])

    annotate_tops(ax,
                  x,
                  [p["prm_resolved"]    for p in pts],
                  [p["comb_resolved"]   for p in pts],
                  [p["prm_submitted"]   for p in pts],
                  [p["comb_submitted"]  for p in pts],
                  benchmark_n=n)

    ax.set_xticks(x)
    ax.set_xticklabels([p["label"] for p in pts], fontsize=10)
    ax.set_ylim(0, n)
    ax.set_yticks(np.arange(0, n + 1, 5))
    ax.set_ylabel("Instances resolved (out of 50)")
    ax.set_title("Critic-guided CWM agent on SWE-Bench Verified Mini\n"
                 "solid = critic-only resolved   ·   hatched = additional gain from base fallback\n"
                 "labels: bold above bar = resolved/submitted with fallback   ·   white inside bar = resolved/submitted critic-only",
                 fontsize=10)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "figure1_headline")


# =====================================================================
# FIGURE 2 — Pareto plot (4 points, linear x, recipe from pareto_minimal)
# =====================================================================

def figure2_pareto():
    rows = list(csv.DictReader(open(RESULTS / "mini50_all.csv")))
    n = 50
    pts = headline_data(rows, n)

    OFFSETS = {
        "base":      (0,  24, "center", "bottom"),
        "untrained": (0, -24, "center", "top"),
        "sft":       (0,  28, "center", "bottom"),
        "opus":      (0,  28, "center", "bottom"),
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pareto frontier line (use combined resolved + cost)
    sorted_pts = sorted(pts, key=lambda p: p["comb_cost"])
    fx, fy = [], []
    best_r = -1
    for p in sorted_pts:
        if p["comb_resolved"] > best_r:
            fx.append(p["comb_cost"])
            fy.append(p["comb_resolved"])
            best_r = p["comb_resolved"]
    ax.plot(fx, fy, color="#444444", linestyle="-", linewidth=1.5,
            alpha=0.55, zorder=1, label="Pareto frontier")

    for p in pts:
        dx, dy, ha, va = OFFSETS[p["key"]]
        ax.scatter([p["comb_cost"]], [p["comb_resolved"]], c=p["color"],
                   marker=p["marker"], s=240,
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(f"{p['label']}\n({int(p['comb_resolved'])}/{n} resolved)",
                    (p["comb_cost"], p["comb_resolved"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=11, ha=ha, va=va, color="black",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=p["color"], lw=1.3, alpha=0.97),
                    arrowprops=dict(arrowstyle="-", color=p["color"], lw=1.2))

    ax.set_xlabel("Avg total cost per instance ($)", fontsize=11)
    ax.set_ylabel("Instances resolved (out of 50, with base fallback)", fontsize=11)
    ax.set_title("Critic cost vs. performance — SWE-Bench Verified Mini", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    max_y = max(p["comb_resolved"] for p in pts)
    ax.set_ylim(-2, max_y + 12)
    max_x = max(p["comb_cost"] for p in pts)
    ax.set_xlim(0, max_x * 1.18)

    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "figure2_pareto")


# =====================================================================
# FIGURE 3 — Opus prompt ablation, exact recipe of appendix/opus_ablation.png
# =====================================================================

def figure3_prompt_ablation():
    """Two panels (mini50 + full500), grouped k=5 vs k=10 bars at each prompt."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    panels = []
    for label, n, csv_name in [
        ("SWE-Bench Verified Mini (50)", 50, "mini50_all.csv"),
        ("SWE-Bench Verified Full (500)", 500, "full500_all.csv"),
    ]:
        rows = list(csv.DictReader(open(RESULTS / csv_name)))
        # Pull base / base+fb numbers
        base_row = find_row(rows, lambda r: r["Critic Model"] == "Base CWM (run 0)")
        fb_row   = find_row(rows, lambda r: r["Critic Model"] == "Base CWM + run-1 fallback")
        base_r = get(base_row, n, "Resolved")
        base_s = get(base_row, n, "Submitted")
        fb_r   = get(fb_row, n, "Resolved") if fb_row else float("nan")
        fb_s   = get(fb_row, n, "Submitted") if fb_row else float("nan")

        # Pull Opus prompt × k matrix
        def opus(infer):
            row_nofb = find_row(rows, lambda r: r["Critic Model"] == f"Opus, run: {infer}")
            row_fb   = find_row(rows, lambda r: r["Critic Model"] == f"Opus, run: {infer} + base fallback")
            return {
                "po": get(row_nofb, n, "Resolved"),
                "co": get(row_fb,   n, "Resolved"),
                "sp": get(row_nofb, n, "Submitted"),
                "sc": get(row_fb,   n, "Submitted"),
            }

        configs = {
            ("detailed", "5"):  opus("detailed-k5"),
            ("detailed", "10"): opus("detailed-k10"),
            ("concise", "5"):   opus("concise-k5"),
            ("concise", "10"):  opus("concise-k10"),
        }
        panels.append((label, n, base_r, base_s, fb_r, fb_s, configs))

    for ax, (panel_label, n, base_r, base_s, fb_r, fb_s, configs) in zip(axes, panels):
        prompts = ["detailed", "concise"]
        ks = ["5", "10"]
        x = np.arange(len(prompts))
        width = 0.35

        for ki, k in enumerate(ks):
            po, co, sp, sc = [], [], [], []
            for p in prompts:
                d = configs[(p, k)]
                po.append(d["po"]); co.append(d["co"])
                sp.append(d["sp"]); sc.append(d["sc"])
            color = "#d62728" if k == "5" else "#ff9896"
            xs = x + (ki - 0.5) * width
            stacked_bar(ax, xs, po, co, width, color)
            annotate_tops(ax, xs, po, co, sp, sc, benchmark_n=n)

        # Reference lines
        if not np.isnan(base_r):
            ax.axhline(base_r, color="#666666", linestyle=":", linewidth=1.2)
        if not np.isnan(fb_r) and not np.isnan(base_r) and fb_r != base_r:
            ax.axhline(fb_r, color="#666666", linestyle="--", linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(prompts)
        ax.set_ylim(0, n)
        ax.set_ylabel(f"Instances resolved (out of {n})")
        ax.set_title(panel_label)
        ax.set_xlabel("Teacher prompt")
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        step = 5 if n == 50 else 50
        ax.set_yticks(np.arange(0, n + 1, step))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#d62728", edgecolor="black", label="k=5 (critic only)"),
        Patch(facecolor=lighten("#d62728"), edgecolor="black", hatch="//", label="k=5 (+ fallback)"),
        Patch(facecolor="#ff9896", edgecolor="black", label="k=10 (critic only)"),
        Patch(facecolor=lighten("#ff9896"), edgecolor="black", hatch="//", label="k=10 (+ fallback)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2, label="Base CWM"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label="Base + run-1 fallback (mini50)"),
    ]
    fig.suptitle("Opus teacher-prompt ablation: detailed prompt boosts Opus by spoonfeeding solutions\n"
                 "(detailed-prompt feedback isn't behavior we want to distill into a small student)",
                 y=0.99, fontsize=11)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.86])
    save(fig, "figure3_prompt_ablation")


figure1_headline()
figure2_pareto()
figure3_prompt_ablation()

print(f"\nMain plots written to: {OUT}")
