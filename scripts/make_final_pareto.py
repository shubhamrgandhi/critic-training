#!/usr/bin/env python3
"""
Final paper Pareto plot — 3 points, full500, with-fallback.

Same visual recipe as the working appendix/pareto_minimal.png.
"""
import csv
import json
import os
from pathlib import Path

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

# Hard-coded numbers (computed above; full500 with-fallback)
points = [
    {
        "label": "CWM base\n(no critic)",
        "resolved": 144, "submitted": 445, "cost": 0.181,
        "color": "#666666", "marker": "s",
        "dx": 0, "dy": 24, "ha": "center", "va": "bottom",
    },
    {
        "label": "CWM + SFT Qwen3-8B critic\n(ours, with base fallback)",
        "resolved": 165, "submitted": 462, "cost": 0.478,
        "color": "#e6b800", "marker": "D",
        "dx": 0, "dy": 28, "ha": "center", "va": "bottom",
    },
    {
        "label": "CWM + Opus critic\n(concise prompt, with base fallback)",
        "resolved": 257, "submitted": 492, "cost": 1.187,
        "color": "#d62728", "marker": "o",
        "dx": 0, "dy": 28, "ha": "center", "va": "bottom",
    },
]

n = 500

fig, ax = plt.subplots(figsize=(10, 7))

# Pareto frontier (sorted by cost, monotone increasing in resolved)
sorted_pts = sorted(points, key=lambda p: p["cost"])
fx, fy = [], []
best = -1
for p in sorted_pts:
    if p["resolved"] > best:
        fx.append(p["cost"])
        fy.append(p["resolved"])
        best = p["resolved"]
ax.plot(fx, fy, color="#444444", linestyle="-", linewidth=1.5,
        alpha=0.55, zorder=1, label="Pareto frontier")

# Scatter + annotate
for p in points:
    ax.scatter([p["cost"]], [p["resolved"]],
               c=p["color"], marker=p["marker"], s=240,
               edgecolors="black", linewidths=1.2, zorder=3)
    ax.annotate(f"{p['label']}\n({p['resolved']}/{n} resolved)",
                (p["cost"], p["resolved"]),
                xytext=(p["dx"], p["dy"]), textcoords="offset points",
                fontsize=11, ha=p["ha"], va=p["va"], color="black",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=p["color"], lw=1.3, alpha=0.97),
                arrowprops=dict(arrowstyle="-", color=p["color"], lw=1.2))

ax.set_xlabel("Avg total cost per instance ($)", fontsize=11)
ax.set_ylabel(f"Instances resolved (out of {n})", fontsize=11)
ax.set_title("Critic cost vs. performance — SWE-Bench Verified (full 500)", fontsize=12)
ax.grid(True, which="both", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

max_y = max(p["resolved"] for p in points)
ax.set_ylim(0, max_y + 80)
max_x = max(p["cost"] for p in points)
ax.set_xlim(0, max_x * 1.18)

ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.tight_layout()

fig.savefig(OUT / "final_pareto_full500.png", bbox_inches="tight")
fig.savefig(OUT / "final_pareto_full500.pdf", bbox_inches="tight")
print(f"Saved: {OUT / 'final_pareto_full500'}.{{png,pdf}}")
