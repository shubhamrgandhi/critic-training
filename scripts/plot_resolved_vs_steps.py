#!/usr/bin/env python3
"""
Plot cumulative resolved instances vs. trajectory steps.

For each run, sorts resolved instances by their step count and plots
a cumulative curve. Shows whether longer trajectories contribute
meaningfully to resolved count or if it plateaus early.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps"

REFERENCE_RUN = (
    "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_"
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think"
)

RUNS = [
    ("base CWM", "singularity_edit_obs_final_only_0_cwm"),
    ("Qwen3-8b", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("SFT + noisy", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"),
    ("SFT + clean", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("RS-SFT + clean", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"),
]


def traj_steps(traj: dict) -> int:
    return sum(
        1 for m in traj.get("messages", [])
        if m.get("role") == "assistant"
        and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
    )


def get_resolved_steps(run_dir: Path, instance_ids: set[str] | None = None) -> tuple[list[int], list[int], int]:
    """Returns (resolved_steps, unresolved_steps, total_instances)."""
    report_path = run_dir / "report.json"
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            resolved_ids = set(json.load(f).get("resolved_ids", []))

    resolved_steps = []
    unresolved_steps = []
    for traj_file in sorted(run_dir.glob("*/*.traj.json")):
        instance_id = traj_file.parent.name
        if instance_ids is not None and instance_id not in instance_ids:
            continue
        with open(traj_file) as f:
            traj = json.load(f)
        steps = traj_steps(traj)
        if instance_id in resolved_ids:
            resolved_steps.append(steps)
        else:
            unresolved_steps.append(steps)

    return resolved_steps, unresolved_steps, len(resolved_steps) + len(unresolved_steps)


def main():
    # Load mini50 subset
    ref_dir = PARENT_DIR / REFERENCE_RUN
    instance_ids = {p.name for p in ref_dir.iterdir() if p.is_dir()}
    n = len(instance_ids)
    print(f"Mini50 subset: {n} instances", file=sys.stderr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.Set1.colors

    ax1 = axes[0]
    ax2 = axes[1]

    for i, (label, dirname) in enumerate(RUNS):
        run_dir = PARENT_DIR / dirname
        if not run_dir.exists():
            print(f"Skipping {label}: not found", file=sys.stderr)
            continue

        resolved_steps, unresolved_steps, total = get_resolved_steps(run_dir, instance_ids)
        if not resolved_steps:
            print(f"Skipping {label}: no resolved instances", file=sys.stderr)
            continue

        resolved_steps.sort()
        total_resolved = len(resolved_steps)
        cumulative = np.arange(1, total_resolved + 1)

        color = colors[i % len(colors)]

        # Left: absolute cumulative
        ax1.step(resolved_steps, cumulative, where='post', label=f"{label} ({total_resolved} resolved)", color=color, linewidth=2)

        # Right: % of total resolved
        ax2.step(resolved_steps, 100 * cumulative / total_resolved, where='post', label=label, color=color, linewidth=2)

    # Add vertical lines at step thresholds
    for ax in [ax1, ax2]:
        for step_mark in [25, 50, 75, 100, 125, 150]:
            ax.axvline(x=step_mark, color='gray', linestyle=':', alpha=0.3)

    ax1.set_xlabel("Trajectory Steps")
    ax1.set_ylabel("Cumulative Resolved Instances")
    ax1.set_title("Cumulative Resolved vs. Steps")
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 155)
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel("Trajectory Steps")
    ax2.set_ylabel("% of Total Resolved")
    ax2.set_title("% of Resolved Instances by Step Threshold")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 155)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f"Resolved Instances vs. Trajectory Length (Mini {n}, Max 150 Steps)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = PARENT_DIR / "mini50_resolved_vs_steps.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}", file=sys.stderr)

    # Also print summary stats
    print("\n=== Step distribution of resolved instances ===")
    for label, dirname in RUNS:
        run_dir = PARENT_DIR / dirname
        if not run_dir.exists():
            continue
        resolved_steps, _, total = get_resolved_steps(run_dir, instance_ids)
        if not resolved_steps:
            continue
        resolved_steps.sort()
        total_resolved = len(resolved_steps)
        for threshold in [25, 50, 75, 100, 125, 150]:
            count = sum(1 for s in resolved_steps if s <= threshold)
            print(f"  {label}: {count}/{total_resolved} ({100*count/total_resolved:.1f}%) resolved within {threshold} steps")
        print()


if __name__ == "__main__":
    main()
