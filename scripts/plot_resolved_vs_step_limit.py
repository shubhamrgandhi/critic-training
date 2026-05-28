#!/usr/bin/env python3
"""
Plot resolved instances vs step limit (plateau plot).

For each run, sweeps step limits from 1..max and counts how many instances
are resolved AND completed within that step limit. Produces a smooth
plateau curve like the OpenAI SWE-bench scaling plot.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# ── Non-prefix runs (results_singularity_max_150_steps) ──
NONPREFIX_DIR = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps"
NONPREFIX_RUNS = [
    ("base CWM", "singularity_edit_obs_final_only_0_cwm"),
    ("Qwen3-8b", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("SFT + noisy", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"),
    ("SFT + clean", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("RS-SFT + clean", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"),
]

# ── Prefix runs (results_singularity_max_150_steps_prefix) ──
PREFIX_DIR = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"
PREFIX_RUNS_MINI = [
    ("base CWM (prefix)", "singularity_edit_obs_final_only_0_cwm"),
    ("Qwen3-8b (prefix)", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("SFT + clean (prefix)", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("RS-SFT + clean (prefix)", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"),
]
PREFIX_RUNS_FULL = [
    ("base CWM (prefix)", "singularity_edit_obs_final_only_0_cwm"),
    ("SFT + clean (prefix)", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
]


def traj_steps(traj: dict) -> int:
    return sum(
        1 for m in traj.get("messages", [])
        if m.get("role") == "assistant"
        and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
    )


def load_resolved_steps(run_dir: Path, instance_ids: set[str] | None = None) -> list[int]:
    """Return list of step counts for resolved instances."""
    report_path = run_dir / "report.json"
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            resolved_ids = set(json.load(f).get("resolved_ids", []))

    resolved_steps = []
    for traj_file in sorted(run_dir.glob("*/*.traj.json")):
        instance_id = traj_file.parent.name
        if instance_ids is not None and instance_id not in instance_ids:
            continue
        if instance_id not in resolved_ids:
            continue
        with open(traj_file) as f:
            traj = json.load(f)
        resolved_steps.append(traj_steps(traj))

    return resolved_steps


def make_plateau_plot(parent_dir: Path, runs: list, instance_ids: set[str] | None,
                      title: str, out_filename: str, max_steps: int = 155):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (label, dirname) in enumerate(runs):
        run_dir = parent_dir / dirname
        if not run_dir.exists():
            print(f"  Skipping {label}: not found", file=sys.stderr)
            continue

        resolved_steps = load_resolved_steps(run_dir, instance_ids)
        if not resolved_steps:
            print(f"  Skipping {label}: no resolved instances", file=sys.stderr)
            continue

        resolved_steps.sort()
        step_limits = np.arange(0, max_steps + 1)
        counts = np.array([sum(1 for s in resolved_steps if s <= sl) for sl in step_limits])

        color = colors[i % len(colors)]
        ax.plot(step_limits, counts, label=f"{label} ({len(resolved_steps)})",
                color=color, linewidth=2.2)

    ax.set_xlabel("Step limit", fontsize=12)
    ax.set_ylabel("Resolved instances", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = parent_dir / out_filename
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}", file=sys.stderr)
    plt.close(fig)


def main():
    # Full 500 instance set from base CWM non-prefix
    base_dir = NONPREFIX_DIR / "singularity_edit_obs_final_only_0_cwm"
    full_ids = {p.name for p in base_dir.iterdir() if p.is_dir()}
    print(f"Full set: {len(full_ids)} instances", file=sys.stderr)

    # Mini50 from reference run
    ref_run = (
        "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_"
        "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think"
    )
    ref_dir = NONPREFIX_DIR / ref_run
    mini_ids = {p.name for p in ref_dir.iterdir() if p.is_dir()} if ref_dir.exists() else None

    # ── Non-prefix plots ──
    print("\n=== Non-prefix runs ===", file=sys.stderr)
    if mini_ids:
        print(f"\nMini {len(mini_ids)} subset:", file=sys.stderr)
        make_plateau_plot(
            NONPREFIX_DIR, NONPREFIX_RUNS, mini_ids,
            f"Resolved vs Step Limit (Mini {len(mini_ids)}, Non-prefix)",
            "mini50_resolved_vs_step_limit.png",
        )

    print(f"\nFull {len(full_ids)}:", file=sys.stderr)
    make_plateau_plot(
        NONPREFIX_DIR, NONPREFIX_RUNS, full_ids,
        f"Resolved vs Step Limit (Full {len(full_ids)}, Non-prefix)",
        "full500_resolved_vs_step_limit.png",
    )

    # ── Prefix plots ──
    if PREFIX_DIR.exists():
        print("\n=== Prefix runs ===", file=sys.stderr)
        prefix_base = PREFIX_DIR / "singularity_edit_obs_final_only_0_cwm"
        if prefix_base.exists():
            prefix_ids = {p.name for p in prefix_base.iterdir() if p.is_dir()}
            print(f"Prefix set: {len(prefix_ids)} instances", file=sys.stderr)

            if mini_ids:
                print(f"\nPrefix Mini {len(mini_ids)} subset:", file=sys.stderr)
                make_plateau_plot(
                    PREFIX_DIR, PREFIX_RUNS_MINI, mini_ids,
                    f"Resolved vs Step Limit (Mini {len(mini_ids)}, Prefix)",
                    "mini50_resolved_vs_step_limit.png",
                )

            make_plateau_plot(
                PREFIX_DIR, PREFIX_RUNS_FULL, prefix_ids,
                f"Resolved vs Step Limit (Full {len(prefix_ids)}, Prefix)",
                "full500_resolved_vs_step_limit.png",
            )


if __name__ == "__main__":
    main()
