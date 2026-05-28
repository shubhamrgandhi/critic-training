#!/usr/bin/env python3
"""
Get exit status counts from traj.json files for each run directory.
Uses info.exit_status from traj.json (more reliable than yaml files).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


RUNS = [
    ("base", "-", "singularity_edit_obs_final_only_0_cwm"),
    ("prm_issue_res_k5", "claude-opus-4-6", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"),
    ("prm_issue_res_k5", "cwm", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_cwm"),
    ("prm_issue_res_k5", "qwen25coder7b", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen25coder7b"),
    ("prm_issue_res_k5", "sweagent7b", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_sweagent7b"),
    ("prm_issue_res_k5", "qwen3-8b", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("prm_issue_res_k5", "qwen3-8b (sft)", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"),
    ("prm_tool_issue_res_k5", "claude-opus-4-6", "singularity_edit_obs_final_only_prm_tool_issue_res_k5_0_cwm_prm_claude-opus-4-6"),
    ("prm_tool_k5", "claude-opus-4-6", "singularity_edit_obs_final_only_prm_tool_k5_0_cwm_prm_claude-opus-4-6"),
]


def get_exit_statuses_from_trajs(run_dir: Path) -> Counter:
    """Read exit_status from each traj.json file."""
    counts = Counter()
    for traj_file in run_dir.glob("*/*.traj.json"):
        try:
            with open(traj_file) as f:
                traj = json.load(f)
            status = traj.get("info", {}).get("exit_status", "unknown")
            counts[status] += 1
        except Exception as e:
            counts["parse_error"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Exit status table from traj.json files")
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity"))
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)

    # Collect all statuses and per-run data
    all_statuses = set()
    run_data = []
    for i, (setting, prm, dirname) in enumerate(RUNS):
        run_dir = parent_dir / dirname
        print(f"[{i+1}/{len(RUNS)}] {setting} / {prm} ...", file=sys.stderr, end=" ", flush=True)
        if not run_dir.exists():
            print("NOT FOUND, skipping", file=sys.stderr)
            run_data.append((setting, prm, Counter()))
            continue
        counts = get_exit_statuses_from_trajs(run_dir)
        total = sum(counts.values())
        print(f"{total} trajectories", file=sys.stderr)
        for s in counts:
            all_statuses.add(s)
        run_data.append((setting, prm, counts))

    sorted_statuses = sorted(all_statuses)

    # Print CSV
    header = "Setting,PRM Model," + ",".join(sorted_statuses) + ",Total"
    print(header)
    for setting, prm, counts in run_data:
        total = sum(counts.values())
        cells = [str(counts.get(s, 0)) for s in sorted_statuses]
        print(f"{setting},{prm}," + ",".join(cells) + f",{total}")


if __name__ == "__main__":
    main()
