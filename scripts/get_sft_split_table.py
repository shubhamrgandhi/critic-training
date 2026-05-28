#!/usr/bin/env python3
"""
Generate resolved counts split by SFT submission status.
Columns: Setting, PRM Model, SFT submits, SFT doesn't submit, Total
"""
import argparse
import json
import sys
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

SFT_DIRNAME = "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"


def main():
    parser = argparse.ArgumentParser(description="SFT split table from traj.json and report.json")
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity"))
    args = parser.parse_args()
    results_dir = Path(args.parent_dir)

    # Build SFT split
    sft_dir = results_dir / SFT_DIRNAME
    print(f"Loading SFT exit statuses from {sft_dir.name} ...", file=sys.stderr)
    sft_statuses = {}
    for traj_file in sft_dir.glob("*/*.traj.json"):
        instance = traj_file.parent.name
        with open(traj_file) as f:
            sft_statuses[instance] = json.load(f).get("info", {}).get("exit_status", "unknown")

    sft_submitted = {i for i, s in sft_statuses.items() if s == "Submitted"}
    sft_not_submitted = {i for i, s in sft_statuses.items() if s != "Submitted"}
    n_sub = len(sft_submitted)
    n_nosub = len(sft_not_submitted)
    assert n_sub + n_nosub == 500, f"SFT split doesn't sum to 500: {n_sub} + {n_nosub}"
    print(f"  SFT submitted: {n_sub}, not submitted: {n_nosub}", file=sys.stderr)

    # Header
    print(f"Setting,PRM Model,SFT submits (/{n_sub}),SFT doesn't submit (/{n_nosub}),Total (/500)")

    for i, (setting, prm, dirname) in enumerate(RUNS):
        run_dir = results_dir / dirname
        print(f"[{i+1}/{len(RUNS)}] {setting} / {prm} ...", file=sys.stderr, end=" ", flush=True)

        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            print(f"{setting},{prm},N/A,N/A,N/A")
            continue

        with open(run_dir / "report.json") as f:
            report = json.load(f)
        resolved_ids = set(report.get("resolved_ids", []))
        resolved = len(resolved_ids)

        r_sub = len(resolved_ids & sft_submitted)
        r_nosub = len(resolved_ids & sft_not_submitted)
        assert r_sub + r_nosub == resolved, f"Split mismatch for {dirname}: {r_sub} + {r_nosub} != {resolved}"

        print(f"{resolved} resolved OK", file=sys.stderr)
        print(f"{setting},{prm},{r_sub},{r_nosub},{resolved}")

    print("\nAll checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
