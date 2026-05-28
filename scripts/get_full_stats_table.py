#!/usr/bin/env python3
"""
Generate full stats table for all settings with:
- Resolved counts and rates
- Avg steps and costs
- Exit status counts
- Split by SFT submission status

All numbers come from traj.json files and report.json.
"""
import argparse
import json
import re
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

SFT_DIRNAME = "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"


def main():
    parser = argparse.ArgumentParser(description="Full stats table from traj.json and report.json")
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
    header = [
        "Setting", "PRM Model",
        "Resolved (/500)", "Res. Rate (/500)", "Res. Rate (/submitted)",
        "Avg Steps (/500)", "Avg Model Cost ($) (/500)", "Avg PRM Cost ($) (/500)", "Avg Total Cost ($) (/500)",
        "ContextWindowExceededError", "LimitsExceeded", "RetryError", "Submitted", "Exit Status Total",
        f"Resolved on SFT-submits (/{n_sub})", f"% on SFT-submits",
        f"Resolved on SFT-not-submits (/{n_nosub})", f"% on SFT-not-submits",
    ]
    print(",".join(header))

    for i, (setting, prm, dirname) in enumerate(RUNS):
        run_dir = results_dir / dirname
        print(f"[{i+1}/{len(RUNS)}] {setting} / {prm} ...", file=sys.stderr, end=" ", flush=True)

        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            print(f"{setting},{prm}," + ",".join(["N/A"] * (len(header) - 2)))
            continue

        # Resolved from report.json
        with open(run_dir / "report.json") as f:
            report = json.load(f)
        resolved_ids = set(report.get("resolved_ids", []))
        resolved = len(resolved_ids)

        # Per-traj stats
        exit_counts = Counter()
        total_steps = 0
        total_cost = 0.0
        total_prm_cost = 0.0
        n_trajs = 0

        for traj_file in run_dir.glob("*/*.traj.json"):
            with open(traj_file) as f:
                traj = json.load(f)
            info = traj.get("info", {})
            exit_counts[info.get("exit_status", "unknown")] += 1

            messages = traj.get("messages", [])
            steps = sum(
                1 for m in messages
                if m.get("role") == "assistant"
                and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
            )
            total_steps += steps

            model_stats = info.get("model_stats") or {}
            total_cost += model_stats.get("instance_cost", 0.0)
            prm_stats = info.get("prm_stats") or {}
            total_prm_cost += prm_stats.get("prm_cost", 0.0)
            n_trajs += 1

        n = n_trajs or 1
        submitted = exit_counts.get("Submitted", 0)
        ctx_err = exit_counts.get("ContextWindowExceededError", 0)
        limits = exit_counts.get("LimitsExceeded", 0)
        retry = exit_counts.get("RetryError", 0)
        exit_total = sum(exit_counts.values())

        # SFT split
        r_sub = len(resolved_ids & sft_submitted)
        r_nosub = len(resolved_ids & sft_not_submitted)

        # Verification
        assert r_sub + r_nosub == resolved, f"SFT split mismatch for {dirname}: {r_sub} + {r_nosub} != {resolved}"
        assert exit_total == n_trajs, f"Exit total mismatch for {dirname}: {exit_total} != {n_trajs}"

        print(f"{n_trajs} trajs, {resolved} resolved, exit_total={exit_total} OK", file=sys.stderr)

        row = [
            setting, prm,
            str(resolved),
            f"{100 * resolved / 500:.2f}",
            f"{100 * resolved / submitted:.2f}" if submitted > 0 else "N/A",
            f"{total_steps / n:.2f}",
            f"{total_cost / n:.4f}",
            f"{total_prm_cost / n:.4f}",
            f"{(total_cost + total_prm_cost) / n:.4f}",
            str(ctx_err), str(limits), str(retry), str(submitted), str(exit_total),
            f"{r_sub}/{n_sub}",
            f"{100 * r_sub / n_sub:.1f}%",
            f"{r_nosub}/{n_nosub}",
            f"{100 * r_nosub / n_nosub:.1f}%",
        ]
        print(",".join(row))

    print("\nAll checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
