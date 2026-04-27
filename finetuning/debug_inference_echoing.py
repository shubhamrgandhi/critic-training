#!/usr/bin/env python3
"""Analyze PRM feedback quality at inference time for trajectory echoing."""

import json
import re
import glob
import os
import sys
from collections import Counter


def classify_feedback(feedback):
    """Classify a feedback string into categories."""
    if not feedback or not feedback.strip():
        return "empty"

    has_detected = "DETECTED:" in feedback
    has_task_status = "TASK_STATUS:" in feedback
    has_overall_guidance = "OVERALL_GUIDANCE:" in feedback
    has_user_env = "[USER/ENVIRONMENT]" in feedback
    has_agent_resp = "[AGENT RESPONSE]" in feedback
    has_bash_block = bool(re.search(r"```bash\n.+?\n```", feedback, re.DOTALL))

    # Trajectory echoing: reproduces agent/environment turns
    if has_user_env or has_agent_resp:
        return "echo_trajectory"

    # Has proper PRM format
    if has_detected and (has_task_status or has_overall_guidance):
        if has_bash_block:
            return "valid_with_code"  # Valid feedback that cites commands
        return "valid"

    # Has bash blocks but no PRM format markers
    if has_bash_block and not has_detected:
        return "echo_code_only"

    # Partial format
    if has_detected:
        return "partial_format"

    return "unknown"


def analyze_run(run_dir, label, max_instances=500):
    traj_files = sorted(glob.glob(os.path.join(run_dir, "*/*.traj.json")))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {run_dir}")
    print(f"{'='*70}")
    print(f"Total trajectories: {len(traj_files)}")

    if not traj_files:
        print("  No trajectories found!")
        return

    category_counts = Counter()
    total_feedbacks = 0
    echo_instances = []
    all_feedback_lengths = []

    for traj_path in traj_files[:max_instances]:
        instance_id = os.path.basename(os.path.dirname(traj_path))
        with open(traj_path) as f:
            traj = json.load(f)

        prm_stats = traj.get("info", {}).get("prm_stats", {})
        feedback_log = prm_stats.get("prm_feedback_log", [])

        instance_echo = 0
        for entry in feedback_log:
            feedback = entry.get("feedback", "")
            total_feedbacks += 1
            all_feedback_lengths.append(len(feedback))

            cat = classify_feedback(feedback)
            category_counts[cat] += 1

            if cat.startswith("echo"):
                instance_echo += 1

        if instance_echo > 0:
            echo_instances.append((instance_id, instance_echo, len(feedback_log)))

    print(f"Total feedbacks analyzed: {total_feedbacks}")
    print(f"\n--- Feedback classification ---")
    for cat in ["valid", "valid_with_code", "partial_format", "echo_trajectory",
                "echo_code_only", "unknown", "empty"]:
        count = category_counts.get(cat, 0)
        pct = 100 * count / total_feedbacks if total_feedbacks else 0
        print(f"  {cat:25s}: {count:5d} ({pct:5.1f}%)")

    echo_total = sum(v for k, v in category_counts.items() if k.startswith("echo"))
    print(f"  {'TOTAL ECHO':25s}: {echo_total:5d} ({100*echo_total/total_feedbacks:.1f}%)")

    if all_feedback_lengths:
        sorted_lens = sorted(all_feedback_lengths)
        n = len(sorted_lens)
        print(f"\n--- Feedback length stats ---")
        print(f"  Mean: {sum(all_feedback_lengths)/n:.0f}, Median: {sorted_lens[n//2]}")
        print(f"  P90: {sorted_lens[int(n*0.9)]}, Max: {max(all_feedback_lengths)}")

    print(f"\n--- Instances with echoing: {len(echo_instances)}/{min(max_instances, len(traj_files))} ---")
    for iid, ec, total in echo_instances[:15]:
        print(f"  {iid}: {ec}/{total} feedbacks are echo")

    # Show examples of each echo type
    for echo_type in ["echo_trajectory", "echo_code_only"]:
        if category_counts.get(echo_type, 0) > 0:
            for traj_path in traj_files[:max_instances]:
                instance_id = os.path.basename(os.path.dirname(traj_path))
                with open(traj_path) as f:
                    traj = json.load(f)
                feedback_log = traj.get("info", {}).get("prm_stats", {}).get("prm_feedback_log", [])
                for entry in feedback_log:
                    feedback = entry.get("feedback", "")
                    if classify_feedback(feedback) == echo_type:
                        print(f"\n--- EXAMPLE: {echo_type} (from {instance_id}) ---")
                        print(feedback[:1000])
                        if len(feedback) > 1000:
                            print("...")
                        break
                else:
                    continue
                break

    return echo_total, total_feedbacks


if __name__ == "__main__":
    base = "/home/srgandhi/tool-overuse/results_singularity_max_150_steps"

    runs = [
        ("SFT + clean",
         f"{base}/singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
        ("RS-SFT + clean",
         f"{base}/singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"),
    ]

    for label, path in runs:
        if os.path.exists(path):
            analyze_run(path, label)
        else:
            print(f"\nSKIPPED (not found): {path}")