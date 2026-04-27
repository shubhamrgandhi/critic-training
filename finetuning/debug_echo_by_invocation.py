#!/usr/bin/env python3
"""Analyze echo rate by invocation number and context length."""

import json
import glob
import os


def analyze_run(run_dir, label, max_instances=500):
    traj_files = sorted(glob.glob(os.path.join(run_dir, "*/*.traj.json")))

    echo_by_inv = {}
    total_by_inv = {}
    echo_lengths = []
    valid_lengths = []

    for traj_path in traj_files[:max_instances]:
        with open(traj_path) as f:
            traj = json.load(f)

        fl = traj.get("info", {}).get("prm_stats", {}).get("prm_feedback_log", [])
        for entry in fl:
            fb = entry.get("feedback", "")
            inv = entry.get("invocation", 0)

            is_echo = "[USER/ENVIRONMENT]" in fb or "[AGENT RESPONSE]" in fb
            # Also check for code-only echo (bash blocks without PRM format)
            if not is_echo:
                import re
                has_bash = bool(re.search(r"```bash\n.+?\n```", fb, re.DOTALL))
                has_prm = "DETECTED:" in fb or "TASK_STATUS:" in fb
                if has_bash and not has_prm:
                    is_echo = True

            total_by_inv[inv] = total_by_inv.get(inv, 0) + 1
            if is_echo:
                echo_by_inv[inv] = echo_by_inv.get(inv, 0) + 1
                echo_lengths.append(len(fb))
            else:
                valid_lengths.append(len(fb))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Echo rate by PRM invocation number:")
    for inv in sorted(total_by_inv.keys())[:30]:
        echoes = echo_by_inv.get(inv, 0)
        total = total_by_inv[inv]
        rate = 100 * echoes / total if total else 0
        bar = "#" * int(rate / 2)
        print(f"  Inv #{inv:2d}: {echoes:4d}/{total:4d} = {rate:5.1f}% {bar}")

    if echo_lengths and valid_lengths:
        echo_lengths.sort()
        valid_lengths.sort()
        ne = len(echo_lengths)
        nv = len(valid_lengths)
        print(f"\n  Echo fb length:  mean={sum(echo_lengths)//ne}, "
              f"median={echo_lengths[ne//2]}, "
              f"p90={echo_lengths[int(ne*0.9)]}")
        print(f"  Valid fb length: mean={sum(valid_lengths)//nv}, "
              f"median={valid_lengths[nv//2]}, "
              f"p90={valid_lengths[int(nv*0.9)]}")


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
            print(f"SKIPPED: {path}")
