#!/usr/bin/env python3
"""Debug training data quality for PRM SFT datasets."""

import json
import re
import sys
from pathlib import Path
from collections import Counter


def analyze_dataset(dataset_path: str, label: str):
    with open(dataset_path) as f:
        lines = f.readlines()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {dataset_path}")
    print(f"{'='*60}")
    print(f"Total samples: {len(lines)}")

    response_lengths = []
    has_detected = 0
    has_task_status = 0
    has_overall_guidance = 0

    # Echo patterns
    multiline_bash_blocks = 0
    inline_bash_mention = 0
    has_user_env_tag = 0
    has_agent_resp_tag = 0

    # Quality issues
    very_long = 0  # >8000 chars
    guidance_lengths = []

    for i, line in enumerate(lines):
        sample = json.loads(line)
        msgs = sample["messages"]
        content = msgs[-1]["content"]
        response_lengths.append(len(content))

        if "DETECTED:" in content:
            has_detected += 1
        if "TASK_STATUS:" in content:
            has_task_status += 1
        if "OVERALL_GUIDANCE:" in content:
            has_overall_guidance += 1

        # Multi-line bash code blocks (actual echoing)
        bash_blocks = re.findall(r"```bash\n.+?\n```", content, re.DOTALL)
        if bash_blocks:
            multiline_bash_blocks += 1
        elif "```bash" in content:
            inline_bash_mention += 1

        if "[USER/ENVIRONMENT]:" in content:
            has_user_env_tag += 1
        if "[AGENT RESPONSE]:" in content:
            has_agent_resp_tag += 1

        if len(content) > 8000:
            very_long += 1

        # Extract OVERALL_GUIDANCE length
        if "OVERALL_GUIDANCE:" in content:
            guid_idx = content.index("OVERALL_GUIDANCE:")
            guidance = content[guid_idx + len("OVERALL_GUIDANCE:"):].strip()
            guidance_lengths.append(len(guidance))

    print(f"\n--- Format compliance ---")
    print(f"Has DETECTED: {has_detected}/{len(lines)} ({100*has_detected/len(lines):.1f}%)")
    print(f"Has TASK_STATUS: {has_task_status}/{len(lines)} ({100*has_task_status/len(lines):.1f}%)")
    print(f"Has OVERALL_GUIDANCE: {has_overall_guidance}/{len(lines)} ({100*has_overall_guidance/len(lines):.1f}%)")

    print(f"\n--- Potential echo patterns ---")
    print(f"Multi-line ```bash blocks: {multiline_bash_blocks} ({100*multiline_bash_blocks/len(lines):.1f}%)")
    print(f"Inline bash mention only: {inline_bash_mention} ({100*inline_bash_mention/len(lines):.1f}%)")
    print(f"[USER/ENVIRONMENT]: tag: {has_user_env_tag} ({100*has_user_env_tag/len(lines):.1f}%)")
    print(f"[AGENT RESPONSE]: tag: {has_agent_resp_tag} ({100*has_agent_resp_tag/len(lines):.1f}%)")

    print(f"\n--- Response lengths ---")
    sorted_lens = sorted(response_lengths)
    print(f"Mean: {sum(response_lengths)/len(response_lengths):.0f} chars")
    print(f"Median: {sorted_lens[len(sorted_lens)//2]}")
    print(f"P25: {sorted_lens[len(sorted_lens)//4]}")
    print(f"P75: {sorted_lens[3*len(sorted_lens)//4]}")
    print(f"P90: {sorted_lens[int(len(sorted_lens)*0.9)]}")
    print(f"P95: {sorted_lens[int(len(sorted_lens)*0.95)]}")
    print(f"Min: {min(response_lengths)}, Max: {max(response_lengths)}")
    print(f"Very long (>8000 chars): {very_long}")

    if guidance_lengths:
        sorted_gl = sorted(guidance_lengths)
        print(f"\n--- OVERALL_GUIDANCE lengths ---")
        print(f"Mean: {sum(guidance_lengths)/len(guidance_lengths):.0f} chars")
        print(f"Median: {sorted_gl[len(sorted_gl)//2]}")
        print(f"P90: {sorted_gl[int(len(sorted_gl)*0.9)]}")

    # Check the input side (user message lengths)
    input_lengths = []
    for line in lines:
        sample = json.loads(line)
        user_content = sample["messages"][1]["content"]
        input_lengths.append(len(user_content))

    sorted_il = sorted(input_lengths)
    print(f"\n--- Input (user) lengths ---")
    print(f"Mean: {sum(input_lengths)/len(input_lengths):.0f} chars")
    print(f"Median: {sorted_il[len(sorted_il)//2]}")
    print(f"P90: {sorted_il[int(len(sorted_il)*0.9)]}")
    print(f"Max: {max(input_lengths)}")

    # Check how many DETECTED:Yes categories per sample
    yes_counts = []
    for line in lines:
        sample = json.loads(line)
        content = sample["messages"][-1]["content"]
        yes_count = content.count("DETECTED: Yes")
        yes_counts.append(yes_count)

    yes_dist = Counter(yes_counts)
    print(f"\n--- Error categories detected per sample ---")
    for k in sorted(yes_dist.keys()):
        print(f"  {k} errors detected: {yes_dist[k]} samples ({100*yes_dist[k]/len(lines):.1f}%)")

    # Show one example with multiline bash if any
    if multiline_bash_blocks > 0:
        for line in lines:
            sample = json.loads(line)
            content = sample["messages"][-1]["content"]
            if re.search(r"```bash\n.+?\n```", content, re.DOTALL):
                print(f"\n--- EXAMPLE: sample with multiline bash block ---")
                print(content[:600])
                print("...")
                break

    # Show one example with [USER/ENVIRONMENT] tag
    if has_user_env_tag > 0:
        for line in lines:
            sample = json.loads(line)
            content = sample["messages"][-1]["content"]
            if "[USER/ENVIRONMENT]:" in content:
                print(f"\n--- EXAMPLE: sample with [USER/ENVIRONMENT] tag ---")
                idx = content.index("[USER/ENVIRONMENT]:")
                start = max(0, idx - 100)
                print(content[start:start+500])
                print("...")
                break


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))

    analyze_dataset(
        f"{base}/prm_sft_data_opus_distill_full_feedback_history_32k/prm_sft_train.jsonl",
        "CLEAN (all instances)"
    )

    analyze_dataset(
        f"{base}/prm_sft_data_opus_distill_full_feedback_history_32k_rejection-sample/prm_sft_train.jsonl",
        "REJECTION-SAMPLE (resolved only)"
    )