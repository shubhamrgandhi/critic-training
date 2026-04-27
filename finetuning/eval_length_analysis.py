#!/usr/bin/env python3
"""Analyze response length vs judge win rate."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "eval_results"
DEFAULT_DATA_PATH = SCRIPT_DIR / "prm_sft_data_opus_distill_full_feedback_history_32k_old" / "prm_sft_train.jsonl"

MODEL_A = "qwen3-8b-base"
MODEL_B = "qwen3-8b-opus-distill-32k-lr5e6"

# Load training data to get prompt (user message) lengths
def load_prompt_lengths(data_path):
    """Load prompt lengths from training JSONL (user message = trajectory prefix)."""
    lengths = {}
    with open(data_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            msgs = item["messages"]
            lengths[i] = len(msgs[1]["content"])  # user message = trajectory
    return lengths

prompt_lengths = load_prompt_lengths(DEFAULT_DATA_PATH)

# Load responses
def load_responses(label):
    path = EVAL_DIR / label / "responses.jsonl"
    data = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data[item["idx"]] = item
    return data

resp_a = load_responses(MODEL_A)
resp_b = load_responses(MODEL_B)

# Load judge results
judge_path = EVAL_DIR / f"judge_{MODEL_A}_vs_{MODEL_B}.jsonl"
judge = {}
with open(judge_path) as f:
    for line in f:
        item = json.loads(line)
        judge[item["idx"]] = item

# Build combined data
rows = []
for idx in sorted(judge.keys()):
    if idx not in resp_a or idx not in resp_b:
        continue
    gt = resp_a[idx]["ground_truth"]
    ra = resp_a[idx]["response"]
    rb = resp_b[idx]["response"]
    winner = judge[idx]["winner_model"]
    rows.append({
        "idx": idx,
        "len_prompt": prompt_lengths.get(idx, 0),
        "len_gt": len(gt),
        "len_base": len(ra),
        "len_sft": len(rb),
        "winner": winner,
    })

print(f"Total samples: {len(rows)}\n")

# --- Overall length stats ---
gt_lens = [r["len_gt"] for r in rows]
base_lens = [r["len_base"] for r in rows]
sft_lens = [r["len_sft"] for r in rows]

print("=== Response Length Stats (characters) ===")
for name, lens in [("Ground truth (Opus)", gt_lens), (f"Base ({MODEL_A})", base_lens), (f"SFT ({MODEL_B})", sft_lens)]:
    print(f"  {name}:")
    print(f"    mean={np.mean(lens):.0f}  median={np.median(lens):.0f}  std={np.std(lens):.0f}")
    print(f"    min={np.min(lens)}  max={np.max(lens)}  p25={np.percentile(lens,25):.0f}  p75={np.percentile(lens,75):.0f}")

# --- Length ratio ---
base_ratios = [r["len_base"] / r["len_gt"] if r["len_gt"] > 0 else 0 for r in rows]
sft_ratios = [r["len_sft"] / r["len_gt"] if r["len_gt"] > 0 else 0 for r in rows]
print(f"\n=== Length Ratio (model / ground_truth) ===")
print(f"  Base: mean={np.mean(base_ratios):.2f}  median={np.median(base_ratios):.2f}")
print(f"  SFT:  mean={np.mean(sft_ratios):.2f}  median={np.median(sft_ratios):.2f}")

# --- Does the longer response win? ---
longer_wins = 0
shorter_wins = 0
same_len_tie = 0
for r in rows:
    if r["winner"] == "tie" or r["winner"] == "parse_error":
        continue
    winner_is_sft = (r["winner"] == MODEL_B)
    sft_longer = r["len_sft"] > r["len_base"]
    if winner_is_sft == sft_longer:
        longer_wins += 1
    else:
        shorter_wins += 1

total_decided = longer_wins + shorter_wins
print(f"\n=== Does the longer response win? ===")
print(f"  Longer response wins:  {longer_wins}/{total_decided} ({100*longer_wins/total_decided:.1f}%)")
print(f"  Shorter response wins: {shorter_wins}/{total_decided} ({100*shorter_wins/total_decided:.1f}%)")

# --- Win rate by length bucket (of ground truth) ---
print(f"\n=== Win rate by ground truth length bucket ===")
buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, float("inf"))]
for lo, hi in buckets:
    bucket_rows = [r for r in rows if lo <= r["len_gt"] < hi]
    if not bucket_rows:
        continue
    sft_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
    base_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
    ties = sum(1 for r in bucket_rows if r["winner"] == "tie")
    n = len(bucket_rows)
    hi_str = f"{hi}" if hi != float("inf") else "inf"
    print(f"  [{lo:>5}, {hi_str:>5}): n={n:>4}  SFT={sft_wins:>4}({100*sft_wins/n:.0f}%)  base={base_wins:>4}({100*base_wins/n:.0f}%)  tie={ties}")

# --- Win rate by SFT response length bucket ---
print(f"\n=== Win rate by SFT response length bucket ===")
for lo, hi in buckets:
    bucket_rows = [r for r in rows if lo <= r["len_sft"] < hi]
    if not bucket_rows:
        continue
    sft_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
    base_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
    ties = sum(1 for r in bucket_rows if r["winner"] == "tie")
    n = len(bucket_rows)
    hi_str = f"{hi}" if hi != float("inf") else "inf"
    print(f"  [{lo:>5}, {hi_str:>5}): n={n:>4}  SFT={sft_wins:>4}({100*sft_wins/n:.0f}%)  base={base_wins:>4}({100*base_wins/n:.0f}%)  tie={ties}")

# --- Win rate by length difference (SFT - base) ---
print(f"\n=== Win rate by length difference (SFT - base) ===")
diff_buckets = [(-float("inf"), -2000), (-2000, -500), (-500, 0), (0, 500), (500, 2000), (2000, float("inf"))]
for lo, hi in diff_buckets:
    bucket_rows = [r for r in rows if lo <= (r["len_sft"] - r["len_base"]) < hi]
    if not bucket_rows:
        continue
    sft_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
    base_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
    n = len(bucket_rows)
    lo_str = f"{lo}" if lo != -float("inf") else "-inf"
    hi_str = f"{hi}" if hi != float("inf") else "inf"
    print(f"  [{lo_str:>5}, {hi_str:>5}): n={n:>4}  SFT={sft_wins:>4}({100*sft_wins/n:.0f}%)  base={base_wins:>4}({100*base_wins/n:.0f}%)")

# --- Win rate by prompt (trajectory prefix) length bucket ---
print(f"\n=== Win rate by prompt/trajectory prefix length bucket (10k-char bins) ===")
prompt_buckets = [(i, i+10000) for i in range(0, 140000, 10000)] + [(140000, float("inf"))]
for lo, hi in prompt_buckets:
    bucket_rows = [r for r in rows if lo <= r["len_prompt"] < hi]
    if not bucket_rows:
        continue
    sft_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
    base_wins = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
    ties = sum(1 for r in bucket_rows if r["winner"] == "tie")
    n = len(bucket_rows)
    hi_str = f"{hi}" if hi != float("inf") else "inf"
    print(f"  [{lo:>5}, {hi_str:>5}): n={n:>4}  SFT={sft_wins:>4}({100*sft_wins/n:.0f}%)  base={base_wins:>4}({100*base_wins/n:.0f}%)  tie={ties}")

# --- Save CSV ---
csv_path = EVAL_DIR / f"length_analysis_{MODEL_A}_vs_{MODEL_B}.csv"
with open(csv_path, "w") as f:
    f.write("idx,len_prompt,len_ground_truth,len_base,len_sft,len_diff_sft_minus_base,len_ratio_base_to_gt,len_ratio_sft_to_gt,winner\n")
    for r in rows:
        gt = r["len_gt"]
        ratio_base = f"{r['len_base']/gt:.3f}" if gt > 0 else "0"
        ratio_sft = f"{r['len_sft']/gt:.3f}" if gt > 0 else "0"
        f.write(f"{r['idx']},{r['len_prompt']},{gt},{r['len_base']},{r['len_sft']},{r['len_sft']-r['len_base']},{ratio_base},{ratio_sft},{r['winner']}\n")

print(f"\nPer-sample CSV saved to: {csv_path}")

# --- Save summary CSV ---
summary_path = EVAL_DIR / "length_analysis_summary.csv"
with open(summary_path, "w") as f:
    # Response length stats
    f.write("Response Length Stats (characters)\n")
    f.write("model,mean,median,std,min,max,p25,p75\n")
    for name, lens in [("Ground truth (Opus)", gt_lens), (f"Base ({MODEL_A})", base_lens), (f"SFT ({MODEL_B})", sft_lens)]:
        f.write(f"{name},{np.mean(lens):.0f},{np.median(lens):.0f},{np.std(lens):.0f},{np.min(lens)},{np.max(lens)},{np.percentile(lens,25):.0f},{np.percentile(lens,75):.0f}\n")

    # Length ratio
    f.write(f"\nLength Ratio (model output length / ground truth length)\n")
    f.write("model,mean_ratio,median_ratio\n")
    f.write(f"Base,{np.mean(base_ratios):.2f},{np.median(base_ratios):.2f}\n")
    f.write(f"SFT,{np.mean(sft_ratios):.2f},{np.median(sft_ratios):.2f}\n")

    # Longer wins
    f.write(f"\nDoes the longer response (of the two models) win the judge comparison?\n")
    f.write("outcome,count,percent\n")
    f.write(f"Longer response wins,{longer_wins},{100*longer_wins/total_decided:.1f}%\n")
    f.write(f"Shorter response wins,{shorter_wins},{100*shorter_wins/total_decided:.1f}%\n")

    # GT length buckets (500-char)
    f.write(f"\nWin rate by ground truth (Opus) response length bucket (500-char bins)\n")
    f.write("gt_length_bucket,n,sft_wins,sft_win%,base_wins,base_win%,ties\n")
    gt_500_buckets = [(i, i+500) for i in range(0, 8000, 500)] + [(8000, float("inf"))]
    for lo, hi in gt_500_buckets:
        bucket_rows = [r for r in rows if lo <= r["len_gt"] < hi]
        if not bucket_rows:
            continue
        sw = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
        bw = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
        ti = sum(1 for r in bucket_rows if r["winner"] == "tie")
        n = len(bucket_rows)
        hi_str = f"{hi}" if hi != float("inf") else "+"
        f.write(f"{lo}-{hi_str},{n},{sw},{100*sw//n}%,{bw},{100*bw//n}%,{ti}\n")

    # SFT response length buckets (500-char)
    f.write(f"\nWin rate by SFT model response length bucket (500-char bins)\n")
    f.write("sft_length_bucket,n,sft_wins,sft_win%,base_wins,base_win%,ties\n")
    sft_500_buckets = [(i, i+500) for i in range(0, 10000, 500)] + [(10000, float("inf"))]
    for lo, hi in sft_500_buckets:
        bucket_rows = [r for r in rows if lo <= r["len_sft"] < hi]
        if not bucket_rows:
            continue
        sw = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
        bw = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
        ti = sum(1 for r in bucket_rows if r["winner"] == "tie")
        n = len(bucket_rows)
        hi_str = f"{hi}" if hi != float("inf") else "+"
        f.write(f"{lo}-{hi_str},{n},{sw},{100*sw//n}%,{bw},{100*bw//n}%,{ti}\n")

    # Length difference buckets
    f.write(f"\nWin rate by length difference (SFT response length minus Base response length)\n")
    f.write("len_diff_bucket,n,sft_wins,sft_win%,base_wins,base_win%\n")
    for lo, hi in diff_buckets:
        bucket_rows = [r for r in rows if lo <= (r["len_sft"] - r["len_base"]) < hi]
        if not bucket_rows:
            continue
        sw = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
        bw = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
        n = len(bucket_rows)
        lo_label = f"SFT much shorter (< -2000)" if lo == -float("inf") else f"SFT shorter ({lo} to {hi})" if hi <= 0 else f"SFT slightly longer ({lo} to {hi})" if hi <= 500 else f"SFT longer ({lo} to {hi})" if hi <= 2000 else f"SFT much longer (> {lo})"
        f.write(f"{lo_label},{n},{sw},{100*sw//n}%,{bw},{100*bw//n}%\n")

    # Prompt length buckets (10k-char)
    f.write(f"\nWin rate by prompt/trajectory prefix length bucket (10k-char bins)\n")
    f.write("prompt_length_bucket,n,sft_wins,sft_win%,base_wins,base_win%,ties\n")
    for lo, hi in prompt_buckets:
        bucket_rows = [r for r in rows if lo <= r["len_prompt"] < hi]
        if not bucket_rows:
            continue
        sw = sum(1 for r in bucket_rows if r["winner"] == MODEL_B)
        bw = sum(1 for r in bucket_rows if r["winner"] == MODEL_A)
        ti = sum(1 for r in bucket_rows if r["winner"] == "tie")
        n = len(bucket_rows)
        hi_str = f"{hi}" if hi != float("inf") else "+"
        f.write(f"{lo}-{hi_str},{n},{sw},{100*sw//n}%,{bw},{100*bw//n}%,{ti}\n")

    # Avg response length by prompt length bucket (10k-char)
    f.write(f"\nAvg response length (chars) by prompt/trajectory prefix length bucket (10k-char bins)\n")
    f.write("prompt_length_bucket,n,avg_gt_opus,avg_base,avg_sft\n")
    for lo, hi in prompt_buckets:
        bucket_rows = [r for r in rows if lo <= r["len_prompt"] < hi]
        if not bucket_rows:
            continue
        avg_gt = np.mean([r["len_gt"] for r in bucket_rows])
        avg_base = np.mean([r["len_base"] for r in bucket_rows])
        avg_sft = np.mean([r["len_sft"] for r in bucket_rows])
        hi_str = f"{hi}" if hi != float("inf") else "+"
        f.write(f"{lo}-{hi_str},{len(bucket_rows)},{avg_gt:.0f},{avg_base:.0f},{avg_sft:.0f}\n")

print(f"Summary CSV saved to: {summary_path}")