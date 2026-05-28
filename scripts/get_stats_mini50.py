#!/usr/bin/env python3
"""
Compute stats for the SWE-Bench Verified Mini subset (50 instances).

The subset is defined by the instance directories in the reference run.
Model costs are recomputed from token counts using full input pricing
(no prompt-caching discounts) for fair comparison across models.

Usage:
    python get_stats_mini50.py
    python get_stats_mini50.py --parent-dir results_singularity_max_150_steps
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Per-token pricing (full input, no cache discounts).
# Source: Anthropic pricing page + litellm_model_registry.json
MODEL_PRICING = {
    # Anthropic Bedrock — Opus uses cache write premium (1.25x input) in stored cost,
    # which is higher than flat prompt_tokens * input_price.  Leave Opus OUT of this
    # table so recompute_prm_cost falls through to the stored cost (most expensive).
    # Generic 32B dense model: ~$0.18/1M (market rate for Qwen3-32B class)
    "facebook/cwm":                     {"input": 1.8e-7,  "output": 1.8e-7},
    "SWE-bench/SWE-agent-LM-32B":      {"input": 7.1e-8,  "output": 2.83e-7},
    # Generic 8B dense model: ~$0.05 input, $0.15 output (market rate for Qwen3-8B class)
    "Qwen/Qwen3-8B":                                               {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn": {"input": 5e-8, "output": 1.5e-7},
}


def recompute_model_cost(traj: dict) -> float:
    """Recompute model cost from token counts using full input pricing."""
    config = traj.get("info", {}).get("config", {})
    model_name = config.get("model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        # Fall back to stored cost if model not in pricing table
        return traj.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0)

    total_cost = 0.0
    for m in traj.get("messages", []):
        if "extra" not in m:
            continue
        usage = m["extra"]["response"].get("usage", {})
        # Use prompt_tokens (total input) at full price — no cache discount
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_cost += input_tokens * pricing["input"] + output_tokens * pricing["output"]
    return total_cost


def recompute_prm_cost(traj: dict) -> float:
    """Recompute PRM cost using full input pricing.

    If per-call token counts are available (newer runs), recompute precisely.
    Otherwise fall back to stored prm_cost.
    """
    info = traj.get("info", {})
    prm_stats = info.get("prm_stats") or {}
    stored_cost = prm_stats.get("prm_cost", 0.0)

    config = info.get("config", {})
    prm_model_name = config.get("prm_model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(prm_model_name)

    if pricing is None:
        return stored_cost

    feedback_log = prm_stats.get("prm_feedback_log", [])

    # Per-call token counts available (newer runs with logged usage)
    has_token_data = any(entry.get("usage") for entry in feedback_log)
    if has_token_data:
        total_cost = 0.0
        for entry in feedback_log:
            usage = entry.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_cost += input_tokens * pricing["input"] + output_tokens * pricing["output"]
        return total_cost

    return stored_cost

# Reference run that defines the 50-instance mini subset
REFERENCE_RUN = "singularity_edit_obs_final_only_0_cwm"

# Summary of each SFT PRM model's training configuration, for the CSV.
# Format string: "<prompt>/k=<int>/<data>/<format>"
# Keys are the PRM run dirname suffixes (everything after "_cwm_prm_") so we can key off dirname.
TRAINING_CONFIG = {
    # Base cases (no training)
    "qwen3-8b": "-",
    "claude-opus-4-6": "-",
    # SFT v1 (clean) — issue_res, k=5, swebench, flattened 3-msg, 2393 samples
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean": "issue_res/k=5/swebench/flattened",
    # RS-SFT — rejection-sampled subset of same data
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample": "issue_res/k=5/swebench/flattened (RS)",
    # SFT v2 flattened/multiturn — instructions, k=10, swebench
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened": "instructions/k=10/swebench/flattened",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn": "instructions/k=10/swebench/multiturn",
    # SFT r2e v2 — instructions, k=10, r2egym
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened": "instructions/k=10/r2egym/flattened",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn": "instructions/k=10/r2egym/multiturn",
    # SFT r2e v1 — issue_res, k=5, r2egym, multiturn
    "qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn": "issue_res/k=5/r2egym/multiturn",
}


def get_training_config(dirname: str) -> str:
    """Extract the SFT training config summary from a run dirname."""
    # Dirnames look like: singularity_..._prm_<prm_model_slug>
    marker = "_cwm_prm_"
    idx = dirname.find(marker)
    if idx < 0:
        return "-"  # base runs, etc.
    prm_slug = dirname[idx + len(marker):]
    return TRAINING_CONFIG.get(prm_slug, "?")

# (group, experiment, prm_label, dirname)
# group: for Google Sheets cell merging
# experiment: specific variant within the group
RUNS = [
    # ── Base CWM (no PRM) ──
    ("Base CWM", "run 0", "-",
     "singularity_edit_obs_final_only_0_cwm"),
    ("Base CWM", "run 1", "-",
     "singularity_edit_obs_final_only_1_cwm"),
    ("Base CWM", "run 2", "-",
     "singularity_edit_obs_final_only_2_cwm"),
    ("Base CWM", "run 3", "-",
     "singularity_edit_obs_final_only_3_cwm"),
    ("Base CWM", "run 4", "-",
     "singularity_edit_obs_final_only_4_cwm"),
    # ── PRM: issue_res prompt, k=5 ──
    ("PRM issue_res k5", "qwen3-8b base", "qwen3-8b (base)",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("PRM issue_res k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("PRM issue_res k5", "RS-SFT", "RS-SFT",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"),
    # ── PRM: issue_res prompt, k=10 ──
    ("PRM issue_res k10", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    # ── PRM: postprocess prompt, k=5 ──
    ("PRM postprocess k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_postprocess_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    # ── PRM: step_aware prompt, k=5 ──
    ("PRM step_aware k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_step_aware_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    # ── PRM: step_aware_explicit prompt, k=5 ──
    ("PRM step_aware_explicit k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_step_aware_explicit_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    # ── PRM: instructions prompt, k=5 ──
    ("PRM instructions k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    # ── PRM: instructions_step_aware prompt, k=10 (new SFT data) ──
    ("PRM instructions_step_aware k10", "SFT v2 (flattened)", "SFT v2 (flattened)",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened"),
    ("PRM instructions_step_aware k10", "SFT v2 (multiturn)", "SFT v2 (multiturn)",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn"),
    ("PRM instructions_step_aware k10", "SFT r2e v2 (flattened)", "SFT r2e v2 (flattened)",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened"),
    ("PRM instructions_step_aware k10", "SFT r2e v2 (multiturn)", "SFT r2e v2 (multiturn)",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn"),
    # ── PRM: step_aware prompt, k=10 (SFT data comparison) ──
    ("PRM step_aware k10", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("PRM step_aware k10", "SFT v2 (multiturn)", "SFT v2 (multiturn)",
     "singularity_edit_obs_final_only_prm_issue_res_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn"),
    ("PRM step_aware k10", "SFT r2e v1", "SFT r2e v1",
     "singularity_edit_obs_final_only_prm_issue_res_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn"),
]

BASE_RUN_DIRS = [
    f"singularity_edit_obs_final_only_{i}_cwm" for i in range(5)
]

BASE_CWM_DIR = "singularity_edit_obs_final_only_0_cwm"


def get_instance_ids(run_dir: Path) -> set[str]:
    """Get mini-50 instance IDs from the HuggingFace dataset.
    Falls back to subdirectories of run_dir if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")
        return set(ds["instance_id"])
    except Exception:
        return {p.name for p in run_dir.iterdir() if p.is_dir()}


def compute_stats(run_dir: Path, instance_ids: set[str],
                   report_filename: str = "report.json") -> dict:
    report_path = run_dir / report_filename
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            resolved_ids = set(json.load(f).get("resolved_ids", []))

    resolved_in_subset = resolved_ids & instance_ids

    exit_counts = Counter()
    total_steps = 0
    total_cost = 0.0
    total_prm_cost = 0.0
    n_found = 0
    missing = []

    for instance_id in sorted(instance_ids):
        inst_dir = run_dir / instance_id
        if not inst_dir.exists():
            missing.append(instance_id)
            continue

        traj_files = list(inst_dir.glob("*.traj.json"))
        if not traj_files:
            missing.append(instance_id)
            continue

        with open(traj_files[0]) as f:
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

        total_cost += recompute_model_cost(traj)
        total_prm_cost += recompute_prm_cost(traj)
        n_found += 1

    submitted = exit_counts.get("Submitted", 0)
    n = len(instance_ids)

    return {
        "n_subset": n,
        "n_found": n_found,
        "n_missing": len(missing),
        "missing": missing,
        "resolved": len(resolved_in_subset),
        "res_rate_subset": 100 * len(resolved_in_subset) / n if n else 0,
        "res_rate_submitted": (100 * len(resolved_in_subset) / submitted) if submitted > 0 else None,
        "avg_steps": total_steps / n if n else 0,
        "avg_model_cost": total_cost / n if n else 0,
        "avg_prm_cost": total_prm_cost / n if n else 0,
        "avg_total_cost": (total_cost + total_prm_cost) / n if n else 0,
        "exit_counts": dict(exit_counts),
        "submitted": submitted,
    }


def load_traj(run_dir: Path, instance_id: str) -> dict | None:
    inst_dir = run_dir / instance_id
    if not inst_dir.exists():
        return None
    traj_files = list(inst_dir.glob("*.traj.json"))
    if not traj_files:
        return None
    with open(traj_files[0]) as f:
        return json.load(f)


def traj_steps(traj: dict) -> int:
    return sum(
        1 for m in traj.get("messages", [])
        if m.get("role") == "assistant"
        and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
    )


def compute_fallback_stats(prm_dir: Path, base_dir: Path,
                           instance_ids: set[str],
                           report_filename: str = "report.json") -> dict:
    """Compute fallback-only stats: for instances where PRM didn't submit,
    what do we get from the base CWM run?

    Returns totals (resolved, steps, cost, exit_counts) for the fallback
    instances only, plus n_fallback and the combined submitted count.
    """
    prm_resolved = set()
    prm_report = prm_dir / report_filename
    if prm_report.exists():
        with open(prm_report) as f:
            prm_resolved = set(json.load(f).get("resolved_ids", []))

    base_resolved = set()
    base_report = base_dir / report_filename
    if base_report.exists():
        with open(base_report) as f:
            base_resolved = set(json.load(f).get("resolved_ids", []))

    # Track fallback instances only
    fb_resolved = 0
    fb_steps = 0
    fb_cost = 0.0
    fb_exit_counts = Counter()
    n_fallback = 0

    # Track combined resolved and submitted
    combined_resolved = 0
    prm_submitted = 0

    for instance_id in sorted(instance_ids):
        prm_traj = load_traj(prm_dir, instance_id)
        if prm_traj is None:
            continue

        prm_exit = prm_traj.get("info", {}).get("exit_status", "unknown")
        if prm_exit == "Submitted":
            prm_submitted += 1
            if instance_id in prm_resolved:
                combined_resolved += 1
        else:
            # Fallback instance — PRM didn't submit, so use base run instead.
            # But if PRM still resolved it (e.g. LimitsExceeded after a correct
            # edit was already applied), count the PRM resolution.
            if instance_id in prm_resolved:
                combined_resolved += 1
                # Still count as fallback for cost/steps accounting
            n_fallback += 1
            base_traj = load_traj(base_dir, instance_id)
            if base_traj is not None:
                base_exit = base_traj.get("info", {}).get("exit_status", "unknown")
                fb_exit_counts[base_exit] += 1
                fb_steps += traj_steps(base_traj)
                fb_cost += recompute_model_cost(base_traj)
                if instance_id not in prm_resolved and instance_id in base_resolved:
                    fb_resolved += 1
                    combined_resolved += 1

    fb_submitted = fb_exit_counts.get("Submitted", 0)

    return {
        "n_fallback": n_fallback,
        "fb_resolved": fb_resolved,
        "fb_steps": fb_steps,
        "fb_cost": fb_cost,
        "fb_exit_counts": dict(fb_exit_counts),
        "fb_submitted": fb_submitted,
        "combined_resolved": combined_resolved,
        "combined_submitted": prm_submitted + fb_submitted,
        "n_subset": len(instance_ids),
    }


def print_csv(headers: list[str], rows: list[list[str]]):
    print(",".join(headers))
    for row in rows:
        print(",".join(row))


def main():
    parser = argparse.ArgumentParser(description="Stats for SWE-Bench Verified Mini (50 samples)")
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"))
    parser.add_argument("--report-file", type=str, default="report.json",
                        help="Report filename to read resolved_ids from (e.g. report-docker.json)")
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    ref_dir = parent / REFERENCE_RUN
    if not ref_dir.exists():
        print(f"ERROR: reference run not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    instance_ids = get_instance_ids(ref_dir)
    n = len(instance_ids)
    print(f"Reference: {REFERENCE_RUN}", file=sys.stderr)
    print(f"Subset size: {n} instances\n", file=sys.stderr)

    all_stats = []
    for group, experiment, prm_label, dirname in RUNS:
        run_dir = parent / dirname
        print(f"  {group} / {experiment} ... ", file=sys.stderr, end="", flush=True)
        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            all_stats.append((group, experiment, prm_label, None))
            continue
        stats = compute_stats(run_dir, instance_ids, args.report_file)
        print(f"found {stats['n_found']}/{n}, resolved {stats['resolved']}", file=sys.stderr)
        if stats["n_missing"] > 0:
            print(f"    WARNING: {stats['n_missing']} missing instances", file=sys.stderr)
        all_stats.append((group, experiment, prm_label, stats))

    # ── Compute fallback stats for non-base runs ──
    base_dir = parent / BASE_CWM_DIR
    fallback_map = {}  # dirname -> fallback stats
    if base_dir.exists():
        print(f"\nComputing fallback stats (base CWM fallback)...", file=sys.stderr)
        for group, experiment, prm_label, dirname in RUNS:
            if group == "Base CWM":
                continue
            prm_dir = parent / dirname
            if not prm_dir.exists():
                continue
            fb = compute_fallback_stats(prm_dir, base_dir, instance_ids, args.report_file)
            print(f"  {group} / {experiment}: {fb['n_fallback']} fallback, "
                  f"+{fb['fb_resolved']} resolved from base, "
                  f"combined {fb['combined_resolved']}", file=sys.stderr)
            fallback_map[dirname] = fb

    # ── Aggregate stats across all base runs (for best-of-5 and critic) ──
    all_base_total_steps = 0
    all_base_total_cost = 0.0
    best_of_5_resolved = set()
    best_of_5_submitted = set()
    n_base_runs = 0
    for base_dirname in BASE_RUN_DIRS:
        base_run_dir = parent / base_dirname
        if not base_run_dir.exists():
            continue
        n_base_runs += 1
        report_path = base_run_dir / args.report_file
        if report_path.exists():
            with open(report_path) as f:
                resolved = set(json.load(f).get("resolved_ids", []))
            best_of_5_resolved |= (resolved & instance_ids)
        for iid in sorted(instance_ids):
            traj = load_traj(base_run_dir, iid)
            if traj is not None:
                all_base_total_steps += traj_steps(traj)
                all_base_total_cost += recompute_model_cost(traj)
                if traj.get("info", {}).get("exit_status") == "Submitted":
                    best_of_5_submitted.add(iid)
    print(f"\nBest-of-{n_base_runs} oracle: {len(best_of_5_resolved)}/{n} resolved", file=sys.stderr)

    # ── Critic inference cost (from all_critic_results.json) ──
    critic_cost = 0.0
    critic_dir = parent / "critic_selected_cwm"
    critic_results_path = critic_dir / "all_critic_results.json"
    if critic_results_path.exists():
        with open(critic_results_path) as f:
            critic_results = json.load(f)
        critic_input_tokens = sum(r.get("formatted_tokens", 0) for r in critic_results)
        critic_output_tokens = len(critic_results) * 500  # estimated ~500 output tokens per call
        # Critic uses CWM model
        critic_pricing = MODEL_PRICING.get("facebook/cwm", {"input": 9e-7, "output": 9e-7})
        critic_cost = (critic_input_tokens * critic_pricing["input"]
                       + critic_output_tokens * critic_pricing["output"])
        print(f"Critic inference cost: ${critic_cost:.3f} ({len(critic_results)} calls, "
              f"{critic_input_tokens:,} input tokens)", file=sys.stderr)

    # ── Stats Table (CSV) — interleaved with fallback rows ──
    # Group column: repeated value for Google Sheets cell merging
    stats_headers = [
        "Group", "Experiment", "Training Config",
        f"Resolved (/{n})",
        f"Submitted (/{n})",
        f"Res. Rate (/{n})",
        "Res. Rate (/submitted)",
        f"Avg Steps (/{n})",
        f"Avg Model Cost ($) (/{n})",
        f"Avg PRM Cost ($) (/{n})",
        f"Avg Total Cost ($) (/{n})",
    ]

    all_base_avg_steps = all_base_total_steps / n if n else 0
    all_base_avg_cost = all_base_total_cost / n if n else 0

    # ── Compute critic_selected stats from its traj files ──
    critic_stats = None
    critic_dir_path = parent / "critic_selected_cwm"
    if critic_dir_path.exists():
        critic_stats = compute_stats(critic_dir_path, instance_ids, args.report_file)

    stats_rows = []
    for (group, experiment, prm_label, stats), (_, _, _, dirname) in zip(
            all_stats, RUNS):
        if stats is None:
            stats_rows.append([group, experiment, get_training_config(dirname)] + ["N/A"] * 8)
            continue
        stats_rows.append([
            group, experiment, get_training_config(dirname),
            str(stats["resolved"]),
            str(stats["submitted"]),
            f"{stats['res_rate_subset']:.2f}",
            f"{stats['res_rate_submitted']:.2f}" if stats["res_rate_submitted"] is not None else "N/A",
            f"{stats['avg_steps']:.2f}",
            f"{stats['avg_model_cost']:.3f}",
            f"{stats['avg_prm_cost']:.3f}",
            f"{stats['avg_total_cost']:.3f}",
        ])

        # After last base run, insert best-of-5 and critic rows
        if group == "Base CWM" and experiment == f"run {n_base_runs - 1}" and n_base_runs > 0:
            bo5_res = len(best_of_5_resolved)
            bo5_sub = len(best_of_5_submitted)
            stats_rows.append([
                "Base CWM", f"best-of-{n_base_runs} oracle", "-",
                str(bo5_res),
                str(bo5_sub),
                f"{100 * bo5_res / n:.2f}",
                f"{100 * bo5_res / bo5_sub:.2f}" if bo5_sub > 0 else "N/A",
                f"{all_base_avg_steps:.2f}",
                f"{all_base_avg_cost:.3f}",
                "0.000",
                f"{all_base_avg_cost:.3f}",
            ])
            if critic_stats is not None:
                critic_total_cost = all_base_avg_cost + critic_cost / n
                stats_rows.append([
                    "Base CWM", "critic_selected", "-",
                    str(critic_stats["resolved"]),
                    str(critic_stats["submitted"]),
                    f"{critic_stats['res_rate_subset']:.2f}",
                    f"{critic_stats['res_rate_submitted']:.2f}" if critic_stats["res_rate_submitted"] is not None else "N/A",
                    f"{all_base_avg_steps:.2f}",
                    f"{all_base_avg_cost:.3f}",
                    f"{critic_cost / n:.3f}",
                    f"{critic_total_cost:.3f}",
                ])

            # Claude Opus PRM (issue_res k5, 75 steps) — reference row from different parent dir
            opus75_dir = Path(str(parent).replace("results_singularity_max_150_steps_prefix",
                                                   "results_singularity_max_75_steps")) / \
                "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"
            if opus75_dir.exists():
                opus75_stats = compute_stats(opus75_dir, instance_ids, args.report_file)
                stats_rows.append([
                    "PRM issue_res k5 (75 steps)", "Claude Opus", "-",
                    str(opus75_stats["resolved"]),
                    str(opus75_stats["submitted"]),
                    f"{opus75_stats['res_rate_subset']:.2f}",
                    f"{opus75_stats['res_rate_submitted']:.2f}" if opus75_stats["res_rate_submitted"] is not None else "N/A",
                    f"{opus75_stats['avg_steps']:.2f}",
                    f"{opus75_stats['avg_model_cost']:.3f}",
                    f"{opus75_stats['avg_prm_cost']:.3f}",
                    f"{opus75_stats['avg_total_cost']:.3f}",
                ])
                opus75_fb = compute_fallback_stats(opus75_dir, base_dir, instance_ids, args.report_file)
                combined_res = opus75_fb["combined_resolved"]
                combined_sub = opus75_fb["combined_submitted"]
                res_rate_sub = (100 * combined_res / combined_sub) if combined_sub > 0 else None
                total_steps = opus75_stats["avg_steps"] + opus75_fb["fb_steps"] / n
                total_model_cost = opus75_stats["avg_model_cost"] + opus75_fb["fb_cost"] / n
                total_prm_cost = opus75_stats["avg_prm_cost"]
                total_cost = total_model_cost + total_prm_cost
                stats_rows.append([
                    "PRM issue_res k5 (75 steps)", "Claude Opus + base fallback", "-",
                    str(combined_res),
                    str(combined_sub),
                    f"{100 * combined_res / n:.2f}",
                    f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
                    f"{total_steps:.2f}",
                    f"{total_model_cost:.3f}",
                    f"{total_prm_cost:.3f}",
                    f"{total_cost:.3f}",
                ])

            # Claude Opus PRM runs (150 steps) — inserted right below best-of-5/critic
            # Format: (group_label, dirname, report_file_override)
            # Each row gets a "Claude Opus" row and a "Claude Opus + base fallback" row.
            # For mini50 numbers, reading report.json and intersecting with mini50 instance_ids
            # gives the same result as a mini50-specific report, so use the default report.json.
            opus_runs = [
                ("PRM issue_res k5", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM issue_res k10", "singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM instructions k5", "singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM instructions k10", "singularity_edit_obs_final_only_prm_issue_res_instructions_k10_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
            ]
            for opus_group, opus_dirname, opus_report_file in opus_runs:
                opus_dir = parent / opus_dirname
                if not opus_dir.exists():
                    continue
                opus_stats = compute_stats(opus_dir, instance_ids, opus_report_file)
                stats_rows.append([
                    opus_group, "Claude Opus", "-",
                    str(opus_stats["resolved"]),
                    str(opus_stats["submitted"]),
                    f"{opus_stats['res_rate_subset']:.2f}",
                    f"{opus_stats['res_rate_submitted']:.2f}" if opus_stats["res_rate_submitted"] is not None else "N/A",
                    f"{opus_stats['avg_steps']:.2f}",
                    f"{opus_stats['avg_model_cost']:.3f}",
                    f"{opus_stats['avg_prm_cost']:.3f}",
                    f"{opus_stats['avg_total_cost']:.3f}",
                ])
                opus_fb = compute_fallback_stats(opus_dir, base_dir, instance_ids, opus_report_file)
                combined_res = opus_fb["combined_resolved"]
                combined_sub = opus_fb["combined_submitted"]
                res_rate_sub = (100 * combined_res / combined_sub) if combined_sub > 0 else None
                total_steps = opus_stats["avg_steps"] + opus_fb["fb_steps"] / n
                total_model_cost = opus_stats["avg_model_cost"] + opus_fb["fb_cost"] / n
                total_prm_cost = opus_stats["avg_prm_cost"]
                total_cost = total_model_cost + total_prm_cost
                stats_rows.append([
                    opus_group, "Claude Opus + base fallback", "-",
                    str(combined_res),
                    str(combined_sub),
                    f"{100 * combined_res / n:.2f}",
                    f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
                    f"{total_steps:.2f}",
                    f"{total_model_cost:.3f}",
                    f"{total_prm_cost:.3f}",
                    f"{total_cost:.3f}",
                ])

        # Add fallback row if applicable
        if dirname in fallback_map and stats is not None:
            fb = fallback_map[dirname]
            combined_res = fb["combined_resolved"]
            combined_sub = fb["combined_submitted"]
            res_rate_sub = (100 * combined_res / combined_sub) if combined_sub > 0 else None
            total_steps = stats["avg_steps"] + fb["fb_steps"] / n
            total_model_cost = stats["avg_model_cost"] + fb["fb_cost"] / n
            total_prm_cost = stats["avg_prm_cost"]
            total_cost = total_model_cost + total_prm_cost
            stats_rows.append([
                group, f"{experiment} + base fallback", get_training_config(dirname),
                str(combined_res),
                str(combined_sub),
                f"{100 * combined_res / n:.2f}",
                f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
                f"{total_steps:.2f}",
                f"{total_model_cost:.3f}",
                f"{total_prm_cost:.3f}",
                f"{total_cost:.3f}",
            ])

    print(f"\n=== SWE-Bench Verified mini ({n} samples) - Stats ===")
    print_csv(stats_headers, stats_rows)

    # ── Exit Status Table (CSV) — interleaved ──
    KNOWN_STATUSES = ["Submitted", "LimitsExceeded", "ContextWindowExceededError"]
    STATUS_LABELS = ["Submitted", "Limits Exceeded", "Context Window Exceeded"]
    exit_headers = ["Group", "Experiment"] + STATUS_LABELS + ["Other", "Total"]

    exit_rows = []
    for (group, experiment, prm_label, stats), (_, _, _, dirname) in zip(
            all_stats, RUNS):
        if stats is None:
            exit_rows.append([group, experiment] + ["N/A"] * 5)
            continue
        ec = stats["exit_counts"]
        known_sum = sum(ec.get(s, 0) for s in KNOWN_STATUSES)
        total = sum(ec.values())
        other = total - known_sum
        exit_rows.append([
            group, experiment,
            *[str(ec.get(s, 0)) for s in KNOWN_STATUSES],
            str(other), str(total),
        ])
        if group == "Base CWM" and experiment == f"run {n_base_runs - 1}" and n_base_runs > 0:
            exit_rows.append(["Base CWM", f"best-of-{n_base_runs} oracle"] + ["N/A"] * 5)
            if critic_stats is not None:
                cec = critic_stats["exit_counts"]
                c_known = sum(cec.get(s, 0) for s in KNOWN_STATUSES)
                c_total = sum(cec.values())
                exit_rows.append([
                    "Base CWM", "critic_selected",
                    *[str(cec.get(s, 0)) for s in KNOWN_STATUSES],
                    str(c_total - c_known), str(c_total),
                ])
            opus75_dir_path = Path(str(parent).replace("results_singularity_max_150_steps_prefix",
                                                        "results_singularity_max_75_steps")) / \
                "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"
            if opus75_dir_path.exists():
                opus75_exit_stats = compute_stats(opus75_dir_path, instance_ids, args.report_file)
                o75ec = opus75_exit_stats["exit_counts"]
                o75_known = sum(o75ec.get(s, 0) for s in KNOWN_STATUSES)
                o75_total = sum(o75ec.values())
                exit_rows.append([
                    "PRM issue_res k5 (75 steps)", "Claude Opus",
                    *[str(o75ec.get(s, 0)) for s in KNOWN_STATUSES],
                    str(o75_total - o75_known), str(o75_total),
                ])
                opus75_fb = compute_fallback_stats(opus75_dir_path, base_dir, instance_ids, args.report_file)
                o75fb_ec = opus75_fb["fb_exit_counts"]
                o75_combined_sub = opus75_fb["combined_submitted"]
                o75fb_limits = sum(o75fb_ec.get(s, 0) for s in ["LimitsExceeded"])
                o75fb_ctx = sum(o75fb_ec.get(s, 0) for s in ["ContextWindowExceededError"])
                o75fb_known = o75fb_ec.get("Submitted", 0) + o75fb_limits + o75fb_ctx
                o75fb_other = sum(o75fb_ec.values()) - o75fb_known
                exit_rows.append([
                    "PRM issue_res k5 (75 steps)", "Claude Opus + base fallback",
                    str(o75_combined_sub),
                    str(o75fb_limits),
                    str(o75fb_ctx),
                    str(o75fb_other),
                    str(opus75_exit_stats["n_found"]),
                ])
            opus_exit_runs = [
                ("PRM issue_res k5", "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM issue_res k10", "singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM instructions k5", "singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
                ("PRM instructions k10", "singularity_edit_obs_final_only_prm_issue_res_instructions_k10_0_cwm_prm_claude-opus-4-6",
                 args.report_file),
            ]
            for opus_group, opus_dirname, opus_report_file in opus_exit_runs:
                opus_dir_path = parent / opus_dirname
                if not opus_dir_path.exists():
                    continue
                opus_exit_stats = compute_stats(opus_dir_path, instance_ids, opus_report_file)
                oec = opus_exit_stats["exit_counts"]
                o_known = sum(oec.get(s, 0) for s in KNOWN_STATUSES)
                o_total = sum(oec.values())
                exit_rows.append([
                    opus_group, "Claude Opus",
                    *[str(oec.get(s, 0)) for s in KNOWN_STATUSES],
                    str(o_total - o_known), str(o_total),
                ])
                opus_fb = compute_fallback_stats(opus_dir_path, base_dir, instance_ids, opus_report_file)
                ofb_ec = opus_fb["fb_exit_counts"]
                o_combined_sub = opus_fb["combined_submitted"]
                ofb_limits = sum(ofb_ec.get(s, 0) for s in ["LimitsExceeded"])
                ofb_ctx = sum(ofb_ec.get(s, 0) for s in ["ContextWindowExceededError"])
                ofb_known = ofb_ec.get("Submitted", 0) + ofb_limits + ofb_ctx
                ofb_other = sum(ofb_ec.values()) - ofb_known
                exit_rows.append([
                    opus_group, "Claude Opus + base fallback",
                    str(o_combined_sub),
                    str(ofb_limits),
                    str(ofb_ctx),
                    str(ofb_other),
                    str(opus_exit_stats["n_found"]),
                ])
        if dirname in fallback_map:
            fb = fallback_map[dirname]
            fb_ec = fb["fb_exit_counts"]
            combined_submitted = fb["combined_submitted"]
            fb_limits = sum(fb_ec.get(s, 0) for s in ["LimitsExceeded"])
            fb_ctx = sum(fb_ec.get(s, 0) for s in ["ContextWindowExceededError"])
            fb_known = fb_ec.get("Submitted", 0) + fb_limits + fb_ctx
            fb_other = sum(fb_ec.values()) - fb_known
            fb_total = fb["n_fallback"]
            exit_rows.append([
                group, f"{experiment} + base fallback",
                str(combined_submitted),
                str(fb_limits),
                str(fb_ctx),
                str(fb_other),
                str(stats["n_found"]),
            ])

    print(f"\n=== SWE-Bench Verified mini ({n} samples) - Exit Statuses ===")
    print_csv(exit_headers, exit_rows)

    # ── Save CSV files ──
    out_dir = parent
    stats_csv = out_dir / "mini50_stats.csv"
    with open(stats_csv, "w") as f:
        f.write(",".join(stats_headers) + "\n")
        for row in stats_rows:
            f.write(",".join(row) + "\n")
    print(f"\nSaved: {stats_csv}", file=sys.stderr)

    exit_csv = out_dir / "mini50_exit_statuses.csv"
    with open(exit_csv, "w") as f:
        f.write(",".join(exit_headers) + "\n")
        for row in exit_rows:
            f.write(",".join(row) + "\n")
    print(f"Saved: {exit_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
