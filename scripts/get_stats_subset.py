#!/usr/bin/env python3
"""
Compute stats for a subset of instances (e.g., SWE-bench Verified Mini).

The subset is defined by the instance directories present in a reference run.
Stats are computed for that same subset across all specified runs.

Outputs two CSV tables:
  1. Stats table: Setting, PRM Model, Resolved, Res. Rate, Avg Steps, Costs
  2. Exit status table: Setting, PRM Model, Submitted, LimitsExceeded, ContextWindowExceededError, Other, Total

Usage:
    python scripts/get_stats_subset.py \
        --parent-dir results_singularity_max_150_steps \
        --reference-run singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think \
        --runs "base:-:singularity_edit_obs_final_only_0_cwm" \
               "prm_issue_res_k5:qwen3-8b:singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b" \
               "prm_issue_res_k5:qwen3-8b (noisy sft):singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6" \
               "prm_issue_res_k5:RS-SFT-Think:singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think"
        --csv-prefix results_singularity_max_150_steps/subset_mini50
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Per-token pricing (full input, no cache discounts).
MODEL_PRICING = {
    # Anthropic Bedrock (us region): $5.00/1M input, $25.00/1M output
    "us.anthropic.claude-opus-4-6-v1":  {"input": 5.0e-6,  "output": 2.5e-5},
    "facebook/cwm":                     {"input": 9e-7,    "output": 9e-7},
    "SWE-bench/SWE-agent-LM-32B":      {"input": 8e-8,  "output": 2.8e-7},
}


def recompute_model_cost(traj: dict) -> float:
    """Recompute model cost from token counts using full input pricing."""
    config = traj.get("info", {}).get("config", {})
    model_name = config.get("model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return traj.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0)
    total_cost = 0.0
    for m in traj.get("messages", []):
        if "extra" not in m:
            continue
        usage = m["extra"]["response"].get("usage", {})
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


def get_instance_ids(run_dir: Path) -> set[str]:
    """Get instance IDs from subdirectories of a run (skip files)."""
    return {p.name for p in run_dir.iterdir() if p.is_dir()}


def compute_stats(run_dir: Path, instance_ids: set[str]) -> dict:
    """Compute stats for a run, restricted to the given instance IDs."""
    # Resolved from report.json
    report_path = run_dir / "report.json"
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        resolved_ids = set(report.get("resolved_ids", []))

    # Only count resolved within our subset
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

    return {
        "n_subset": len(instance_ids),
        "n_found": n_found,
        "n_missing": len(missing),
        "missing": missing,
        "resolved": len(resolved_in_subset),
        "res_rate_subset": 100 * len(resolved_in_subset) / len(instance_ids) if instance_ids else 0,
        "res_rate_submitted": (100 * len(resolved_in_subset) / submitted) if submitted > 0 else None,
        "avg_steps": total_steps / n_found if n_found else 0,
        "avg_model_cost": total_cost / n_found if n_found else 0,
        "avg_prm_cost": total_prm_cost / n_found if n_found else 0,
        "avg_total_cost": (total_cost + total_prm_cost) / n_found if n_found else 0,
        "exit_counts": dict(exit_counts),
        "submitted": submitted,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute stats for a subset of instances")
    parser.add_argument("--parent-dir", type=str, required=True,
                        help="Parent directory containing all run directories")
    parser.add_argument("--reference-run", type=str, required=True,
                        help="Directory name of the reference run (defines the subset)")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="setting:label:dirname triples for each run to evaluate")
    parser.add_argument("--csv-prefix", type=str, default=None,
                        help="Optional: save CSVs to <prefix>_stats.csv and <prefix>_exit_statuses.csv")
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    ref_dir = parent / args.reference_run
    if not ref_dir.exists():
        print(f"ERROR: reference run directory not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    instance_ids = get_instance_ids(ref_dir)
    n = len(instance_ids)
    print(f"Reference run: {args.reference_run}", file=sys.stderr)
    print(f"Subset size: {n} instances", file=sys.stderr)
    print(file=sys.stderr)

    # Collect all stats
    all_stats = []
    for run_spec in args.runs:
        parts = run_spec.split(":", 2)
        if len(parts) != 3:
            print(f"ERROR: expected setting:label:dirname, got: {run_spec}", file=sys.stderr)
            sys.exit(1)
        setting, label, dirname = parts
        run_dir = parent / dirname
        print(f"Processing: {setting} / {label} ({dirname}) ...", file=sys.stderr, end=" ")

        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            all_stats.append((setting, label, None))
            continue

        stats = compute_stats(run_dir, instance_ids)
        print(f"found {stats['n_found']}/{n}, resolved {stats['resolved']}", file=sys.stderr)

        if stats["n_missing"] > 0:
            print(f"  WARNING: {stats['n_missing']} instances missing from this run", file=sys.stderr)

        all_stats.append((setting, label, stats))

    # ── Table 1: Stats ──
    stats_header = [
        "Setting", "PRM Model",
        f"Resolved (/{n})", f"Res. Rate (/{n})", "Res. Rate (/submitted)",
        f"Avg Steps (/{n})", f"Avg Model Cost ($) (/{n})",
        f"Avg PRM Cost ($) (/{n})", f"Avg Total Cost ($) (/{n})",
    ]

    stats_rows = []
    for setting, label, stats in all_stats:
        if stats is None:
            stats_rows.append([setting, label] + ["N/A"] * (len(stats_header) - 2))
            continue
        stats_rows.append([
            setting, label,
            str(stats["resolved"]),
            f"{stats['res_rate_subset']:.2f}",
            f"{stats['res_rate_submitted']:.2f}" if stats["res_rate_submitted"] is not None else "N/A",
            f"{stats['avg_steps']:.2f}",
            f"{stats['avg_model_cost']:.4f}",
            f"{stats['avg_prm_cost']:.4f}",
            f"{stats['avg_total_cost']:.4f}",
        ])

    print("\n=== Stats Table ===")
    print(",".join(stats_header))
    for row in stats_rows:
        print(",".join(row))

    # ── Table 2: Exit Statuses ──
    KNOWN_STATUSES = ["Submitted", "LimitsExceeded", "ContextWindowExceededError"]

    exit_header = ["Setting", "PRM Model"] + KNOWN_STATUSES + ["Other", "Total"]

    exit_rows = []
    for setting, label, stats in all_stats:
        if stats is None:
            exit_rows.append([setting, label] + ["N/A"] * (len(exit_header) - 2))
            continue
        ec = stats["exit_counts"]
        known_sum = sum(ec.get(s, 0) for s in KNOWN_STATUSES)
        total = sum(ec.values())
        other = total - known_sum
        exit_rows.append([
            setting, label,
            *[str(ec.get(s, 0)) for s in KNOWN_STATUSES],
            str(other),
            str(total),
        ])

    print("\n=== Exit Status Table ===")
    print(",".join(exit_header))
    for row in exit_rows:
        print(",".join(row))

    # Save CSVs if requested
    if args.csv_prefix:
        prefix = Path(args.csv_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        stats_csv = Path(f"{prefix}_stats.csv")
        with open(stats_csv, "w") as f:
            f.write(",".join(stats_header) + "\n")
            for row in stats_rows:
                f.write(",".join(row) + "\n")
        print(f"\nSaved stats to: {stats_csv}", file=sys.stderr)

        exit_csv = Path(f"{prefix}_exit_statuses.csv")
        with open(exit_csv, "w") as f:
            f.write(",".join(exit_header) + "\n")
            for row in exit_rows:
                f.write(",".join(row) + "\n")
        print(f"Saved exit statuses to: {exit_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
