#!/usr/bin/env python3
"""
Compute stats for the full SWE-Bench Verified set (500 instances), max 75 steps.

Usage:
    python get_stats_full500_75steps.py
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

MODEL_PRICING = {
    "us.anthropic.claude-opus-4-6-v1":  {"input": 5.0e-6,  "output": 2.5e-5},
    "facebook/cwm":                     {"input": 9e-7,    "output": 9e-7},
    "SWE-bench/SWE-agent-LM-32B":      {"input": 8e-8,  "output": 2.8e-7},
}


def recompute_model_cost(traj: dict) -> float:
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


REFERENCE_RUN = "singularity_edit_obs_final_only_0_cwm"

# (setting, prm_label, dirname)
RUNS = [
    ("base", "-",
     "singularity_edit_obs_final_only_0_cwm"),
    ("prm_issue_res_k5", "Claude-Opus-4.6",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"),
    ("prm_issue_res_k5", "cwm",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_cwm"),
    ("prm_issue_res_k5", "qwen25coder7b",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen25coder7b"),
    ("prm_issue_res_k5", "sweagent7b",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_sweagent7b"),
    ("prm_issue_res_k5", "Qwen3-8b",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b"),
    ("prm_issue_res_k5", "Qwen3-8b (noisy sft)",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_noisy"),
    ("prm_tool_issue_res_k5", "Claude-Opus-4.6",
     "singularity_edit_obs_final_only_prm_tool_issue_res_k5_0_cwm_prm_claude-opus-4-6"),
    ("prm_tool_k5", "Claude-Opus-4.6",
     "singularity_edit_obs_final_only_prm_tool_k5_0_cwm_prm_claude-opus-4-6"),
]

BASE_CWM_DIR = "singularity_edit_obs_final_only_0_cwm"


def get_instance_ids(run_dir: Path) -> set[str]:
    return {p.name for p in run_dir.iterdir() if p.is_dir()}


def compute_stats(run_dir: Path, instance_ids: set[str]) -> dict:
    report_path = run_dir / "report.json"
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
                           instance_ids: set[str]) -> dict:
    prm_resolved = set()
    prm_report = prm_dir / "report.json"
    if prm_report.exists():
        with open(prm_report) as f:
            prm_resolved = set(json.load(f).get("resolved_ids", []))

    base_resolved = set()
    base_report = base_dir / "report.json"
    if base_report.exists():
        with open(base_report) as f:
            base_resolved = set(json.load(f).get("resolved_ids", []))

    fb_resolved = 0
    fb_steps = 0
    fb_cost = 0.0
    fb_exit_counts = Counter()
    n_fallback = 0
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
            n_fallback += 1
            base_traj = load_traj(base_dir, instance_id)
            if base_traj is not None:
                base_exit = base_traj.get("info", {}).get("exit_status", "unknown")
                fb_exit_counts[base_exit] += 1
                fb_steps += traj_steps(base_traj)
                fb_cost += recompute_model_cost(base_traj)
                if instance_id in base_resolved:
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
    parser = argparse.ArgumentParser(description="Stats for full SWE-Bench Verified (500), max 75 steps")
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity_max_75_steps"))
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
    for setting, prm_label, dirname in RUNS:
        run_dir = parent / dirname
        print(f"  {setting} / {prm_label} ... ", file=sys.stderr, end="", flush=True)
        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            all_stats.append((setting, prm_label, None))
            continue
        stats = compute_stats(run_dir, instance_ids)
        print(f"found {stats['n_found']}/{n}, resolved {stats['resolved']}", file=sys.stderr)
        if stats["n_missing"] > 0:
            print(f"    WARNING: {stats['n_missing']} missing instances", file=sys.stderr)
        all_stats.append((setting, prm_label, stats))

    # ── Compute fallback stats for non-base runs ──
    base_dir = parent / BASE_CWM_DIR
    fallback_map = {}
    if base_dir.exists():
        print(f"\nComputing fallback stats (base CWM fallback)...", file=sys.stderr)
        for setting, prm_label, dirname in RUNS:
            if setting == "base":
                continue
            prm_dir = parent / dirname
            if not prm_dir.exists():
                continue
            fb = compute_fallback_stats(prm_dir, base_dir, instance_ids)
            print(f"  {setting} / {prm_label}: {fb['n_fallback']} fallback, "
                  f"+{fb['fb_resolved']} resolved from base, "
                  f"combined {fb['combined_resolved']}", file=sys.stderr)
            fallback_map[dirname] = fb

    # ── Stats Table (CSV) — interleaved with fallback rows ──
    stats_headers = [
        "Setting", "PRM Model",
        f"Resolved (/{n})",
        f"Res. Rate (/{n})",
        "Res. Rate (/submitted)",
        f"Avg Steps (/{n})",
        f"Avg Model Cost ($) (/{n})",
        f"Avg PRM Cost ($) (/{n})",
        f"Avg Total Cost ($) (/{n})",
    ]

    stats_rows = []
    for (setting, prm_label, stats), (_, _, dirname) in zip(all_stats, RUNS):
        if stats is None:
            stats_rows.append([setting, prm_label] + ["N/A"] * 7)
            continue
        stats_rows.append([
            setting, prm_label,
            str(stats["resolved"]),
            f"{stats['res_rate_subset']:.2f}",
            f"{stats['res_rate_submitted']:.2f}" if stats["res_rate_submitted"] is not None else "N/A",
            f"{stats['avg_steps']:.2f}",
            f"{stats['avg_model_cost']:.3f}",
            f"{stats['avg_prm_cost']:.3f}",
            f"{stats['avg_total_cost']:.3f}",
        ])
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
                f"{setting} + base fallback", prm_label,
                str(combined_res),
                f"{100 * combined_res / n:.2f}",
                f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
                f"{total_steps:.2f}",
                f"{total_model_cost:.3f}",
                f"{total_prm_cost:.3f}",
                f"{total_cost:.3f}",
            ])

    print(f"\n=== Full SWE-Bench Verified ({n} samples, max 75 steps) - Stats ===")
    print_csv(stats_headers, stats_rows)

    # ── Exit Status Table (CSV) — interleaved ──
    KNOWN_STATUSES = ["Submitted", "LimitsExceeded", "ContextWindowExceededError"]
    STATUS_LABELS = ["Submitted", "Limits Exceeded", "Context Window Exceeded"]
    exit_headers = ["Setting", "PRM Model"] + STATUS_LABELS + ["Other", "Total"]

    exit_rows = []
    for (setting, prm_label, stats), (_, _, dirname) in zip(all_stats, RUNS):
        if stats is None:
            exit_rows.append([setting, prm_label] + ["N/A"] * 5)
            continue
        ec = stats["exit_counts"]
        known_sum = sum(ec.get(s, 0) for s in KNOWN_STATUSES)
        total = sum(ec.values())
        other = total - known_sum
        exit_rows.append([
            setting, prm_label,
            *[str(ec.get(s, 0)) for s in KNOWN_STATUSES],
            str(other), str(total),
        ])
        if dirname in fallback_map:
            fb = fallback_map[dirname]
            fb_ec = fb["fb_exit_counts"]
            combined_submitted = fb["combined_submitted"]
            fb_limits = fb_ec.get("LimitsExceeded", 0)
            fb_ctx = fb_ec.get("ContextWindowExceededError", 0)
            fb_known = fb_ec.get("Submitted", 0) + fb_limits + fb_ctx
            fb_other = sum(fb_ec.values()) - fb_known
            exit_rows.append([
                f"{setting} + base fallback", prm_label,
                str(combined_submitted),
                str(fb_limits),
                str(fb_ctx),
                str(fb_other),
                str(stats["n_found"]),
            ])

    print(f"\n=== Full SWE-Bench Verified ({n} samples, max 75 steps) - Exit Statuses ===")
    print_csv(exit_headers, exit_rows)

    # ── Save CSV files ──
    out_dir = parent
    stats_csv = out_dir / "full500_stats.csv"
    with open(stats_csv, "w") as f:
        f.write(",".join(stats_headers) + "\n")
        for row in stats_rows:
            f.write(",".join(row) + "\n")
    print(f"\nSaved: {stats_csv}", file=sys.stderr)

    exit_csv = out_dir / "full500_exit_statuses.csv"
    with open(exit_csv, "w") as f:
        f.write(",".join(exit_headers) + "\n")
        for row in exit_rows:
            f.write(",".join(row) + "\n")
    print(f"Saved: {exit_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
