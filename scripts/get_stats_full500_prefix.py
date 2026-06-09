#!/usr/bin/env python3
"""
Compute stats for full SWE-Bench Verified (500 instances), prefix runs.

Usage:
    python get_stats_full500_prefix.py
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)
BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)


def load_gold_patches(dataset: str = "princeton-nlp/SWE-bench_Verified",
                      split: str = "test") -> dict:
    """Return {instance_id: gold patch string}. Empty dict on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        return {row["instance_id"]: row["patch"] for row in ds}
    except Exception as e:
        print(f"WARNING: could not load gold patches ({e}); "
              f"localization will be N/A", file=sys.stderr)
        return {}


def files_in_patch(patch: str) -> set:
    if not patch:
        return set()
    return set(DIFF_FILE_RE.findall(patch))


def normalize_file_path(p: str) -> str:
    """SWE-bench gold patches use repo-relative paths. Strip leading ./ or
    testbed/ prefixes that occasionally appear in agent patches."""
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("testbed/"):
        p = p[len("testbed/"):]
    return p


def localized(model_patch: str, gold_patch: str) -> bool:
    """File-level recall: at least one file the agent modified appears in the
    gold patch's file set."""
    m = {normalize_file_path(f) for f in files_in_patch(model_patch)}
    g = {normalize_file_path(f) for f in files_in_patch(gold_patch)}
    if not m or not g:
        return False
    return bool(m & g)


def is_stuck_in_loop(traj: dict, k: int = 3) -> bool:
    """True if the agent emits the same bash command in K consecutive assistant
    steps at any point in the trajectory. Compares the extracted bash command
    rather than full content. Falls back to full content if no bash block."""
    msgs = traj.get("messages", [])
    cmds = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        bashes = BASH_BLOCK_RE.findall(content)
        if len(bashes) == 1:
            cmds.append(bashes[0].strip())
        else:
            cmds.append(content.strip())

    if len(cmds) < k:
        return False
    for i in range(len(cmds) - k + 1):
        window = cmds[i: i + k]
        if all(c == window[0] for c in window):
            return True
    return False

# Per-token pricing (full input, no cache discounts).
# See scripts/get_stats_mini50.py for rationale.
MODEL_PRICING = {
    # Opus intentionally omitted so recompute_prm_cost falls through to the stored cost,
    # which includes cache-write premium (1.25x input) — the most expensive, as the user requested.
    "facebook/cwm":                     {"input": 8e-8,  "output": 2.8e-7},
    "SWE-bench/SWE-agent-LM-32B":       {"input": 8e-8,  "output": 2.8e-7},
    "Qwen/Qwen3-8B":                                                                                          {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                                       {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                            {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                                   {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                                   {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":           {"input": 5e-8, "output": 2e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn":   {"input": 5e-8, "output": 2e-7},
    "bedrock/qwen.qwen3-32b-v1:0":                                                                            {"input": 8e-8, "output": 2.8e-7},
    "bedrock/qwen.qwen3-next-80b-a3b":                                                                        {"input": 9e-8, "output": 7.8e-7},
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
    """Recompute PRM cost using full input pricing."""
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


# Reference run that defines the 500-instance set
REFERENCE_RUN = "singularity_edit_obs_final_only_0_cwm"

# Complete full500 runs only (preds.json has 500 entries AND report.json exists).
# (group, experiment, prm_label, dirname)
RUNS = [
    # Base
    ("Base CWM", "run 0", "-",
     "singularity_edit_obs_final_only_0_cwm"),
    # Claude Opus PRM runs — at the top right below base
    ("PRM issue_res k5", "Claude Opus", "Claude Opus",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"),
    ("PRM issue_res k10", "Claude Opus", "Claude Opus",
     "singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_claude-opus-4-6"),
    ("PRM instructions k5", "Claude Opus", "Claude Opus",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_claude-opus-4-6"),
    ("PRM instructions k10", "Claude Opus", "Claude Opus",
     "singularity_edit_obs_final_only_prm_issue_res_instructions_k10_0_cwm_prm_claude-opus-4-6"),
    # Qwen3-8B SFT (clean) PRM runs
    ("PRM issue_res k5", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
    ("PRM issue_res k10", "SFT v1 (clean)", "SFT v1 (clean)",
     "singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"),
]

BASE_CWM_DIR = "singularity_edit_obs_final_only_0_cwm"


def get_instance_ids(run_dir: Path) -> set[str]:
    """Get the 500 instance IDs for SWE-Bench Verified.
    Falls back to subdirectories of run_dir if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        return set(ds["instance_id"])
    except Exception:
        return {p.name for p in run_dir.iterdir() if p.is_dir()}


def compute_stats(run_dir: Path, instance_ids: set[str],
                  gold_patches: dict | None = None) -> dict:
    report_path = run_dir / "report.json"
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            resolved_ids = set(json.load(f).get("resolved_ids", []))

    resolved_in_subset = resolved_ids & instance_ids

    # Load preds.json for localization (file-level recall vs gold patch).
    preds = {}
    preds_path = run_dir / "preds.json"
    if preds_path.exists():
        try:
            with open(preds_path) as f:
                preds = json.load(f)
        except Exception:
            preds = {}

    exit_counts = Counter()
    total_steps = 0
    total_cost = 0.0
    total_prm_cost = 0.0
    n_found = 0
    n_loops = 0
    n_traj_seen = 0
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

        n_traj_seen += 1
        if is_stuck_in_loop(traj, k=3):
            n_loops += 1

    # Localization: walk preds.json (matches compute_extra_metrics convention,
    # denominator is full subset size).
    n_localized = 0
    n_localized_denom = 0
    if gold_patches:
        for inst_id in instance_ids:
            entry = preds.get(inst_id)
            if not entry:
                continue
            patch = (entry.get("model_patch") or "").strip()
            if not patch:
                continue
            gold = gold_patches.get(inst_id)
            if gold is None:
                continue
            n_localized_denom += 1
            if localized(patch, gold):
                n_localized += 1

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
        "n_localized": n_localized,
        "n_localized_denom": n_localized_denom,
        "n_loops": n_loops,
        "n_traj_seen": n_traj_seen,
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
                           instance_ids: set,
                           gold_patches: dict | None = None) -> dict:
    """Compute fallback-only stats: for instances where PRM didn't submit,
    what do we get from the base CWM run?"""
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

    # Preds for combined localization computation.
    prm_preds = {}
    base_preds = {}
    try:
        with open(prm_dir / "preds.json") as f:
            prm_preds = json.load(f)
    except Exception:
        pass
    try:
        with open(base_dir / "preds.json") as f:
            base_preds = json.load(f)
    except Exception:
        pass

    fb_resolved = 0
    fb_steps = 0
    fb_cost = 0.0
    fb_exit_counts = Counter()
    n_fallback = 0
    combined_resolved = 0
    prm_submitted = 0
    fb_loops = 0
    fb_traj_seen = 0
    prm_submitted_loops = 0  # loops in PRM trajs that *did* submit

    for instance_id in sorted(instance_ids):
        prm_traj = load_traj(prm_dir, instance_id)
        if prm_traj is None:
            continue

        prm_exit = prm_traj.get("info", {}).get("exit_status", "unknown")
        if prm_exit == "Submitted":
            prm_submitted += 1
            if is_stuck_in_loop(prm_traj, k=3):
                prm_submitted_loops += 1
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
                fb_traj_seen += 1
                if is_stuck_in_loop(base_traj, k=3):
                    fb_loops += 1
                if instance_id in base_resolved:
                    fb_resolved += 1
                    combined_resolved += 1

    fb_submitted = fb_exit_counts.get("Submitted", 0)

    # Combined localization: PRM patch when PRM submitted, else base patch.
    combined_localized = 0
    combined_localized_denom = 0
    if gold_patches:
        for inst_id in instance_ids:
            prm_traj = load_traj(prm_dir, inst_id)
            if prm_traj is None:
                continue
            prm_exit = prm_traj.get("info", {}).get("exit_status", "unknown")
            if prm_exit == "Submitted":
                entry = prm_preds.get(inst_id)
            else:
                entry = base_preds.get(inst_id)
            if not entry:
                continue
            patch = (entry.get("model_patch") or "").strip()
            if not patch:
                continue
            gold = gold_patches.get(inst_id)
            if gold is None:
                continue
            combined_localized_denom += 1
            if localized(patch, gold):
                combined_localized += 1

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
        "fb_loops": fb_loops,
        "fb_traj_seen": fb_traj_seen,
        "prm_submitted_loops": prm_submitted_loops,
        "combined_localized": combined_localized,
        "combined_localized_denom": combined_localized_denom,
    }


def print_csv(headers: list[str], rows: list[list[str]]):
    print(",".join(headers))
    for row in rows:
        print(",".join(row))


def main():
    parser = argparse.ArgumentParser(description="Stats for full SWE-Bench Verified (500 samples)")
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"))
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

    print("Loading gold patches...", file=sys.stderr)
    gold_patches = load_gold_patches()
    print(f"  {len(gold_patches)} gold patches loaded", file=sys.stderr)

    all_stats = []
    for group, experiment, prm_label, dirname in RUNS:
        run_dir = parent / dirname
        print(f"  {group} / {experiment} ... ", file=sys.stderr, end="", flush=True)
        if not run_dir.exists():
            print("NOT FOUND", file=sys.stderr)
            all_stats.append((group, experiment, prm_label, None))
            continue
        # Skip runs that don't have a report.json yet (full500 eval not complete)
        if not (run_dir / "report.json").exists():
            print("NO REPORT (skipping)", file=sys.stderr)
            all_stats.append((group, experiment, prm_label, None))
            continue
        stats = compute_stats(run_dir, instance_ids, gold_patches=gold_patches)
        print(f"found {stats['n_found']}/{n}, resolved {stats['resolved']}", file=sys.stderr)
        if stats["n_missing"] > 0:
            print(f"    WARNING: {stats['n_missing']} missing instances", file=sys.stderr)
        all_stats.append((group, experiment, prm_label, stats))

    # ── Compute fallback stats for non-base runs ──
    base_dir = parent / BASE_CWM_DIR
    fallback_map = {}
    if base_dir.exists():
        print(f"\nComputing fallback stats (base CWM fallback)...", file=sys.stderr)
        for group, experiment, prm_label, dirname in RUNS:
            if group == "Base CWM":
                continue
            prm_dir = parent / dirname
            if not prm_dir.exists() or not (prm_dir / "report.json").exists():
                continue
            fb = compute_fallback_stats(prm_dir, base_dir, instance_ids,
                                        gold_patches=gold_patches)
            print(f"  {group} / {experiment}: {fb['n_fallback']} fallback, "
                  f"+{fb['fb_resolved']} resolved from base, "
                  f"combined {fb['combined_resolved']}", file=sys.stderr)
            fallback_map[dirname] = fb

    # ── Stats Table (CSV) — interleaved with fallback rows ──
    stats_headers = [
        "Group", "Experiment",
        f"Resolved (/{n})",
        f"Submitted (/{n})",
        f"Res. Rate (/{n})",
        "Res. Rate (/submitted)",
        f"Avg Steps (/{n})",
        f"Avg Model Cost ($) (/{n})",
        f"Avg PRM Cost ($) (/{n})",
        f"Avg Total Cost ($) (/{n})",
        f"Localization (/{n})",
        f"Stuck-in-loop (/{n})",
    ]

    def _pct(num: int, denom: int) -> str:
        if not denom:
            return "N/A"
        return f"{100 * num / denom:.2f}"

    stats_rows = []
    for (group, experiment, prm_label, stats), (_, _, _, dirname) in zip(
            all_stats, RUNS):
        if stats is None:
            continue  # skip incomplete runs entirely
        stats_rows.append([
            group, experiment,
            str(stats["resolved"]),
            str(stats["submitted"]),
            f"{stats['res_rate_subset']:.2f}",
            f"{stats['res_rate_submitted']:.2f}" if stats["res_rate_submitted"] is not None else "N/A",
            f"{stats['avg_steps']:.2f}",
            f"{stats['avg_model_cost']:.3f}",
            f"{stats['avg_prm_cost']:.3f}",
            f"{stats['avg_total_cost']:.3f}",
            _pct(stats["n_localized"], n),
            _pct(stats["n_loops"], n),
        ])
        if dirname in fallback_map:
            fb = fallback_map[dirname]
            combined_res = fb["combined_resolved"]
            combined_sub = fb["combined_submitted"]
            res_rate_sub = (100 * combined_res / combined_sub) if combined_sub > 0 else None
            total_steps = stats["avg_steps"] + fb["fb_steps"] / n
            total_model_cost = stats["avg_model_cost"] + fb["fb_cost"] / n
            total_prm_cost = stats["avg_prm_cost"]
            total_cost = total_model_cost + total_prm_cost
            # Combined loop count: PRM-submitted instances use PRM traj loops,
            # fallback instances use base traj loops.
            combined_loops = fb["prm_submitted_loops"] + fb["fb_loops"]
            stats_rows.append([
                group, f"{experiment} + base fallback",
                str(combined_res),
                str(combined_sub),
                f"{100 * combined_res / n:.2f}",
                f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
                f"{total_steps:.2f}",
                f"{total_model_cost:.3f}",
                f"{total_prm_cost:.3f}",
                f"{total_cost:.3f}",
                _pct(fb["combined_localized"], n),
                _pct(combined_loops, n),
            ])

    print(f"\n=== Full SWE-Bench Verified ({n} samples) - Stats ===")
    print_csv(stats_headers, stats_rows)

    # ── Exit Status Table (CSV) — interleaved ──
    KNOWN_STATUSES = ["Submitted", "LimitsExceeded", "ContextWindowExceededError"]
    STATUS_LABELS = ["Submitted", "Limits Exceeded", "Context Window Exceeded"]
    exit_headers = ["Group", "Experiment"] + STATUS_LABELS + ["Other", "Total"]

    exit_rows = []
    for (group, experiment, prm_label, stats), (_, _, _, dirname) in zip(
            all_stats, RUNS):
        if stats is None:
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
        if dirname in fallback_map:
            fb = fallback_map[dirname]
            fb_ec = fb["fb_exit_counts"]
            combined_submitted = fb["combined_submitted"]
            fb_limits = fb_ec.get("LimitsExceeded", 0)
            fb_ctx = fb_ec.get("ContextWindowExceededError", 0)
            fb_known = fb_ec.get("Submitted", 0) + fb_limits + fb_ctx
            fb_other = sum(fb_ec.values()) - fb_known
            exit_rows.append([
                group, f"{experiment} + base fallback",
                str(combined_submitted),
                str(fb_limits),
                str(fb_ctx),
                str(fb_other),
                str(stats["n_found"]),
            ])

    print(f"\n=== Full SWE-Bench Verified ({n} samples) - Exit Statuses ===")
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
