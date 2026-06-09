#!/usr/bin/env python3
"""Compute additional metrics for a SWE-bench results directory.

Adds, on top of the existing report.json:
  * Localization rate: fraction of submitted instances whose model_patch
    touches at least one file that the gold patch also touches.
  * Stuck-in-loop rate: fraction of all instances where the agent emitted
    3 (or more) identical consecutive bash commands at any point.

Both metrics read from preds.json + traj.json and the SWE-bench Verified
dataset's gold 'patch' column. Nothing is written to existing files; output
is printed to stdout.

This module is a thin wrapper around the helpers in
``get_stats_full500_prefix.py`` so the underlying definitions stay in one
place.

Usage:
    python3 compute_extra_metrics.py <results_dir> [<results_dir2> ...]
    python3 compute_extra_metrics.py --preds-file preds-autosubmit.json <results_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_stats_full500_prefix import (
    files_in_patch,
    is_stuck_in_loop,
    load_gold_patches,
    localized,
    normalize_file_path,
)

__all__ = [
    "files_in_patch",
    "is_stuck_in_loop",
    "load_gold_patches",
    "localized",
    "normalize_file_path",
    "analyze_run",
]


def analyze_run(
    results_dir: Path,
    gold_patches: dict,
    preds_filename: str = "preds.json",
) -> dict:
    preds_path = results_dir / preds_filename
    if not preds_path.exists():
        raise FileNotFoundError(f"No {preds_filename} in {results_dir}")
    preds = json.loads(preds_path.read_text())

    n_total = len(preds)

    n_with_patch = sum(1 for v in preds.values() if (v.get("model_patch") or "").strip())

    n_localized = 0
    n_localized_denom = 0
    for inst_id, entry in preds.items():
        patch = (entry.get("model_patch") or "").strip()
        if not patch:
            continue
        gold = gold_patches.get(inst_id)
        if gold is None:
            continue
        n_localized_denom += 1
        if localized(patch, gold):
            n_localized += 1

    n_loops = 0
    n_traj_seen = 0
    for inst_id in preds:
        traj_path = results_dir / inst_id / f"{inst_id}.traj.json"
        if not traj_path.exists():
            continue
        try:
            traj = json.loads(traj_path.read_text())
        except Exception:
            continue
        n_traj_seen += 1
        if is_stuck_in_loop(traj, k=3):
            n_loops += 1

    n_resolved = None
    report_path = results_dir / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            n_resolved = len(report.get("resolved_ids", []))
        except Exception:
            pass

    return {
        "results_dir": str(results_dir),
        "preds_file": preds_filename,
        "n_total": n_total,
        "n_with_patch": n_with_patch,
        "n_resolved": n_resolved,
        "n_localized": n_localized,
        "n_localized_denom": n_localized_denom,
        "n_traj_seen": n_traj_seen,
        "n_loops": n_loops,
    }


def fmt_pct(n: int, d: int) -> str:
    if not d:
        return "n/a"
    return f"{100*n/d:.1f}%"


def print_report(stats: dict) -> None:
    print(f"\n=== {stats['results_dir']} ({stats['preds_file']}) ===")
    print(f"  Total preds:           {stats['n_total']}")
    print(f"  With non-empty patch:  {stats['n_with_patch']}  ({fmt_pct(stats['n_with_patch'], stats['n_total'])})")
    if stats['n_resolved'] is not None:
        print(f"  Resolved (report):     {stats['n_resolved']}  ({fmt_pct(stats['n_resolved'], stats['n_total'])})")
    print(f"  Localization rate:     {stats['n_localized']}/{stats['n_total']}  ({fmt_pct(stats['n_localized'], stats['n_total'])})  [denom: all 500]")
    print(f"  Stuck-in-loop rate:    {stats['n_loops']}/{stats['n_traj_seen']}  ({fmt_pct(stats['n_loops'], stats['n_traj_seen'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", nargs="+", type=Path,
                    help="One or more results directories to analyze")
    ap.add_argument("--preds-file", default="preds.json",
                    help="Predictions filename inside each results dir (default: preds.json)")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json", action="store_true", help="Output a JSON list of stats")
    args = ap.parse_args()

    print(f"Loading gold patches from {args.dataset} ({args.split})...", file=sys.stderr)
    gold = load_gold_patches(args.dataset, args.split)
    print(f"  {len(gold)} gold patches", file=sys.stderr)

    all_stats = []
    for d in args.results_dirs:
        if not d.is_dir():
            print(f"Skipping (not a dir): {d}", file=sys.stderr)
            continue
        try:
            stats = analyze_run(d, gold, preds_filename=args.preds_file)
        except FileNotFoundError as e:
            print(f"Skipping ({e})", file=sys.stderr)
            continue
        all_stats.append(stats)
        if not args.json:
            print_report(stats)

    if args.json:
        print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
