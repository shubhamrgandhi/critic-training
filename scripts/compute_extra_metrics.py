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

Usage:
    python3 compute_extra_metrics.py <results_dir> [<results_dir2> ...]
    python3 compute_extra_metrics.py --preds-file preds-autosubmit.json <results_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)
BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)


def load_gold_patches(dataset: str = "princeton-nlp/SWE-bench_Verified", split: str = "test") -> dict[str, str]:
    """Return {instance_id: gold patch string}."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    return {row["instance_id"]: row["patch"] for row in ds}


def files_in_patch(patch: str) -> set[str]:
    if not patch:
        return set()
    return set(DIFF_FILE_RE.findall(patch))


def normalize_file_path(p: str) -> str:
    """SWE-bench gold patches use repo-relative paths. Agent patches are
    against /testbed (the cloned repo at base_commit). Both should already be
    repo-relative because git diff produces a/<repo-relative> b/<repo-relative>.
    Strip any leading ./ or testbed/ prefix just in case."""
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("testbed/"):
        p = p[len("testbed/"):]
    return p


def localized(model_patch: str, gold_patch: str) -> bool:
    """File-level recall: at least one file the agent modified appears in the
    gold patch's file set. This is the convention most SWE-bench papers report
    (Agentless, SWE-agent analyses, etc.)."""
    m = {normalize_file_path(f) for f in files_in_patch(model_patch)}
    g = {normalize_file_path(f) for f in files_in_patch(gold_patch)}
    if not m or not g:
        return False
    return bool(m & g)


def is_stuck_in_loop(traj: dict, k: int = 3) -> bool:
    """True if the agent emits the same bash command in K consecutive assistant
    steps at any point in the trajectory.

    We compare the extracted bash command (the first triple-backticked block)
    rather than the full assistant message, so that loop detection is about
    repeated *actions*, not repeated reasoning text. Falls back to full content
    if no bash block found.
    """
    msgs = traj.get("messages", [])
    cmds: list[str] = []
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
            # Format error or no/multiple bash blocks — treat as the raw text
            cmds.append(content.strip())

    if len(cmds) < k:
        return False
    for i in range(len(cmds) - k + 1):
        window = cmds[i : i + k]
        if all(c == window[0] for c in window):
            return True
    return False


def analyze_run(
    results_dir: Path,
    gold_patches: dict[str, str],
    preds_filename: str = "preds.json",
) -> dict:
    preds_path = results_dir / preds_filename
    if not preds_path.exists():
        raise FileNotFoundError(f"No {preds_filename} in {results_dir}")
    preds = json.loads(preds_path.read_text())

    n_total = len(preds)

    # Submission detection: a "submitted" pred has a non-empty patch (matches
    # the SWE-bench convention where empty patches are categorized separately).
    n_with_patch = sum(1 for v in preds.values() if (v.get("model_patch") or "").strip())

    # Localization
    n_localized = 0
    n_localized_denom = 0  # patches we could check (have patch and gold)
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

    # Stuck-in-loop: walk trajectories
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

    # Resolved (from existing report.json if present, for context)
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
