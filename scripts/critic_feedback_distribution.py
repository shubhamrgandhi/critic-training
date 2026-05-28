#!/usr/bin/env python3
"""For each given results dir, walk every trajectory and collect the TASK_STATUS
the critic reported on each invocation. Report:
  - Total distribution (across all instances with critic feedback)
  - Distribution restricted to RESOLVED instances only
  - Distribution restricted to UNRESOLVED instances only

Status keys recognized:
  "On track"
  "Needs correction"
  "Critical intervention required"

Anything else (parse failures, blank, no TASK_STATUS line, etc.) goes to "Other".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

STATUS_LINE_RE = re.compile(r"TASK_STATUS[\*_\s:]*[\*_\s]*\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)

CANONICAL = {
    "on track": "On track",
    "needs correction": "Needs correction",
    "critical intervention required": "Critical intervention required",
}


def normalize_status(raw: str) -> str:
    if not raw:
        return "Other"
    # Strip markdown emphasis chars (**bold**), quotes, periods, whitespace, hashes
    s = raw.strip()
    s = re.sub(r"^[\*_`\"'#\s]+", "", s)
    s = re.sub(r"[\*_`\"'.\s]+$", "", s)
    s = s.split("#")[0].strip().lower()
    for key, canon in CANONICAL.items():
        if s == key or s.startswith(key):
            return canon
    return "Other"


def extract_statuses_from_traj(traj: dict) -> list[str]:
    """Pull every PRM feedback's TASK_STATUS for one instance.

    Two possible sources in the saved trajectory:
      info.prm_stats.prm_feedback_history  → list of {"after_step": N, "feedback": "..."}
      messages with _supervisor=True → content contains the feedback
    Prefer feedback_history because it's more direct."""
    out: list[str] = []
    info = traj.get("info") or {}
    prm = info.get("prm_stats") or {}

    history = prm.get("prm_feedback_history") or prm.get("prm_feedback_log") or []
    for entry in history:
        text = entry.get("feedback") if isinstance(entry, dict) else str(entry)
        if not text:
            continue
        m = STATUS_LINE_RE.search(text)
        if m:
            out.append(normalize_status(m.group(1)))
        else:
            out.append("Other")

    if out:
        return out

    # Fallback: scan supervisor messages
    for msg in traj.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        if not msg.get("_supervisor"):
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        m = STATUS_LINE_RE.search(content)
        if m:
            out.append(normalize_status(m.group(1)))
        else:
            out.append("Other")
    return out


def analyze_run(results_dir: Path, resolved_ids: set[str]) -> dict:
    """Walk every instance dir, accumulate status counts split by resolved/unresolved."""
    total_counter = Counter()
    resolved_counter = Counter()
    unresolved_counter = Counter()

    instances_with_feedback = 0
    instances_total = 0
    feedback_total = 0

    for inst in sorted(os.listdir(results_dir)):
        p = results_dir / inst
        if not p.is_dir():
            continue
        traj_path = p / f"{inst}.traj.json"
        if not traj_path.exists():
            continue
        try:
            traj = json.loads(traj_path.read_text())
        except Exception:
            continue
        instances_total += 1
        statuses = extract_statuses_from_traj(traj)
        if not statuses:
            continue
        instances_with_feedback += 1
        feedback_total += len(statuses)
        is_resolved = inst in resolved_ids
        for s in statuses:
            total_counter[s] += 1
            (resolved_counter if is_resolved else unresolved_counter)[s] += 1

    return {
        "results_dir": str(results_dir),
        "instances_total": instances_total,
        "instances_with_feedback": instances_with_feedback,
        "feedback_total": feedback_total,
        "total_counter": dict(total_counter),
        "resolved_counter": dict(resolved_counter),
        "unresolved_counter": dict(unresolved_counter),
    }


def load_resolved_ids(results_dir: Path) -> set[str]:
    rep = results_dir / "report.json"
    if not rep.exists():
        return set()
    try:
        return set(json.loads(rep.read_text()).get("resolved_ids", []))
    except Exception:
        return set()


CATS = ["On track", "Needs correction", "Critical intervention required", "Other"]


def fmt_row(label: str, counter: dict, total: int) -> str:
    parts = [f"{label:<24}"]
    parts.append(f"n={total:>5}")
    for cat in CATS:
        c = counter.get(cat, 0)
        pct = (100 * c / total) if total else 0
        parts.append(f"{cat[:8]:>8}: {pct:5.1f}%")
    return "  ".join(parts)


def print_report(stats: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"   path: {stats['results_dir']}")
    n_inst = stats["instances_total"]
    n_with_fb = stats["instances_with_feedback"]
    n_fb = stats["feedback_total"]
    print(f"   instances: {n_with_fb}/{n_inst} have feedback, total feedback entries: {n_fb}")
    total = sum(stats["total_counter"].values())
    res_n = sum(stats["resolved_counter"].values())
    unres_n = sum(stats["unresolved_counter"].values())

    # Header
    cat_hdrs = "   ".join(f"{c[:8]:>8}" for c in CATS)
    print(f"   {'split':<13} {'feedbacks':>9}  {cat_hdrs}")
    for label_, counter, denom in [
        ("Total       ", stats["total_counter"], total),
        ("Resolved    ", stats["resolved_counter"], res_n),
        ("Unresolved  ", stats["unresolved_counter"], unres_n),
    ]:
        cells = []
        for cat in CATS:
            c = counter.get(cat, 0)
            pct = (100 * c / denom) if denom else 0
            cells.append(f"{pct:>7.1f}%")
        print(f"   {label_} {denom:>9}  {'   '.join(cells)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+",
                    help="One or more 'LABEL=PATH' or just 'PATH' entries")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = []
    for raw in args.entries:
        if "=" in raw:
            label, _, path_str = raw.partition("=")
        else:
            label, path_str = "", raw
        d = Path(path_str)
        if not d.is_dir():
            print(f"Skip (not a dir): {d}", file=sys.stderr)
            continue
        resolved = load_resolved_ids(d)
        stats = analyze_run(d, resolved)
        out.append(stats)
        if not args.json:
            print_report(stats, label or d.name)
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
