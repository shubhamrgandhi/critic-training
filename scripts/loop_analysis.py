#!/usr/bin/env python3
"""Thorough analysis of stuck-vs-unstuck and resolved-vs-unresolved in critic runs.

Reports for each run:
  * 2x2 contingency: {resolved, unresolved} x {stuck, unstuck}
  * Distribution of where in the trajectory the loop starts (early vs late)
  * Trajectory-length distributions per cell
  * Whether the loop forms after a critic call or independent of it:
      - distance (in steps) from the most recent critic call to the first
        repeated command in the loop window
  * Cross-run, length-matched loop rate: among instances with >=L steps
    in BOTH the base run and the critic run, what's the loop rate?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASH_RE = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)


def get_command_sequence(traj: dict) -> list[tuple[int, str, bool]]:
    """Return list of (step_index, command_or_text, is_supervisor) for the
    trajectory's assistant messages. step_index is the agent step number
    (1-indexed). Supervisor entries are inserted at their position in messages
    but not counted as agent steps."""
    out = []
    step = 0
    for i, m in enumerate(traj.get("messages", []) or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        c = m.get("content", "") or ""
        is_sup = bool(m.get("_supervisor"))
        if role == "assistant" and not is_sup:
            step += 1
            bashes = BASH_RE.findall(c if isinstance(c, str) else "")
            cmd = bashes[0].strip() if len(bashes) == 1 else (c.strip() if isinstance(c, str) else "")
            out.append((step, cmd, False))
        elif is_sup:
            # Supervisor injection appears between agent steps; mark its position
            # using the current step count (i.e. it was injected after step `step`)
            out.append((step, "<critic>", True))
    return out


def find_first_loop_window(seq: list[tuple[int, str, bool]], k: int = 3) -> int | None:
    """Find the first 0-indexed agent-step position where K consecutive
    assistant commands match. Returns the step index of the first of K, or
    None if no loop."""
    cmds = [(s, c) for s, c, is_sup in seq if not is_sup]
    for i in range(len(cmds) - k + 1):
        if all(cmds[i + j][1] == cmds[i][1] for j in range(k)):
            return cmds[i][0]
    return None


def critic_step_positions(seq: list[tuple[int, str, bool]]) -> list[int]:
    """Step numbers (after which) the critic was invoked."""
    return [s for s, c, is_sup in seq if is_sup]


def analyze_run(results_dir: Path, k_loop: int = 3) -> dict:
    rep_path = results_dir / "report.json"
    resolved_ids = set()
    if rep_path.exists():
        try:
            resolved_ids = set(json.loads(rep_path.read_text()).get("resolved_ids", []))
        except Exception:
            pass

    rows = []  # per-instance dicts
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
        seq = get_command_sequence(traj)
        loop_start = find_first_loop_window(seq, k=k_loop)
        crit_steps = critic_step_positions(seq)
        n_steps = max([s for s, c, is_sup in seq if not is_sup], default=0)
        # Distance from most recent critic call before loop_start to loop_start
        dist_to_loop = None
        if loop_start is not None and crit_steps:
            prior_critic = [c for c in crit_steps if c < loop_start]
            if prior_critic:
                dist_to_loop = loop_start - prior_critic[-1]
            else:
                dist_to_loop = loop_start  # critic never fired before loop
        info = traj.get("info", {}) or {}
        rows.append({
            "id": inst,
            "resolved": inst in resolved_ids,
            "stuck": loop_start is not None,
            "loop_start": loop_start,
            "n_steps": n_steps,
            "n_critic_calls": len(crit_steps),
            "had_critic_before_loop": dist_to_loop is not None,
            "dist_critic_to_loop": dist_to_loop,
            "exit_status": info.get("exit_status"),
        })

    return {"results_dir": str(results_dir), "rows": rows}


def print_report(name: str, stats: dict, base_stats: dict | None = None) -> None:
    rows = stats["rows"]
    n = len(rows)
    n_res = sum(r["resolved"] for r in rows)
    n_unres = n - n_res
    n_stuck = sum(r["stuck"] for r in rows)
    n_unstuck = n - n_stuck

    res_stuck = sum(1 for r in rows if r["resolved"] and r["stuck"])
    res_unstuck = sum(1 for r in rows if r["resolved"] and not r["stuck"])
    unres_stuck = sum(1 for r in rows if not r["resolved"] and r["stuck"])
    unres_unstuck = sum(1 for r in rows if not r["resolved"] and not r["stuck"])

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    print(f"\n=== {name} (N={n}) ===")
    print(f"  path: {stats['results_dir']}")
    print()
    print(f"  Overall stuck rate: {n_stuck}/{n} = {pct(n_stuck, n)}")
    print(f"  Resolution rate:    {n_res}/{n} = {pct(n_res, n)}")
    print()
    print(f"  ┌─────────────────────┬──────────┬──────────┐")
    print(f"  │                     │  Stuck   │ Unstuck  │")
    print(f"  ├─────────────────────┼──────────┼──────────┤")
    print(f"  │ Resolved            │ {res_stuck:>4}     │ {res_unstuck:>4}     │  ({pct(res_stuck, n_res)} of res stuck)")
    print(f"  │ Unresolved          │ {unres_stuck:>4}     │ {unres_unstuck:>4}     │  ({pct(unres_stuck, n_unres)} of unres stuck)")
    print(f"  └─────────────────────┴──────────┴──────────┘")
    if n_stuck:
        print(f"     of stuck:  {pct(res_stuck, n_stuck)} resolved, {pct(unres_stuck, n_stuck)} unresolved")
    if n_unstuck:
        print(f"     of unstuck:{pct(res_unstuck, n_unstuck)} resolved, {pct(unres_unstuck, n_unstuck)} unresolved")
    print()
    # Trajectory length stats per cell
    def avg_len(filter_fn):
        xs = [r["n_steps"] for r in rows if filter_fn(r)]
        return (sum(xs) / len(xs)) if xs else 0
    print(f"  Avg n_steps:")
    print(f"     resolved+stuck:    {avg_len(lambda r: r['resolved'] and r['stuck']):.1f}")
    print(f"     resolved+unstuck:  {avg_len(lambda r: r['resolved'] and not r['stuck']):.1f}")
    print(f"     unresolved+stuck:  {avg_len(lambda r: not r['resolved'] and r['stuck']):.1f}")
    print(f"     unresolved+unstuck:{avg_len(lambda r: not r['resolved'] and not r['stuck']):.1f}")
    print()
    # Loop-start distribution among stuck
    starts = [r["loop_start"] for r in rows if r["stuck"]]
    if starts:
        starts_sorted = sorted(starts)
        med = starts_sorted[len(starts_sorted)//2]
        print(f"  Loop start step (among stuck):")
        print(f"     min={min(starts)}, median={med}, max={max(starts)}")
        bins = Counter()
        for s in starts:
            if s <= 20: bins["1-20"] += 1
            elif s <= 50: bins["21-50"] += 1
            elif s <= 100: bins["51-100"] += 1
            else: bins["101+"] += 1
        for label in ["1-20", "21-50", "51-100", "101+"]:
            print(f"     steps {label}: {bins.get(label, 0)} ({pct(bins.get(label, 0), len(starts))})")
    print()
    # Critic-loop adjacency
    has_crit_calls = [r for r in rows if r["n_critic_calls"] > 0]
    stuck_with_crit_before = [r for r in has_crit_calls if r["stuck"] and r["had_critic_before_loop"]]
    if has_crit_calls:
        print(f"  Critic-loop adjacency (among instances with >=1 critic call before any loop):")
        print(f"     stuck instances with critic before loop: {len(stuck_with_crit_before)}/{sum(1 for r in has_crit_calls if r['stuck'])}")
        dists = [r["dist_critic_to_loop"] for r in stuck_with_crit_before if r["dist_critic_to_loop"] is not None]
        if dists:
            dists.sort()
            print(f"     dist (steps) from last critic to loop start:")
            print(f"        min={min(dists)}, median={dists[len(dists)//2]}, max={max(dists)}")
            in_first_5 = sum(1 for d in dists if d <= 5)
            print(f"        <=5 steps after critic: {in_first_5}/{len(dists)} ({pct(in_first_5, len(dists))})")

    # Length-controlled comparison vs base
    if base_stats is not None:
        base_rows = {r["id"]: r for r in base_stats["rows"]}
        common_ids = [r["id"] for r in rows if r["id"] in base_rows]
        # Loop rate among instances >= L steps in BOTH
        for L in (30, 50, 100):
            both_long = [
                r for r in rows
                if r["id"] in base_rows and r["n_steps"] >= L and base_rows[r["id"]]["n_steps"] >= L
            ]
            if not both_long:
                continue
            this_loops = sum(1 for r in both_long if r["stuck"])
            base_loops = sum(1 for r in both_long if base_rows[r["id"]]["stuck"])
            print()
            print(f"  Length-matched loop rate (instances with >= {L} steps in both runs, n={len(both_long)}):")
            print(f"     this run: {this_loops}/{len(both_long)} = {pct(this_loops, len(both_long))}")
            print(f"     base:     {base_loops}/{len(both_long)} = {pct(base_loops, len(both_long))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", nargs="+", type=Path)
    ap.add_argument("--base-dir", type=Path,
                    help="Base run for length-matched comparison (defaults to first results_dir)")
    ap.add_argument("--k", type=int, default=3, help="Loop window size (default 3)")
    args = ap.parse_args()

    base_stats = analyze_run(args.base_dir, k_loop=args.k) if args.base_dir else None

    for d in args.results_dirs:
        if not d.is_dir():
            print(f"Skip: {d}", file=sys.stderr)
            continue
        stats = analyze_run(d, k_loop=args.k)
        # Use first run as base if --base-dir not provided
        bs = base_stats if base_stats and d != args.base_dir else None
        print_report(d.name, stats, bs)


if __name__ == "__main__":
    main()
