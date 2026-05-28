#!/usr/bin/env python3
"""Retroactively strip .venv/ (and other noise) from existing trajectory files.

Walks a directory recursively, finds all *.traj.json files, and rewrites each
one by stripping noisy diff sections from:
  - info.submission
  - every message's content

Skips files that don't contain noise. Prints summary stats.

Usage:
    python3 clean_traj_venv.py <root_dir> [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running this script standalone without installing the package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mini-swe-agent" / "src"))

from minisweagent.run.utils.diff_cleanup import clean_diff_text, clean_message_content  # noqa: E402


def clean_traj_file(path: Path, dry_run: bool) -> tuple[bool, int, int]:
    """Returns (changed, bytes_before, bytes_after)."""
    try:
        before_size = path.stat().st_size
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARN: failed to read {path}: {e}", file=sys.stderr)
        return False, 0, 0

    changed = False

    info = data.get("info") or {}
    sub = info.get("submission")
    if isinstance(sub, str):
        new_sub = clean_diff_text(sub)
        if new_sub != sub:
            info["submission"] = new_sub
            changed = True

    msgs = data.get("messages") or []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        new_content = clean_message_content(content)
        if new_content != content:
            msg["content"] = new_content
            changed = True

    if not changed:
        return False, before_size, before_size

    if dry_run:
        new_text = json.dumps(data, indent=2)
        return True, before_size, len(new_text.encode("utf-8"))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    after_size = path.stat().st_size
    return True, before_size, after_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path,
                        help="Directory to walk recursively for *.traj.json files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write changes; just show what would be cleaned")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: {args.root} not found", file=sys.stderr)
        sys.exit(1)

    trajs = sorted(args.root.rglob("*.traj.json"))
    print(f"Found {len(trajs)} trajectory files under {args.root}")
    if not trajs:
        return

    total_before = 0
    total_after = 0
    n_changed = 0
    n_skipped = 0
    biggest_savings = []  # list of (savings_bytes, path)

    for i, p in enumerate(trajs, 1):
        changed, before, after = clean_traj_file(p, args.dry_run)
        total_before += before
        total_after += after
        if changed:
            n_changed += 1
            biggest_savings.append((before - after, str(p)))
        else:
            n_skipped += 1
        if i % 100 == 0 or i == len(trajs):
            print(f"  Progress: {i}/{len(trajs)}  changed={n_changed}  saved_so_far={(total_before-total_after)/1e9:.2f} GB")

    biggest_savings.sort(reverse=True)
    print()
    print("=" * 60)
    print(f"Total files:     {len(trajs)}")
    print(f"Cleaned:         {n_changed}")
    print(f"Untouched:       {n_skipped}")
    print(f"Bytes before:    {total_before:,} ({total_before/1e9:.2f} GB)")
    print(f"Bytes after:     {total_after:,} ({total_after/1e9:.2f} GB)")
    print(f"Savings:         {(total_before-total_after)/1e9:.2f} GB")
    if biggest_savings:
        print()
        print("Top 10 biggest savings:")
        for sv, p in biggest_savings[:10]:
            print(f"  {sv/1e6:>8.1f} MB  {p}")
    if args.dry_run:
        print("\n(dry-run, no files written)")


if __name__ == "__main__":
    main()
