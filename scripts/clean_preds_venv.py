#!/usr/bin/env python3
"""Strip .venv/ (and other noise) from patches in a preds.json file.

matplotlib R2E-Gym containers ship with a pre-existing .venv/ in /testbed that
isn't gitignored, so `git add -A && git diff --cached` sweeps its entire contents
into the submission. This script removes those diff hunks post-hoc so preds.json
contains only real code changes.

Usage:
    python3 clean_preds_venv.py <preds.json> [--dry-run]

By default, writes back to the same file after backing up to <file>.bak.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


# Patterns of top-level paths whose diffs should always be stripped.
# These are virtualenv / build-artifact / test-output paths that agents
# should never be legitimately modifying.
STRIP_PATH_PATTERNS = [
    r"\.venv(/|$)",
    r"venv(/|$)",
    r"__pycache__(/|$)",
    r"\.git(/|$)",
    r"\.pytest_cache(/|$)",
    r"\.mypy_cache(/|$)",
]

STRIP_RE = re.compile("|".join(STRIP_PATH_PATTERNS))


def split_diff_into_file_sections(patch: str) -> list[tuple[str | None, str]]:
    """Split a unified diff into (file_path, section_text) tuples.

    Anything before the first `diff --git` line is a preamble (path=None).
    Each subsequent `diff --git a/PATH b/PATH` starts a new file section.
    """
    sections: list[tuple[str | None, str]] = []
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    for i, part in enumerate(parts):
        if i == 0 and not part.startswith("diff --git "):
            if part.strip():
                sections.append((None, part))
            continue
        m = re.match(r"diff --git a/(\S+) b/\S+", part)
        file_path = m.group(1) if m else None
        sections.append((file_path, part))
    return sections


def clean_patch(patch: str) -> tuple[str, dict]:
    """Remove diff sections whose file path matches STRIP_PATH_PATTERNS.

    Returns (cleaned_patch, stats).
    """
    sections = split_diff_into_file_sections(patch)
    kept = []
    stripped_files = 0
    stripped_bytes = 0
    for file_path, section in sections:
        if file_path and STRIP_RE.search(file_path):
            stripped_files += 1
            stripped_bytes += len(section)
            continue
        kept.append(section)
    return "".join(kept), {
        "stripped_files": stripped_files,
        "stripped_bytes": stripped_bytes,
        "kept_files": sum(1 for f, _ in sections if f and not STRIP_RE.search(f)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip .bak backup (default: backup first)")
    args = parser.parse_args()

    if not args.preds_path.exists():
        print(f"ERROR: {args.preds_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.preds_path} ({args.preds_path.stat().st_size / 1e9:.2f} GB)...")
    with open(args.preds_path) as f:
        data = json.load(f)

    total_before = 0
    total_after = 0
    total_stripped_files = 0
    affected_instances = 0
    for inst_id, entry in data.items():
        patch = entry.get("model_patch", "")
        before = len(patch)
        total_before += before
        cleaned, stats = clean_patch(patch)
        entry["model_patch"] = cleaned
        total_after += len(cleaned)
        if stats["stripped_files"] > 0:
            affected_instances += 1
            total_stripped_files += stats["stripped_files"]

    savings = total_before - total_after
    print(f"\nInstances: {len(data)}")
    print(f"Affected:  {affected_instances}")
    print(f"Stripped {total_stripped_files:,} file diffs ({savings:,} bytes = {savings/1e9:.2f} GB)")
    print(f"Before: {total_before/1e9:.2f} GB  →  After: {total_after/1e9:.2f} GB")

    if args.dry_run:
        print("\n(dry-run, no files written)")
        return

    if not args.no_backup:
        bak = args.preds_path.with_suffix(args.preds_path.suffix + ".bak")
        if bak.exists():
            print(f"\nBackup {bak} already exists, not overwriting.")
        else:
            print(f"\nBacking up to {bak}...")
            shutil.copy2(args.preds_path, bak)

    print(f"Writing cleaned preds back to {args.preds_path}...")
    with open(args.preds_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Done. New size: {args.preds_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
