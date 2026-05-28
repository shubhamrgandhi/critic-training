#!/usr/bin/env python3
"""
Remove subdirectories and preds.json entries for instances that have
a given exit status (e.g. CalledProcessError, RetryError) in any
exit_statuses_*.yaml file.
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import yaml

DEFAULT_DIR = "/home/srgandhi/tool-overuse/results_singularity/singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_sweagent7b"

# Only these error types are infrastructure failures that should ever be
# removed by this script. Anything else (Submitted, LimitsExceeded,
# ContextWindowExceededError, etc.) is a legitimate completed run and must
# never be deleted, even if a stale yaml file lists it under that status.
ALLOWED_ERROR_TYPES = {"RetryError", "CalledProcessError", "Exception"}


def get_actual_exit_status(results_dir: Path, instance: str) -> str | None:
    """Return the exit_status recorded in the instance's trajectory file, or None
    if the trajectory file doesn't exist / is unreadable.

    The trajectory file is the authoritative source — yaml files can become stale
    when a previously-failed instance is re-run and succeeds.
    """
    traj = results_dir / instance / f"{instance}.traj.json"
    if not traj.exists():
        return None
    try:
        with open(traj) as f:
            data = json.load(f)
        return data.get("info", {}).get("exit_status")
    except (json.JSONDecodeError, IOError):
        return None


def get_error_instances(results_dir: Path, error_type: str) -> set[str]:
    instances = set()
    yaml_files = list(results_dir.glob("exit_statuses_*.yaml"))
    if not yaml_files:
        print(f"No exit_statuses_*.yaml files found in {results_dir}", file=sys.stderr)
        return instances

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        by_status = data.get("instances_by_exit_status", {})
        for instance in by_status.get(error_type, []):
            instances.add(instance)

    return instances


def remove_instance_dirs(results_dir: Path, instances: set[str], dry_run: bool) -> list[str]:
    removed = []
    for instance in sorted(instances):
        instance_dir = results_dir / instance
        if instance_dir.is_dir():
            if dry_run:
                print(f"[dry-run] Would remove directory: {instance_dir}")
            else:
                shutil.rmtree(instance_dir)
                print(f"Removed directory: {instance_dir}")
            removed.append(instance)
        else:
            print(f"Directory not found (skipping): {instance_dir}")
    return removed


def remove_from_preds(results_dir: Path, instances: set[str], dry_run: bool):
    preds_file = results_dir / "preds.json"
    if not preds_file.exists():
        print(f"preds.json not found at {preds_file}", file=sys.stderr)
        return

    if preds_file.stat().st_size == 0:
        print(f"preds.json is empty at {preds_file} (skipping)")
        return

    with open(preds_file) as f:
        preds = json.load(f)

    to_remove = [k for k in preds if k in instances]
    if not to_remove:
        print("No matching entries found in preds.json")
        return

    for key in to_remove:
        if dry_run:
            print(f"[dry-run] Would remove from preds.json: {key}")
        else:
            del preds[key]
            print(f"Removed from preds.json: {key}")

    if not dry_run:
        with open(preds_file, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"Saved updated preds.json ({len(preds)} entries remaining)")


def remove_from_exit_statuses(results_dir: Path, instances: set[str], error_type: str, dry_run: bool):
    yaml_files = list(results_dir.glob("exit_statuses_*.yaml"))
    if not yaml_files:
        print("No exit_statuses_*.yaml files found.", file=sys.stderr)
        return

    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        modified = False

        # Remove from instances_by_exit_status.<error_type>
        by_status = data.get("instances_by_exit_status", {})
        if error_type in by_status:
            if dry_run:
                print(f"[dry-run] Would remove {error_type} key from {yaml_file.name}")
            else:
                del by_status[error_type]
                modified = True
                print(f"Removed {error_type} key from {yaml_file.name}")

        # Remove from exit_status_by_instance
        by_instance = data.get("exit_status_by_instance", {})
        to_remove = [k for k in by_instance if k in instances]
        for key in to_remove:
            if dry_run:
                print(f"[dry-run] Would remove {key} from exit_status_by_instance in {yaml_file.name}")
            else:
                del by_instance[key]
                modified = True
                print(f"Removed {key} from exit_status_by_instance in {yaml_file.name}")

        if modified and not dry_run:
            with open(yaml_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Saved updated {yaml_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove instances with a given error status from a results directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DIR,
        help=f"Results directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--error-type",
        type=str,
        default="CalledProcessError",
        help="Exit status to filter on (default: CalledProcessError)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making any changes",
    )
    args = parser.parse_args()

    results_dir = Path(args.directory)
    if not results_dir.is_dir():
        print(f"Error: directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Safeguard 1: only allow infrastructure-failure error types
    if args.error_type not in ALLOWED_ERROR_TYPES:
        print(
            f"Error: --error-type {args.error_type!r} is not allowed.\n"
            f"This script can only remove infrastructure failures. Allowed types: "
            f"{sorted(ALLOWED_ERROR_TYPES)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = get_error_instances(results_dir, args.error_type)
    if not candidates:
        print(f"No {args.error_type} instances found.")
        return

    # Safeguard 2: cross-check each candidate against its trajectory file.
    # Yaml entries can be stale — an instance that previously failed may have
    # been re-run and succeeded. The trajectory file is authoritative.
    instances = set()
    skipped = []
    for inst in sorted(candidates):
        actual = get_actual_exit_status(results_dir, inst)
        if actual is None:
            # No trajectory file — instance dir may be empty (a previous failed
            # attempt that never produced a traj). Safe to remove.
            instances.add(inst)
        elif actual == args.error_type:
            instances.add(inst)
        else:
            skipped.append((inst, actual))

    if skipped:
        print(
            f"Skipping {len(skipped)} instance(s) whose trajectory has a different status "
            f"(yaml is stale; trajectory is authoritative):"
        )
        for inst, actual in skipped:
            print(f"  {inst}: yaml says {args.error_type}, traj says {actual}")
        print()

    if not instances:
        print(f"No {args.error_type} instances confirmed by trajectory file. Nothing to remove.")
        return

    print(f"Found {len(instances)} {args.error_type} instance(s) to remove: {sorted(instances)}\n")

    remove_instance_dirs(results_dir, instances, dry_run=args.dry_run)
    print()
    remove_from_preds(results_dir, instances, dry_run=args.dry_run)
    print()
    remove_from_exit_statuses(results_dir, instances, args.error_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
