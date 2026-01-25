#!/usr/bin/env python3
"""
Script to delete subdirectories for instances with RetryError status.

This script reads a YAML file containing instance exit statuses and deletes
the subdirectories corresponding to instances that have a RetryError status.
"""

import argparse
import json
import shutil
from pathlib import Path
import yaml
import sys


def load_yaml_file(yaml_path):
    """Load and parse the YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def get_retry_error_instances(data):
    """Extract instance IDs with RetryError status."""
    try:
        retry_errors = data['instances_by_exit_status']['RetryError']
        return retry_errors
    except KeyError as e:
        print(f"Error: Expected key not found in YAML structure: {e}")
        sys.exit(1)


def load_preds_json(json_path):
    """Load the preds.json file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: preds.json not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing preds.json file: {e}")
        return None


def save_preds_json(json_path, data):
    """Save the updated preds.json file."""
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving preds.json: {e}")
        return False


def remove_from_preds_json(json_path, instance_ids, dry_run=False):
    """Remove entries from preds.json for the given instance IDs."""
    preds_data = load_preds_json(json_path)
    
    if preds_data is None:
        return 0, 0
    
    removed_count = 0
    not_found_count = 0
    
    print(f"\nProcessing preds.json: {json_path}")
    print(f"Total entries in preds.json: {len(preds_data)}")
    print("-" * 60)
    
    for instance_id in instance_ids:
        if instance_id in preds_data:
            if dry_run:
                print(f"[DRY RUN] Would remove from preds.json: {instance_id}")
            else:
                del preds_data[instance_id]
                print(f"Removed from preds.json: {instance_id}")
            removed_count += 1
        else:
            not_found_count += 1
    
    if not dry_run and removed_count > 0:
        if save_preds_json(json_path, preds_data):
            print(f"\nSuccessfully updated preds.json")
            print(f"Remaining entries: {len(preds_data)}")
        else:
            print(f"\nFailed to save updated preds.json")
    
    print("-" * 60)
    print(f"preds.json Summary:")
    print(f"  Removed: {removed_count}")
    print(f"  Not found: {not_found_count}")
    
    return removed_count, not_found_count


def delete_instance_directories(base_dir, instance_ids, dry_run=False):
    """Delete subdirectories for the given instance IDs."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    if not base_path.is_dir():
        print(f"Error: Base path is not a directory: {base_dir}")
        sys.exit(1)
    
    deleted_count = 0
    not_found_count = 0
    error_count = 0
    
    print(f"Base directory: {base_dir}")
    print(f"Found {len(instance_ids)} RetryError instances to delete")
    print("-" * 60)
    
    for instance_id in instance_ids:
        instance_path = base_path / instance_id
        
        if instance_path.exists() and instance_path.is_dir():
            try:
                if dry_run:
                    print(f"[DRY RUN] Would delete: {instance_path}")
                    deleted_count += 1
                else:
                    shutil.rmtree(instance_path)
                    print(f"Deleted: {instance_path}")
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting {instance_path}: {e}")
                error_count += 1
        else:
            print(f"Not found: {instance_path}")
            not_found_count += 1
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Deleted: {deleted_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Errors: {error_count}")
    
    return deleted_count, not_found_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Delete subdirectories for instances with RetryError status",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'yaml_file',
        help='Path to the YAML file containing exit statuses (the base directory will be extracted from this path)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--skip-preds',
        action='store_true',
        help='Skip removing entries from preds.json'
    )
    
    args = parser.parse_args()
    
    # Extract base directory from YAML file path
    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {args.yaml_file}")
        sys.exit(1)
    
    base_dir = yaml_path.parent
    preds_json_path = base_dir / "preds.json"
    
    # Load YAML file
    print(f"Loading YAML file: {args.yaml_file}")
    data = load_yaml_file(args.yaml_file)
    
    # Get RetryError instances
    retry_error_instances = get_retry_error_instances(data)
    
    # Delete directories
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be deleted ***\n")
    else:
        try:
            yaml_path.unlink()
            print(f"\nDeleted YAML file: {args.yaml_file}")
        except Exception as e:
            print(f"\nError deleting YAML file: {e}")
    
    delete_instance_directories(base_dir, retry_error_instances, args.dry_run)
    
    # Remove from preds.json
    if not args.skip_preds:
        print()
        remove_from_preds_json(preds_json_path, retry_error_instances, args.dry_run)
    else:
        print("\nSkipping preds.json update (--skip-preds flag used)")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - Run without --dry-run to actually delete ***")


if __name__ == "__main__":
    main()