#!/usr/bin/env python3
"""
Wrapper script to sample test set trajectories and run judge analysis.

This script:
1. Discovers all model-setting pairs in the results directory
2. For each pair, randomly selects one of the 3 runs
3. Samples N instances from the available set
4. Creates a temporary directory with symlinks to sampled trajectories
5. Runs the judge script on the sampled data

Example usage:
    python sample_and_judge_test_set.py \\
        --results-dir /path/to/results \\
        --output-dir /path/to/judge_outputs \\
        --sample-size 50 \\
        --setting-filter edit_obs_final_only \\
        --k 5
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import rich
from rich.progress import Progress, SpinnerColumn, TextColumn


def parse_directory_name(dir_name: str):
    """
    Parse directory name to extract setting, run_number, and model_name.
    Format: {setting}_{run_number}_{model_name}
    
    Returns: (setting, run_number, model_name) or None if parsing fails
    """
    parts = dir_name.split('_')
    
    # We need at least 3 parts: setting, run_number, model_name
    if len(parts) < 3:
        return None
    
    # The run number should be the second-to-last or third-to-last part
    # Try to find it by looking for a digit
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit():
            run_number = int(parts[i])
            setting = '_'.join(parts[:i])
            model_name = '_'.join(parts[i+1:])
            return setting, run_number, model_name
    
    return None


def discover_model_runs(results_dir: Path):
    """
    Discover all model-setting-run combinations.
    
    Returns: dict of {(setting, model_name): [run_dirs]}
    """
    model_runs = defaultdict(list)
    
    for dir_path in results_dir.iterdir():
        if not dir_path.is_dir():
            continue
        
        parsed = parse_directory_name(dir_path.name)
        if parsed is None:
            continue
        
        setting, run_number, model_name = parsed
        key = (setting, model_name)
        model_runs[key].append((run_number, dir_path))
    
    # Sort runs by run number for each key
    for key in model_runs:
        model_runs[key].sort(key=lambda x: x[0])
    
    return model_runs


def get_all_instances(run_dir: Path):
    """
    Get all instance IDs from a run directory.
    Only includes instances with valid trajectory files.
    
    Returns: set of instance IDs
    """
    instances = set()
    
    # Look for trajectory files in subdirectories
    for traj_file in run_dir.glob("*/*.traj.json"):
        try:
            # Try to load the file to ensure it's valid JSON
            with open(traj_file, 'r') as f:
                data = json.load(f)
                instance_id = data.get('instance_id')
                if instance_id:
                    # Verify the instance has messages (basic validation)
                    messages = data.get('messages', [])
                    if messages:
                        instances.add(instance_id)
                    else:
                        rich.print(f"[yellow]    Skipping {instance_id}: no messages[/yellow]")
                else:
                    rich.print(f"[yellow]    Skipping {traj_file.name}: no instance_id[/yellow]")
        except json.JSONDecodeError as e:
            rich.print(f"[yellow]    Skipping {traj_file.name}: invalid JSON - {e}[/yellow]")
        except Exception as e:
            rich.print(f"[yellow]    Skipping {traj_file.name}: {e}[/yellow]")
    
    return instances


def sample_instances_and_run(model_runs: dict, sample_size: int, seed: int, 
                              sampling_strategy: str, temp_dir: Path, output_base_dir: Path, 
                              judge_script: Path, judge_args: dict):
    """
    For each model-setting pair:
    1. Find instances that exist across all runs
    2. Randomly select one run
    3. Sample instances based on strategy
    4. Create temp directory with symlinks
    5. Run judge script
    
    sampling_strategy:
        - 'common': prefer instances that exist in all runs
        - 'selected_run': sample only from selected run
        - 'any': sample from union of all runs (may not exist in selected run)
    """
    random.seed(seed)
    
    rich.print(f"\n[cyan]Found {len(model_runs)} model-setting pairs[/cyan]")
    rich.print(f"[cyan]Sampling strategy: {sampling_strategy}[/cyan]")
    
    # PHASE 1: Find global instances that exist across all model-setting pairs
    rich.print(f"\n[yellow]Phase 1: Finding common instances across all models...[/yellow]")
    
    all_common_instances_per_model = {}
    for (setting, model_name), runs in model_runs.items():
        # Get instances from all runs for this model
        instances_per_run = {}
        for run_num, run_dir in runs:
            instances = get_all_instances(run_dir)
            instances_per_run[run_num] = instances
        
        # Find instances common to all runs for this model
        common = set.intersection(*instances_per_run.values()) if instances_per_run else set()
        all_common_instances_per_model[(setting, model_name)] = {
            'common': common,
            'per_run': instances_per_run
        }
        rich.print(f"  {setting}/{model_name}: {len(common)} common instances")
    
    # Find instances that are common across ALL model-setting pairs
    global_common_instances = set.intersection(
        *[data['common'] for data in all_common_instances_per_model.values()]
    ) if all_common_instances_per_model else set()
    
    rich.print(f"\n[green]Global common instances (across all models): {len(global_common_instances)}[/green]")
    
    # Initialize variables for global sampling
    global_sampled_instances = None
    use_global_sample = False
    
    # Sample the global instances once
    if sampling_strategy == 'common':
        if len(global_common_instances) >= sample_size:
            global_sampled_instances = random.sample(list(global_common_instances), sample_size)
            rich.print(f"[green]✓ Sampled {len(global_sampled_instances)} instances globally (all models will use these)[/green]")
            use_global_sample = True
        else:
            rich.print(f"[yellow]⚠ Only {len(global_common_instances)} global common instances, will sample per-model[/yellow]")
            use_global_sample = False
    else:
        rich.print(f"[yellow]Strategy '{sampling_strategy}' doesn't use global sampling[/yellow]")
        use_global_sample = False
    
    # PHASE 2: Process each model with global or per-model sampling
    rich.print(f"\n[yellow]Phase 2: Processing each model...[/yellow]")
    
    results = []
    
    for (setting, model_name), runs in model_runs.items():
        rich.print(f"\n[green]Processing: {setting} / {model_name}[/green]")
        rich.print(f"  Available runs: {[r[0] for r in runs]}")
        
        # Get pre-computed instance data
        instances_per_run = all_common_instances_per_model[(setting, model_name)]['per_run']
        common_instances = all_common_instances_per_model[(setting, model_name)]['common']
        
        for run_num, count in instances_per_run.items():
            rich.print(f"  Run {run_num}: {len(count)} instances")
        rich.print(f"  Common instances (this model): {len(common_instances)}")
        
        # Find instances that exist in ANY run (union)
        all_instances = set.union(*instances_per_run.values()) if instances_per_run else set()
        rich.print(f"  Total unique instances (this model): {len(all_instances)}")
        
        # Randomly select one run
        selected_run_num, selected_run_dir = random.choice(runs)
        rich.print(f"  Selected run: {selected_run_num}")
        
        # Get instances available in the selected run
        available_instances = instances_per_run[selected_run_num]
        rich.print(f"  Instances in selected run: {len(available_instances)}")
        
        # Determine which instances to use
        pool_description = ""  # Initialize pool_description
        
        if use_global_sample:
            # Use the globally sampled instances (SAME for all models)
            sampled_instances = global_sampled_instances
            pool_description = "global common instances (shared across all models)"
            rich.print(f"  ✓ Using {len(sampled_instances)} globally sampled instances")
        else:
            # Sample per-model based on strategy
            if sampling_strategy == 'common':
                # Prefer common instances for this model, fallback to selected run
                if len(common_instances) >= sample_size:
                    sampling_pool = common_instances
                    pool_description = "common instances (this model)"
                elif len(available_instances) >= sample_size:
                    sampling_pool = available_instances
                    pool_description = f"run {selected_run_num} (fallback)"
                else:
                    sampling_pool = available_instances
                    pool_description = f"run {selected_run_num} (all available)"
            elif sampling_strategy == 'selected_run':
                sampling_pool = available_instances
                pool_description = f"run {selected_run_num}"
            elif sampling_strategy == 'any':
                sampling_pool = all_instances
                pool_description = "all runs (union)"
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            
            # Sample from the pool
            if len(sampling_pool) >= sample_size:
                sampled_instances = random.sample(list(sampling_pool), sample_size)
                rich.print(f"  ✓ Sampled {len(sampled_instances)} from {pool_description}")
            else:
                sampled_instances = list(sampling_pool)
                rich.print(f"  [yellow]Warning: Only {len(sampled_instances)} instances in {pool_description}, using all[/yellow]")
        
        # Create temporary directory for this model-setting pair
        temp_model_dir = temp_dir / f"{setting}_{model_name}"
        temp_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks to sampled trajectories
        linked_count = 0
        for instance_id in sampled_instances:
            # Find the trajectory file for this instance
            traj_files = list(selected_run_dir.glob(f"*/{instance_id}.traj.json"))
            
            if not traj_files:
                rich.print(f"    [yellow]Warning: Trajectory not found for {instance_id}[/yellow]")
                continue
            
            traj_file = traj_files[0]
            
            # Create subdirectory in temp (maintain structure)
            temp_subdir = temp_model_dir / traj_file.parent.name
            temp_subdir.mkdir(parents=True, exist_ok=True)
            
            # Create symlink
            link_path = temp_subdir / traj_file.name
            if not link_path.exists():
                link_path.symlink_to(traj_file.resolve())
                linked_count += 1
        
        rich.print(f"  Created {linked_count} symlinks")
        
        # Prepare output directory for this model-setting pair
        output_dir = output_base_dir / f"{setting}_{model_name}" / "policy_v2"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run judge script
        rich.print(f"  Running judge script...")
        
        cmd = [
            sys.executable,
            str(judge_script),
            "--results-dir", str(temp_model_dir),
            "--output-dir", str(output_dir),
            "--model", judge_args['model'],
            "--max-workers", str(judge_args['max_workers']),
            "--k", str(judge_args['k'])
        ]
        
        if judge_args.get('api_key'):
            cmd.extend(["--api-key", judge_args['api_key']])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            rich.print(f"  [green]✓ Judge completed successfully[/green]")
            
            # Save metadata about sampling
            metadata = {
                "setting": setting,
                "model_name": model_name,
                "selected_run": selected_run_num,
                "selected_run_dir": str(selected_run_dir),
                "sampling_strategy": sampling_strategy,
                "used_global_sample": use_global_sample,
                "global_common_instances": len(global_common_instances) if global_common_instances else 0,
                "total_instances_in_selected_run": len(available_instances),
                "common_instances_across_all_runs_this_model": len(common_instances),
                "total_unique_instances_across_all_runs_this_model": len(all_instances),
                "sampled_instances": len(sampled_instances),
                "sample_size_requested": sample_size,
                "seed": seed,
                "sampled_from": pool_description,
                "instances_per_run": {str(run_num): len(instances) for run_num, instances in instances_per_run.items()},
                "sampled_instance_ids": sorted(sampled_instances)
            }
            
            metadata_file = output_dir / "sampling_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results.append({
                "setting": setting,
                "model_name": model_name,
                "run": selected_run_num,
                "status": "success",
                "output_dir": str(output_dir)
            })
            
        except subprocess.CalledProcessError as e:
            rich.print(f"  [red]✗ Judge failed with error:[/red]")
            rich.print(f"  [red]{e.stderr}[/red]")
            results.append({
                "setting": setting,
                "model_name": model_name,
                "run": selected_run_num,
                "status": "failed",
                "error": str(e)
            })
        
        # Clean up temp directory for this model
        shutil.rmtree(temp_model_dir)
        rich.print(f"  Cleaned up temporary directory")
    
    # Return results along with global sampling info
    return {
        'results': results,
        'use_global_sample': use_global_sample,
        'global_common_instances': global_common_instances,
        'global_sampled_instances': global_sampled_instances
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample test set trajectories and run judge analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/usr0/home/srgandhi/tool-overuse/results",
        help="Base directory containing all result directories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory for judge outputs (will create subdirs per model-setting)"
    )
    
    parser.add_argument(
        "--judge-script",
        type=str,
        default="./run-judge-policy-majority-vote.py",
        help="Path to the judge script"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of instances to sample per model-setting pair (default: 50)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        choices=['common', 'selected_run', 'any'],
        default='common',
        help=(
            "Strategy for sampling instances: "
            "'common' = prefer instances in all runs (default), "
            "'selected_run' = sample from selected run only, "
            "'any' = sample from union of all runs"
        )
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model to use for judge analysis (default: gpt-5-mini-2025-08-07)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or use OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers for judge (default: 32)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of judge calls per trajectory for majority voting (default: 5)"
    )
    
    parser.add_argument(
        "--setting-filter",
        type=str,
        default=None,
        help="Only process model-setting pairs matching this setting prefix (e.g., 'edit_obs_final_only')"
    )
    
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directories after processing"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        rich.print(f"[red]Results directory not found: {results_dir}[/red]")
        return 1
    
    judge_script = Path(args.judge_script).expanduser().resolve()
    if not judge_script.exists():
        rich.print(f"[red]Judge script not found: {judge_script}[/red]")
        return 1
    
    output_base_dir = Path(args.output_dir).expanduser().resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory
    temp_dir = output_base_dir / "_temp_sampled_trajectories"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    rich.print("[green]Starting test set sampling and judge analysis...[/green]")
    rich.print(f"  Results directory: {results_dir}")
    rich.print(f"  Output directory: {output_base_dir}")
    rich.print(f"  Judge script: {judge_script}")
    rich.print(f"  Sample size: {args.sample_size}")
    rich.print(f"  Random seed: {args.seed}")
    rich.print(f"  K (majority voting): {args.k}")
    
    # Discover all model runs
    rich.print("\n[cyan]Discovering model-setting pairs...[/cyan]")
    model_runs = discover_model_runs(results_dir)
    
    # Filter by setting if requested
    if args.setting_filter:
        filtered_runs = {
            k: v for k, v in model_runs.items() 
            if k[0].startswith(args.setting_filter)
        }
        rich.print(f"  Filtered from {len(model_runs)} to {len(filtered_runs)} pairs matching '{args.setting_filter}'")
        model_runs = filtered_runs
    
    if not model_runs:
        rich.print("[red]No model-setting pairs found![/red]")
        return 1
    
    # Prepare judge arguments
    judge_args = {
        'model': args.model,
        'max_workers': args.max_workers,
        'k': args.k,
        'api_key': args.api_key
    }
    
    # Process all model-setting pairs
    sampling_output = sample_instances_and_run(
        model_runs=model_runs,
        sample_size=args.sample_size,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        temp_dir=temp_dir,
        output_base_dir=output_base_dir,
        judge_script=judge_script,
        judge_args=judge_args
    )
    
    # Extract results and metadata
    results = sampling_output['results']
    use_global_sample = sampling_output['use_global_sample']
    global_common_instances = sampling_output['global_common_instances']
    global_sampled_instances = sampling_output['global_sampled_instances']
    
    # Clean up temp directory
    if not args.keep_temp and temp_dir.exists():
        shutil.rmtree(temp_dir)
        rich.print(f"\n[green]Cleaned up temporary directory[/green]")
    
    # Print summary
    rich.print("\n[green]=" * 80 + "[/green]")
    rich.print("[green]SUMMARY[/green]")
    rich.print("[green]=" * 80 + "[/green]")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    rich.print(f"\nTotal model-setting pairs processed: {len(results)}")
    rich.print(f"  Successful: {len(successful)}")
    rich.print(f"  Failed: {len(failed)}")
    
    if successful:
        rich.print("\n[green]Successful runs:[/green]")
        for r in successful:
            rich.print(f"  ✓ {r['setting']} / {r['model_name']} (run {r['run']})")
            rich.print(f"    Output: {r['output_dir']}")
    
    if failed:
        rich.print("\n[red]Failed runs:[/red]")
        for r in failed:
            rich.print(f"  ✗ {r['setting']} / {r['model_name']} (run {r['run']})")
            rich.print(f"    Error: {r.get('error', 'Unknown')}")
    
    # Save overall summary
    summary_file = output_base_dir / "sampling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "sample_size": args.sample_size,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "used_global_sample": use_global_sample,
            "global_common_instances": len(global_common_instances),
            "global_sampled_instances": sorted(global_sampled_instances) if global_sampled_instances else None,
            "total_pairs": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }, f, indent=2)
    
    rich.print(f"\n[blue]Summary saved to: {summary_file}[/blue]")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())