#!/usr/bin/env python3
"""
Analyze mini SWE-agent trajectories to compute statistics across models.
Supports multiple runs per model-setting pair and computes mean ± std-dev.
Processes multiple settings and shows them all in a single table.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import re
import pandas as pd
import numpy as np
from tabulate import tabulate


def load_trajectory(trajectory_path: Path) -> Dict:
    """Load a trajectory file."""
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {trajectory_path}: {e}")
        return None


def extract_token_stats(trajectory: Dict) -> Dict:
    """Extract token usage statistics from a trajectory."""
    messages = trajectory.get('messages', [])
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    num_steps = 0
    
    for message in messages:
        if message.get('role') == 'assistant':
            # Check if this message contains a bash command
            content = message.get('content', '')
            bash_pattern = r'```bash\n(.*?)\n```'
            bash_matches = re.findall(bash_pattern, content, re.DOTALL)
            
            if bash_matches:
                num_steps += 1
                
                # Extract token usage from response
                extra = message.get("extra") or {}
                resp = extra.get("response") or {}
                usage = resp.get("usage") or {}
                
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                step_total = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += step_total
    
    # Extract instance cost from trajectory info
    info = trajectory.get('info', {})
    model_stats = info.get('model_stats', {})
    instance_cost = model_stats.get('instance_cost', 0.0)
    
    return {
        'num_steps': num_steps,
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'total_tokens': total_tokens,
        'cost': instance_cost
    }


def load_results_json(results_path: Path) -> Dict:
    """Load results.json file."""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return {}


def analyze_model_directory(model_dir: Path) -> Dict:
    """Analyze all trajectories in a model directory."""
    stats_list = []
    
    # Find all trajectory files
    traj_files = list(model_dir.glob("*/*.traj.json"))
    
    for traj_file in traj_files:
        trajectory = load_trajectory(traj_file)
        if trajectory:
            stats = extract_token_stats(trajectory)
            if stats['num_steps'] > 0:  # Only include if there are steps
                stats_list.append(stats)
    
    if not stats_list:
        return None
    
    # Calculate averages
    num_trajectories = len(stats_list)
    avg_stats = {
        'num_trajectories': num_trajectories,
        'avg_steps': sum(s['num_steps'] for s in stats_list) / num_trajectories,
        'avg_prompt_tokens': sum(s['prompt_tokens'] for s in stats_list) / num_trajectories,
        'avg_completion_tokens': sum(s['completion_tokens'] for s in stats_list) / num_trajectories,
        'avg_total_tokens': sum(s['total_tokens'] for s in stats_list) / num_trajectories,
        'avg_cost': sum(s['cost'] for s in stats_list) / num_trajectories,
    }
    
    # Load resolution rate from results.json
    results_path = model_dir / "results.json"
    if results_path.exists():
        results = load_results_json(results_path)
        total_instances = results.get('total_instances', 0)
        resolved_instances = results.get('resolved_instances', 0)
        avg_stats['resolved_instances'] = resolved_instances
        avg_stats['total_instances'] = total_instances
    else:
        avg_stats['resolved_instances'] = None
        avg_stats['total_instances'] = None
    
    return avg_stats


def parse_directory_name(dir_name: str):
    """Parse directory name to extract setting, run_id, and model name.
    
    Expected format: {setting}_{run_id}_{model_name}
    """
    parts = dir_name.split('_')
    
    # Find the run_id (should be 0, 1, or 2)
    run_id = None
    run_idx = None
    for i, part in enumerate(parts):
        if part in ['0', '1', '2']:
            run_id = int(part)
            run_idx = i
            break
    
    if run_id is None:
        return None, None, None
    
    # Everything before run_id is setting
    setting = '_'.join(parts[:run_idx])
    
    # Everything after run_id is model_name
    model_name = '_'.join(parts[run_idx + 1:])
    
    return setting, run_id, model_name


def format_mean_std(values, decimals=2, scale=1.0):
    """Format mean ± std-dev string."""
    if not values:
        return "N/A"
    
    scaled_values = [v * scale for v in values]
    mean = np.mean(scaled_values)
    std = np.std(scaled_values, ddof=1) if len(scaled_values) > 1 else 0
    
    format_str = f"{{:.{decimals}f}}"
    if std > 0:
        return f"{format_str.format(mean)} ± {format_str.format(std)}"
    else:
        return format_str.format(mean)


def process_setting(parent_dir: Path, setting: str, model_order: List[str]):
    """Process a single setting and return list of row dictionaries."""
    
    # Find all model directories and parse their names
    all_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    
    # Group directories by (setting, model_name)
    grouped_dirs = {}
    for d in all_dirs:
        parsed_setting, run_id, model_name = parse_directory_name(d.name)
        
        if parsed_setting == setting and model_name:
            key = (parsed_setting, model_name)
            if key not in grouped_dirs:
                grouped_dirs[key] = {}
            grouped_dirs[key][run_id] = d
    
    if not grouped_dirs:
        print(f"No directories found for setting: {setting}")
        return []
    
    print(f"\nFound {len(grouped_dirs)} model(s) for setting '{setting}'")
    
    # Collect statistics for each model
    rows = []
    total_instances = None
    
    for (parsed_setting, model_name), run_dirs in sorted(grouped_dirs.items()):
        print(f"Analyzing {model_name}...")
        
        # Collect stats from each run
        run_stats = {}
        for run_id in [0, 1, 2]:  # Check all possible runs
            if run_id in run_dirs:
                model_dir = run_dirs[run_id]
                print(f"  Run {run_id}...")
                stats = analyze_model_directory(model_dir)
                if stats:
                    run_stats[run_id] = stats
                    if total_instances is None and stats['total_instances'] is not None:
                        total_instances = stats['total_instances']
        
        if not run_stats:
            print(f"  Warning: No valid trajectories found for {model_name}")
            continue
        
        # Add individual run rows
        for run_id in [0, 1, 2]:
            if run_id in run_stats:
                stats = run_stats[run_id]
                row = {
                    'Setting': setting,
                    'Model': f"{model_name} (run {run_id})",
                    'Avg Steps': round(stats['avg_steps'], 2),
                    'Avg Total Tokens (M)': round(stats['avg_total_tokens'] / 1e6, 3),
                    'Avg Cost (USD)': round(stats['avg_cost'], 4),
                    'Resolved': stats['resolved_instances'] if stats['resolved_instances'] is not None else '-',
                    'model_name': model_name,  # For sorting
                    'is_aggregate': False,
                    'total_instances': total_instances
                }
                rows.append(row)
            else:
                # Add placeholder row for missing run
                row = {
                    'Setting': setting,
                    'Model': f"{model_name} (run {run_id})",
                    'Avg Steps': '-',
                    'Avg Total Tokens (M)': '-',
                    'Avg Cost (USD)': '-',
                    'Resolved': '-',
                    'model_name': model_name,
                    'is_aggregate': False,
                    'total_instances': total_instances
                }
                rows.append(row)
        
        # Add aggregated row
        all_steps = [s['avg_steps'] for s in run_stats.values()]
        all_tokens = [s['avg_total_tokens'] for s in run_stats.values()]
        all_costs = [s['avg_cost'] for s in run_stats.values()]
        all_resolved = [s['resolved_instances'] for s in run_stats.values() 
                       if s['resolved_instances'] is not None]
        
        agg_row = {
            'Setting': setting,
            'Model': f"{model_name} (mean ± std)",
            'Avg Steps': format_mean_std(all_steps, decimals=2),
            'Avg Total Tokens (M)': format_mean_std(all_tokens, decimals=3, scale=1e-6),
            'Avg Cost (USD)': format_mean_std(all_costs, decimals=4),
            'Resolved': format_mean_std(all_resolved, decimals=1) if all_resolved else '-',
            'model_name': model_name,
            'is_aggregate': True,
            'total_instances': total_instances
        }
        rows.append(agg_row)
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory statistics across models with multiple runs and settings"
    )
    
    parser.add_argument(
        "--parent-dir",
        type=str,
        default="/usr0/home/srgandhi/tool-overuse/results/",
        help="Parent directory containing model subdirectories"
    )
    
    parser.add_argument(
        "--settings",
        type=str,
        nargs='+',
        default=["base", "edit_obs_diff", "edit_obs_final_only", 
                 "prompt_efficient_edit_obs_diff", "prompt_efficient_edit_obs_final_only"],
        help="List of setting names to process"
    )
    
    args = parser.parse_args()
    
    parent_dir = Path(args.parent_dir)
    settings = args.settings
    
    if not parent_dir.exists():
        print(f"Error: Parent directory not found: {parent_dir}")
        return 1
    
    # Define model order for sorting
    model_order = [
        'Qwen25-Coder-32B-Instruct',
        'SWE-agent-LM-32B',
        'Qwen3-Coder-30B-A3B-Instruct',
        'Devstral-Small-2507',
        'cwm-sft',
        'cwm'
    ]
    
    # Define setting order
    setting_order = {s: i for i, s in enumerate(settings)}
    
    # Process each setting and collect all rows
    print(f"\n{'='*100}")
    print(f"Processing {len(settings)} settings")
    print('='*100)
    
    all_rows = []
    for setting in settings:
        print(f"\nProcessing setting: {setting}")
        rows = process_setting(parent_dir, setting, model_order)
        all_rows.extend(rows)
    
    if not all_rows:
        print("No statistics collected")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Get total instances for column naming
    total_instances = None
    for row in all_rows:
        if row.get('total_instances') is not None:
            total_instances = row['total_instances']
            break
    
    # Sort by setting, then model order, then run type
    def sort_key(row):
        # Setting order
        setting_idx = setting_order.get(row['Setting'], len(setting_order))
        
        # Model order
        try:
            model_idx = model_order.index(row['model_name'])
        except ValueError:
            model_idx = len(model_order)
        
        # Aggregate rows come after individual runs
        aggregate_order = 1 if row['is_aggregate'] else 0
        
        return (setting_idx, model_idx, aggregate_order, row['Model'])
    
    df['sort_key'] = df.apply(sort_key, axis=1)
    df = df.sort_values('sort_key').drop(['sort_key', 'model_name', 'is_aggregate', 'total_instances'], axis=1).reset_index(drop=True)
    
    # Rename the Resolved column to include total instances
    if total_instances is not None:
        df = df.rename(columns={'Resolved': f'Resolved (/{total_instances})'})
    
    # Save to CSV
    output_csv = parent_dir / "all_settings_stats_detailed.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*100}")
    print(f"Statistics saved to CSV: {output_csv}")
    print('='*100)
    
    # Pretty print to console
    print(f"\n{'='*160}")
    print(f"Trajectory Statistics - All Settings")
    print('='*160)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())