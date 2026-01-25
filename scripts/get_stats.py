#!/usr/bin/env python3
"""
Analyze mini SWE-agent trajectories to compute statistics across models.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import re


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
        if total_instances > 0:
            avg_stats['resolution_rate'] = resolved_instances / total_instances
        else:
            avg_stats['resolution_rate'] = 0.0
    else:
        avg_stats['resolution_rate'] = None
    
    return avg_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory statistics across models"
    )
    
    parser.add_argument(
        "--parent-dir",
        type=str,
        default="/usr0/home/srgandhi/tool-overuse/results/",
        help="Parent directory containing model subdirectories"
    )
    
    parser.add_argument(
        "--setting",
        type=str,
        default="base",
        help="Setting name to filter directories (e.g., 'base')"
    )
    
    args = parser.parse_args()
    
    parent_dir = Path(args.parent_dir)
    setting = args.setting
    
    if not parent_dir.exists():
        print(f"Error: Parent directory not found: {parent_dir}")
        return 1
    
    # Find all model directories matching the setting
    pattern = f"{setting}_*"
    model_dirs = [d for d in parent_dir.glob(pattern) if d.is_dir()]
    
    if not model_dirs:
        print(f"No directories found matching pattern: {pattern}")
        return 1
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Collect statistics for each model
    model_stats = {}
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name.replace(f"{setting}_", "")
        print(f"Analyzing {model_name}...")
        
        stats = analyze_model_directory(model_dir)
        if stats:
            model_stats[model_name] = stats
        else:
            print(f"  Warning: No valid trajectories found for {model_name}")
    
    if not model_stats:
        print("No statistics collected")
        return 1
    
    # Write results to file
    output_file = parent_dir / f"{setting}_stats.txt"
    with open(output_file, 'w') as f:
        f.write(f"Trajectory Statistics for Setting: {setting}\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name]
            
            # Convert tokens to millions
            avg_prompt_m = stats['avg_prompt_tokens'] / 1_000_000
            avg_completion_m = stats['avg_completion_tokens'] / 1_000_000
            avg_total_m = stats['avg_total_tokens'] / 1_000_000
            
            f.write(f"Model: {model_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of trajectories:      {stats['num_trajectories']}\n")
            f.write(f"  Avg. number of steps:        {stats['avg_steps']:.2f}\n")
            f.write(f"  Avg. prompt tokens (M):      {avg_prompt_m:.3f}\n")
            f.write(f"  Avg. completion tokens (M):  {avg_completion_m:.3f}\n")
            f.write(f"  Avg. total tokens (M):       {avg_total_m:.3f}\n")
            f.write(f"  Avg. cost (USD):             ${stats['avg_cost']:.4f}\n")
            
            if stats['resolution_rate'] is not None:
                f.write(f"  Resolution rate:             {stats['resolution_rate']:.2%}\n")
            else:
                f.write(f"  Resolution rate:             N/A\n")
            
            f.write("\n")
    
    print(f"\nStatistics saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())