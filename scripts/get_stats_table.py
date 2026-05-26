#!/usr/bin/env python3
"""
Analyze mini SWE-agent trajectories to compute statistics across models.
Supports multiple runs per model-setting pair and computes mean ± std-dev.
Auto-discovers settings from directory names.

Directory naming convention:
  {setting}_{run_id}_{agent_model}[_prm_{prm_model}]

Examples:
  singularity_edit_obs_final_only_0_cwm
  singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b
  singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6

Usage:
    python get_stats_table.py
    python get_stats_table.py --parent-dir results_singularity
    python get_stats_table.py --settings singularity_edit_obs_final_only singularity_edit_obs_final_only_prm_tool_k5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd
import numpy as np
from tabulate import tabulate


def load_trajectory(trajectory_path: Path) -> Optional[Dict]:
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {trajectory_path}: {e}")
        return None


def extract_instance_stats(trajectory: Dict) -> Dict:
    """Extract per-instance stats from a trajectory."""
    messages = trajectory.get('messages', [])

    num_steps = 0
    for message in messages:
        if message.get('role') == 'assistant':
            content = message.get('content', '')
            if re.search(r'```bash\n.*?\n```', content, re.DOTALL):
                num_steps += 1

    info = trajectory.get('info', {})
    model_stats = info.get('model_stats') or {}
    instance_cost = model_stats.get('instance_cost', 0.0)

    prm_stats = info.get('prm_stats') or {}
    prm_cost = prm_stats.get('prm_cost', 0.0)

    return {
        'num_steps': num_steps,
        'cost': instance_cost,
        'prm_cost': prm_cost,
    }


def load_report(run_dir: Path) -> Optional[Dict]:
    """Load report.json or results.json from a run directory."""
    for name in ['report.json', 'results.json']:
        path = run_dir / name
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return None


def analyze_run_directory(run_dir: Path) -> Optional[Dict]:
    """Analyze all trajectories in a run directory.

    Averages are computed over total_instances (from report.json), not just
    the instances that produced trajectories.  Instances without trajectories
    contribute 0 steps / 0 cost to the averages.
    """
    traj_files = list(run_dir.glob("*/*.traj.json"))
    stats_list = []

    for traj_file in traj_files:
        trajectory = load_trajectory(traj_file)
        if trajectory:
            stats = extract_instance_stats(trajectory)
            if stats['num_steps'] > 0:
                stats_list.append(stats)

    report = load_report(run_dir)
    resolved_instances = report.get('resolved_instances') if report else None
    total_instances = report.get('total_instances') if report else None

    if not stats_list and resolved_instances is None:
        return None

    # Average over actual trajectory count (instances that ran), not
    # total_instances from report (which is the full benchmark size).
    num_with_trajs = len(stats_list)
    n = num_with_trajs or 1

    total_steps = sum(s['num_steps'] for s in stats_list)
    total_cost = sum(s['cost'] for s in stats_list)
    total_prm_cost = sum(s['prm_cost'] for s in stats_list)

    result = {
        'num_instances_with_trajs': num_with_trajs,
        'total_instances': total_instances,
        'resolved_instances': resolved_instances,
        # num_ran = actual instances this run covered
        'num_ran': num_with_trajs,
        'avg_steps': total_steps / n,
        'avg_cost': total_cost / n,
        'avg_prm_cost': total_prm_cost / n,
        'total_cost': total_cost,
        'total_prm_cost': total_prm_cost,
    }

    return result


def parse_directory_name(dir_name: str):
    """Parse directory name into (setting, run_id, agent_model, prm_model).

    Expected format: {setting}_{run_id}_{agent_model}[_prm_{prm_model}]
    run_id is a single digit (0, 1, 2, ...).
    prm_model is None when no _prm_ suffix is present.
    """
    parts = dir_name.split('_')

    # Find the run_id: first single-digit part
    run_id = None
    run_idx = None
    for i, part in enumerate(parts):
        if re.fullmatch(r'\d', part):
            run_id = int(part)
            run_idx = i
            break

    if run_id is None:
        return None, None, None, None

    setting = '_'.join(parts[:run_idx])
    rest = '_'.join(parts[run_idx + 1:])  # e.g. "cwm" or "cwm_prm_qwen3-8b"

    if '_prm_' in rest:
        agent_model, prm_model = rest.split('_prm_', 1)
    else:
        agent_model = rest
        prm_model = None

    return setting, run_id, agent_model, prm_model


def discover_settings(parent_dir: Path) -> List[str]:
    """Auto-discover all unique settings from directory names."""
    settings = set()
    for d in parent_dir.iterdir():
        if d.is_dir():
            setting, run_id, agent_model, prm_model = parse_directory_name(d.name)
            if setting and agent_model:
                settings.add(setting)
    return sorted(settings)


def format_mean_std(values, decimals=2, scale=1.0):
    if not values:
        return "N/A"
    scaled = [v * scale for v in values]
    mean = np.mean(scaled)
    std = np.std(scaled, ddof=1) if len(scaled) > 1 else 0
    fmt = f"{{:.{decimals}f}}"
    if std > 0:
        return f"{fmt.format(mean)} ± {fmt.format(std)}"
    return fmt.format(mean)


def process_setting(parent_dir: Path, setting: str):
    """Process a single setting and return row dicts."""
    all_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    # Group by (agent_model, prm_model) -> {run_id: Path}
    grouped = {}
    for d in all_dirs:
        parsed_setting, run_id, agent_model, prm_model = parse_directory_name(d.name)
        if parsed_setting == setting and agent_model:
            key = (agent_model, prm_model)
            grouped.setdefault(key, {})[run_id] = d

    if not grouped:
        print(f"  No directories found for setting: {setting}")
        return []

    print(f"  Found {len(grouped)} model combination(s)")

    rows = []
    for (agent_model, prm_model), run_dirs in sorted(grouped.items()):
        run_stats = {}
        for run_id, run_dir in sorted(run_dirs.items()):
            stats = analyze_run_directory(run_dir)
            if stats:
                run_stats[run_id] = stats

        if not run_stats:
            print(f"    Warning: No valid trajectories for agent={agent_model} prm={prm_model}")
            continue

        prm_display = prm_model if prm_model else '-'

        # Individual run rows (run number omitted from display per user request)
        for run_id in sorted(run_stats):
            s = run_stats[run_id]
            resolved_raw = s['resolved_instances']
            num_ran = s['num_ran']

            resolved_str = f"{resolved_raw}/{num_ran}" if resolved_raw is not None else '-'

            rows.append({
                'Setting': setting,
                'Agent Model': agent_model,
                'PRM Model': prm_display,
                'Num Instances': num_ran,
                'Resolved': resolved_str,
                'Avg Steps': round(s['avg_steps'], 2),
                'Avg Model Cost ($)': round(s['avg_cost'], 4),
                'Avg PRM Cost ($)': round(s['avg_prm_cost'], 4),
                'Total Cost ($)': round(s['total_cost'] + s['total_prm_cost'], 4),
                '_agent': agent_model,
                '_prm': prm_model,
                '_is_agg': False,
                '_resolved_raw': resolved_raw,
                '_num_ran': num_ran,
                '_avg_steps_raw': s['avg_steps'],
                '_avg_cost_raw': s['avg_cost'],
                '_avg_prm_cost_raw': s['avg_prm_cost'],
                '_total_cost_raw': s['total_cost'] + s['total_prm_cost'],
                '_setting_raw': setting,
            })

        # Aggregate row only when multiple runs exist
        if len(run_stats) > 1:
            all_resolved = [s['resolved_instances'] for s in run_stats.values()
                            if s['resolved_instances'] is not None]
            agg_num_ran = max(s['num_ran'] for s in run_stats.values())
            resolved_str = '-'
            resolved_raw_mean = None
            if all_resolved:
                resolved_raw_mean = np.mean(all_resolved)
                resolved_str = format_mean_std(all_resolved, decimals=1) + f"/{agg_num_ran}"

            avg_steps_vals = [s['avg_steps'] for s in run_stats.values()]
            avg_cost_vals = [s['avg_cost'] for s in run_stats.values()]
            avg_prm_cost_vals = [s['avg_prm_cost'] for s in run_stats.values()]
            total_cost_vals = [s['total_cost'] + s['total_prm_cost'] for s in run_stats.values()]

            rows.append({
                'Setting': setting,
                'Agent Model': agent_model,
                'PRM Model': prm_display + ' (mean±std)',
                'Num Instances': agg_num_ran,
                'Resolved': resolved_str,
                'Avg Steps': format_mean_std(avg_steps_vals),
                'Avg Model Cost ($)': format_mean_std(avg_cost_vals, decimals=4),
                'Avg PRM Cost ($)': format_mean_std(avg_prm_cost_vals, decimals=4),
                'Total Cost ($)': format_mean_std(total_cost_vals, decimals=4),
                '_agent': agent_model,
                '_prm': prm_model,
                '_is_agg': True,
                '_resolved_raw': resolved_raw_mean,
                '_num_ran': agg_num_ran,
                '_avg_steps_raw': np.mean(avg_steps_vals),
                '_avg_cost_raw': np.mean(avg_cost_vals),
                '_avg_prm_cost_raw': np.mean(avg_prm_cost_vals),
                '_total_cost_raw': np.mean(total_cost_vals),
                '_setting_raw': setting,
            })

    return rows


BASE_SETTING = "singularity_edit_obs_final_only"


def quick_stats(run_dir: Path) -> int:
    """Print quick stats for a single run directory."""
    if not run_dir.exists():
        print(f"Error: {run_dir} not found")
        return 1

    stats = analyze_run_directory(run_dir)
    if not stats:
        print("No instances with trajectories found.")
        return 1

    report = load_report(run_dir)
    resolved = report.get('resolved_instances') if report else None
    n = stats['num_ran']

    resolved_str = f"{resolved}/{n}" if resolved is not None else "-"
    resolve_rate = f"{resolved / n * 100:.1f}%" if resolved is not None else "-"
    avg_total_cost = stats['avg_cost'] + stats['avg_prm_cost']

    print(f"Run dir:              {run_dir.name}")
    print(f"Instances ran:        {n}")
    print(f"Resolved:             {resolved_str} ({resolve_rate})")
    print(f"Avg Steps:            {stats['avg_steps']:.2f}")
    print(f"Avg Model Cost ($):   {stats['avg_cost']:.4f}")
    print(f"Avg PRM Cost ($):     {stats['avg_prm_cost']:.4f}")
    print(f"Avg Total Cost ($):   {avg_total_cost:.4f}")

    # CSV for pasting into Google Sheets horizontally
    print("\n--- Copy-paste for Google Sheets ---")
    print(f"{resolved_str},{stats['avg_steps']:.2f},{stats['avg_cost']:.4f},{stats['avg_prm_cost']:.4f},{avg_total_cost:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory statistics table for results_singularity runs"
    )
    parser.add_argument(
        "--parent-dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "results_singularity"),
        help="Parent directory containing run subdirectories",
    )
    parser.add_argument(
        "--settings", type=str, nargs='+', default=None,
        help="Settings to include (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (defaults to <parent-dir>/stats_table.csv)",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Single run directory to get quick stats for (skips full table)",
    )
    args = parser.parse_args()

    if args.run_dir:
        return quick_stats(Path(args.run_dir))

    parent_dir = Path(args.parent_dir)
    if not parent_dir.exists():
        print(f"Error: directory not found: {parent_dir}")
        return 1

    settings = args.settings or discover_settings(parent_dir)
    if not settings:
        print("No settings found. Check directory names.")
        return 1

    # Ensure base setting is processed first
    if BASE_SETTING in settings:
        settings = [BASE_SETTING] + [s for s in settings if s != BASE_SETTING]

    print(f"Settings to process: {settings}")

    all_rows = []
    for setting in settings:
        print(f"\nProcessing: {setting}")
        all_rows.extend(process_setting(parent_dir, setting))

    if not all_rows:
        print("No statistics collected.")
        return 1

    df = pd.DataFrame(all_rows)

    # Resolve rate (%)
    def resolve_rate(row):
        r = row['_resolved_raw']
        if r is None:
            return None
        n = row['_num_ran']
        return round(r / n * 100, 1)

    df['Resolve Rate (%)'] = df.apply(resolve_rate, axis=1)

    # Build baseline lookup: (agent_model, prm=None) -> stats from BASE_SETTING
    base_rows = df[df['_setting_raw'] == BASE_SETTING]
    baseline = {}  # agent_model -> dict
    for _, row in base_rows.iterrows():
        agent = row['_agent']
        if agent not in baseline or row['_is_agg']:
            baseline[agent] = {
                'avg_steps': row['_avg_steps_raw'],
                'avg_cost': row['_avg_cost_raw'],
                'avg_prm_cost': row['_avg_prm_cost_raw'],
                'resolve_rate': resolve_rate(row),
            }

    # Delta columns relative to base setting
    def delta_col(row, key, baseline_key):
        if row['_setting_raw'] == BASE_SETTING:
            return '-'
        base = baseline.get(row['_agent'])
        if base is None or base[baseline_key] is None:
            return 'N/A'
        val = row.get(key)
        if val is None:
            return 'N/A'
        return f"{val - base[baseline_key]:+.2f}"

    df['Δ Resolve Rate (%)'] = df.apply(lambda r: delta_col(r, 'Resolve Rate (%)', 'resolve_rate'), axis=1)
    df['Δ Avg Steps'] = df.apply(lambda r: delta_col(r, '_avg_steps_raw', 'avg_steps'), axis=1)

    def delta_total_cost(row):
        if row['_setting_raw'] == BASE_SETTING:
            return '-'
        base = baseline.get(row['_agent'])
        if base is None:
            return 'N/A'
        base_total = (base['avg_cost'] or 0) + (base['avg_prm_cost'] or 0)
        row_total = (row.get('_avg_cost_raw') or 0) + (row.get('_avg_prm_cost_raw') or 0)
        return f"{row_total - base_total:+.4f}"

    df['Δ Avg Total Cost ($)'] = df.apply(delta_total_cost, axis=1)

    # Shorten setting names
    def shorten_setting(s):
        if s == BASE_SETTING:
            return "base"
        prefix = BASE_SETTING + "_"
        if s.startswith(prefix):
            return s[len(prefix):]
        return s

    df['Setting'] = df['Setting'].apply(shorten_setting)

    # Sort: setting order, agent model, prm model, individual before aggregate
    setting_order = {s: i for i, s in enumerate(settings)}
    df['_sort'] = df.apply(
        lambda r: (
            setting_order.get(r['_setting_raw'], 999),
            r['_agent'],
            r['_prm'] or '',
            r['_is_agg'],
        ),
        axis=1,
    )
    internal_cols = ['_sort', '_agent', '_prm', '_is_agg', '_resolved_raw', '_num_ran',
                     '_avg_steps_raw', '_avg_cost_raw', '_avg_prm_cost_raw', '_total_cost_raw',
                     '_setting_raw']
    df = df.sort_values('_sort').drop(columns=internal_cols).reset_index(drop=True)

    # Save CSV
    output_csv = Path(args.output) if args.output else parent_dir / "stats_table.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    # Save filtered CSV without v0 rows
    df_no_v0 = df[~df['Setting'].str.contains('v0', case=False, na=False)]
    output_no_v0 = output_csv.with_name(output_csv.stem + "_no_v0.csv")
    df_no_v0.to_csv(output_no_v0, index=False)
    print(f"Saved (no v0): {output_no_v0}")

    # Print table
    print()
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print()

    return 0


if __name__ == "__main__":
    exit(main())
