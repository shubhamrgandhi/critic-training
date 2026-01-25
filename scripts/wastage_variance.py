#!/usr/bin/env python3
"""
Analyze variance in redundant steps across instances from judge results.

This script visualizes the distribution of redundant steps per instance to understand
the high variance in judge evaluations (e.g., some instances have 50 redundant steps, 
others only 2).

Inputs:
  --inputs  : folder containing *_judge_result.json files
  --outdir  : output folder for plots and analysis (default: <inputs>/redundancy_variance)

Outputs:
  - redundancy_distribution.png: Histogram of redundant step counts
  - redundancy_vs_total_steps.png: Scatter plot showing relationship
  - redundancy_boxplot.png: Box plot showing quartiles and outliers
  - redundancy_variance_stats.csv: Statistical summary
  - instance_redundancy_details.csv: Per-instance details
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import csv

def load_judge_results(inputs: Path):
    """Load all judge result files."""
    files = list(inputs.glob("*_judge_result.json"))
    
    records = []
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    records.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return records

def extract_redundancy_stats(records):
    """Extract redundancy statistics per instance."""
    instance_stats = []
    
    for rec in records:
        instance_id = rec.get("instance_id") or Path(rec.get("trajectory_file", "unknown")).stem
        
        # Get judge response summary
        judge_response = rec.get("judge_response", {})
        summary = judge_response.get("summary", {})
        steps = judge_response.get("steps", [])
        
        # Count redundant steps
        redundant_steps = sum(1 for s in steps if s.get("redundant", False))
        total_steps = len(steps)
        
        # Get efficiency score if available
        efficiency_score = summary.get("efficiency_score", None)
        
        # Count by category
        category_counts = defaultdict(int)
        for step in steps:
            if step.get("redundant", False):
                category = step.get("category", "unknown")
                category_counts[category] += 1
        
        # Get action type distribution for redundant steps
        redundant_action_types = defaultdict(int)
        for step in steps:
            if step.get("redundant", False):
                action_type = step.get("action_type", "unknown")
                redundant_action_types[action_type] += 1
        
        instance_stats.append({
            "instance_id": instance_id,
            "total_steps": total_steps,
            "redundant_steps": redundant_steps,
            "essential_steps": total_steps - redundant_steps,
            "redundancy_rate": redundant_steps / total_steps if total_steps > 0 else 0,
            "efficiency_score": efficiency_score,
            "top_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "category_counts": dict(category_counts),
            "redundant_action_types": dict(redundant_action_types)
        })
    
    return instance_stats

def calculate_statistics(instance_stats):
    """Calculate overall statistics."""
    redundant_counts = [s["redundant_steps"] for s in instance_stats]
    total_counts = [s["total_steps"] for s in instance_stats]
    redundancy_rates = [s["redundancy_rate"] for s in instance_stats]
    
    stats = {
        "n_instances": len(instance_stats),
        "redundant_steps": {
            "mean": np.mean(redundant_counts),
            "median": np.median(redundant_counts),
            "std": np.std(redundant_counts),
            "min": min(redundant_counts),
            "max": max(redundant_counts),
            "q1": np.percentile(redundant_counts, 25),
            "q3": np.percentile(redundant_counts, 75),
            "iqr": np.percentile(redundant_counts, 75) - np.percentile(redundant_counts, 25),
            "cv": np.std(redundant_counts) / np.mean(redundant_counts) if np.mean(redundant_counts) > 0 else 0
        },
        "total_steps": {
            "mean": np.mean(total_counts),
            "median": np.median(total_counts),
            "std": np.std(total_counts),
            "min": min(total_counts),
            "max": max(total_counts)
        },
        "redundancy_rate": {
            "mean": np.mean(redundancy_rates),
            "median": np.median(redundancy_rates),
            "std": np.std(redundancy_rates),
            "min": min(redundancy_rates),
            "max": max(redundancy_rates)
        }
    }
    
    return stats

def plot_redundancy_distribution(instance_stats, outdir: Path):
    """Create histogram of redundant step counts."""
    redundant_counts = [s["redundant_steps"] for s in instance_stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = min(30, len(set(redundant_counts)))
    n, bins, patches = ax1.hist(redundant_counts, bins=bins, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(redundant_counts), color='red', linestyle='--', 
                label=f'Mean: {np.mean(redundant_counts):.1f}')
    ax1.axvline(np.median(redundant_counts), color='green', linestyle='--', 
                label=f'Median: {np.median(redundant_counts):.1f}')
    ax1.set_xlabel('Number of Redundant Steps per Instance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Redundant Steps Across Instances')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add text box with variance info
    variance_text = (f'Std Dev: {np.std(redundant_counts):.1f}\n'
                    f'Range: {min(redundant_counts)}-{max(redundant_counts)}\n'
                    f'CV: {np.std(redundant_counts)/np.mean(redundant_counts):.2f}')
    ax1.text(0.95, 0.95, variance_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # CDF
    sorted_counts = np.sort(redundant_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(sorted_counts, cdf, linewidth=2)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
    ax2.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='25th/75th percentile')
    ax2.axhline(0.75, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Number of Redundant Steps per Instance')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Redundant Steps')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(outdir / "redundancy_distribution.png", dpi=200)
    plt.close()

def plot_redundancy_vs_total(instance_stats, outdir: Path):
    """Create scatter plot of redundant steps vs total steps."""
    total_steps = [s["total_steps"] for s in instance_stats]
    redundant_steps = [s["redundant_steps"] for s in instance_stats]
    redundancy_rates = [s["redundancy_rate"] for s in instance_stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute numbers
    scatter = ax1.scatter(total_steps, redundant_steps, alpha=0.6, 
                         c=redundancy_rates, cmap='RdYlBu_r', s=50)
    ax1.plot([0, max(total_steps)], [0, max(total_steps)], 'k--', alpha=0.3, 
            label='100% redundancy line')
    ax1.plot([0, max(total_steps)], [0, max(total_steps)*0.5], 'g--', alpha=0.3,
            label='50% redundancy line')
    ax1.set_xlabel('Total Steps per Instance')
    ax1.set_ylabel('Redundant Steps per Instance')
    ax1.set_title('Redundant Steps vs Total Steps')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Redundancy Rate', rotation=270, labelpad=15)
    
    # Redundancy rate by total steps
    ax2.scatter(total_steps, [r*100 for r in redundancy_rates], alpha=0.6, s=50)
    ax2.axhline(np.mean(redundancy_rates)*100, color='red', linestyle='--', 
               label=f'Mean: {np.mean(redundancy_rates)*100:.1f}%')
    ax2.set_xlabel('Total Steps per Instance')
    ax2.set_ylabel('Redundancy Rate (%)')
    ax2.set_title('Redundancy Rate vs Total Steps')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "redundancy_vs_total_steps.png", dpi=200)
    plt.close()

def plot_boxplot_analysis(instance_stats, outdir: Path):
    """Create box plot and violin plot analysis."""
    redundant_counts = [s["redundant_steps"] for s in instance_stats]
    
    # Group instances by redundancy level
    low = [s for s in instance_stats if s["redundant_steps"] <= 10]
    medium = [s for s in instance_stats if 10 < s["redundant_steps"] <= 30]
    high = [s for s in instance_stats if s["redundant_steps"] > 30]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Box plot
    ax1 = axes[0, 0]
    bp = ax1.boxplot([redundant_counts], labels=['All Instances'], 
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax1.set_ylabel('Redundant Steps')
    ax1.set_title('Box Plot of Redundant Steps Distribution')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add outlier annotations
    outliers = []
    q1, q3 = np.percentile(redundant_counts, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for i, val in enumerate(redundant_counts):
        if val < lower_bound or val > upper_bound:
            outliers.append((instance_stats[i]["instance_id"], val))
    
    # Violin plot
    ax2 = axes[0, 1]
    parts = ax2.violinplot([redundant_counts], positions=[1], 
                           showmeans=True, showmedians=True)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['All Instances'])
    ax2.set_ylabel('Redundant Steps')
    ax2.set_title('Violin Plot of Redundant Steps Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    # Redundancy by groups
    ax3 = axes[1, 0]
    group_data = []
    group_labels = []
    group_colors = []
    
    if low:
        group_data.append([s["redundant_steps"] for s in low])
        group_labels.append(f'Low\n(≤10)\nn={len(low)}')
        group_colors.append('green')
    if medium:
        group_data.append([s["redundant_steps"] for s in medium])
        group_labels.append(f'Medium\n(11-30)\nn={len(medium)}')
        group_colors.append('yellow')
    if high:
        group_data.append([s["redundant_steps"] for s in high])
        group_labels.append(f'High\n(>30)\nn={len(high)}')
        group_colors.append('red')
    
    if group_data:
        bp2 = ax3.boxplot(group_data, labels=group_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax3.set_ylabel('Redundant Steps')
    ax3.set_title('Redundant Steps by Severity Group')
    ax3.grid(axis='y', alpha=0.3)
    
    # Top categories causing redundancy
    ax4 = axes[1, 1]
    category_totals = defaultdict(int)
    for s in instance_stats:
        for cat, count in s["category_counts"].items():
            category_totals[cat] += count
    
    if category_totals:
        top_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        cats, counts = zip(*top_cats)
        ax4.barh(range(len(cats)), counts)
        ax4.set_yticks(range(len(cats)))
        ax4.set_yticklabels(cats)
        ax4.set_xlabel('Total Redundant Steps')
        ax4.set_title('Top 10 Redundancy Categories')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "redundancy_boxplot.png", dpi=200)
    plt.close()
    
    return outliers

def write_analysis_files(instance_stats, stats, outliers, outdir: Path):
    """Write CSV files with analysis results."""
    
    # Write detailed instance stats
    with open(outdir / "instance_redundancy_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "total_steps", "redundant_steps", "essential_steps", 
                   "redundancy_rate", "efficiency_score", "top_category"])
        
        # Sort by redundant steps descending to see worst offenders first
        sorted_stats = sorted(instance_stats, key=lambda x: x["redundant_steps"], reverse=True)
        for s in sorted_stats:
            w.writerow([
                s["instance_id"],
                s["total_steps"],
                s["redundant_steps"],
                s["essential_steps"],
                f"{s['redundancy_rate']:.3f}",
                f"{s['efficiency_score']:.3f}" if s['efficiency_score'] is not None else "N/A",
                s["top_category"] or "N/A"
            ])
    
    # Write summary statistics
    with open(outdir / "redundancy_variance_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_instances", stats["n_instances"]])
        w.writerow([""])
        w.writerow(["Redundant Steps Statistics", ""])
        w.writerow(["mean", f"{stats['redundant_steps']['mean']:.2f}"])
        w.writerow(["median", f"{stats['redundant_steps']['median']:.1f}"])
        w.writerow(["std_dev", f"{stats['redundant_steps']['std']:.2f}"])
        w.writerow(["coefficient_of_variation", f"{stats['redundant_steps']['cv']:.3f}"])
        w.writerow(["min", stats['redundant_steps']['min']])
        w.writerow(["max", stats['redundant_steps']['max']])
        w.writerow(["q1", f"{stats['redundant_steps']['q1']:.1f}"])
        w.writerow(["q3", f"{stats['redundant_steps']['q3']:.1f}"])
        w.writerow(["iqr", f"{stats['redundant_steps']['iqr']:.1f}"])
        w.writerow([""])
        w.writerow(["Redundancy Rate Statistics", ""])
        w.writerow(["mean", f"{stats['redundancy_rate']['mean']:.3f}"])
        w.writerow(["median", f"{stats['redundancy_rate']['median']:.3f}"])
        w.writerow(["std_dev", f"{stats['redundancy_rate']['std']:.3f}"])
        w.writerow(["min", f"{stats['redundancy_rate']['min']:.3f}"])
        w.writerow(["max", f"{stats['redundancy_rate']['max']:.3f}"])
        w.writerow([""])
        w.writerow(["Outliers (>1.5 IQR)", ""])
        for instance_id, count in outliers[:10]:  # Top 10 outliers
            w.writerow([instance_id, count])

def main():
    parser = argparse.ArgumentParser(
        description="Analyze variance in redundant steps across instances from judge results"
    )
    parser.add_argument(
        "--inputs", 
        required=True, 
        type=str, 
        help="Folder containing *_judge_result.json files"
    )
    parser.add_argument(
        "--outdir", 
        default=None, 
        type=str, 
        help="Output folder for plots and analysis (default: <inputs>/redundancy_variance)"
    )
    
    args = parser.parse_args()
    
    inputs = Path(args.inputs).expanduser().resolve()
    if not inputs.exists() or not inputs.is_dir():
        raise SystemExit(f"Input folder not found: {inputs}")
    
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (inputs / "redundancy_variance")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    print("Loading judge results...")
    records = load_judge_results(inputs)
    print(f"Loaded {len(records)} instance results")
    
    if not records:
        print("No judge result files found.")
        return
    
    # Extract statistics
    print("Extracting redundancy statistics...")
    instance_stats = extract_redundancy_stats(records)
    
    # Calculate overall statistics
    stats = calculate_statistics(instance_stats)
    
    # Print summary
    print(f"\nRedundancy Variance Analysis:")
    print(f"  Instances analyzed: {stats['n_instances']}")
    print(f"  Redundant steps per instance:")
    print(f"    Mean: {stats['redundant_steps']['mean']:.1f}")
    print(f"    Std Dev: {stats['redundant_steps']['std']:.1f}")
    print(f"    Range: {stats['redundant_steps']['min']}-{stats['redundant_steps']['max']}")
    print(f"    Coefficient of Variation: {stats['redundant_steps']['cv']:.3f}")
    print(f"    Median: {stats['redundant_steps']['median']:.1f}")
    print(f"    IQR: {stats['redundant_steps']['iqr']:.1f}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_redundancy_distribution(instance_stats, outdir)
    plot_redundancy_vs_total(instance_stats, outdir)
    outliers = plot_boxplot_analysis(instance_stats, outdir)
    
    # Write analysis files
    print("Writing analysis files...")
    write_analysis_files(instance_stats, stats, outliers, outdir)
    
    print(f"\nAnalysis complete! Results saved to: {outdir}")
    print(f"  - redundancy_distribution.png")
    print(f"  - redundancy_vs_total_steps.png")
    print(f"  - redundancy_boxplot.png")
    print(f"  - redundancy_variance_stats.csv")
    print(f"  - instance_redundancy_details.csv")

if __name__ == "__main__":
    main()