#!/usr/bin/env python3
"""
Analyze judge analysis outputs and plot agent token totals vs wastage across multiple models.

Inputs:
  --parent-dir : parent directory containing model subdirectories 
                 (default: /usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5)
  --setting    : setting prefix for model directories (default: edit_obs_final_only_dev)
  --outdir     : output folder (default: {parent-dir}/{setting}_wastage_analysis)

The script automatically discovers all subdirectories matching {setting}_*/policy_v2
and creates comparative visualizations across all models.

Outputs:
  PNGs:
    - steps_comparison.png (subplots showing step distribution per model)
    - tokens_by_action_type_comparison.png (grouped bar chart comparing models)
    - tokens_by_category_comparison.png (grouped bar chart comparing models)
  CSVs:
    - {model}_tokens_by_action_type.csv
    - {model}_tokens_by_category.csv
    - {model}_per_step_tokens.csv
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import matplotlib.pyplot as plt
import numpy as np

# Fixed order for action types in plots
ACTION_TYPE_ORDER = ["read", "write", "search", "execute", "env", "other"]

# Fixed order for redundancy categories in plots
CATEGORY_ORDER = [
    "essential",
    # Information Redundancy
    "duplicate_read",
    "duplicate_search",
    "duplicate_status_check",
    # Inefficient Information Access
    "overly_broad_read",
    "inefficient_search_strategy",
    "unfocused_browsing",
    "tangential_exploration",
    # Failed Strategy Persistence
    "repeated_failed_command",
    "circular_debugging",
    # Unnecessary Verification
    "obvious_confirmation",
    "excessive_intermediate_testing",
    "redundant_success_validation",
]

# Fixed model-to-color mapping
# Each model family gets a consistent color regardless of which models are present
MODEL_COLOR_MAP = {
    'qwen2.5': '#1f77b4',  # blue
    'qwen25': '#1f77b4',   # blue (same as qwen2.5)
    'swe-agent-lm': '#ff7f0e',  # orange
    'sweagent': '#ff7f0e',  # orange (alternative name)
    'qwen3': '#2ca02c',  # green
    'devstral': '#d62728',  # red
    'cwm-sft': '#9467bd',  # purple
    'cwm': '#8c564b',  # brown
}

# Model display name mapping (directory name -> display name)
MODEL_DISPLAY_NAMES = {
    'qwen25-coder-32b-instruct': 'Qwen2.5-Coder-32B-Instruct',
    'qwen25-coder-32b-instruct_1': 'Qwen2.5-Coder-32B-Instruct (Only Edit Result)',
    'qwen25': 'Qwen2.5',
    # Add more mappings as needed
}

# Preferred model ordering (for consistent plot ordering)
MODEL_ORDER = [
    'qwen2.5', 'qwen25',  # Qwen2.5 variants
    'swe-agent-lm', 'swe-agent-lm-32b',  # SWE-agent variants
    'qwen3', 'qwen3-coder-30b',  # Qwen3 variants
    'devstral', 'devstral-small-2507',  # Devstral variants
    'cwm-sft',  # CWM SFT
    'cwm',  # CWM SFT + RL
]

def get_model_color(model_name):
    """
    Get the consistent color for a model based on its name.
    Returns the color from MODEL_COLOR_MAP or a default color if not found.
    """
    lower_name = model_name.lower()
    
    # Check for exact or partial matches in the color map
    for pattern, color in MODEL_COLOR_MAP.items():
        if pattern in lower_name:
            return color
    
    # Default fallback color if model not found
    return '#7f7f7f'  # gray

def get_display_name(model_name):
    """Get display name for a model, applying any mappings."""
    lower_name = model_name.lower()
    # Check exact matches first
    if lower_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[lower_name]
    # Check if any key is a substring
    for key, display in MODEL_DISPLAY_NAMES.items():
        if key in lower_name:
            return display
    # Default: return original with proper casing preserved
    return model_name

def sort_models(model_names):
    """Sort model names according to MODEL_ORDER preference."""
    def sort_key(name):
        lower_name = name.lower()
        # Check for matches in MODEL_ORDER
        for idx, pattern in enumerate(MODEL_ORDER):
            if pattern in lower_name:
                return (idx, name)
        # If not in MODEL_ORDER, put at end and sort alphabetically
        return (len(MODEL_ORDER), name)
    
    return sorted(model_names, key=sort_key)

def discover_model_dirs(parent_dir: Path, setting: str):
    """
    Discover all model directories matching {setting}_*/policy_v2 pattern.
    Returns list of (model_name, policy_v2_path) tuples.
    """
    model_dirs = []
    pattern = f"{setting}_*"
    
    for setting_dir in parent_dir.glob(pattern):
        if not setting_dir.is_dir():
            continue
        policy_dir = setting_dir / "policy_v2"
        if policy_dir.exists() and policy_dir.is_dir():
            # Extract model name (everything after setting_)
            model_name = setting_dir.name[len(setting) + 1:]
            model_dirs.append((model_name, policy_dir))
    
    return sorted(model_dirs)

def load_records(inputs: Path):
    """Load judge result records from directory."""
    files = list(inputs.glob("*_judge_result.json"))
    agg = inputs / "all_judge_results.json"
    if agg.exists():
        files.append(agg)
    files = list({p.resolve() for p in files})

    records = []
    for p in files:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            records.extend([d for d in data if isinstance(d, dict)])
        elif isinstance(data, dict):
            records.append(data)
    return records

def join_per_step(records):
    """
    Returns a list of per-step dicts with:
      instance_id, step_number, action_type, category, redundant(bool),
      total_tokens (agent), wasted_tokens (agent), immediate_wasted, snowball_wasted
    """
    rows = []

    for rec in records:
        instance_id = rec.get("instance_id") or Path(rec.get("trajectory_file", "unknown")).stem

        # Judge steps metadata (action_type, category, redundant) for ALL steps
        steps_meta = {}
        for st in (rec.get("judge_response", {}) or {}).get("steps", []) or []:
            sn = st.get("step_number")
            if sn is None:
                continue
            steps_meta[sn] = {
                "action_type": st.get("action_type") or "unknown",
                "category": st.get("category") or ("essential" if not st.get("redundant") else "unknown"),
                "redundant": bool(st.get("redundant", False)),
            }

        # Agent per-step total tokens (ALL steps where available)
        usage_by_step = {s.get("step_number"): (s.get("total_tokens") or 0)
                         for s in (rec.get("agent_token_usage_by_step") or []) if isinstance(s, dict)}

        # Wastage per redundant step with breakdown
        wastage_by_step = {}
        for w in (rec.get("agent_token_wastage") or []):
            if isinstance(w, dict):
                sn = w.get("step_number")
                tokens_wasted = w.get("tokens_wasted") or 0
                snowball = (w.get("num_subsequent_steps") or 0) * (w.get("context_increase_per_step") or 0)
                # immediate = tokens_wasted - snowball
                immediate = w.get("completion_wasted") or 0
                wastage_by_step[sn] = {
                    "total": tokens_wasted,
                    "immediate": immediate,
                    "snowball": snowball
                }

        # Build unified step rows (only for steps we have judge metadata)
        for sn, meta in steps_meta.items():
            total_tokens = int(usage_by_step.get(sn, 0) or 0)
            wastage = wastage_by_step.get(sn, {"total": 0, "immediate": 0, "snowball": 0})
            wasted_tokens = int(wastage["total"])
            immediate_wasted = int(wastage["immediate"])
            snowball_wasted = int(wastage["snowball"])
            
            category = meta["category"] if meta["redundant"] else "essential"
            rows.append({
                "instance_id": instance_id,
                "step_number": sn,
                "action_type": meta["action_type"],
                "category": category,
                "redundant": meta["redundant"],
                "total_tokens": total_tokens,
                "wasted_tokens": wasted_tokens,
                "immediate_wasted": immediate_wasted,
                "snowball_wasted": snowball_wasted,
                "non_wasted_tokens": max(0, total_tokens - (snowball_wasted + immediate_wasted)),
            })
    return rows

def aggregate(rows, key):
    """
    key: 'action_type' or 'category'
    Returns list of (name, dict) sorted appropriately, where dict has:
      total_tokens, wasted_tokens, immediate_wasted, snowball_wasted, non_wasted_tokens (all averaged per instance), steps_count, redundant_steps
    """
    # First, sum by instance
    instance_data = defaultdict(lambda: defaultdict(lambda: {
        "total_tokens": 0,
        "wasted_tokens": 0,
        "immediate_wasted": 0,
        "snowball_wasted": 0,
        "non_wasted_tokens": 0,
        "steps_count": 0,
        "redundant_steps": 0
    }))
    
    for r in rows:
        k = r.get(key) or "unknown"
        # Combine all action types starting with 'other' into 'other' category
        if key == "action_type" and k.startswith("other"):
            k = "other"
        
        instance_id = r["instance_id"]
        instance_data[k][instance_id]["total_tokens"] += int(r["total_tokens"])
        instance_data[k][instance_id]["wasted_tokens"] += int(r["wasted_tokens"])
        instance_data[k][instance_id]["immediate_wasted"] += int(r["immediate_wasted"])
        instance_data[k][instance_id]["snowball_wasted"] += int(r["snowball_wasted"])
        instance_data[k][instance_id]["non_wasted_tokens"] += int(r["non_wasted_tokens"])
        instance_data[k][instance_id]["steps_count"] += 1
        instance_data[k][instance_id]["redundant_steps"] += 1 if r["redundant"] else 0
    
    # Get total number of unique instances across all categories
    all_instances = set()
    for r in rows:
        all_instances.add(r["instance_id"])
    num_instances = len(all_instances)
    
    # Now aggregate and divide by total number of instances
    agg = {}
    for k, instances in instance_data.items():
        total_tokens_sum = sum(inst["total_tokens"] for inst in instances.values())
        wasted_tokens_sum = sum(inst["wasted_tokens"] for inst in instances.values())
        immediate_wasted_sum = sum(inst["immediate_wasted"] for inst in instances.values())
        snowball_wasted_sum = sum(inst["snowball_wasted"] for inst in instances.values())
        non_wasted_tokens_sum = sum(inst["non_wasted_tokens"] for inst in instances.values())
        steps_count_sum = sum(inst["steps_count"] for inst in instances.values())
        redundant_steps_sum = sum(inst["redundant_steps"] for inst in instances.values())
        
        agg[k] = {
            "total_tokens": total_tokens_sum / num_instances,
            "wasted_tokens": wasted_tokens_sum / num_instances,
            "immediate_wasted": immediate_wasted_sum / num_instances,
            "snowball_wasted": snowball_wasted_sum / num_instances,
            "non_wasted_tokens": non_wasted_tokens_sum / num_instances,
            "steps_count": steps_count_sum,
            "redundant_steps": redundant_steps_sum
        }

    if key == "category":
        items = list(agg.items())
        essential = [(k, v) for k, v in items if k == "essential"]
        others = [(k, v) for k, v in items if k != "essential"]
        others.sort(key=lambda kv: kv[1]["wasted_tokens"], reverse=True)
        return essential + others
    else:
        items = list(agg.items())
        ordered = []
        
        for action_type in ACTION_TYPE_ORDER:
            for k, v in items:
                if k == action_type:
                    ordered.append((k, v))
                    break
        
        ordered_keys = {k for k, _ in ordered}
        remaining = [(k, v) for k, v in items if k not in ordered_keys]
        remaining.sort(key=lambda kv: kv[1]["total_tokens"], reverse=True)
        
        return ordered + remaining

def format_number(n):
    """Format number in K or M with appropriate suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(int(n))

def plot_steps_comparison(model_data, outfile):
    """
    Create subplots comparing step distributions across models.
    model_data: dict of {model_name: rows}
    """
    n_models = len(model_data)
    if n_models == 0:
        return
    
    # Sort models according to preference
    sorted_model_names = sort_models(list(model_data.keys()))
    
    # Create subplot grid (2 rows x 3 cols for up to 6 models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # First pass: determine global y-axis limits
    max_y = 0
    all_histograms = []
    
    for model_name in sorted_model_names:
        rows = model_data[model_name]
        steps_per_trajectory = Counter(r["instance_id"] for r in rows)
        step_counts = list(steps_per_trajectory.values())
        
        if step_counts:
            counts, _ = np.histogram(step_counts, bins=20)
            max_y = max(max_y, counts.max())
            avg_steps = sum(step_counts) / len(step_counts)
            median_steps = float(np.median(step_counts))
            all_histograms.append((model_name, step_counts, avg_steps, median_steps))
        else:
            all_histograms.append((model_name, None, None, None))
    
    # Add 10% padding to max_y
    # max_y = max_y * 1.1
    max_y = 25
    
    # Second pass: plot with uniform y-axis
    for idx, (model_name, step_counts, avg_steps, median_steps) in enumerate(all_histograms):
        ax = axes[idx]
        
        if step_counts is None:
            ax.set_visible(False)
            continue
        
        # Use consistent color for this model
        color = get_model_color(model_name)
        
        ax.hist(
            step_counts,
            bins=20,
            edgecolor='black',
            alpha=0.7,
            color=color
        )
        
        # average (red)
        ax.axvline(avg_steps, color='red', linestyle='--', linewidth=2.5, label='Mean')
        # median (blue)
        ax.axvline(median_steps, color='blue', linestyle=':', linewidth=2.5, label='Median')
        
        # Add text box (show both)
        ax.text(
            0.98,
            0.95,
            f'Avg: {avg_steps:.1f}\nMed: {median_steps:.1f}',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
        
        ax.set_xlabel("Number of Steps per Trajectory", fontsize=12)
        ax.set_ylabel("Number of Trajectories", fontsize=12)
        ax.set_title(get_display_name(model_name), fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=11)
        
        # Set uniform y-axis limits
        ax.set_ylim(0, max_y)
        
        # optional: show legend for the two lines
        ax.legend(
            loc='upper left',
            fontsize=10,
            frameon=True,
            facecolor='white',
            edgecolor='black'
        )
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        "Step Distribution Comparison Across Models",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outfile}")


def plot_grouped_comparison(model_aggregates, key_type, outfile):
    """
    Create grouped bar chart comparing models with immediate and snowball wastage breakdown.
    model_aggregates: dict of {model_name: aggregated_data}
    key_type: 'action_type' or 'category'
    """
    if not model_aggregates:
        return
    
    # Get all unique categories across all models
    all_categories = set()
    for agg_data in model_aggregates.values():
        all_categories.update([k for k, v in agg_data])
    
    # Sort categories appropriately
    if key_type == "action_type":
        categories = [cat for cat in ACTION_TYPE_ORDER if cat in all_categories]
        remaining = sorted([cat for cat in all_categories if cat not in ACTION_TYPE_ORDER])
        categories.extend(remaining)
    else:
        # Use CATEGORY_ORDER for redundancy categories
        categories = [cat for cat in CATEGORY_ORDER if cat in all_categories]
        remaining = sorted([cat for cat in all_categories if cat not in CATEGORY_ORDER])
        categories.extend(remaining)
    
    # Sort models according to preference
    model_names = sort_models(list(model_aggregates.keys()))
    n_models = len(model_names)
    n_categories = len(categories)
    
    # Prepare data matrices
    immediate_matrix = np.zeros((n_models, n_categories))
    snowball_matrix = np.zeros((n_models, n_categories))
    non_wasted_matrix = np.zeros((n_models, n_categories))
    
    # Calculate total wastage per model for legend
    model_totals = {}
    
    for i, model_name in enumerate(model_names):
        agg_dict = {k: v for k, v in model_aggregates[model_name]}
        
        total_wasted = 0
        total_tokens = 0
        
        for j, cat in enumerate(categories):
            if cat in agg_dict:
                immediate = agg_dict[cat]["immediate_wasted"]
                snowball = agg_dict[cat]["snowball_wasted"]
                non_wasted = agg_dict[cat]["non_wasted_tokens"]
                immediate_matrix[i, j] = immediate
                snowball_matrix[i, j] = snowball
                non_wasted_matrix[i, j] = non_wasted
                total_wasted += immediate + snowball
                total_tokens += immediate + snowball + non_wasted
        
        model_totals[model_name] = {
            'total_wasted': total_wasted,
            'total_tokens': total_tokens,
            'percentage': (total_wasted / total_tokens * 100) if total_tokens > 0 else 0
        }
    
    # Calculate percentage wasted for each model-category combination
    immediate_pct_matrix = np.zeros((n_models, n_categories))
    snowball_pct_matrix = np.zeros((n_models, n_categories))
    for i in range(n_models):
        for j in range(n_categories):
            total = immediate_matrix[i, j] + snowball_matrix[i, j] + non_wasted_matrix[i, j]
            if total > 0:
                immediate_pct_matrix[i, j] = (immediate_matrix[i, j] / total) * 100
                snowball_pct_matrix[i, j] = (snowball_matrix[i, j] / total) * 100
    
    # Create grouped bar chart
    fig_width = max(14, n_categories * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    x = np.arange(n_categories)
    width = 0.8 / n_models  # Width of each bar
    
    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        # Use consistent color for this model
        color = get_model_color(model_name)
        
        # Plot non-wasted (bottom)
        ax.bar(x + offset, non_wasted_matrix[i] / 1e6, width, 
               label=f'{model_name}' if i == 0 else '',
               color=color, alpha=0.7)
        
        # Plot immediate wasted (middle)
        ax.bar(x + offset, immediate_matrix[i] / 1e6, width,
               bottom=non_wasted_matrix[i] / 1e6,
               color=color, alpha=0.4, hatch='//')
        
        # Plot snowball wasted (top)
        ax.bar(x + offset, snowball_matrix[i] / 1e6, width,
               bottom=(non_wasted_matrix[i] + immediate_matrix[i]) / 1e6,
               color=color, alpha=0.2, hatch='xx')
        
        # Add percentage labels ONLY for action_type plot
        if key_type == "action_type":
            for j in range(n_categories):
                total_height = (immediate_matrix[i, j] + snowball_matrix[i, j] + non_wasted_matrix[i, j]) / 1e6
                if total_height > 0:
                    # Immediate wastage label (at middle of immediate section)
                    if immediate_pct_matrix[i, j] > 0:
                        immediate_height = (non_wasted_matrix[i, j] + immediate_matrix[i, j] / 2) / 1e6
                        ax.annotate(f'{immediate_pct_matrix[i, j]:.1f}%',
                                   xy=(x[j] + offset, immediate_height),
                                   xytext=(0, 0),
                                   textcoords='offset points',
                                   ha='center', va='center', fontsize=8, rotation=0,
                                   color='black', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                            edgecolor=color, alpha=0.9, linewidth=0.8))
                    
                    # Snowball wastage label (at middle of snowball section)
                    if snowball_pct_matrix[i, j] > 0:
                        snowball_height = (non_wasted_matrix[i, j] + immediate_matrix[i, j] + snowball_matrix[i, j] / 2) / 1e6
                        ax.annotate(f'{snowball_pct_matrix[i, j]:.1f}%',
                                   xy=(x[j] + offset, snowball_height),
                                   xytext=(0, 0),
                                   textcoords='offset points',
                                   ha='center', va='center', fontsize=8, rotation=0,
                                   color='black', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                            edgecolor=color, alpha=0.9, linewidth=0.8))
    
    # Custom legend with total wastage information
    from matplotlib.patches import Patch
    legend_elements = []

    for i, model_name in enumerate(model_names):
        totals = model_totals[model_name]
        wasted_str = format_number(totals['total_wasted'])
        total_str = format_number(totals['total_tokens'])
        pct_val = f"{totals['percentage']:.1f}"
        display_name = get_display_name(model_name)

        # bold only the pct
        pct_bold = rf"$\mathbf{{{pct_val}\%}}$"

        label = f"{display_name} ({wasted_str}/{total_str}, {pct_bold})"

        # Use consistent color for this model
        color = get_model_color(model_name)

        legend_elements.append(
            Patch(
                facecolor=color,
                alpha=0.7,
                label=label
            )
        )

    # separator
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))

    # wastage indicators
    legend_elements.append(Patch(facecolor='gray', alpha=0.7, label='Non-wasted'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.4, hatch='//', label='Immediate wasted'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.2, hatch='xx', label='Snowball wasted'))

    from matplotlib.font_manager import FontProperties
    # no bold here – just size
    legend_font = FontProperties(size=12)

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        prop=legend_font,
        ncol=1,
        framealpha=0.9
    )

    
    ax.set_ylabel('Average Tokens per Instance (Millions)', fontsize=13, fontweight='bold')
    ax.set_xlabel(key_type.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    
    title = f"Token Usage Comparison by {key_type.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    ax.set_ylim(0, 1.8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outfile}")

def write_csv(rows, by_action, by_category, outdir: Path, model_name: str):
    """Write CSV files for a single model."""
    model_prefix = model_name.replace('/', '_').replace(' ', '_')
    
    # Detailed per-step
    with open(outdir / f"{model_prefix}_per_step_tokens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id","step_number","action_type","category",
                    "redundant","total_tokens","wasted_tokens","immediate_wasted","snowball_wasted","non_wasted_tokens"])
        for r in sorted(rows, key=lambda x: (x["instance_id"], x["step_number"])):
            w.writerow([
                r["instance_id"], r["step_number"], r["action_type"], r["category"],
                int(r["redundant"]), r["total_tokens"], r["wasted_tokens"], 
                r["immediate_wasted"], r["snowball_wasted"], r["non_wasted_tokens"]
            ])

    # Aggregated by action type
    with open(outdir / f"{model_prefix}_tokens_by_action_type.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_type","steps_count","redundant_steps","avg_total_tokens","avg_wasted_tokens",
                    "avg_immediate_wasted","avg_snowball_wasted","avg_non_wasted_tokens"])
        for k, v in by_action:
            w.writerow([k, v["steps_count"], v["redundant_steps"], v["total_tokens"], v["wasted_tokens"],
                       v["immediate_wasted"], v["snowball_wasted"], v["non_wasted_tokens"]])

    # Aggregated by category
    with open(outdir / f"{model_prefix}_tokens_by_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category","steps_count","redundant_steps","avg_total_tokens","avg_wasted_tokens",
                    "avg_immediate_wasted","avg_snowball_wasted","avg_non_wasted_tokens"])
        for k, v in by_category:
            w.writerow([k, v["steps_count"], v["redundant_steps"], v["total_tokens"], v["wasted_tokens"],
                       v["immediate_wasted"], v["snowball_wasted"], v["non_wasted_tokens"]])

def main():
    ap = argparse.ArgumentParser(
        description="Compare agent token wastage across multiple models.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--parent-dir", 
        default="/usr0/home/srgandhi/tool-overuse/judge_swe_bench_verified",
        type=str, 
        help="Parent directory containing model subdirectories"
    )
    ap.add_argument(
        "--setting", 
        default="edit_obs_temp_07_dev",
        type=str, 
        help="Setting prefix for model directories (e.g., edit_obs_temp_07_dev)"
    )
    ap.add_argument(
        "--outdir", 
        default=None, 
        type=str, 
        help="Output folder (default: {parent-dir}/{setting}_wastage_analysis)"
    )
    args = ap.parse_args()

    parent_dir = Path(args.parent_dir).expanduser().resolve()
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise SystemExit(f"Parent directory not found: {parent_dir}")

    # Discover model directories
    model_dirs = discover_model_dirs(parent_dir, args.setting)
    if not model_dirs:
        raise SystemExit(f"No model directories found matching pattern {args.setting}_*/policy_v2 in {parent_dir}")
    
    # Sort model directories according to preference
    model_names = [name for name, _ in model_dirs]
    sorted_model_names = sort_models(model_names)
    model_dirs_dict = {name: path for name, path in model_dirs}
    sorted_model_dirs = [(name, model_dirs_dict[name]) for name in sorted_model_names]
    
    print(f"Found {len(sorted_model_dirs)} models:")
    for model_name, _ in sorted_model_dirs:
        display_name = get_display_name(model_name)
        color = get_model_color(model_name)
        if display_name != model_name:
            print(f"  - {display_name} (from {model_name}) [{color}]")
        else:
            print(f"  - {model_name} [{color}]")
    print()

    # Set output directory
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (parent_dir / f"{args.setting}_wastage_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and process data for each model
    all_model_data = {}
    all_model_by_action = {}
    all_model_by_category = {}

    for model_name, policy_dir in sorted_model_dirs:
        display_name = get_display_name(model_name)
        print(f"Processing {display_name}...")
        records = load_records(policy_dir)
        rows = join_per_step(records)
        
        if not rows:
            print(f"  Warning: No step rows found for {display_name}")
            continue
        
        all_model_data[model_name] = rows
        all_model_by_action[model_name] = aggregate(rows, key="action_type")
        all_model_by_category[model_name] = aggregate(rows, key="category")
        
        # Write individual model CSVs
        write_csv(rows, all_model_by_action[model_name], 
                 all_model_by_category[model_name], outdir, model_name)
        
        print(f"  Processed {len(rows)} steps from {len(set(r['instance_id'] for r in rows))} trajectories")

    if not all_model_data:
        raise SystemExit("No data found for any model.")

    print("\nGenerating comparison plots...")
    
    # Create comparison plots
    plot_steps_comparison(all_model_data, outdir / "steps_comparison.png")
    plot_grouped_comparison(all_model_by_action, "action_type", 
                           outdir / "tokens_by_action_type_comparison.png")
    plot_grouped_comparison(all_model_by_category, "category",
                           outdir / "tokens_by_category_comparison.png")

    print(f"\n✓ All outputs saved to: {outdir}")
    print(f"  - Comparison plots: 3 PNG files")
    print(f"  - Individual CSVs: {len(all_model_data) * 3} files")


if __name__ == "__main__":
    main()