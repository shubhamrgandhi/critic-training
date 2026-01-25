#!/usr/bin/env python3
"""
Analyze judge analysis outputs and plot agent token totals vs wastage.

Inputs:
  --inputs  : folder containing *_judge_result.json and/or all_judge_results.json
  --outdir  : output folder for PNGs/CSVs (default: <inputs>/wastage_plots)

Outputs:
  PNGs:
    - tokens_total_vs_wasted_by_action_type.png
    - tokens_total_vs_wasted_by_category.png
  CSVs:
    - tokens_by_action_type.csv
    - tokens_by_category.csv
    - per_step_tokens.csv   (optional detail per step)

Definition:
  total_tokens (per step)   := agent_token_usage_by_step[step].total_tokens
  wasted_tokens (per step)  := agent_token_wastage[step].tokens_wasted (0 if not redundant)
  non_wasted_tokens         := max(0, total_tokens - wasted_tokens)

Grouping:
  - Action type: use judge_response.steps[*].action_type
  - Category:
      * For redundant steps: use the overuse category from judge (e.g., 'duplicate_read')
      * For essential steps: 'essential'
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

# Fixed order for action types in plots
ACTION_TYPE_ORDER = ["read", "write", "search", "execute", "env", "other"]

def load_records(inputs: Path):
    # Collect per-trajectory files plus aggregate file if present
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
      total_tokens (agent), wasted_tokens (agent)
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

        # Wastage per redundant step
        wasted_by_step = {w.get("step_number"): (w.get("tokens_wasted") or 0)
                          for w in (rec.get("agent_token_wastage") or []) if isinstance(w, dict)}

        # Build unified step rows (only for steps we have judge metadata)
        for sn, meta in steps_meta.items():
            total_tokens = int(usage_by_step.get(sn, 0) or 0)
            wasted_tokens = int(wasted_by_step.get(sn, 0) or 0)
            # Category for essentials should be 'essential'
            category = meta["category"] if meta["redundant"] else "essential"
            rows.append({
                "instance_id": instance_id,
                "step_number": sn,
                "action_type": meta["action_type"],
                "category": category,
                "redundant": meta["redundant"],
                "total_tokens": total_tokens,
                "wasted_tokens": wasted_tokens,
                "non_wasted_tokens": max(0, total_tokens - wasted_tokens),
            })
    return rows

def aggregate(rows, key):
    """
    key: 'action_type' or 'category'
    Returns list of (name, dict) sorted appropriately, where dict has:
      total_tokens, wasted_tokens, non_wasted_tokens, steps_count, redundant_steps
    """
    agg = defaultdict(lambda: {
        "total_tokens": 0,
        "wasted_tokens": 0,
        "non_wasted_tokens": 0,
        "steps_count": 0,
        "redundant_steps": 0
    })
    for r in rows:
        k = r.get(key) or "unknown"
        # Combine all action types starting with 'other' into 'other' category
        if key == "action_type" and k.startswith("other"):
            k = "other"
        agg[k]["total_tokens"] += int(r["total_tokens"])
        agg[k]["wasted_tokens"] += int(r["wasted_tokens"])
        agg[k]["non_wasted_tokens"] += int(r["non_wasted_tokens"])
        agg[k]["steps_count"] += 1
        agg[k]["redundant_steps"] += 1 if r["redundant"] else 0

    if key == "category":
        # For category: put 'essential' first, then sort others by wasted_tokens descending
        items = list(agg.items())
        essential = [(k, v) for k, v in items if k == "essential"]
        others = [(k, v) for k, v in items if k != "essential"]
        others.sort(key=lambda kv: kv[1]["wasted_tokens"], reverse=True)
        return essential + others
    else:
        # For action_type: use fixed order
        items = list(agg.items())
        ordered = []
        
        # Add items in fixed order if they exist
        for action_type in ACTION_TYPE_ORDER:
            for k, v in items:
                if k == action_type:
                    ordered.append((k, v))
                    break
        
        # Add any remaining items not in fixed order at the end
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
        return str(n)

def plot_stacked_total_vs_wasted(bar_data, title, outfile, global_totals):
    labels = [k for k, _ in bar_data]
    total = [v["total_tokens"] for _, v in bar_data]
    wasted = [v["wasted_tokens"] for _, v in bar_data]
    non_wasted = [v["non_wasted_tokens"] for _, v in bar_data]

    # Use global totals passed in
    total_wasted = global_totals["wasted"]
    total_essential = global_totals["essential"]
    
    # Debug: verify totals match
    sum_wasted = sum(wasted)
    sum_non_wasted = sum(non_wasted)
    print(f"Plot '{title}':")
    print(f"  Sum of bars - Wasted: {sum_wasted}, Non-wasted: {sum_non_wasted}")
    print(f"  Global totals - Wasted: {total_wasted}, Essential: {total_essential}")
    print()

    # Size scales with number of bars, but keep sane bounds
    fig_w = max(6, min(18, 0.6 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(labels, non_wasted, label="Non-wasted tokens")
    ax.bar(labels, wasted, bottom=non_wasted, label="Wasted tokens")
    ax.set_title(title)
    ax.set_ylabel("Sum of agent tokens across steps")
    ax.set_xlabel("")
    
    # Place legend in upper right
    legend = ax.legend(loc='upper right')
    
    # Add info box below legend - use actual sum from bars to verify
    info_text = f"Total Wasted: {format_number(sum_wasted)}\nTotal Essential: {format_number(sum_non_wasted)}"
    ax.text(0.98, 0.85, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Add formatted numeric labels on tops
    for i, (nw, w) in enumerate(zip(non_wasted, wasted)):
        tot = nw + w
        if tot > 0:
            ax.annotate(format_number(tot), (i, tot), ha="center", va="bottom", 
                       fontsize=8, xytext=(0, 2), textcoords="offset points")
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_steps_histogram(rows, outfile):
    """Plot histogram of number of steps per trajectory with average line."""
    from collections import Counter
    
    # Count steps per trajectory
    steps_per_trajectory = Counter(r["instance_id"] for r in rows)
    step_counts = list(steps_per_trajectory.values())
    
    if not step_counts:
        return
    
    avg_steps = sum(step_counts) / len(step_counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(step_counts, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(avg_steps, color='red', linestyle='--', linewidth=2, 
               label=f'Avg: {avg_steps:.1f} steps')
    ax.set_xlabel("Number of steps per trajectory")
    ax.set_ylabel("Number of trajectories")
    ax.set_title("Distribution of Steps per Trajectory")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def write_csv(rows, by_action, by_category, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Detailed per-step
    with open(outdir / "per_step_tokens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id","step_number","action_type","category",
                    "redundant","total_tokens","wasted_tokens","non_wasted_tokens"])
        for r in sorted(rows, key=lambda x: (x["instance_id"], x["step_number"])):
            w.writerow([
                r["instance_id"], r["step_number"], r["action_type"], r["category"],
                int(r["redundant"]), r["total_tokens"], r["wasted_tokens"], r["non_wasted_tokens"]
            ])

    # Aggregated by action type
    with open(outdir / "tokens_by_action_type.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_type","steps_count","redundant_steps","total_tokens","wasted_tokens","non_wasted_tokens"])
        for k, v in by_action:
            w.writerow([k, v["steps_count"], v["redundant_steps"], v["total_tokens"], v["wasted_tokens"], v["non_wasted_tokens"]])

    # Aggregated by category
    with open(outdir / "tokens_by_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category","steps_count","redundant_steps","total_tokens","wasted_tokens","non_wasted_tokens"])
        for k, v in by_category:
            w.writerow([k, v["steps_count"], v["redundant_steps"], v["total_tokens"], v["wasted_tokens"], v["non_wasted_tokens"]])

def main():
    ap = argparse.ArgumentParser(description="Plot agent total tokens vs wastage by action type and category.")
    ap.add_argument("--inputs", required=True, type=str, help="Folder with judge analysis JSON outputs.")
    ap.add_argument("--outdir", default=None, type=str, help="Output folder (default: <inputs>/wastage_plots)")
    args = ap.parse_args()

    inputs = Path(args.inputs).expanduser().resolve()
    if not inputs.exists() or not inputs.is_dir():
        raise SystemExit(f"Input folder not found: {inputs}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (inputs / "wastage_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(inputs)
    rows = join_per_step(records)
    if not rows:
        print("No step rows found. Ensure your *_judge_result.json / all_judge_results.json files are present.")
        return

    # Calculate global totals once from raw rows
    global_totals = {
        "wasted": sum(r["wasted_tokens"] for r in rows),
        "essential": sum(r["non_wasted_tokens"] for r in rows)
    }

    by_action = aggregate(rows, key="action_type")
    by_category = aggregate(rows, key="category")

    # Plots: pass global totals to ensure consistency
    plot_stacked_total_vs_wasted(by_action, "Agent Tokens: Total vs Wasted by Action Type",
                                 outdir / "tokens_total_vs_wasted_by_action_type.png",
                                 global_totals)
    plot_stacked_total_vs_wasted(by_category, "Agent Tokens: Total vs Wasted by Category",
                                 outdir / "tokens_total_vs_wasted_by_category.png",
                                 global_totals)

    # CSVs
    write_csv(rows, by_action, by_category, outdir)

    plot_steps_histogram(rows, outdir / "steps_per_trajectory.png")

    print(f"Saved plots and CSVs to: {outdir}")
    print(f"Saved plots (including steps histogram) and CSVs to: {outdir}")


if __name__ == "__main__":
    main()