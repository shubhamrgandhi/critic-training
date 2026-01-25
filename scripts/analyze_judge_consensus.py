#!/usr/bin/env python3
"""
Analyze judge consensus from majority voting metadata.

Inputs:
  --inputs  : folder containing *_judge_result.json and/or all_judge_results.json
  --outdir  : output folder for consensus report (default: <inputs>/consensus_analysis)

Outputs:
  consensus_report.txt: consensus metrics averaged over all instances
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_records(inputs: Path):
    """Load judge result JSON files (same logic as original script)."""
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


def analyze_consensus(records):
    """
    Analyze consensus from majority_voting_metadata.
    Returns dict with consensus metrics.
    """
    all_agreement_rates = []
    unanimous_count = 0  # 100%
    strong_count = 0     # 80-99%
    moderate_count = 0   # 70-79%
    split_count = 0      # 60-69%
    weak_count = 0       # 50-59%
    total_steps = 0
    
    for rec in records:
        metadata = rec.get("majority_voting_metadata", {})
        if not metadata:
            continue
            
        step_details = metadata.get("step_voting_details", [])
        if not step_details:
            continue
        
        for step in step_details:
            total_votes = step.get("total_votes", 0)
            if total_votes == 0:
                continue
            
            redundant_votes = step.get("redundant_votes", 0)
            essential_votes = step.get("essential_votes", 0)
            
            # Agreement rate = votes for majority decision / total votes
            majority_votes = max(redundant_votes, essential_votes)
            agreement_rate = majority_votes / total_votes
            
            all_agreement_rates.append(agreement_rate)
            total_steps += 1
            
            # Mutually exclusive categories
            if agreement_rate == 1.0:
                unanimous_count += 1
            elif agreement_rate >= 0.8:
                strong_count += 1
            elif agreement_rate >= 0.7:
                moderate_count += 1
            elif agreement_rate >= 0.6:
                split_count += 1
            else:
                weak_count += 1
    
    if total_steps == 0:
        return None
    
    return {
        "total_steps": total_steps,
        "total_instances": len(records),
        "avg_agreement_rate": sum(all_agreement_rates) / len(all_agreement_rates),
        "unanimous_pct": (unanimous_count / total_steps) * 100,
        "strong_pct": (strong_count / total_steps) * 100,
        "moderate_pct": (moderate_count / total_steps) * 100,
        "split_pct": (split_count / total_steps) * 100,
        "weak_pct": (weak_count / total_steps) * 100,
        "unanimous_count": unanimous_count,
        "strong_count": strong_count,
        "moderate_count": moderate_count,
        "split_count": split_count,
        "weak_count": weak_count,
    }


def write_consensus_report(metrics, outfile: Path):
    """Write consensus metrics to a text file."""
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("JUDGE CONSENSUS ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total instances analyzed: {metrics['total_instances']}\n")
        f.write(f"Total steps analyzed: {metrics['total_steps']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("OVERALL CONSENSUS\n")
        f.write("-" * 60 + "\n\n")
        
        f.write(f"Average Agreement Rate: {metrics['avg_agreement_rate']:.2%}\n")
        f.write(f"  (Mean % of votes that agreed with majority decision)\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("CONSENSUS DISTRIBUTION (mutually exclusive)\n")
        f.write("-" * 60 + "\n\n")
        
        f.write(f"Unanimous (100%):        {metrics['unanimous_pct']:5.1f}%")
        f.write(f"  ({metrics['unanimous_count']:4d} steps)\n")
        
        f.write(f"Strong (80-99%):         {metrics['strong_pct']:5.1f}%")
        f.write(f"  ({metrics['strong_count']:4d} steps)\n")
        
        f.write(f"Moderate (70-79%):       {metrics['moderate_pct']:5.1f}%")
        f.write(f"  ({metrics['moderate_count']:4d} steps)\n")
        
        f.write(f"Split (60-69%):          {metrics['split_pct']:5.1f}%")
        f.write(f"  ({metrics['split_count']:4d} steps)\n")
        
        f.write(f"Weak (50-59%):           {metrics['weak_pct']:5.1f}%")
        f.write(f"  ({metrics['weak_count']:4d} steps)\n")
        
        total_pct = (metrics['unanimous_pct'] + metrics['strong_pct'] + 
                     metrics['moderate_pct'] + metrics['split_pct'] + metrics['weak_pct'])
        f.write(f"{'-' * 40}\n")
        f.write(f"Total:                   {total_pct:5.1f}%\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 60 + "\n\n")
        
        avg = metrics['avg_agreement_rate']
        unanimous_strong = metrics['unanimous_pct'] + metrics['strong_pct']
        
        f.write(f"High Confidence Steps (≥80%): {unanimous_strong:.1f}%\n")
        f.write(f"Uncertain Steps (<70%): {metrics['split_pct'] + metrics['weak_pct']:.1f}%\n\n")
        
        if avg >= 0.85:
            f.write("✓ High confidence: Judge shows strong consensus across votes.\n")
        elif avg >= 0.70:
            f.write("⚠ Moderate confidence: Some disagreement in judge decisions.\n")
        else:
            f.write("✗ Low confidence: Significant disagreement in judge votes.\n")
        
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Analyze judge consensus from majority voting.")
    ap.add_argument("--inputs", required=True, type=str, 
                    help="Folder with judge analysis JSON outputs.")
    ap.add_argument("--outdir", default=None, type=str, 
                    help="Output folder (default: <inputs>/consensus_analysis)")
    args = ap.parse_args()

    inputs = Path(args.inputs).expanduser().resolve()
    if not inputs.exists() or not inputs.is_dir():
        raise SystemExit(f"Input folder not found: {inputs}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (inputs / "consensus_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(inputs)
    if not records:
        print("No judge result files found.")
        return

    metrics = analyze_consensus(records)
    if not metrics:
        print("No majority voting metadata found in records.")
        return

    outfile = outdir / "consensus_report.txt"
    write_consensus_report(metrics, outfile)
    
    print(f"✓ Consensus analysis saved to: {outfile}")
    print(f"\nQuick Summary:")
    print(f"  Average Agreement: {metrics['avg_agreement_rate']:.1%}")
    print(f"  Unanimous (100%): {metrics['unanimous_pct']:.1f}%")
    print(f"  Strong (80-99%): {metrics['strong_pct']:.1f}%")
    print(f"  Split/Uncertain (<70%): {metrics['split_pct'] + metrics['moderate_pct'] + metrics['weak_pct']:.1f}%")


if __name__ == "__main__":
    main()