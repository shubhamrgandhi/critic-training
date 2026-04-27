#!/usr/bin/env python3
"""Step 2: Compute BLEU and ROUGE scores comparing two models' PRM responses.

Reads responses from eval_results/<model-a>/responses.jsonl and
eval_results/<model-b>/responses.jsonl, computes BLEU-1/2/4 and
ROUGE-1/2/L for each against the Claude Opus ground truth.

Usage:
    python eval_compute_scores.py --model-a qwen3-8b-base --model-b opus-distill-32k-lr5e6
"""

import argparse
import json
import sys
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval_results"


def compute_bleu(reference: str, hypothesis: str) -> dict:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if not hyp_tokens or not ref_tokens:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}

    smoothie = SmoothingFunction().method1
    return {
        "bleu1": sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        "bleu2": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "bleu4": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }


def compute_rouge(reference: str, hypothesis: str, scorer: rouge_scorer.RougeScorer) -> dict:
    if not hypothesis.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def load_responses(eval_dir: Path, model_label: str) -> dict:
    """Load responses for a model, keyed by idx."""
    path = eval_dir / model_label / "responses.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run eval_generate_responses.py first.")
        sys.exit(1)
    data = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data[item["idx"]] = item
    return data


def score_model(data: dict, label: str, scorer: rouge_scorer.RougeScorer) -> dict:
    """Compute BLEU/ROUGE for one model's responses vs ground truth."""
    all_bleu = {"bleu1": [], "bleu2": [], "bleu4": []}
    all_rouge = {"rouge1": [], "rouge2": [], "rougeL": []}
    per_sample = {}

    for idx in tqdm(sorted(data.keys()), desc=f"Scoring {label}", unit="sample"):
        d = data[idx]
        ref = d["ground_truth"]
        hyp = d["response"]

        bleu = compute_bleu(ref, hyp)
        rouge = compute_rouge(ref, hyp, scorer)

        for k in all_bleu:
            all_bleu[k].append(bleu[k])
        for k in all_rouge:
            all_rouge[k].append(rouge[k])

        per_sample[idx] = {"bleu": bleu, "rouge": rouge}

    avg_bleu = {k: sum(v) / len(v) for k, v in all_bleu.items()}
    avg_rouge = {k: sum(v) / len(v) for k, v in all_rouge.items()}

    return {
        "n_samples": len(data),
        "avg_bleu": avg_bleu,
        "avg_rouge": avg_rouge,
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute BLEU/ROUGE scores")
    parser.add_argument("--model-a", type=str, required=True,
                        help="First model label (e.g. qwen3-8b-base)")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Second model label (e.g. opus-distill-32k-lr5e6)")
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Load both models' responses
    data_a = load_responses(eval_dir, args.model_a)
    data_b = load_responses(eval_dir, args.model_b)
    print(f"Model A ({args.model_a}): {len(data_a)} samples")
    print(f"Model B ({args.model_b}): {len(data_b)} samples")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = {"models": {}, "comparison": {}}

    for label, data in [(args.model_a, data_a), (args.model_b, data_b)]:
        model_results = score_model(data, label, scorer)
        results["models"][label] = {
            "n_samples": model_results["n_samples"],
            "avg_bleu": model_results["avg_bleu"],
            "avg_rouge": model_results["avg_rouge"],
            "_per_sample": model_results["per_sample"],
        }

        print(f"\n{label} ({model_results['n_samples']} samples):")
        ab = model_results["avg_bleu"]
        ar = model_results["avg_rouge"]
        print(f"  BLEU-1: {ab['bleu1']:.4f}  BLEU-2: {ab['bleu2']:.4f}  BLEU-4: {ab['bleu4']:.4f}")
        print(f"  ROUGE-1: {ar['rouge1']:.4f}  ROUGE-2: {ar['rouge2']:.4f}  ROUGE-L: {ar['rougeL']:.4f}")

    # Comparison
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {args.model_b} vs {args.model_a}")
    print(f"{'=' * 60}")
    for metric_group in ["avg_bleu", "avg_rouge"]:
        for metric in results["models"][args.model_a][metric_group]:
            val_a = results["models"][args.model_a][metric_group][metric]
            val_b = results["models"][args.model_b][metric_group][metric]
            diff = val_b - val_a
            arrow = "+" if diff > 0 else ""
            print(f"  {metric:8s}  {args.model_a}={val_a:.4f}  {args.model_b}={val_b:.4f}  delta={arrow}{diff:.4f}")

    # Per-sample comparison using already-computed scores
    shared = sorted(set(data_a.keys()) & set(data_b.keys()))
    per_sample_a = results["models"][args.model_a].get("_per_sample", {})
    per_sample_b = results["models"][args.model_b].get("_per_sample", {})
    if shared and per_sample_a and per_sample_b:
        results["comparison"]["shared_samples"] = len(shared)
        a_wins = b_wins = ties = 0
        for idx in shared:
            ra = per_sample_a.get(idx, {}).get("rouge", {}).get("rougeL", 0.0)
            rb = per_sample_b.get(idx, {}).get("rouge", {}).get("rougeL", 0.0)
            if ra > rb + 0.01:
                a_wins += 1
            elif rb > ra + 0.01:
                b_wins += 1
            else:
                ties += 1
        results["comparison"]["rougeL_wins"] = {
            args.model_a: a_wins, args.model_b: b_wins, "tie": ties
        }
        print(f"\n  Per-sample ROUGE-L wins (on {len(shared)} shared samples):")
        print(f"    {args.model_a}: {a_wins}  {args.model_b}: {b_wins}  ties: {ties}")

    # Strip per-sample data before saving (too large)
    for label in results["models"]:
        results["models"][label].pop("_per_sample", None)

    output_path = eval_dir / f"scores_{args.model_a}_vs_{args.model_b}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {output_path}")


if __name__ == "__main__":
    main()
