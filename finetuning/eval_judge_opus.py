#!/usr/bin/env python3
"""Step 3: Use Claude Opus as a judge to compare two models' PRM responses.

For each sample where both models have responses, asks Claude Opus to judge
which PRM response would be more helpful for guiding the coding agent.

The judge sees: the PRM task description, the trajectory context, and both
responses (randomly ordered to avoid position bias). It picks a winner or
declares a tie.

Usage:
    python eval_judge_opus.py --model-a qwen3-8b-base --model-b opus-distill-32k-lr5e6
    python eval_judge_opus.py --model-a qwen3-8b-base --model-b opus-distill-32k-lr5e6 --limit 100
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

import litellm
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval_results"
DEFAULT_DATA_PATH = SCRIPT_DIR / "prm_sft_data_opus_distill_full_feedback_history_32k" / "prm_sft_train.jsonl"

PRM_SYSTEM_PROMPT_SUMMARY = """\
The PRM (Process Reward Model) is a supervisor that monitors an LLM-based coding agent \
solving software engineering tasks. It analyzes the agent's trajectory for 12 error categories \
(task spec violations, step repetition, hallucinations, tool selection errors, etc.) and provides:
- DETECTED: Yes/No for each category with evidence and recovery actions
- TASK_STATUS: On track / Needs correction / Critical intervention required
- OVERALL_GUIDANCE: Specific actionable guidance for the agent

The PRM's goal is to detect trajectory-level errors and provide corrective guidance \
to prevent task failure. Good PRM feedback is specific, actionable, and correctly \
identifies real issues while avoiding false positives."""

JUDGE_SYSTEM_PROMPT = f"""\
You are an expert evaluator comparing two Process Reward Model (PRM) responses for quality.

{PRM_SYSTEM_PROMPT_SUMMARY}

You will be shown:
1. The agent's trajectory context (what the PRM was reviewing)
2. Two PRM responses labeled [A] and [B] (order is randomized)

Evaluate which PRM response would be MORE HELPFUL for guiding the coding agent toward \
successfully completing its task. Consider:

- **Accuracy**: Does it correctly identify real issues vs false positives? Are the \
DETECTED labels appropriate given the trajectory?
- **Specificity**: Does it provide specific evidence and actionable recovery instructions, \
or is it vague/generic?
- **Completeness**: Does it cover the important issues without missing critical problems?
- **Actionability**: Would the OVERALL_GUIDANCE actually help the agent fix its approach?
- **Format compliance**: Does it follow the expected response structure?

Respond with EXACTLY this format:
WINNER: A or B or TIE
REASONING: Your explanation here (can be multiple sentences)"""


def truncate_context(user_content: str, max_chars: int = 15000) -> str:
    """Truncate trajectory context for the judge — keep enough for accuracy evaluation."""
    if len(user_content) <= max_chars:
        return user_content
    head = 3000
    tail = max_chars - head - 100
    return user_content[:head] + "\n\n[...middle of trajectory truncated for brevity...]\n\n" + user_content[-tail:]


def parse_judge_output(text: str) -> dict:
    """Parse WINNER and REASONING from judge output."""
    winner = "UNKNOWN"
    reasoning_lines = []
    in_reasoning = False

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("WINNER:"):
            val = stripped.split(":", 1)[1].strip().upper()
            if "TIE" in val:
                winner = "TIE"
            elif "A" in val and "B" not in val:
                winner = "A"
            elif "B" in val and "A" not in val:
                winner = "B"
            else:
                winner = val
            in_reasoning = False
        elif stripped.upper().startswith("REASONING:"):
            reasoning_lines.append(stripped.split(":", 1)[1].strip())
            in_reasoning = True
        elif in_reasoning and stripped:
            reasoning_lines.append(stripped)

    return {"winner": winner, "reasoning": " ".join(reasoning_lines)}


def judge_pair(user_context: str, response_a: str, response_b: str) -> dict:
    """Ask Claude Opus to judge which response is better."""
    context_summary = truncate_context(user_context)

    user_prompt = f"""## Trajectory Context

{context_summary}

## PRM Response [A]

{response_a}

## PRM Response [B]

{response_b}

Which PRM response (A or B) would be more helpful for guiding the coding agent? \
Or is it a TIE? Remember to evaluate accuracy, specificity, completeness, and actionability."""

    response = litellm.completion(
        model="bedrock/us.anthropic.claude-opus-4-6-v1",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    text = response.choices[0].message.content or ""
    parsed = parse_judge_output(text)
    parsed["raw_judge_output"] = text
    return parsed


def save_results(existing: dict, output_path: Path):
    """Atomically write results to disk."""
    all_results = sorted(existing.values(), key=lambda x: x["idx"])
    tmp = output_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(output_path)


def main():
    parser = argparse.ArgumentParser(description="Claude Opus judge evaluation")
    parser.add_argument("--model-a", type=str, required=True,
                        help="First model label (e.g. qwen3-8b-base)")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Second model label (e.g. opus-distill-32k-lr5e6)")
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to training data JSONL (for trajectory context)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N samples (0 = all)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent judge requests (default: 16)")
    parser.add_argument("--save-every", type=int, default=50, help="Save every N completions (default: 50)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_path = eval_dir / f"judge_{args.model_a}_vs_{args.model_b}.jsonl"

    # Load both models' responses
    responses_a = {}
    path_a = eval_dir / args.model_a / "responses.jsonl"
    with open(path_a) as f:
        for line in f:
            item = json.loads(line)
            responses_a[item["idx"]] = item

    responses_b = {}
    path_b = eval_dir / args.model_b / "responses.jsonl"
    with open(path_b) as f:
        for line in f:
            item = json.loads(line)
            responses_b[item["idx"]] = item

    # Find shared indices
    shared_idxs = sorted(set(responses_a.keys()) & set(responses_b.keys()))
    print(f"Model A ({args.model_a}): {len(responses_a)} samples")
    print(f"Model B ({args.model_b}): {len(responses_b)} samples")
    print(f"Shared samples: {len(shared_idxs)}")

    if not shared_idxs:
        print("ERROR: No shared samples between models.")
        sys.exit(1)

    if args.limit > 0:
        shared_idxs = shared_idxs[:args.limit]
        print(f"Limited to {len(shared_idxs)} samples")

    # Load existing results for resumability
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                existing[item["idx"]] = item
        print(f"Loaded {len(existing)} existing judge results")

    to_judge_idxs = [idx for idx in shared_idxs if idx not in existing]
    print(f"Need to judge {len(to_judge_idxs)} samples")

    if to_judge_idxs:
        # Load training data for trajectory context
        print(f"Loading training data for context from {args.data_path}...")
        training_context = {}
        with open(args.data_path) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                training_context[i] = item["messages"][1]["content"]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        lock = threading.Lock()
        completed = 0
        errors = 0

        # Running win counts for progress bar
        wins = {args.model_a: 0, args.model_b: 0, "tie": 0, "parse_error": 0}
        # Count existing results too
        for r in existing.values():
            w = r.get("winner_model", "other")
            if w in wins:
                wins[w] += 1
            else:
                wins["other"] += 1

        def progress_postfix():
            valid = wins[args.model_a] + wins[args.model_b] + wins["tie"]
            if valid == 0:
                return {"A": 0, "B": 0, "tie": 0, "err": errors, "parse_err": wins["parse_error"]}
            return {
                args.model_a: f"{wins[args.model_a]}({100*wins[args.model_a]/valid:.0f}%)",
                args.model_b: f"{wins[args.model_b]}({100*wins[args.model_b]/valid:.0f}%)",
                "tie": f"{wins['tie']}({100*wins['tie']/valid:.0f}%)",
                "err": errors,
                "parse_err": wins["parse_error"],
            }

        summary_path = output_path.with_suffix(".summary.json")

        def save_summary():
            valid = wins[args.model_a] + wins[args.model_b] + wins["tie"]
            summary = {
                "model_a": args.model_a,
                "model_b": args.model_b,
                "total_judged": valid + wins["parse_error"],
                "total_valid": valid,
                "total_to_judge": len(shared_idxs),
                "wins_model_a": wins[args.model_a],
                "wins_model_b": wins[args.model_b],
                "ties": wins["tie"],
                "parse_errors": wins["parse_error"],
                "api_errors": errors,
            }
            if valid > 0:
                summary["pct_model_a"] = round(100 * wins[args.model_a] / valid, 1)
                summary["pct_model_b"] = round(100 * wins[args.model_b] / valid, 1)
                summary["pct_tie"] = round(100 * wins["tie"] / valid, 1)
            tmp = summary_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(summary, f, indent=2)
            tmp.rename(summary_path)

        pbar = tqdm(total=len(to_judge_idxs), desc="Judging", unit="sample")
        pbar.set_postfix(progress_postfix())

        def process_sample(idx):
            user_context = training_context.get(idx, "")
            resp_a = responses_a[idx]["response"]
            resp_b = responses_b[idx]["response"]

            # Randomize order to avoid position bias
            coin = random.random() < 0.5
            if coin:
                judge_a, judge_b = resp_a, resp_b
                label_a, label_b = args.model_a, args.model_b
            else:
                judge_a, judge_b = resp_b, resp_a
                label_a, label_b = args.model_b, args.model_a

            try:
                result = judge_pair(user_context, judge_a, judge_b)
            except Exception as e:
                return idx, None, str(e)

            if result["winner"] == "A":
                result["winner_model"] = label_a
            elif result["winner"] == "B":
                result["winner_model"] = label_b
            elif result["winner"] == "TIE":
                result["winner_model"] = "tie"
            else:
                result["winner_model"] = "parse_error"

            result["idx"] = idx
            result["order"] = {"A": label_a, "B": label_b}
            return idx, result, None

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, idx): idx for idx in to_judge_idxs}

            for future in as_completed(futures):
                idx, result, err = future.result()

                with lock:
                    if err:
                        errors += 1
                        pbar.set_postfix(progress_postfix())
                        pbar.update(1)
                        continue

                    existing[idx] = result
                    w = result.get("winner_model", "other")
                    if w in wins:
                        wins[w] += 1
                    else:
                        wins["other"] += 1
                    completed += 1
                    pbar.set_postfix(progress_postfix())
                    pbar.update(1)

                    if completed % args.save_every == 0:
                        save_results(existing, output_path)
                        save_summary()

        pbar.close()
        print(f"\nJudged {completed} samples ({errors} errors)")

    # Final save
    save_results(existing, output_path)

    # Save final summary
    total_all = sum(1 for r in existing.values())
    wins_final = {}
    for r in existing.values():
        w = r.get("winner_model", "other")
        wins_final[w] = wins_final.get(w, 0) + 1
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "total_judged": total_all,
        "wins_model_a": wins_final.get(args.model_a, 0),
        "wins_model_b": wins_final.get(args.model_b, 0),
        "ties": wins_final.get("tie", 0),
    }
    if total_all > 0:
        summary["pct_model_a"] = round(100 * wins_final.get(args.model_a, 0) / total_all, 1)
        summary["pct_model_b"] = round(100 * wins_final.get(args.model_b, 0) / total_all, 1)
        summary["pct_tie"] = round(100 * wins_final.get("tie", 0) / total_all, 1)
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary
    winners = {}
    for r in existing.values():
        w = r.get("winner_model", "other")
        winners[w] = winners.get(w, 0) + 1

    total = sum(winners.values())
    print(f"\n{'=' * 50}")
    print(f"JUDGE RESULTS ({total} samples)")
    print(f"{args.model_a} vs {args.model_b}")
    print(f"{'=' * 50}")
    if total:
        for label in [args.model_a, args.model_b, "tie"]:
            count = winners.get(label, 0)
            print(f"  {label:30s}  {count:4d}  ({100*count/total:.1f}%)")
        other = total - winners.get(args.model_a, 0) - winners.get(args.model_b, 0) - winners.get("tie", 0)
        if other:
            print(f"  {'(parse errors)':30s}  {other:4d}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
