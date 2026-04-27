#!/usr/bin/env python3
"""Step 1: Generate PRM responses from a model for evaluation.

Reads the training data JSONL, extracts system+user prompts (same prompts
the model was trained on), and generates responses via OpenAI-compatible
vLLM API. Each model's responses are saved to a separate directory under
eval_results/<model-label>/responses.jsonl.

The script is resumable — skips samples that already have responses.
Saves periodically to avoid losing progress on crash.

Usage:
    # Generate from finetuned model
    python eval_generate_responses.py \
        --model-name qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6 \
        --model-label opus-distill-32k-lr5e6 \
        --api-base http://babel-n5-28:8071/v1

    # Generate from base Qwen3-8B
    python eval_generate_responses.py \
        --model-name Qwen/Qwen3-8B \
        --model-label qwen3-8b-base \
        --api-base http://babel-m5-28:8071/v1

    # Limit to N samples for quick testing
    python eval_generate_responses.py ... --limit 50
"""

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "prm_sft_data_opus_distill_full_feedback_history_32k" / "prm_sft_train.jsonl"
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval_results"


def load_training_data(data_path: str) -> list[dict]:
    """Load training JSONL and return list of samples with prompts and ground truth."""
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            msgs = item["messages"]
            assert len(msgs) == 3 and msgs[0]["role"] == "system"
            samples.append({
                "idx": i,
                "system": msgs[0]["content"],
                "user": msgs[1]["content"],
                "ground_truth": msgs[2]["content"],
            })
    return samples


def generate_response(client: OpenAI, model_name: str, system: str, user: str,
                      disable_thinking: bool = False) -> dict:
    """Generate a single response from a vLLM-served model.

    Returns dict with 'response' and 'reasoning'.
    vLLM 0.15.x returns reasoning in message.reasoning_content when thinking is enabled.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    enable_thinking = not disable_thinking
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        n=1,
        extra_body=extra_body,
    )

    choice = response.choices[0].message
    raw = choice.content or ""

    # vLLM 0.15.x may put reasoning in a separate field
    reasoning = getattr(choice, "reasoning_content", None) or ""

    # Fallback: extract inline reasoning from content
    # Model may output <think>...</think>, or just ...reasoning...</think> (missing open tag)
    if not reasoning:
        think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        elif "</think>" in raw:
            # Missing <think> open tag — everything before </think> is reasoning
            parts = raw.split("</think>", 1)
            reasoning = parts[0].strip()

    # Strip all thinking/reasoning from the response content
    if "</think>" in raw:
        cleaned = raw.split("</think>", 1)[-1].strip()
    else:
        cleaned = strip_think_tags(raw)

    return {"response": cleaned, "reasoning": reasoning, "raw": raw}


def save_results(results: dict, output_path: Path):
    """Atomically write results to disk."""
    tmp = output_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for idx in sorted(results.keys()):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")
    tmp.rename(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate PRM responses for evaluation")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to training data JSONL")
    parser.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR),
                        help="Root eval results directory")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name as served by vLLM")
    parser.add_argument("--model-label", type=str, required=True,
                        help="Label for this model (used as subdirectory name, e.g. "
                             "qwen3-8b-base, opus-distill-32k-lr5e6)")
    parser.add_argument("--api-base", type=str, required=True,
                        help="vLLM API base URL (e.g. http://babel-n5-28:8071/v1)")
    parser.add_argument("--disable-thinking", action="store_true", default=False,
                        help="Disable thinking for Qwen3 models")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N samples (0 = all)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of concurrent requests (default: 32)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N completions (default: 25)")
    args = parser.parse_args()

    # Auto-detect thinking based on label if --disable-thinking not explicitly passed
    if not args.disable_thinking and "think" not in args.model_label:
        args.disable_thinking = True
    print(f"Thinking: {'enabled' if not args.disable_thinking else 'disabled'} (label={args.model_label})")

    # Output: eval_results/<model-label>/responses.jsonl
    output_dir = Path(args.eval_dir) / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "responses.jsonl"

    # Load training data
    print(f"Loading training data from {args.data_path}...")
    samples = load_training_data(args.data_path)
    print(f"Loaded {len(samples)} samples")

    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    samples_by_idx = {s["idx"]: s for s in samples}

    # Load existing results for resumability
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                existing[item["idx"]] = item
        print(f"Loaded {len(existing)} existing results from {output_path}")

    to_generate = [s for s in samples if s["idx"] not in existing]
    print(f"Need to generate {len(to_generate)} responses for model_label={args.model_label}")

    if not to_generate:
        print("Nothing to generate. Done.")
        return

    # Initialize client
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"Connected to {args.api_base}. Available models: {available}")
    except Exception as e:
        print(f"ERROR: Cannot connect to {args.api_base}: {e}")
        sys.exit(1)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    completed = 0
    errors = 0
    start_time = time.time()

    use_disable_thinking = args.disable_thinking

    def process_sample(sample):
        try:
            result = generate_response(
                client, args.model_name,
                sample["system"], sample["user"],
                disable_thinking=use_disable_thinking,
            )
            return sample["idx"], result, None
        except Exception as e:
            return sample["idx"], None, str(e)

    pbar = tqdm(total=len(to_generate), desc=f"Generating ({args.model_label})", unit="sample")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in to_generate}

        for future in as_completed(futures):
            idx, result, err = future.result()

            with lock:
                if err:
                    errors += 1
                    pbar.set_postfix(errors=errors)
                    pbar.update(1)
                    continue

                existing[idx] = {
                    "idx": idx,
                    "ground_truth": samples_by_idx[idx]["ground_truth"],
                    "response": result["response"],
                    "reasoning": result["reasoning"],
                    "raw": result["raw"],
                }

                completed += 1
                pbar.update(1)

                if completed % args.save_every == 0:
                    save_results(existing, output_path)

    pbar.close()

    save_results(existing, output_path)

    elapsed = time.time() - start_time
    print(f"\nDone. Generated {completed} responses in {elapsed:.1f}s ({errors} errors)")
    print(f"Results saved to {output_path}")
    print(f"Total samples: {len(existing)}")

    # Save metadata
    meta = {
        "model_name": args.model_name,
        "model_label": args.model_label,
        "api_base": args.api_base,
        "data_path": args.data_path,
        "n_samples": len(existing),
        "n_errors": errors,
        "disable_thinking": use_disable_thinking,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
