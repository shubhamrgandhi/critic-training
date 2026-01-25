#!/usr/bin/env python
"""
Check determinism of an OpenAI-compatible chat model.

Example usage:

  python determinism_check.py \
    --model "mistralai/Devstral-Small-2507" \
    --api-base "http://localhost:8083/v1" \
    --n 5 \
    --prompt "Write a Python function to compute Fibonacci numbers with random words."

Or, using a prompt from file:

  python determinism_check.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --api-base "http://localhost:8085/v1" \
    --n 10 \
    --prompt @prompt.txt
"""

import argparse
import hashlib
from difflib import unified_diff

import litellm
import os
os.environ["LITELLM_LOG"] = "DEBUG"

def load_prompt(prompt_arg: str) -> str:
    """
    If prompt_arg starts with @ treat it as a path and read the file.
    Otherwise use it directly as the prompt string.
    """
    if prompt_arg.startswith("@"):
        path = prompt_arg[1:]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return prompt_arg


def call_model(
    model: str,
    api_base: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    seed: int | None,
):
    """
    One chat completion call with deterministic settings (as far as the backend allows).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "api_base": api_base,
        "api_key": api_key,
        "custom_llm_provider": "openai",  # <<< this is the key line
    }

    if seed is not None:
        kwargs["seed"] = seed

    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content or ""



def main():
    parser = argparse.ArgumentParser(description="Determinism checker for OpenAI-compatible chat models.")
    parser.add_argument("--model", required=True, help="Model name (e.g. 'Qwen/Qwen3-Coder-30B-A3B-Instruct').")
    parser.add_argument("--api-base", required=True, help="API base URL (e.g. 'http://localhost:8085/v1').")
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key passed to the server (many local servers ignore it but still require a header).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of repeated calls to run with the same prompt.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt text, or @file.txt to read the prompt from a file.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful coding assistant.",
        help="System prompt to use (keep fixed across runs).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to pass to the backend (if supported).",
    )

    args = parser.parse_args()

    user_prompt = load_prompt(args.prompt)

    outputs: list[str] = []

    print(
        f"\nRunning {args.n} calls with identical inputs:\n"
        f"  model        = {args.model}\n"
        f"  api_base     = {args.api_base}\n"
        f"  temperature  = 0.0\n"
        f"  top_p        = 1.0\n"
        f"  max_tokens   = {args.max_tokens}\n"
        f"  seed         = {args.seed}\n"
    )

    for i in range(args.n):
        text = call_model(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        outputs.append(text)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        print(f"Run {i}: length={len(text)} chars, hash={h}")

    print("\nSummary")
    print("=" * 60)

    if not outputs:
        print("No outputs collected.")
        return

    ref = outputs[0]
    ref_hash = hashlib.md5(ref.encode("utf-8")).hexdigest()
    print(f"Reference: run 0, hash={ref_hash}, length={len(ref)} chars")

    all_identical = True
    for i, out in enumerate(outputs[1:], start=1):
        identical = (out == ref)
        h = hashlib.md5(out.encode("utf-8")).hexdigest()
        print(f"Run {i}: identical_to_run0={identical}, hash={h}, length={len(out)}")

        if not identical and all_identical:
            all_identical = False
            print("\nFirst differing run:", i)
            print("Showing unified diff for first 200 characters:\n")
            diff = unified_diff(
                ref[:200].splitlines(),
                out[:200].splitlines(),
                fromfile="run0",
                tofile=f"run{i}",
                lineterm="",
            )
            for line in diff:
                print(line)

    if all_identical:
        print("\nAll runs are byte-identical. Any variance you see in the full agent is very likely coming from")
        print("the agent logic / environment (tool selection, repo state, test flakiness, etc.), not the model.")
    else:
        print("\nAt least one run differed from run 0.")
        print("This suggests non-determinism in the model server / decoding stack even with temperature=0.")


if __name__ == "__main__":
    main()
