#!/usr/bin/env python3
"""Probe backend determinism without importing mini-swe-agent.

Reads a mini-swe-agent-style config YAML, loads the model settings, and calls
the OpenAI-compatible endpoint directly via litellm. Prints how many distinct
outputs are seen when sending the exact same prompt repeatedly in:
  1) Sequential calls (single thread)
  2) Parallel calls (ThreadPool)

If you see more than 1 unique output, the backend is non-deterministic under
that access pattern and config (e.g., seed ignored, speculative decoding,
parallel scheduler).

Examples:
    # Small prompt
    python scripts/check_model_determinism_standalone.py --repeats 5 --workers 4

    # Large prompt read from file and repeated to simulate mini-SWE-agent context
    python scripts/check_model_determinism_standalone.py \\
      --prompt-file /path/to/long_prompt.txt \\
      --repeat-chunks 3 \\
      --repeats 3 \\
      --workers 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import litellm
except ModuleNotFoundError:  # pragma: no cover - defensive UX
    print("litellm is required. Try: pip install 'litellm>=1.43'", file=sys.stderr)
    sys.exit(1)

import yaml


def load_model_params(config_path: Path | None) -> tuple[str, dict]:
    """Load model_name and kwargs from a mini-swe-agent config YAML."""
    if config_path is None:
        return "", {}
    cfg = yaml.safe_load(config_path.read_text()) or {}
    model_cfg = cfg.get("model", {}) or {}
    registry_path = model_cfg.get("litellm_model_registry")
    if registry_path and Path(registry_path).is_file():
        litellm.utils.register_model(json.loads(Path(registry_path).read_text()))
    model_name = model_cfg.get("model_name") or ""
    kwargs = model_cfg.get("model_kwargs", {}) or {}
    return model_name, kwargs


def call_model(model_name: str, kwargs: dict, prompt: str) -> str:
    """Call the OpenAI-compatible endpoint and return stripped content."""
    resp = litellm.completion(model=model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
    return (resp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]


def run_sequential(model_name: str, kwargs: dict, prompt: str, repeats: int) -> list[str]:
    return [call_model(model_name, kwargs, prompt) for _ in range(repeats)]


def run_parallel(model_name: str, kwargs: dict, prompt: str, repeats: int, workers: int) -> list[str]:
    outputs: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call_model, model_name, kwargs, prompt) for _ in range(repeats)]
        for fut in concurrent.futures.as_completed(futures):
            outputs.append(fut.result())
    return outputs


def summarize(label: str, outputs: Iterable[str]) -> None:
    counts = Counter(outputs)
    print(f"\n[{label}] {len(counts)} unique outputs across {sum(counts.values())} calls")
    for text, count in counts.most_common():
        preview = text.replace("\n", "\\n")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        print(f"  {count:3d}x | {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check backend determinism without mini-swe-agent imports.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml"),
        help="Path to mini-swe-agent YAML config (default: base Qwen config). Set to '' to skip.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Override model name.")
    parser.add_argument("--api-base", default="http://localhost:8085/v1", help="Override api_base sent to OpenAI compat.")
    parser.add_argument("--prompt", default="Say the exact word DETERMINISM.", help="Prompt to send.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional path to a file containing the prompt (e.g., a long mini-SWE-agent task).",
    )
    parser.add_argument(
        "--repeat-chunks",
        type=int,
        default=1,
        help="Repeat the chosen prompt this many times to simulate long context.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of queries per test.")
    parser.add_argument("--workers", type=int, default=4, help="ThreadPool workers for parallel test.")
    args = parser.parse_args()

    config_path: Path | None = args.config if str(args.config) else None
    model_name, kwargs = load_model_params(config_path)

    if not model_name:
        model_name = args.model_name
    kwargs.setdefault("api_base", args.api_base)
    kwargs.setdefault("custom_llm_provider", "openai")
    kwargs.setdefault("temperature", 0.0)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("top_k", -1)
    kwargs.setdefault("n", 1)
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("max_tokens", 2048)
    kwargs.setdefault("drop_params", True)

    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = args.prompt_file.read_text()
    prompt_text = prompt_text * max(args.repeat_chunks, 1)

    print(f"Model: {model_name}")
    print(f"Params: {kwargs}")
    print(f"Prompt length (chars): {len(prompt_text)}")

    seq = run_sequential(model_name, kwargs, prompt_text, args.repeats)
    par = run_parallel(model_name, kwargs, prompt_text, args.repeats, args.workers)

    summarize("Sequential (single thread)", seq)
    summarize(f"Parallel ({args.workers} workers)", par)


if __name__ == "__main__":
    main()
