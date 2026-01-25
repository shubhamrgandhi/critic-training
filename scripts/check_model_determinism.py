#!/usr/bin/env python3
"""Probe whether the configured model replies deterministically.

The script loads the model config (same schema mini-SWE-agent uses), asks the
model the exact same prompt multiple times, and reports how many distinct
outputs it saw in:
1) A single shared-model sequential run.
2) A fresh-model sequential run (re-instantiate for each call).
3) A fresh-model parallel run (one model instance per thread).

Usage:
    python scripts/check_model_determinism.py \
        --config mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml \
        --prompt "Repeat the word determinism." \
        --repeats 5 \
        --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "mini-swe-agent" / "src"))
from minisweagent.models import get_model  # noqa: E402


def load_model_loader(config_path: Path) -> Callable[[], object]:
    config = yaml.safe_load(config_path.read_text())
    model_config = config.get("model", {})

    def loader():
        return get_model(config=model_config)

    return loader


def ask(model, prompt: str) -> str:
    """Send a simple user-only message and return stripped content."""
    response = model.query([{"role": "user", "content": prompt}])
    return (response.get("content") or "").strip()


def run_sequential(model_factory: Callable[[], object], prompt: str, repeats: int, shared: bool) -> list[str]:
    outputs: list[str] = []
    model = model_factory() if shared else None
    for _ in range(repeats):
        m = model if shared else model_factory()
        outputs.append(ask(m, prompt))
    return outputs


def run_parallel(model_factory: Callable[[], object], prompt: str, repeats: int, workers: int) -> list[str]:
    outputs: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, model_factory(), prompt) for _ in range(repeats)]
        for fut in concurrent.futures.as_completed(futures):
            outputs.append(fut.result())
    return outputs


def summarize(label: str, outputs: Iterable[str]) -> None:
    counts = Counter(outputs)
    print(f"\n[{label}] {len(counts)} unique outputs across {sum(counts.values())} calls")
    for text, count in counts.most_common():
        # Keep it short in case the model emits long answers.
        preview = text.replace("\n", "\\n")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        print(f"  {count:3d}x | {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check backend determinism by repeating identical prompts.")
    parser.add_argument("--config", required=True, type=Path, help="Path to mini-SWE-agent YAML config.")
    parser.add_argument("--prompt", default="Say the exact word DETERMINISM.", help="Prompt to send.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of queries per test.")
    parser.add_argument("--workers", type=int, default=4, help="Workers for parallel test.")
    args = parser.parse_args()

    model_loader = load_model_loader(args.config)

    seq_shared = run_sequential(model_loader, args.prompt, args.repeats, shared=True)
    seq_fresh = run_sequential(model_loader, args.prompt, args.repeats, shared=False)
    par_fresh = run_parallel(model_loader, args.prompt, args.repeats, args.workers)

    summarize("Sequential (shared model instance)", seq_shared)
    summarize("Sequential (fresh model each call)", seq_fresh)
    summarize(f"Parallel {args.workers} workers (fresh model per thread)", par_fresh)


if __name__ == "__main__":
    main()
