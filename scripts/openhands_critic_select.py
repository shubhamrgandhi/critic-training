#!/usr/bin/env python3
"""
OpenHands-style critic-based Best-of-N selection across multiple base runs.

Uses the rubric prompt from the OpenHands Critic paper (Wang et al., 2026,
arxiv:2603.03800) Section G.2 ("Segment WITHOUT user feedback") to score
trajectories from multiple runs of the same SWE-bench instance, then selects
the trajectory with the fewest detected issues (lowest issue count = best).

The system prompt, instruction message, and trajectory formatting are matched
to the OpenHands implementation (critic-rubrics/critic_rubrics/rubrics/
trajectory/{trajectory.py, converter.py}).  Since mini-swe-agent uses plain
bash code blocks rather than OpenAI tool_calls, the converter is adapted
accordingly: assistant messages with ```bash blocks are tagged inline, and
user messages containing <returncode> are prefixed as execution results.

Usage:
    python scripts/openhands_critic_select.py \
        --runs-dir results_singularity_max_150_steps_prefix \
        --run-names singularity_edit_obs_final_only_{0,1,2,3,4}_cwm \
        --output results_singularity_max_150_steps_prefix/critic_selected_cwm \
        --api-base http://localhost:8071/v1 \
        --model Qwen/Qwen3-8B \
        --workers 8
"""

import argparse
import json
import glob
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Tokenizer (loaded once, shared across threads — encode() is thread-safe)
# ---------------------------------------------------------------------------

_tokenizer: AutoTokenizer | None = None


def _get_tokenizer(model_name: str = "Qwen/Qwen3-8B") -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ---------------------------------------------------------------------------
# OpenHands Critic Rubric — EXACT system prompt from
# critic-rubrics/critic_rubrics/rubrics/trajectory/trajectory.py
# (Section G.2 — WITHOUT user feedback)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """You are an AI conversation annotator analyzing agent-environment interactions to identify failure patterns. You are NOT participating in the conversation; you are an external observer evaluating what went wrong.

========================
CONVERSATION STRUCTURE
========================
- Focus on the LAST AGENT MESSAGE.

========================
CONTEXT SOURCES
========================
Use all evidence: screenshots, code, logs, specs, file trees, error messages, prompts/system messages, and tool traces. Prefer short verbatim quotes (<=25 words) when supporting a claim.

========================
DETECTION FRAMEWORK
========================
Multiple issues can co-occur. For each issue:
1) Set the corresponding boolean to TRUE.
2) Provide a short, specific rationale quoting concrete evidence (agent actions, errors).

AGENT BEHAVIORAL ISSUES
- misunderstood_intention: Agent misunderstood the user's goal/intent.
  - Examples: User asked for a summary and agent produced a rewrite; user wanted high-level bullets but agent delivered full code.

- did_not_follow_instruction: Agent ignored or failed to comply with explicit instructions/system constraints.
  - Examples: User: 'Do NOT push to main.' Agent pushes to main; System says not to create pull request unless user asks for it and user didn't ask for it, agent creates pull request; user asked for bullet points only, agent gives long prose.

- insufficient_analysis: Didn't explore existing materials sufficiently (prior code/docs/examples) before acting.
  - Examples: User points to an existing function/file that is relavant OR already solves it; agent reinvents it.

- insufficient_clarification: Failed to ask necessary questions before acting when requirements were ambiguous.
  - Examples: Agent proceeds despite unclear acceptance criteria (e.g., locales, time zones, error thresholds) then is corrected later.

- improper_tool_use_or_setup: Misused tools/commands or had missing/incorrect dependencies/setup.
  - Examples: wrong command syntax, using inappropriate tools for the task

- loop_behavior: Repeats the same failed action 3+ times without strategy change.
  - Examples: repeat the same failed action 3+ times without changing approach).

- insufficient_testing: Skipped reasonable verification/tests for non-trivial or risky changes (note: trivial edits may be acceptable).
  - Examples: No run/validation for a new parser; no check that a migration applies cleanly; no sanity check of output.

- insufficient_debugging: Did not investigate or reduce failing behavior when needed to make progress.
  - Examples: Ignores stack trace; no isolation of failure; proceeds while errors persist.

- incomplete_implementation: Delivered unfinished or non-functioning work.
  - Examples: TODO/FIXME left; stub methods; code that cannot run.

- file_management_errors: Wrong paths, overwrites, misplaced/extra files (including unnecessary files).
  - Examples: Writes into wrong directory; overwrites config; creates unwanted artifacts.

- scope_creep: Implemented unrequested features without approval.
  - Examples: Adds a dashboard or endpoint not asked for.

- risky_actions_or_permission: Risky steps without user's explicit consent.
  - Examples: git push to main; deleting existing files in a repo (deleting files created by agent itself is fine); altering credentials.

- other_agent_issue: Any agent-side problem not covered above.

INFRASTRUCTURE (EXTERNAL vs AGENT-CAUSED)
- infrastructure_external_issue: Environment/platform limits outside agent control.
  - Examples: Provider outage; disk full on managed runner; missing enterprise API key; network failure not caused by agent.

- infrastructure_agent_caused_issue: Infrastructure fault introduced by the agent's prior actions.
  - Examples: Agent leaves a server running on port 8000; later start on 8000 fails; agent fills the disk with logs earlier, causing later writes to fail.

========================
QUALITY STANDARDS
========================
- Evidence Threshold: Mark TRUE only with specific evidence; prefer short quotes.
- Conservative Defaults: When uncertain, mark FALSE and briefly explain why.
- No speculation: Tie every flagged issue to observable behavior or quoted text."""

# ---------------------------------------------------------------------------
# OpenHands Critic Rubric — EXACT instruction message from
# critic-rubrics/critic_rubrics/rubrics/trajectory/trajectory.py
#
# Adapted: the original says "Fill the annotate_conversation function" which
# relies on tool_call / function-calling output.  Since we use Qwen3-8B as a
# generative model, we replace that single line with a JSON output instruction
# while keeping every other sentence identical.
# ---------------------------------------------------------------------------

ANNOTATION_INSTRUCTION_MESSAGE = """=== END OF CONVERSATION TO ANALYZE ===

Analyze the conversation above and output your annotations as a JSON object.

Goal
- Set only the booleans that clearly apply.

What to record
1) Agent behavioral issues (select all that apply)
   - misunderstood_intention, did_not_follow_instruction, insufficient_analysis, insufficient_clarification,
     improper_tool_use_or_setup, loop_behavior, insufficient_testing, insufficient_debugging,
     incomplete_implementation, file_management_errors, scope_creep, risky_actions_or_permission,
     other_agent_issue.
   - Rationale: cite code/commands/errors or a short quote and explain in one sentence.

2) Infrastructure
   - infrastructure_external_issue_detected for environment/platform limits beyond agent control.
   - infrastructure_agent_caused_issue_detected for faults introduced by the agent's prior actions (e.g., orphaned server on port 8000).
   - Rationale: include the error/status line or brief description.


Evidence & quality
- Prefer concrete, minimal quotes; avoid speculation. If evidence is insufficient, leave the flag false.

Quick disambiguation (common splits)
- insufficient_analysis vs insufficient_clarification: didn't look for existing work vs didn't ask when requirements were ambiguous.
- insufficient_testing vs insufficient_debugging: skipped reasonable verification vs didn't investigate a failing state enough to make progress.

Output ONLY a JSON object with the following schema (no other text):

```json
{
  "misunderstood_intention_detected": <true/false>,
  "misunderstood_intention_rationale": "<one sentence or empty>",
  "did_not_follow_instruction_detected": <true/false>,
  "did_not_follow_instruction_rationale": "<one sentence or empty>",
  "insufficient_analysis_detected": <true/false>,
  "insufficient_analysis_rationale": "<one sentence or empty>",
  "insufficient_clarification_detected": <true/false>,
  "insufficient_clarification_rationale": "<one sentence or empty>",
  "improper_tool_use_or_setup_detected": <true/false>,
  "improper_tool_use_or_setup_rationale": "<one sentence or empty>",
  "loop_behavior_detected": <true/false>,
  "loop_behavior_rationale": "<one sentence or empty>",
  "insufficient_testing_detected": <true/false>,
  "insufficient_testing_rationale": "<one sentence or empty>",
  "insufficient_debugging_detected": <true/false>,
  "insufficient_debugging_rationale": "<one sentence or empty>",
  "incomplete_implementation_detected": <true/false>,
  "incomplete_implementation_rationale": "<one sentence or empty>",
  "file_management_errors_detected": <true/false>,
  "file_management_errors_rationale": "<one sentence or empty>",
  "scope_creep_detected": <true/false>,
  "scope_creep_rationale": "<one sentence or empty>",
  "risky_actions_or_permission_detected": <true/false>,
  "risky_actions_or_permission_rationale": "<one sentence or empty>",
  "other_agent_issue_detected": <true/false>,
  "other_agent_issue_rationale": "<one sentence or empty>",
  "infrastructure_external_issue_detected": <true/false>,
  "infrastructure_external_issue_rationale": "<one sentence or empty>",
  "infrastructure_agent_caused_issue_detected": <true/false>,
  "infrastructure_agent_caused_issue_rationale": "<one sentence or empty>"
}
```"""

RUBRIC_FEATURES = [
    "misunderstood_intention",
    "did_not_follow_instruction",
    "insufficient_analysis",
    "insufficient_clarification",
    "improper_tool_use_or_setup",
    "loop_behavior",
    "insufficient_testing",
    "insufficient_debugging",
    "incomplete_implementation",
    "file_management_errors",
    "scope_creep",
    "risky_actions_or_permission",
    "other_agent_issue",
    "infrastructure_external_issue",
    "infrastructure_agent_caused_issue",
]


# ---------------------------------------------------------------------------
# Trajectory formatting — matches OpenHands converter (converter.py) adapted
# for mini-swe-agent message format (bash code blocks, not tool_calls).
#
# Structure mirrors transform_for_annotator():
#   1. Critic system message (CRITIC_SYSTEM_PROMPT)
#   2. First user message wrapped with original system message + first task
#   3. Alternating assistant/user messages
#   4. Last assistant tagged with << BEGIN/END LAST AGENT MESSAGE >>
#   5. Annotation instruction appended at the end
#
# Left truncation at turn boundaries when the trajectory exceeds max_tokens.
# ---------------------------------------------------------------------------

def build_critic_messages(
    messages: list[dict],
    max_tokens: int = 120000,
) -> tuple[list[dict], bool]:
    """Build the multi-turn critic prompt from a mini-swe-agent trajectory.

    Returns (critic_messages, was_truncated) where critic_messages is a list
    of {"role": ..., "content": ...} dicts ready for litellm.completion().
    """
    filtered = [m for m in messages if not m.get("_supervisor")]
    if not filtered:
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": "(empty trajectory)\n" + ANNOTATION_INSTRUCTION_MESSAGE},
        ], False

    # Separate system message from the conversation
    system_msg = None
    conversation = []
    for m in filtered:
        if m.get("role") == "system" and system_msg is None:
            system_msg = m.get("content", "") or ""
        else:
            conversation.append(m)

    if not conversation:
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": "(empty trajectory)\n" + ANNOTATION_INSTRUCTION_MESSAGE},
        ], False

    # Find last assistant index (for << LAST AGENT MESSAGE >> tagging)
    last_asst_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "assistant":
            last_asst_idx = i
            break

    # Build formatted conversation turns (before truncation)
    # First user message gets the original system message prepended
    formatted_turns = []  # list of {"role": str, "content": str}
    for i, m in enumerate(conversation):
        role = m.get("role", "user")
        content = m.get("content", "") or ""

        if i == 0:
            # First message: wrap with original system message (matches converter.py)
            first_block = (
                "<< BEGIN ORIGINAL SYSTEM MESSAGE>>\n"
                f"{system_msg or '(no system message)'}\n"
                "<< END ORIGINAL SYSTEM MESSAGE >>\n\n"
                "<< BEGIN FIRST USER MESSAGE >>\n"
                f"{content}\n"
                "<< END FIRST USER MESSAGE >>"
            )
            formatted_turns.append({"role": "user", "content": first_block})

        elif role == "assistant":
            if i == last_asst_idx:
                tagged = (
                    "<< BEGIN LAST AGENT MESSAGE >>\n"
                    f"{content}\n"
                    "<< END LAST AGENT MESSAGE >>"
                )
                formatted_turns.append({"role": "assistant", "content": tagged})
            else:
                formatted_turns.append({"role": "assistant", "content": content})

        elif role == "user":
            # User messages containing <returncode> are execution results
            if "<returncode>" in content:
                formatted_turns.append({
                    "role": "user",
                    "content": f"EXECUTION RESULT of [bash]:\n{content}",
                })
            else:
                formatted_turns.append({"role": "user", "content": content})

    # Compute token costs per turn for truncation
    separator_overhead = 4  # typical chat template overhead per message
    turn_tokens = [_count_tokens(t["content"]) + separator_overhead for t in formatted_turns]

    # Fixed overhead: critic system prompt + annotation instruction
    system_tokens = _count_tokens(CRITIC_SYSTEM_PROMPT) + separator_overhead
    instruction_tokens = _count_tokens(ANNOTATION_INSTRUCTION_MESSAGE) + separator_overhead
    fixed_overhead = system_tokens + instruction_tokens
    budget = max_tokens - fixed_overhead

    total_turn_tokens = sum(turn_tokens)
    was_truncated = total_turn_tokens > budget

    if was_truncated:
        # Left truncation at turn boundaries: drop oldest turns first
        truncation_marker_content = "... (earlier context truncated) ..."
        marker_tokens = _count_tokens(truncation_marker_content) + separator_overhead

        available = budget - marker_tokens
        kept_indices = []
        tokens_used = 0
        for i in range(len(formatted_turns) - 1, -1, -1):
            if tokens_used + turn_tokens[i] > available:
                break
            kept_indices.append(i)
            tokens_used += turn_tokens[i]
        kept_indices.reverse()

        truncated_turns = [{"role": "user", "content": truncation_marker_content}]
        for i in kept_indices:
            truncated_turns.append(formatted_turns[i])
        formatted_turns = truncated_turns

    # Merge consecutive same-role messages (required by some chat templates)
    merged_turns = []
    for turn in formatted_turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            merged_turns[-1]["content"] += "\n\n" + turn["content"]
        else:
            merged_turns.append(dict(turn))

    # Append annotation instruction to the last message
    if merged_turns:
        last = merged_turns[-1]
        if last["role"] == "user":
            last["content"] += "\n\n" + ANNOTATION_INSTRUCTION_MESSAGE
        else:
            merged_turns.append({"role": "user", "content": ANNOTATION_INSTRUCTION_MESSAGE})
    else:
        merged_turns.append({"role": "user", "content": ANNOTATION_INSTRUCTION_MESSAGE})

    # Prepend critic system message
    critic_messages = [{"role": "system", "content": CRITIC_SYSTEM_PROMPT}] + merged_turns

    return critic_messages, was_truncated


# ---------------------------------------------------------------------------
# Critic scoring
# ---------------------------------------------------------------------------

def parse_critic_response(response_text: str) -> dict:
    """Extract the JSON rubric from the critic's response."""
    # Try to find JSON block in markdown
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try direct JSON parse
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find any JSON object
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def count_issues(rubric: dict) -> int:
    """Count the number of detected issues. Lower is better."""
    count = 0
    for feature in RUBRIC_FEATURES:
        # Check both "X_detected" (OpenHands format) and "X" (plain format)
        for key in (f"{feature}_detected", feature):
            val = rubric.get(key)
            if val is True or (isinstance(val, str) and val.lower() == "true"):
                count += 1
                break
    return count


def extract_rationales(rubric: dict) -> dict:
    """Extract rationale strings for detected issues."""
    rationales = {}
    for feature in RUBRIC_FEATURES:
        for det_key in (f"{feature}_detected", feature):
            val = rubric.get(det_key)
            if val is True or (isinstance(val, str) and val.lower() == "true"):
                rat_key = f"{feature}_rationale"
                rationales[feature] = rubric.get(rat_key, "")
                break
    return rationales


def score_trajectory(
    instance_id: str,
    run_name: str,
    traj_path: str,
    model: str,
    api_base: str,
    max_tokens: int = 120000,
    max_retries: int = 3,
) -> dict:
    """Score a single trajectory using the critic model."""
    with open(traj_path) as f:
        traj = json.load(f)

    messages = traj.get("messages", [])
    exit_status = traj.get("info", {}).get("exit_status", "Unknown")
    submission = traj.get("info", {}).get("submission", "")

    n_messages = len([m for m in messages if not m.get("_supervisor")])
    critic_messages, was_truncated = build_critic_messages(messages, max_tokens=max_tokens)

    # Count tokens in the full prompt
    formatted_tokens = sum(_count_tokens(m["content"]) for m in critic_messages)

    rubric = {}
    raw_response = ""
    error = None

    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=critic_messages,
                api_base=api_base,
                temperature=0.0,
                max_tokens=2048,
                custom_llm_provider="openai",
                drop_params=True,
            )
            raw_response = response.choices[0].message.content or ""
            # Strip think tags if present
            raw_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
            rubric = parse_critic_response(raw_response)
            if rubric:
                break
        except Exception as e:
            error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    issue_count = count_issues(rubric)
    rationales = extract_rationales(rubric)

    return {
        "instance_id": instance_id,
        "run_name": run_name,
        "traj_path": traj_path,
        "exit_status": exit_status,
        "submission": submission,
        "rubric": rubric,
        "issue_count": issue_count,
        "rationales": rationales,
        "n_messages": n_messages,
        "formatted_tokens": formatted_tokens,
        "was_truncated": was_truncated,
        "raw_response": raw_response,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_trajectories(runs_dir: str, run_names: list[str]) -> dict[str, list[dict]]:
    """Find all trajectories grouped by instance_id."""
    instances: dict[str, list[dict]] = {}
    for run_name in run_names:
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            print(f"WARNING: Run directory not found: {run_dir}", file=sys.stderr)
            continue
        traj_files = glob.glob(os.path.join(run_dir, "*", "*.traj.json"))
        for traj_path in traj_files:
            instance_id = os.path.basename(os.path.dirname(traj_path))
            if instance_id not in instances:
                instances[instance_id] = []
            instances[instance_id].append({
                "run_name": run_name,
                "traj_path": traj_path,
            })
    return instances


def copy_trajectory(src_run_dir: str, instance_id: str, dest_dir: str):
    """Copy the full instance subdirectory from the source run to dest."""
    src = os.path.join(src_run_dir, instance_id)
    dst = os.path.join(dest_dir, instance_id)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="OpenHands-style critic Best-of-N selection across base runs."
    )
    parser.add_argument(
        "--runs-dir", required=True,
        help="Parent directory containing all run directories",
    )
    parser.add_argument(
        "--run-names", required=True, nargs="+",
        help="Names of run directories to evaluate",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for selected trajectories, preds.json, and scores",
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8071/v1",
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="Model name for litellm",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--min-runs", type=int, default=2,
        help="Minimum number of runs an instance must appear in to be included",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=120000,
        help="Max tokens for trajectory in critic prompt (left truncation at turn boundaries). "
             "Default 120K leaves headroom for critic prompt + generation within 128K context.",
    )
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3-8B",
        help="HuggingFace tokenizer name for token counting",
    )
    args = parser.parse_args()

    # Initialize tokenizer
    print(f"Loading tokenizer {args.tokenizer}...")
    _get_tokenizer(args.tokenizer)

    # Find all trajectories
    print(f"Scanning {len(args.run_names)} runs in {args.runs_dir}...")
    instances = find_trajectories(args.runs_dir, args.run_names)

    # Filter by min_runs
    filtered = {
        iid: trajs for iid, trajs in instances.items()
        if len(trajs) >= args.min_runs
    }
    total_trajs = sum(len(v) for v in filtered.values())
    print(f"Found {len(filtered)} instances with >= {args.min_runs} runs "
          f"({total_trajs} total trajectories to score)")

    if not filtered:
        print("No instances to process. Check --runs-dir and --run-names.")
        sys.exit(1)

    # Build work items — only score trajectories that submitted a patch
    work_items = []
    skipped_no_submission = 0
    for instance_id, trajs in sorted(filtered.items()):
        for t in trajs:
            # Quick check: only score if trajectory has a submission
            try:
                with open(t["traj_path"]) as f:
                    traj_info = json.load(f).get("info", {})
                if traj_info.get("exit_status") != "Submitted" or not traj_info.get("submission"):
                    skipped_no_submission += 1
                    continue
            except Exception:
                skipped_no_submission += 1
                continue
            work_items.append((instance_id, t["run_name"], t["traj_path"]))
    print(f"Skipped {skipped_no_submission} non-submitted trajectories (not scored)")

    # Resume logic: load previously scored results from cache JSONL
    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, ".critic_cache.jsonl")
    cached_results: dict[tuple[str, str], dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = (entry["instance_id"], entry["run_name"])
                    cached_results[key] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Loaded {len(cached_results)} cached results from previous run")

    # Split into cached vs new work
    results: list[dict] = []
    new_work_items = []
    for instance_id, run_name, traj_path in work_items:
        key = (instance_id, run_name)
        if key in cached_results:
            entry = cached_results[key]
            # Restore submission from trajectory (stripped from cache to save space)
            if "submission" not in entry or not entry["submission"]:
                try:
                    with open(traj_path) as f:
                        traj = json.load(f)
                    entry["submission"] = traj.get("info", {}).get("submission", "")
                except Exception:
                    entry["submission"] = ""
            results.append(entry)
        else:
            new_work_items.append((instance_id, run_name, traj_path))

    if not new_work_items:
        print(f"All {len(work_items)} trajectories already scored (cached). Skipping inference.")
    else:
        print(f"Scoring {len(new_work_items)} new trajectories with {args.workers} workers "
              f"({len(results)} cached)...")

        cache_file = open(cache_path, "a")
        cache_lock = __import__("threading").Lock()

        def _write_cache(entry: dict):
            serializable = {k: v for k, v in entry.items() if k != "submission"}
            with cache_lock:
                cache_file.write(json.dumps(serializable, default=str) + "\n")
                cache_file.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    score_trajectory,
                    instance_id, run_name, traj_path,
                    args.model, args.api_base, args.max_tokens,
                ): (instance_id, run_name)
                for instance_id, run_name, traj_path in new_work_items
            }

            with tqdm(total=len(new_work_items), desc="Scoring", unit="traj") as pbar:
                for future in as_completed(futures):
                    instance_id, run_name = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        _write_cache(result)
                        pbar.set_postfix(last=f"{instance_id[:30]}={result['issue_count']}issues")
                    except Exception as e:
                        error_result = {
                            "instance_id": instance_id,
                            "run_name": run_name,
                            "issue_count": 999,
                            "error": str(e),
                        }
                        results.append(error_result)
                        _write_cache(error_result)
                        pbar.set_postfix(last=f"{instance_id[:30]}=ERROR")
                    pbar.update(1)

        cache_file.close()

    # Group results by instance and select best
    print("\nSelecting best trajectory per instance...")
    by_instance: dict[str, list[dict]] = {}
    for r in results:
        iid = r["instance_id"]
        if iid not in by_instance:
            by_instance[iid] = []
        by_instance[iid].append(r)

    # Selection strategy:
    # 1. Only consider trajectories that submitted a patch (exit_status == "Submitted")
    # 2. Among submitted, fewest critic issues wins
    # 3. If no submitted trajectories, fall back to fewest issues overall
    preds = {}
    selections = []
    for instance_id, candidates in sorted(by_instance.items()):
        def sort_key(c):
            is_submitted = 0 if c.get("exit_status") == "Submitted" else 1
            return (is_submitted, c["issue_count"])

        candidates.sort(key=sort_key)
        best = candidates[0]

        submission = best.get("submission", "")
        if submission:
            preds[instance_id] = {
                "model_name_or_path": "critic_selected_cwm",
                "instance_id": instance_id,
                "model_patch": submission,
            }

        selections.append({
            "instance_id": instance_id,
            "selected_run": best.get("run_name", ""),
            "issue_count": best["issue_count"],
            "exit_status": best.get("exit_status", ""),
            "n_candidates": len(candidates),
            "was_truncated": best.get("was_truncated", False),
            "all_scores": [
                {
                    "run": c.get("run_name", ""),
                    "issues": c["issue_count"],
                    "exit_status": c.get("exit_status", ""),
                    "n_messages": c.get("n_messages", 0),
                    "formatted_tokens": c.get("formatted_tokens", 0),
                    "was_truncated": c.get("was_truncated", False),
                    "rubric": c.get("rubric", {}),
                    "rationales": c.get("rationales", {}),
                }
                for c in candidates
            ],
        })

    # Write outputs
    os.makedirs(args.output, exist_ok=True)

    # 1. preds.json — same format as base runs
    preds_path = os.path.join(args.output, "preds.json")
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"\nWrote {len(preds)} predictions to {preds_path}")

    # 2. Copy selected trajectory subdirectories
    print("Copying selected trajectory directories...")
    for sel in selections:
        instance_id = sel["instance_id"]
        selected_run = sel["selected_run"]
        src_run_dir = os.path.join(args.runs_dir, selected_run)
        copy_trajectory(src_run_dir, instance_id, args.output)
    print(f"Copied {len(selections)} trajectory directories to {args.output}")

    # 3. critic_scores.json — per-instance selection details with all rubrics
    scores_path = os.path.join(args.output, "critic_scores.json")
    with open(scores_path, "w") as f:
        json.dump(selections, f, indent=2)
    print(f"Wrote {len(selections)} selection details to {scores_path}")

    # 4. all_critic_results.json — full scoring details for every trajectory
    all_results_path = os.path.join(args.output, "all_critic_results.json")
    results_for_save = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "submission"}
        results_for_save.append(entry)
    with open(all_results_path, "w") as f:
        json.dump(results_for_save, f, indent=2, default=str)
    print(f"Wrote {len(results_for_save)} individual scores to {all_results_path}")

    # Summary statistics
    issue_counts = [s["issue_count"] for s in selections]
    submitted_count = sum(1 for s in selections if s["exit_status"] == "Submitted")
    truncated_count = sum(1 for r in results if r.get("was_truncated", False))
    error_count = sum(1 for r in results if r.get("error"))
    print(f"\n=== Summary ===")
    print(f"Instances selected: {len(selections)}")
    print(f"Submitted: {submitted_count}/{len(selections)}")
    print(f"Issue counts (selected): min={min(issue_counts)}, max={max(issue_counts)}, "
          f"mean={sum(issue_counts)/len(issue_counts):.1f}")
    print(f"Truncated trajectories: {truncated_count}/{len(results)}")
    print(f"Scoring errors: {error_count}/{len(results)}")

    # Per-run breakdown
    run_selected_counts: dict[str, int] = {}
    for s in selections:
        run = s["selected_run"]
        run_selected_counts[run] = run_selected_counts.get(run, 0) + 1
    print(f"\nSelected from each run:")
    for run, count in sorted(run_selected_counts.items()):
        print(f"  {run}: {count}")

    # Issue distribution across all scored trajectories
    all_issue_counts = [r["issue_count"] for r in results if r["issue_count"] < 999]
    if all_issue_counts:
        print(f"\nIssue distribution (all {len(all_issue_counts)} trajectories):")
        from collections import Counter
        dist = Counter(all_issue_counts)
        for count in sorted(dist.keys()):
            print(f"  {count} issues: {dist[count]} trajectories")


if __name__ == "__main__":
    main()
