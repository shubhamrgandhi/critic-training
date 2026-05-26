#!/usr/bin/env python3
"""Prepare SFT training data for distilling Claude Opus PRM into Qwen3-8B.

Reads trajectory files from a PRM-enhanced run and reconstructs the exact
inputs the PRM (Claude Opus) received at each invocation, paired with its
response.  Outputs data in LlamaFactory sharegpt format (JSONL with messages).

Two output formats (controlled by --format):

  flattened  — 3-message format: system + single user message (with
               [STEP-ENVIRONMENT]/[STEP-AGENT] tags) + assistant.
               Standard SFT loss on the single assistant turn.

  multiturn  — Real multi-turn chat: system, then alternating user/assistant
               messages matching the actual trajectory, ending with the PRM
               response.  Requires mask_history=true in LlamaFactory so loss
               is computed only on the final assistant turn (PRM response),
               not on the agent's assistant turns.
"""

import argparse
import json
import os
import re
from tqdm import tqdm
import sys
from pathlib import Path


# Tag constants for flattened format — matches the original training/inference
# format used in previous experiments.
TAG_ENVIRONMENT = "[USER/ENVIRONMENT]"
TAG_AGENT = "[AGENT RESPONSE]"
TAG_FEEDBACK_PREFIX = "[PREVIOUS_FEEDBACK"  # followed by " #N]:"


def extract_prm_system_prompt(traj: dict) -> str:
    """Extract the PRM system prompt from the trajectory config."""
    config = traj.get("info", {}).get("config", {})
    agent_config = config.get("agent", {})
    prompt = agent_config.get("prm_template", "")
    if not prompt:
        raise ValueError("No prm_template found in trajectory config")
    return prompt


def build_after_step_map(traj: dict) -> dict[int, int]:
    """Build a map from invocation number (1-indexed) to after_step.

    Uses the feedback log first (authoritative), falls back to supervisor
    message _prm_after_step metadata.
    """
    step_map: dict[int, int] = {}

    fl = traj.get("info", {}).get("prm_stats", {}).get("prm_feedback_log", [])
    for entry in fl:
        inv = entry.get("invocation", 0)
        step = entry.get("after_step")
        if inv and step is not None:
            step_map[inv] = step

    if not step_map:
        messages = traj.get("messages", [])
        inv_counter = 0
        for msg in messages:
            if msg.get("_supervisor"):
                inv_counter += 1
                step = msg.get("_prm_after_step")
                if step is not None:
                    step_map[inv_counter] = step

    return step_map


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def is_valid_prm_feedback(text: str) -> bool:
    """Check if text looks like actual PRM feedback rather than an agent response.

    PRM feedback follows a structured format starting with error category analysis.
    Agent responses typically contain bash/python code blocks or conversational text.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # PRM feedback should contain these structural markers
    prm_markers = ["DETECTED:", "SPECIFICATION ERRORS", "REASONING ERRORS",
                    "COORDINATION ERRORS", "TASK_STATUS:", "OVERALL_GUIDANCE:"]
    has_prm_markers = sum(1 for m in prm_markers if m in stripped)
    if has_prm_markers >= 2:
        return True
    # Agent responses typically start with conversational text or code
    first_line = stripped.split("\n")[0].strip()
    agent_markers = ["```", "Let me", "I'll", "I need", "Now let", "Okay",
                     "cd ", "cat ", "find ", "grep ", "python ", "pip "]
    if any(first_line.startswith(m) for m in agent_markers):
        return False
    return has_prm_markers >= 1


def extract_current_feedback(message_content: str) -> str:
    """Extract just the current feedback portion from a supervisor message.

    Returns empty string if the extracted content doesn't look like PRM feedback
    (e.g. if an agent response was erroneously placed in the Current Feedback section).
    """
    if "## Current Feedback" in message_content:
        idx = message_content.index("## Current Feedback")
        after_header = message_content[idx:]
        lines = after_header.split("\n", 1)
        feedback = lines[1].strip() if len(lines) > 1 else ""
        if feedback and not is_valid_prm_feedback(feedback):
            return ""  # Skip — this is an agent response, not PRM feedback
        return feedback
    feedback = message_content.replace("SUPERVISOR FEEDBACK:", "").strip()
    if feedback and not is_valid_prm_feedback(feedback):
        return ""
    return feedback


def extract_text_content(content) -> str:
    """Extract plain text from message content which may be a string or list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", item.get("value", str(item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content else ""


def clean_message_for_training(msg: dict) -> dict:
    """Strip internal metadata keys from a message, keeping only role and string content."""
    return {"role": msg["role"], "content": extract_text_content(msg.get("content", ""))}


def find_supervisor_indices(messages: list[dict]) -> list[int]:
    """Return indices of all supervisor messages in the trajectory."""
    return [i for i, m in enumerate(messages) if m.get("_supervisor", False)]


def reconstruct_prm_samples(traj: dict, prm_system_prompt: str) -> list[dict]:
    """Given a trajectory dict, reconstruct each PRM invocation as a training sample.

    Each sample is a list of messages (sharegpt format) ending with the PRM's
    assistant response.

    Returns list of {"messages": [...], "instance_id": str, "invocation": int}
    """
    messages = traj["messages"]
    instance_id = traj.get("instance_id", "unknown")
    sup_indices = find_supervisor_indices(messages)

    if not sup_indices:
        return []

    after_step_map = build_after_step_map(traj)

    samples = []
    previous_feedbacks: list[dict] = []

    for inv_num, sup_idx in enumerate(sup_indices):
        sup_msg = messages[sup_idx]
        inv_number = inv_num + 1  # 1-indexed

        # Extract the PRM's actual response.  Newer runs store raw feedback
        # as the supervisor message content; older runs wrap it in
        # "## Current Feedback".
        content = extract_text_content(sup_msg.get("content", ""))
        if "## Current Feedback" in content:
            prm_response = extract_current_feedback(content)
        else:
            prm_response = strip_think_tags(content)
            if prm_response and not is_valid_prm_feedback(prm_response):
                prm_response = ""

        if not prm_response.strip():
            continue

        # --- Reconstruct PRM input messages ---
        # Match the exact format used at inference in _invoke_prm_inner():
        #   system: PRM system prompt
        #   user:   single flattened message with tagged blocks

        prm_messages = [{"role": "system", "content": prm_system_prompt}]

        # Task context (from instance_template message, index 1)
        task_context = extract_text_content(messages[1].get("content", ""))
        prm_messages.append({
            "role": "user",
            "content": task_context,
        })

        # Agent conversation up to (but not including) this supervisor message.
        # Skip system (idx 0), instance template (idx 1), and all supervisor msgs.
        for msg in messages[2:sup_idx]:
            if msg.get("_supervisor"):
                continue
            prm_messages.append(clean_message_for_training(msg))

        # Cap feedback history to most recent MAX entries, matching inference
        # (default_prm.py MAX_PRM_FEEDBACK_HISTORY = 5).
        MAX_PRM_FEEDBACK_HISTORY = 5
        recent_feedbacks = previous_feedbacks[-MAX_PRM_FEEDBACK_HISTORY:]

        history_context = ""
        if recent_feedbacks:
            history_context = "\n\n## Previous Supervisor Feedback History:\n"
            for j, fb in enumerate(recent_feedbacks, 1):
                history_context += (
                    f"\n### Feedback #{j} (after step {fb['after_step']}):\n"
                    f"{fb['feedback']}\n"
                )

        prm_messages.append({
            "role": "user",
            "content": (
                f"{history_context}\n\n"
                "Please provide your supervisor feedback now based on the trajectory above."
            ),
        })

        # PRM's response (training target)
        prm_messages.append({
            "role": "assistant",
            "content": prm_response,
        })

        prm_messages = merge_consecutive_roles(prm_messages)

        samples.append({
            "messages": prm_messages,
            "instance_id": instance_id,
            "invocation": inv_number,
        })

        after_step = after_step_map.get(inv_number)
        if after_step is None:
            after_step = sup_msg.get("_prm_after_step", inv_number * 10)
        previous_feedbacks.append({
            "after_step": after_step,
            "feedback": prm_response,
        })

    return samples


def merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role (except system)."""
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"] and msg["role"] != "system":
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)
    return merged


def format_for_llamafactory(sample: dict) -> dict:
    """Convert a sample to LlamaFactory legacy sharegpt SFT format.

    Flattens the multi-turn trajectory into a 3-message conversation:
      system: PRM system prompt
      user:   all trajectory context (task + agent turns + feedback history)
      assistant: PRM response (only this gets loss in legacy mode)

    This avoids intermediate assistant turns being trained on.

    The user message is structured as:
      1. Task context block: [STEP-ENVIRONMENT]: <task context>
      2. Trajectory blocks: alternating [STEP-AGENT] and [STEP-ENVIRONMENT]
      3. Feedback blocks: [PREV-FEEDBACK #N]: ... (one per past feedback)
      4. Final prompt: [STEP-ENVIRONMENT]: Please provide your supervisor feedback ...

    Tags use distinct names (not [USER/ENVIRONMENT] / [AGENT RESPONSE]) to
    prevent the finetuned model from hallucinating agent-style entries.
    These tags must match the ones used at inference in default_prm.py.
    """
    messages = sample["messages"]

    system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""

    prm_response = messages[-1]["content"]
    assert messages[-1]["role"] == "assistant", "Last message must be assistant (PRM response)"

    context_parts = []
    for msg in messages[1:-1]:
        if msg["role"] == "assistant":
            context_parts.append(f"{TAG_AGENT}: {msg['content']}")
        elif msg["role"] == "user":
            content = msg["content"]
            if "## Previous Supervisor Feedback History:" in content:
                hist_start = content.index("## Previous Supervisor Feedback History:")
                before_hist = content[:hist_start].strip()
                hist_and_prompt = content[hist_start:]

                prompt_marker = "Please provide your supervisor feedback now"
                prompt_idx = hist_and_prompt.find(prompt_marker)
                if prompt_idx >= 0:
                    hist_section = hist_and_prompt[:prompt_idx].strip()
                    prompt_section = hist_and_prompt[prompt_idx:].strip()
                else:
                    hist_section = hist_and_prompt.strip()
                    prompt_section = ""

                if before_hist:
                    context_parts.append(f"{TAG_ENVIRONMENT}: {before_hist}")

                fb_blocks = re.split(r"(?=### Feedback #\d+)", hist_section)
                for fb_block in fb_blocks:
                    fb_block = fb_block.strip()
                    if not fb_block or fb_block == "## Previous Supervisor Feedback History:":
                        continue
                    fb_match = re.match(r"### Feedback #(\d+)", fb_block)
                    if fb_match:
                        fb_num = fb_match.group(1)
                        fb_content = re.sub(r"^### Feedback #\d+[^\n]*\n?", "", fb_block).strip()
                        context_parts.append(f"{TAG_FEEDBACK_PREFIX} #{fb_num}]: {fb_content}")

                if prompt_section:
                    context_parts.append(f"{TAG_ENVIRONMENT}: {prompt_section}")
            else:
                context_parts.append(f"{TAG_ENVIRONMENT}: {content}")
        else:
            context_parts.append(f"[{msg['role'].upper()}]: {content}")

    user_content = "\n\n".join(context_parts)

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prm_response},
        ]
    }


def format_multiturn_for_llamafactory(sample: dict) -> dict:
    """Convert a sample to LlamaFactory multi-turn sharegpt SFT format.

    Preserves the natural multi-turn structure of the trajectory:
      system:    PRM system prompt
      user:      task context
      assistant: agent step 1
      user:      env output 1
      assistant: agent step 2
      user:      env output 2
      ...
      user:      [previous feedbacks] + final prompt
      assistant: PRM response  ← only this gets loss (via mask_history=true)

    No tags needed — the chat template itself provides structural boundaries
    via <|im_start|>user / <|im_start|>assistant tokens.  The model learns
    to generate PRM feedback when it's its turn, not to continue agent output.
    """
    messages = sample["messages"]

    system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""
    prm_response = messages[-1]["content"]
    assert messages[-1]["role"] == "assistant", "Last message must be assistant (PRM response)"

    out_messages = [{"role": "system", "content": system_content}]

    for msg in messages[1:-1]:
        out_messages.append({"role": msg["role"], "content": msg["content"]})

    # Merge consecutive same-role messages (can happen at user/user boundaries
    # where feedback history is appended after the last env output)
    merged = [out_messages[0]]
    for msg in out_messages[1:]:
        if msg["role"] == merged[-1]["role"] and msg["role"] != "system":
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    # The final assistant turn is the PRM response
    merged.append({"role": "assistant", "content": prm_response})

    return {"messages": merged}


def smart_truncate_multiturn(
    formatted_sample: dict,
    tokenizer,
    max_tokens: int,
    max_feedbacks: int = 0,
) -> dict:
    """Truncate multi-turn samples by dropping oldest complete turn pairs.

    A "turn pair" is a (user, assistant) pair representing one agent step
    (environment output + agent response).  Dropping a pair keeps the
    conversation coherent — never splits a turn mid-way.

    Preserves:
      - System message (index 0)
      - Task context (first user message, index 1)
      - Last user message (feedbacks + prompt)
      - Last assistant message (PRM response — training target)

    Drops oldest turn pairs first (keeps most recent trajectory).
    """
    msgs = formatted_sample["messages"]
    total_tokens = sum(len(tokenizer.encode(m["content"])) for m in msgs)

    if total_tokens <= max_tokens:
        return formatted_sample

    # Fixed messages: system (0), task context (1), last user (-2), last assistant (-1)
    fixed_indices = {0, 1, len(msgs) - 2, len(msgs) - 1}
    fixed_tokens = sum(
        len(tokenizer.encode(msgs[i]["content"])) for i in fixed_indices
    ) + 100  # overhead

    available = max_tokens - fixed_tokens
    if available < 1000:
        available = 1000

    # Variable messages: everything between task context and final user/assistant
    variable = msgs[2:-2]

    # Group into turn pairs (user, assistant).  The first variable message
    # should be assistant (agent's first response after task context).
    # Group consecutive messages into (user, assistant) pairs where each
    # pair represents one complete agent step.
    pairs: list[tuple[list[dict], int]] = []
    i = 0
    while i < len(variable):
        pair_msgs = [variable[i]]
        pair_tokens = len(tokenizer.encode(variable[i]["content"]))
        # If this is a user message followed by assistant, group them
        if i + 1 < len(variable) and variable[i]["role"] != variable[i + 1]["role"]:
            pair_msgs.append(variable[i + 1])
            pair_tokens += len(tokenizer.encode(variable[i + 1]["content"]))
            i += 2
        else:
            i += 1
        pairs.append((pair_msgs, pair_tokens))

    # Keep most recent pairs that fit within budget
    kept_reversed: list[tuple[list[dict], int]] = []
    used = 0
    for pair_msgs, pair_tokens in reversed(pairs):
        if used + pair_tokens > available:
            break
        kept_reversed.append((pair_msgs, pair_tokens))
        used += pair_tokens
    kept_pairs = list(reversed(kept_reversed))

    dropped_pairs = len(pairs) - len(kept_pairs)
    dropped_msgs = sum(len(p) for p, _ in pairs) - sum(len(p) for p, _ in kept_pairs)

    result_msgs = [msgs[0], msgs[1]]  # system + task context
    if dropped_pairs > 0:
        result_msgs.append({
            "role": "user",
            "content": f"[...{dropped_msgs} earlier trajectory messages ({dropped_pairs} turn pairs) omitted...]",
        })
    for pair_msgs, _ in kept_pairs:
        result_msgs.extend(pair_msgs)
    result_msgs.append(msgs[-2])  # final user (feedbacks + prompt)
    result_msgs.append(msgs[-1])  # PRM response

    # Re-merge consecutive same-role after insertion of truncation marker
    merged = [result_msgs[0]]
    for msg in result_msgs[1:]:
        if msg["role"] == merged[-1]["role"] and msg["role"] != "system":
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    return {"messages": merged}


def split_user_into_blocks(user_content: str) -> list[tuple[str, str]]:
    """Split the flattened user message into categorised (tag, text) blocks.

    Returns a list of (category, raw_block_text) where category is one of:
      "task"      — the first [STEP-ENVIRONMENT] block (task context)
      "trajectory"— [STEP-AGENT] or [STEP-ENVIRONMENT] blocks (agent steps)
      "feedback"  — [PREV-FEEDBACK #N] blocks
      "prompt"    — the final [STEP-ENVIRONMENT] "Please provide your supervisor
                     feedback now …" block
    """
    tag_names = [
        re.escape(TAG_AGENT),
        re.escape(TAG_ENVIRONMENT),
        re.escape(TAG_FEEDBACK_PREFIX) + r" #\d+\]",
        r"\[CRITICAL CONTEXT\]",
    ]
    pattern = r"\n\n(?=(?:" + "|".join(tag_names) + r"):)"
    raw_blocks = [p for p in re.split(pattern, user_content) if p.strip()]

    categorised: list[tuple[str, str]] = []
    seen_task = False
    for block in raw_blocks:
        if block.startswith(TAG_FEEDBACK_PREFIX):
            categorised.append(("feedback", block))
        elif block.startswith(f"{TAG_ENVIRONMENT}:") and "Please provide your supervisor feedback now" in block:
            categorised.append(("prompt", block))
        elif block.startswith(f"{TAG_ENVIRONMENT}:") and not seen_task:
            categorised.append(("task", block))
            seen_task = True
        else:
            categorised.append(("trajectory", block))
    return categorised


def smart_truncate_for_token_budget(
    formatted_sample: dict,
    tokenizer,
    max_tokens: int,
    max_feedbacks: int = 0,
    feedback_budget_ratio: float = 0.3,
) -> dict:
    """Truncate long samples with principled budget allocation.

    Design principles:
      1. Never cut a block mid-way — either keep or drop an entire block.
      2. Drop OLDEST blocks first (keep most recent trajectory steps / feedbacks).
      3. Feedback history is capped to the most recent ``max_feedbacks`` entries
         (0 = no cap) BEFORE token budgeting, so it never crowds out trajectory.
      4. Remaining token budget is split between trajectory and feedback with
         ``feedback_budget_ratio`` controlling the feedback share.

    Always preserved in full:
      - System prompt
      - Task context (first user block)
      - Final prompt ("Please provide your supervisor feedback now …")
      - PRM response (assistant message — training target)
    """
    msgs = formatted_sample["messages"]
    sys_tokens = len(tokenizer.encode(msgs[0]["content"]))
    asst_tokens = len(tokenizer.encode(msgs[2]["content"]))
    user_text = msgs[1]["content"]

    # --- 1. Split into categorised blocks ---
    blocks = split_user_into_blocks(user_text)

    task_blocks = [(i, b) for i, (cat, b) in enumerate(blocks) if cat == "task"]
    traj_blocks = [(i, b) for i, (cat, b) in enumerate(blocks) if cat == "trajectory"]
    fb_blocks   = [(i, b) for i, (cat, b) in enumerate(blocks) if cat == "feedback"]
    prompt_blocks = [(i, b) for i, (cat, b) in enumerate(blocks) if cat == "prompt"]

    # --- 2. Cap feedback count (keep most recent N) ---
    if max_feedbacks > 0 and len(fb_blocks) > max_feedbacks:
        fb_blocks = fb_blocks[-max_feedbacks:]

    # --- 3. Tokenize fixed blocks ---
    task_text = "\n\n".join(b for _, b in task_blocks)
    prompt_text = "\n\n".join(b for _, b in prompt_blocks)
    task_tokens = len(tokenizer.encode(task_text)) if task_text else 0
    prompt_tokens = len(tokenizer.encode(prompt_text)) if prompt_text else 0

    fixed_tokens = sys_tokens + asst_tokens + task_tokens + prompt_tokens + 100  # overhead
    available = max_tokens - fixed_tokens
    if available < 1000:
        available = 1000

    # --- 4. Tokenize variable blocks ---
    traj_with_tokens = [(b, len(tokenizer.encode(b))) for _, b in traj_blocks]
    fb_with_tokens   = [(b, len(tokenizer.encode(b))) for _, b in fb_blocks]

    total_traj = sum(t for _, t in traj_with_tokens)
    total_fb   = sum(t for _, t in fb_with_tokens)

    # If everything fits, no truncation needed
    if total_traj + total_fb <= available:
        # Still need to rebuild if we capped feedbacks
        if max_feedbacks > 0 and len([(cat, _) for cat, _ in blocks if cat == "feedback"]) != len(fb_blocks):
            return _rebuild_sample(msgs, task_blocks, traj_with_tokens, fb_with_tokens, prompt_blocks, 0, 0)
        return formatted_sample

    # --- 5. Budget split ---
    fb_budget = int(available * feedback_budget_ratio)
    traj_budget = available - fb_budget

    # If one category is under budget, give its surplus to the other
    if total_fb <= fb_budget:
        traj_budget += fb_budget - total_fb
        fb_budget = total_fb
    elif total_traj <= traj_budget:
        fb_budget += traj_budget - total_traj
        traj_budget = total_traj

    # --- 6. Truncate trajectory: drop OLDEST blocks first (keep recent) ---
    traj_kept, traj_dropped = _keep_recent_blocks(traj_with_tokens, traj_budget)

    # --- 7. Truncate feedback: drop OLDEST feedbacks first (keep recent) ---
    fb_kept, fb_dropped = _keep_recent_blocks(fb_with_tokens, fb_budget)

    return _rebuild_sample(msgs, task_blocks, traj_kept, fb_kept, prompt_blocks, traj_dropped, fb_dropped)


def _keep_recent_blocks(
    blocks_with_tokens: list[tuple[str, int]],
    budget: int,
) -> tuple[list[tuple[str, int]], int]:
    """Keep as many RECENT blocks as fit within budget, dropping oldest first.

    Returns (kept_blocks_with_tokens, num_dropped).
    """
    if not blocks_with_tokens:
        return blocks_with_tokens, 0

    total = sum(t for _, t in blocks_with_tokens)
    if total <= budget:
        return blocks_with_tokens, 0

    # Walk from the end (most recent), accumulate until budget is hit
    kept_reversed: list[tuple[str, int]] = []
    used = 0
    for block_text, block_tokens in reversed(blocks_with_tokens):
        if used + block_tokens > budget:
            break
        kept_reversed.append((block_text, block_tokens))
        used += block_tokens

    kept = list(reversed(kept_reversed))
    dropped = len(blocks_with_tokens) - len(kept)
    return kept, dropped


def _rebuild_sample(
    msgs: list[dict],
    task_blocks: list[tuple[int, str]],
    traj_kept: list[tuple[str, int]],
    fb_kept: list[tuple[str, int]],
    prompt_blocks: list[tuple[int, str]],
    traj_dropped: int,
    fb_dropped: int,
) -> dict:
    """Reassemble the user message from kept blocks."""
    parts: list[str] = []

    # Task context
    for _, b in task_blocks:
        parts.append(b)

    # Trajectory truncation marker
    if traj_dropped > 0:
        parts.append(f"[...{traj_dropped} earlier trajectory step(s) omitted...]")

    # Kept trajectory blocks (already in chronological order — most recent)
    for b, _ in traj_kept:
        parts.append(b)

    # Feedback truncation marker
    if fb_dropped > 0:
        parts.append(f"[...{fb_dropped} earlier supervisor feedback(s) omitted...]")

    # Kept feedback blocks (already in chronological order — most recent)
    for b, _ in fb_kept:
        parts.append(b)

    # Final prompt
    for _, b in prompt_blocks:
        parts.append(b)

    user_content = "\n\n".join(parts)

    return {
        "messages": [
            msgs[0],
            {"role": "user", "content": user_content},
            msgs[2],
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PRM SFT data from trajectory files for LlamaFactory training."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to results directory containing trajectory subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for prepared training data.",
    )
    parser.add_argument(
        "--format",
        choices=["flattened", "multiturn"],
        default="flattened",
        help="Output format. 'flattened' = 3-message format with tags (standard SFT). "
             "'multiturn' = real chat turns with mask_history=true (loss only on last assistant).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (rest is validation). "
             "Default 1.0 means all data is used for training (no val split).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Max tokens per sample. If >0, long samples are smart-truncated. "
             "Truncation drops oldest trajectory steps and oldest feedbacks first, "
             "never cutting mid-block. System prompt, task context, final prompt, "
             "and PRM response are always preserved in full.",
    )
    parser.add_argument(
        "--max_feedbacks",
        type=int,
        default=0,
        help="Max number of past PRM feedbacks to include per sample (0 = no cap). "
             "When set, only the N most recent feedbacks are kept. Applied before "
             "token budgeting so feedback never crowds out trajectory.",
    )
    parser.add_argument(
        "--feedback_budget_ratio",
        type=float,
        default=0.3,
        help="Fraction of the variable token budget allocated to feedback history "
             "(rest goes to trajectory). Only used when truncation is needed. "
             "Default 0.3 means 30%% feedback, 70%% trajectory.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer to use for token counting (only needed with --max_tokens).",
    )
    parser.add_argument(
        "--report_json",
        default=None,
        help="Path to report.json for rejection sampling. "
             "If provided, only samples from resolved instances are kept.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load resolved instance IDs for rejection sampling
    resolved_ids = None
    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.exists():
            print(f"ERROR: Report file not found: {report_path}")
            sys.exit(1)
        with open(report_path) as f:
            report = json.load(f)
        resolved_ids = set(report["resolved_ids"])
        print(f"Rejection sampling: keeping only {len(resolved_ids)} resolved instances", flush=True)

    # Collect all trajectory files
    traj_files = sorted(results_dir.glob("*/*.traj.json"))
    print(f"Found {len(traj_files)} trajectory files in {results_dir}", flush=True)

    # Extract PRM system prompt from first valid trajectory
    prm_system_prompt = None
    for traj_path in traj_files:
        try:
            with open(traj_path) as f:
                traj = json.load(f)
            prm_system_prompt = extract_prm_system_prompt(traj)
            break
        except (json.JSONDecodeError, IOError, ValueError):
            continue

    if not prm_system_prompt:
        print("ERROR: Could not extract PRM system prompt from any trajectory")
        sys.exit(1)
    print(f"Extracted PRM system prompt ({len(prm_system_prompt)} chars)", flush=True)

    all_samples = []
    skipped = 0
    errors = 0
    filtered = 0
    denoised = 0

    for traj_path in tqdm(traj_files, desc="Extracting samples"):
        try:
            with open(traj_path) as f:
                traj = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Failed to read {traj_path.name}: {e}")
            errors += 1
            continue

        instance_id = traj.get("instance_id", "unknown")
        if resolved_ids is not None and instance_id not in resolved_ids:
            filtered += 1
            continue

        samples = reconstruct_prm_samples(traj, prm_system_prompt)
        if not samples:
            skipped += 1
            continue

        # Track denoised count (feedback log entries minus valid samples)
        fl = traj.get("info", {}).get("prm_stats", {}).get("prm_feedback_log", [])
        denoised += len(fl) - len(samples)

        all_samples.extend(samples)

    print(f"\nExtracted {len(all_samples)} PRM invocation samples "
          f"from {len(traj_files) - skipped - errors - filtered} trajectories")
    print(f"  Skipped (no supervisor messages): {skipped}")
    print(f"  Errors (failed to read): {errors}")
    print(f"  Denoised (invalid feedback removed): {denoised}")
    if resolved_ids is not None:
        print(f"  Filtered (unresolved instances): {filtered}")

    if not all_samples:
        print("ERROR: No samples extracted. Exiting.")
        sys.exit(1)

    # Print some statistics
    invocation_counts = {}
    for s in all_samples:
        inv = s["invocation"]
        invocation_counts[inv] = invocation_counts.get(inv, 0) + 1
    print("\nSamples per invocation number:")
    for inv in sorted(invocation_counts):
        print(f"  Invocation #{inv}: {invocation_counts[inv]} samples")

    # Compute message length stats
    total_tokens_approx = 0
    for s in all_samples:
        for msg in s["messages"]:
            total_tokens_approx += len(msg["content"]) // 4  # rough char-to-token
    avg_tokens = total_tokens_approx // len(all_samples) if all_samples else 0
    print(f"\nApprox avg tokens per sample: {avg_tokens}")

    # Split into train/val by instance_id (not by sample, to avoid data leakage)
    instance_ids = sorted(set(s["instance_id"] for s in all_samples))
    split_idx = int(len(instance_ids) * args.train_ratio)
    train_ids = set(instance_ids[:split_idx])
    val_ids = set(instance_ids[split_idx:])

    train_samples = [s for s in all_samples if s["instance_id"] in train_ids]
    val_samples = [s for s in all_samples if s["instance_id"] in val_ids]

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val "
          f"({len(train_ids)} / {len(val_ids)} instances)")

    # Load tokenizer if smart truncation is enabled
    tokenizer = None
    if args.max_tokens > 0:
        from transformers import AutoTokenizer
        print(f"\nLoading tokenizer {args.tokenizer} for smart truncation (max_tokens={args.max_tokens})...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("Tokenizer loaded.", flush=True)

    # Select formatter based on --format
    use_multiturn = args.format == "multiturn"
    print(f"\nFormat: {args.format}" + (" (requires mask_history=true in training config)" if use_multiturn else ""), flush=True)

    # Write output files
    train_path = output_dir / "prm_sft_train.jsonl"

    truncated_count = 0
    output_pairs = [(train_path, train_samples)]
    if val_samples:
        val_path = output_dir / "prm_sft_val.jsonl"
        output_pairs.append((val_path, val_samples))

    for path, samples in output_pairs:
        split_name = "train" if "train" in str(path) else "val"
        with open(path, "w") as f:
            for s in tqdm(samples, desc=f"Writing {split_name}"):
                if use_multiturn:
                    formatted = format_multiturn_for_llamafactory(s)
                    if tokenizer is not None:
                        original = formatted
                        formatted = smart_truncate_multiturn(
                            formatted, tokenizer, args.max_tokens,
                            max_feedbacks=args.max_feedbacks,
                        )
                        if len(formatted["messages"]) != len(original["messages"]):
                            truncated_count += 1
                else:
                    formatted = format_for_llamafactory(s)
                    if tokenizer is not None:
                        original = formatted
                        formatted = smart_truncate_for_token_budget(
                            formatted, tokenizer, args.max_tokens,
                            max_feedbacks=args.max_feedbacks,
                            feedback_budget_ratio=args.feedback_budget_ratio,
                        )
                        if formatted["messages"][1]["content"] != original["messages"][1]["content"]:
                            truncated_count += 1
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples)} samples to {path}")

    if tokenizer is not None:
        print(f"Smart-truncated {truncated_count} samples")
        if args.max_feedbacks > 0:
            print(f"  Max feedbacks per sample: {args.max_feedbacks}")
        if not use_multiturn:
            print(f"  Feedback budget ratio: {args.feedback_budget_ratio}")
        print(f"  Strategy: drop oldest {'messages' if use_multiturn else 'trajectory steps & feedbacks'} first")

    # Write dataset_info.json for LlamaFactory legacy sharegpt format
    sharegpt_tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    }
    dataset_info = {
        "prm_sft_train": {
            "file_name": "prm_sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": sharegpt_tags,
        },
    }
    if val_samples:
        dataset_info["prm_sft_val"] = {
            "file_name": "prm_sft_val.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": sharegpt_tags,
        }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Wrote dataset info to {info_path}")

    # Derive PRM interval from after_step values in first valid trajectory
    prm_interval = None
    for traj_path in traj_files[:10]:
        try:
            with open(traj_path) as f:
                traj = json.load(f)
            step_map = build_after_step_map(traj)
            if len(step_map) >= 2:
                steps = sorted(step_map.values())
                prm_interval = steps[1] - steps[0]
                break
            elif step_map:
                prm_interval = list(step_map.values())[0]
                break
        except Exception:
            continue

    metadata = {
        "source_results_dir": str(results_dir),
        "prm_model": "us.anthropic.claude-opus-4-6-v1",
        "agent_model": "facebook/cwm",
        "prm_interval": prm_interval,
        "format": args.format,
        "feedback_history_format": "full_untruncated",
        "max_tokens": args.max_tokens,
        "max_feedbacks": args.max_feedbacks,
        "feedback_budget_ratio": args.feedback_budget_ratio,
        "num_truncated_samples": truncated_count,
        "num_denoised": denoised,
        "num_trajectories": len(traj_files) - skipped - errors - filtered,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_instance_ids": sorted(train_ids),
    }
    if val_samples:
        metadata["val_instance_ids"] = sorted(val_ids)
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
