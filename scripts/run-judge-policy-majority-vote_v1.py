#!/usr/bin/env python3
"""
Judge script for analyzing tool overuse in mini SWE-agent trajectories.
Performs step-wise analysis of each action to identify redundant tool usage.
NOW WITH PARALLEL PROCESSING AND MAJORITY VOTING!
"""

import argparse
import json
import os
import re
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import threading

import rich
from litellm import completion
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# --- Pricing (per 1M tokens) ---
INPUT_PRICE_PER_MTOK = 0.250   # $ per 1M input tokens
OUTPUT_PRICE_PER_MTOK = 2.000  # $ per 1M output tokens

JUDGE_SYSTEM_MESSAGE = """
You are an expert judge analyzing mini SWE-agent shell trajectories for TOOL OVERUSE. Your objective is to MAXIMIZE RECALL of redundant actions while remaining consistent and precise.

The agent works in an isolated repo; assume no hidden external edits. There is EXACTLY ONE final submission command; it is ALWAYS essential.

=====================================================
A) GLOBAL CONSTRAINTS (APPLY THROUGHOUT)
=====================================================
GC1. NO WHOLE-FILE READS (HARD BAN)
- The agent must never read an entire file. It must always hunt for something specific via search → targeted line-range read.
- If a command dumps the entire file (any size) → redundant.
- If a command reads a very broad span (>200 lines OR >3KB) for a targeted need → redundant.

GC2. READ-AFTER-OWN-EDIT IS REDUNDANT
- Any read of lines that were just edited (even small/local) is redundant. The agent must trust its own write or validate via execution/tests, not by re-reading.

GC3. SINGLE FINAL SUBMISSION
- The unique submission command
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`
  is ALWAYS essential. Never mark it redundant for any reason.

=====================================================
B) STATE TO TRACK WHILE READING THE TRAJECTORY
=====================================================
1) FILE-VIEW MAP: For each file, maintain merged viewed line-intervals.
   - A read is a duplicate if ≥50% of its requested span has been seen previously.
2) WRITE LOG: Track writes by file and line-spans (if available).
3) TEST/STATUS LOG: Track outcomes of git status/add/diff/commit and test/build runs.

=====================================================
C) DECISION PROCEDURE (STRICT ORDER)
=====================================================
For each step:
1) Identify action_type: {search, read, write, execute, env, other (...)}.
2) Check ESSENTIAL policies (E1–E4). If any applies → ESSENTIAL.
3) Else check REDUNDANT policies (R1–R13). If any applies → REDUNDANT (cite the rule).
4) Else → ESSENTIAL (default).

=====================================================
D) ESSENTIAL STEP POLICIES (any one match → ESSENTIAL)
=====================================================
E1. FIRST-TIME, TARGETED INFORMATION ACQUISITION
- Retrieves information not previously seen AND does so with targeted scope.
  Examples: first search for symbols/paths; first read of an unseen line-range (not whole-file; use line ranges).
  Tests:
    • If ≥50% overlap with previously viewed lines → NOT E1.
    • Whole-file reads are NEVER E1 (see GC1/R4).

E2. REPOSITORY MODIFICATION
- Changes repository state in a way that can advance the task.
  Examples: source edits, file creation/deletion, dependency/config updates, script creation.

E3. EXECUTION / VALIDATION AFTER MATERIAL CHANGE
- Runs tests/builds/programs to observe effects of a prior material (behavior-affecting) change.

E4. FINAL SUBMISSION (UNIQUE, TASK-DEFINED)
- The command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached` is ALWAYS essential.

=====================================================
E) REDUNDANT STEP POLICIES (any one match → REDUNDANT)
=====================================================
(Include the policy code(s) in violated_policies; choose exactly one category from the mapping in Section F.)

R1. DUPLICATE CONTENT RETRIEVAL (category: duplicate_read)
- Re-reading content already retrieved earlier (≥50% line-span overlap) without external changes.
- Reading after the agent’s own write is always redundant (see GC2), even for small/local edits.

R2. REPEATED SEARCHES (category: duplicate_search)
- Repeating identical or substantially similar searches without new parameters or new intent.

R3. REDUNDANT STATE CHECKS (category: duplicate_status_check)
- Re-checking system state with no intervening action that could change it.
  Examples: repeated `git status`, repeated `git diff --cached`, repeated `wc -l`, repeated `pwd`.

R4. OVERLY BROAD READ (category: overly_broad_read)
- Any whole-file read (hard ban; see GC1) OR very broad reads (>200 lines OR >3KB) when hunting a specific target.
  Correction: search (`grep -n`, `git grep`, ripgrep) → then narrow line-ranged read (e.g., `sed -n 'L-20,L+20p'`).

R5. COMMAND CONSOLIDATION MISS (category: overly_broad_read)
- Multiple near-identical queries that could be combined into one command (e.g., separate greps instead of one alternation).
  Correction: combine patterns/wildcards.

R6. UNFOCUSED DIRECTORY EXPLORATION (category: overly_broad_read)
- Broad listings (e.g., `ls -la` in large dirs) instead of filtered discovery with patterns/constraints.
  Correction: use `find`/`grep -R` with narrow filters.

R7. IRRELEVANT SCOPE EXPLORATION (category: overly_broad_read)
- Exploring directories/files clearly unrelated to the stated task/component.
  Correction: skip or redirect scope per task.

R8. UNMODIFIED RETRIES (category: repeated_failed_command)
- Re-running a failing command unchanged (same args/paths/env) without addressing the cause.

R9. NO STRATEGY ADAPTATION (category: circular_debugging)
- Cycling the same ineffective debugging/fix approach >2 times without material adjustment.

R10. OBVIOUS CONFIRMATIONS (category: obvious_confirmation)
- Verifying states that are logically certain from prior successful outputs.
  Examples: checking file existence immediately after successful creation; `git diff` right after `git status` shows no changes.

R11. FRAGMENTED TEST AUTHORING (category: excessive_intermediate_testing)
- Writing many tiny test/check steps across multiple actions where a single, comprehensive step suffices.
  Correction: combine into one step.

R12. TESTING WITHOUT MEANINGFUL CHANGE (category: excessive_intermediate_testing)
- Running tests/builds with no material change since the last run (comments/whitespace/docstring-only changes included).

R13. REDUNDANT SUCCESS VALIDATION (category: redundant_success_validation)
- Re-running already passing tests with no intervening implementation change.

=====================================================
F) CATEGORY ↔ POLICY MAPPING (NO OVERLAP)
=====================================================
- duplicate_read → R1
- duplicate_search → R2
- duplicate_status_check → R3
- overly_broad_read → R4, R5, R6, R7   (choose the best-fitting of these four)
- repeated_failed_command → R8
- circular_debugging → R9
- obvious_confirmation → R10
- excessive_intermediate_testing → R11, R12
- redundant_success_validation → R13
- essential → (for E1–E4)

=====================================================
G) OUTPUT FORMAT (KEEP EXACT SCHEMA)
=====================================================
For each step, produce an object with EXACTLY these fields:

{
  "step_number": <int>,
  "command": "<verbatim command>",
  "action_type": "search|read|write|execute|env|other (...)",
  "redundant": true|false,
  "category": "essential|duplicate_read|duplicate_search|duplicate_status_check|overly_broad_read|repeated_failed_command|circular_debugging|obvious_confirmation|excessive_intermediate_testing|redundant_success_validation",
  "violated_policies": ["R#"...],          # [] when essential
  "reasoning": "<1–3 sentences tied to E# or R#>",
  "correction": null | "skip" | "modify" | "combine"
}

Rules:
- If ESSENTIAL: redundant=false, category="essential", violated_policies=[], correction=null.
- If REDUNDANT: redundant=true, choose exactly ONE category (per Section F), include at least one R#, set correction:
    • "skip" for pure waste,
    • "modify" for better parameters (e.g., narrowed grep),
    • "combine" for consolidation opportunities.

=====================================================
H) QUICK CANONICAL DECISIONS
=====================================================
- Read 220–235 → Write 223–230 → Read 220–240:
  Redundant (R1 duplicate_read; also violates GC2 read-after-own-edit; exceeds narrow validation, which is disallowed).
- `cat file.py` to “look around”:
  Redundant (R4 overly_broad_read; GC1 hard ban on whole-file reads).
- `ls -la` at repo root instead of filtered search:
  Redundant (R6 unfocused directory exploration).
- Re-run `git diff --cached` with no new edits:
  Redundant (R3 duplicate_status_check).
- First-time precise search → line-ranged read of unseen region:
  Essential (E1).
- After code change, run tests once:
  Essential (E3).
- Final submission command:
  Essential (E4) ALWAYS.

Be strict and consistent. Prefer marking re-reads, broad reads, broad listings, repeated checks, and unmodified retries as REDUNDANT to maximize recall.

# OUTPUT FORMAT

You MUST respond with your analysis in a JSON code block using triple backticks. Provide the JSON in the following exact structure:

```json
{
  "trajectory_id": "instance_id_here",
  "steps": [
    {
      "step_number": 1,
      "command": "cat large_file.py",
      "action_type": "read",
      "redundant": true,
      "category": "overly_broad_read",
      "violated_policies": ["R4"],
      "reasoning": "Violates R4 (Overly Broad Read): whole-file dump while hunting a specific symbol; must search first, then line-range.",
      "correction": "modify"
    },
    {
      "step_number": 2,
      "command": "grep -n 'def process_data' -R .",
      "action_type": "search",
      "redundant": false,
      "category": "essential",
      "violated_policies": [],
      "reasoning": "Essential (E1): first-time, targeted search to locate the symbol before a narrow read.",
      "correction": null
    }
  ],
  "summary": {
    "total_steps": 2,
    "redundant_steps": 1,
    "efficiency_score": 0.5,
    "main_inefficiencies": ["overly_broad_read"],
    "violated_policies_summary": {
      "R4": 1
    },
    "action_type_distribution": {
      "search": 1,
      "read": 1,
      "write": 0,
      "execute": 0,
      "env": 0,
      "other": 0
    }
  }
}
```

# CRITICAL REQUIREMENTS

1) Output MUST be a JSON code block using triple backticks (```json ... ```).
2) Use booleans for "redundant" (true/false), not strings.
3) Use null (not the string "null") for "correction" when the step is essential.
4) Use an empty array [] for "violated_policies" when the step is essential.
5) For redundant steps, cite R-codes in both "violated_policies" (e.g., ["R1"]) and "reasoning" (e.g., "Violates R1: ..."). 
   For essential steps, reference the E-policy in reasoning (e.g., "Essential (E3): ...").
6) Ensure all "step_number" values match the actual trajectory ordering.
7) Compute "efficiency_score" as (total_steps - redundant_steps) / total_steps, rounded as needed by the caller.
8) Include "violated_policies_summary" with counts per R-code used across all steps.
9) If any R-policy applies, mark the step as redundant=true, select exactly ONE category, and set "correction" to:
   - "skip" for pure waste,
   - "modify" for better parameters (e.g., narrower grep),
   - "combine" when multiple actions should be consolidated.
10) "category" must be one of:
    - "essential" (for E1–E4), or one of:
      "duplicate_read", "duplicate_search", "duplicate_status_check",
      "overly_broad_read", "repeated_failed_command", "circular_debugging",
      "obvious_confirmation", "excessive_intermediate_testing", "redundant_success_validation"."""

JUDGE_USER_MESSAGE = """Analyze the following mini SWE-agent trajectory for tool overuse patterns. Classify each action type and identify redundancies.

# TRAJECTORY TO ANALYZE
{trajectory_content}

Remember to:
1. Classify each command into the correct action type
2. Identify redundant vs essential actions
3. Assign specific categories from the taxonomy
4. Provide clear reasoning
5. Output your analysis in a JSON code block using triple backticks"""


def load_trajectory(trajectory_path: Path) -> Dict[str, Any]:
    """Load a mini SWE-agent trajectory file."""
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        rich.print(f"[red]Error loading trajectory {trajectory_path}: {e}[/red]")
        return None


def extract_steps_from_trajectory(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract steps from mini SWE-agent trajectory format."""
    steps = []
    messages = trajectory.get('messages', [])
    
    step_number = 1
    for i, message in enumerate(messages):
        if message.get('role') == 'assistant':
            # Extract command from assistant message
            content = message.get('content', '')
            
            # Look for bash code blocks
            bash_pattern = r'```bash\n(.*?)\n```'
            bash_matches = re.findall(bash_pattern, content, re.DOTALL)
            
            if bash_matches:
                command = bash_matches[0].strip()
                
                # Get the output from the next user message if available
                output = ""
                returncode = 0
                if i + 1 < len(messages) and messages[i + 1].get('role') == 'user':
                    next_content = messages[i + 1].get('content', '')
                    # Extract returncode and output
                    returncode_match = re.search(r'<returncode>(\d+)</returncode>', next_content)
                    output_match = re.search(r'<output>(.*?)</output>', next_content, re.DOTALL)
                    
                    if returncode_match:
                        returncode = int(returncode_match.group(1))
                    if output_match:
                        output = output_match.group(1).strip()
                
                # Pull per-step agent usage from provider response if present
                agent_prompt_tokens = 0
                agent_completion_tokens = 0
                agent_total_tokens = 0
                agent_model = None

                extra = message.get("extra") or {}
                resp = (extra.get("response") or {})
                usage = resp.get("usage") or {}
                agent_prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                agent_completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                agent_total_tokens = usage.get("total_tokens") or (agent_prompt_tokens + agent_completion_tokens)
                agent_model = resp.get("model")

                steps.append({
                    'step_number': step_number,
                    'command': command,
                    'output': output,
                    'returncode': returncode,
                    'thought': content.split('```bash')[0].replace('THOUGHT:', '').strip() if 'THOUGHT:' in content else '',
                    'agent_token_usage': {
                        'model': agent_model,
                        'prompt_tokens': agent_prompt_tokens,
                        'completion_tokens': agent_completion_tokens,
                        'total_tokens': agent_total_tokens,
                    }
                })

                step_number += 1
    
    return steps


def format_trajectory_for_judge(trajectory: Dict[str, Any], steps: List[Dict[str, Any]]) -> str:
    """Format trajectory data for the judge."""
    instance_id = trajectory.get('instance_id', 'unknown')
    task_description = ""
    
    # Extract task from the first user message
    messages = trajectory.get('messages', [])
    if messages and messages[0].get('role') == 'user':
        content = messages[0].get('content', '')
        # Extract PR description
        pr_match = re.search(r'<pr_description>(.*?)</pr_description>', content, re.DOTALL)
        if pr_match:
            task_description = pr_match.group(1).strip()
    
    formatted = f"""
INSTANCE ID: {instance_id}

TASK DESCRIPTION:
{task_description}

TRAJECTORY STEPS:
"""
    
    for step in steps:
        formatted += f"""
--- STEP {step['step_number']} ---
THOUGHT: {step['thought']}
COMMAND: {step['command']}
RETURN CODE: {step['returncode']}
OUTPUT: {step['output'][:500]}{'...' if len(step['output']) > 500 else ''}
"""
    
    return formatted


def call_judge(trajectory_content: str, model: str = "gpt-5-mini", api_key: str = None) -> Optional[Dict[str, Any]]:
    """Call the LLM judge to analyze the trajectory and return analysis + token/cost meta."""
    try:
        user_message = JUDGE_USER_MESSAGE.format(trajectory_content=trajectory_content)
        
        # Store the full messages for later reference
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": user_message}
        ]

        response = completion(
            model=model,
            messages=messages,
            api_key=api_key
        )

        # ---- Extract usage safely across providers ----
        usage = None
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
        elif hasattr(response, "usage"):
            usage = response.usage
        else:
            usage = {}

        # Normalize token counts
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        total_tokens = input_tokens + output_tokens

        # Cost calculation
        input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK
        output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MTOK
        total_cost = input_cost + output_cost

        # Extract the response content (for judge JSON)
        content = response["choices"][0]["message"]["content"] if isinstance(response, dict) \
            else response.choices[0].message.content

        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            try:
                analysis = json.loads(json_content)
            except json.JSONDecodeError as e:
                analysis = {
                    "error": f"Invalid JSON in code block: {str(e)}",
                    "raw_content": content,
                    "extracted_json": json_content
                }
        else:
            analysis = {
                "error": "No JSON code block found in response",
                "raw_content": content
            }

        return {
            "analysis": analysis,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "cost_usd": {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6)
            },
            "raw_judge_response": content,
            "judge_prompt": {
                "system": JUDGE_SYSTEM_MESSAGE,
                "user": user_message
            }
        }

    except Exception as e:
        return {
            "analysis": {"error": str(e), "raw_content": None},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost_usd": {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0},
            "raw_judge_response": None,
            "judge_prompt": None
        }


def majority_vote_steps(all_judge_responses: List[Dict[str, Any]], num_steps: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Take majority vote on redundancy for each step and randomly select other fields from majority bucket.
    
    Returns:
        - List of steps with majority voted fields
        - Metadata about voting process
    """
    voted_steps = []
    voting_metadata = {
        "total_steps": num_steps,
        "step_voting_details": []
    }
    
    for step_num in range(1, num_steps + 1):
        # Collect all judgments for this step across k calls
        step_judgments = []
        for judge_resp in all_judge_responses:
            analysis = judge_resp.get("analysis", {})
            if "error" in analysis:
                continue
            steps = analysis.get("steps", [])
            # Find the step with matching step_number
            for step in steps:
                if step.get("step_number") == step_num:
                    step_judgments.append(step)
                    break
        
        if not step_judgments:
            continue
        
        # Count redundant votes
        redundant_votes = [s.get("redundant", False) for s in step_judgments]
        redundant_counter = Counter(redundant_votes)
        majority_redundant = redundant_counter.most_common(1)[0][0]
        
        # Filter to majority bucket
        majority_bucket = [s for s in step_judgments if s.get("redundant") == majority_redundant]
        
        # Randomly select one from majority bucket
        selected_step = random.choice(majority_bucket)
        
        # Record voting details
        voting_detail = {
            "step_number": step_num,
            "total_votes": len(step_judgments),
            "redundant_votes": redundant_counter.get(True, 0),
            "essential_votes": redundant_counter.get(False, 0),
            "majority_decision": "redundant" if majority_redundant else "essential",
            "selected_from_call": step_judgments.index(selected_step) + 1
        }
        voting_metadata["step_voting_details"].append(voting_detail)
        
        voted_steps.append(selected_step)
    
    return voted_steps, voting_metadata


def aggregate_summary(voted_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from voted steps."""
    total_steps = len(voted_steps)
    redundant_steps = sum(1 for s in voted_steps if s.get("redundant", False))
    
    category_counts = {}
    action_type_counts = {"search": 0, "read": 0, "write": 0, "execute": 0, "env": 0, "other": 0}
    policy_violations = {}
    
    for step in voted_steps:
        # Count categories
        category = step.get("category", "")
        if category and category != "essential":
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count action types
        action_type = step.get("action_type", "")
        if action_type in action_type_counts:
            action_type_counts[action_type] += 1
        
        # Count policy violations
        for policy in step.get("violated_policies", []):
            policy_violations[policy] = policy_violations.get(policy, 0) + 1
    
    efficiency_score = (total_steps - redundant_steps) / total_steps if total_steps > 0 else 0.0
    
    return {
        "total_steps": total_steps,
        "redundant_steps": redundant_steps,
        "efficiency_score": round(efficiency_score, 4),
        "main_inefficiencies": sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True),
        "violated_policies_summary": policy_violations,
        "action_type_distribution": action_type_counts
    }


def process_single_trajectory(traj_file: Path, output_dir: Path, model: str, api_key: str, k: int) -> Dict[str, Any]:
    """Process a single trajectory file with k judge calls and majority voting."""
    # Load trajectory
    trajectory = load_trajectory(traj_file)
    if not trajectory:
        return None
    
    # Extract steps
    steps = extract_steps_from_trajectory(trajectory)
    if not steps:
        return None
    
    instance_id = trajectory.get('instance_id', traj_file.stem)
    
    # Format for judge
    trajectory_content = format_trajectory_for_judge(trajectory, steps)
    
    # Save judge prompt
    prompts_dir = output_dir / "judge_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / f"{instance_id}_prompt.json"
    with open(prompt_file, 'w') as f:
        json.dump({
            "instance_id": instance_id,
            "system_message": JUDGE_SYSTEM_MESSAGE,
            "user_message": JUDGE_USER_MESSAGE.format(trajectory_content=trajectory_content),
            "formatted_trajectory": trajectory_content
        }, f, indent=2)
    
    # Call judge k times
    all_judge_responses = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for i in range(k):
        judge_result = call_judge(trajectory_content, model, api_key)
        all_judge_responses.append(judge_result)
        
        # Accumulate tokens and costs
        usage = judge_result.get("usage", {})
        cost = judge_result.get("cost_usd", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_cost += cost.get("total_cost", 0.0)
    
    # Save all raw judge responses
    raw_responses_dir = output_dir / "raw_judge_responses"
    raw_responses_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_file = raw_responses_dir / f"{instance_id}_all_responses.json"
    with open(raw_responses_file, 'w') as f:
        json.dump({
            "instance_id": instance_id,
            "k": k,
            "responses": all_judge_responses
        }, f, indent=2)
    
    # Perform majority voting
    voted_steps, voting_metadata = majority_vote_steps(all_judge_responses, len(steps))
    
    # Generate summary from voted steps
    summary = aggregate_summary(voted_steps)
    
    # Build final judge response
    final_judge_response = {
        "trajectory_id": instance_id,
        "steps": voted_steps,
        "summary": summary
    }
    
    # --- Compute agent token wastage for redundant steps (snowball rule) ---
    usage_by_step = {
        s["step_number"]: (s.get("agent_token_usage") or {})
        for s in steps
    }

    wastage_records = []
    total_steps = len(steps)
    
    for step_report in voted_steps:
        sn = step_report.get("step_number")
        if not sn or not step_report.get("redundant"):
            continue

        cur = usage_by_step.get(sn, {})
        nxt = usage_by_step.get(sn + 1, {})

        cur_prompt = int(cur.get("prompt_tokens") or 0)
        cur_completion = int(cur.get("completion_tokens") or 0)
        nxt_prompt = int(nxt.get("prompt_tokens") or 0)

        num_subsequent_steps = total_steps - sn
        context_increase = max(0, nxt_prompt - cur_prompt)
        
        prompt_wasted = cur_prompt + (num_subsequent_steps * context_increase)
        completion_wasted = cur_completion
        tokens_wasted = prompt_wasted + completion_wasted

        wastage_records.append({
            "step_number": sn,
            "prompt_wasted": prompt_wasted,
            "completion_wasted": completion_wasted,
            "tokens_wasted": tokens_wasted,
            "num_subsequent_steps": num_subsequent_steps,
            "context_increase_per_step": context_increase
        })

    # Build result
    result = {
        "trajectory_file": str(traj_file),
        "instance_id": instance_id,
        "num_steps": len(steps),
        "judge_response": final_judge_response,
        "judge_token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "judge_cost_usd": {
            "input_cost": round((total_input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK, 6),
            "output_cost": round((total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MTOK, 6),
            "total_cost": round(total_cost, 6)
        },
        "majority_voting_metadata": voting_metadata,
        "k_calls": k,
        "agent_token_usage_by_step": [
            {
                "step_number": s["step_number"],
                "model": s.get("agent_token_usage", {}).get("model"),
                "prompt_tokens": s.get("agent_token_usage", {}).get("prompt_tokens", 0),
                "completion_tokens": s.get("agent_token_usage", {}).get("completion_tokens", 0),
                "total_tokens": s.get("agent_token_usage", {}).get("total_tokens", 0),
            } for s in steps
        ],
        "agent_token_wastage": wastage_records,
        "agent_token_wastage_totals": {
            "prompt_wasted": sum(w["prompt_wasted"] for w in wastage_records),
            "completion_wasted": sum(w["completion_wasted"] for w in wastage_records),
            "tokens_wasted": sum(w["tokens_wasted"] for w in wastage_records),
        },
        "timestamp": time.time()
    }

    # Save individual result
    result_file = output_dir / f"{instance_id}_judge_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def process_trajectories(results_dir: Path, output_dir: Path, limit: Optional[int] = None, 
                        model: str = "gpt-5-mini", api_key: str = None, max_workers: int = 5, k: int = 5):
    """Process all trajectories in parallel with k judge calls per trajectory."""
    
    # Find all trajectory files
    trajectory_files = list(results_dir.glob("*/*.traj.json"))
    
    if not trajectory_files:
        rich.print(f"[red]No trajectory files found in {results_dir}[/red]")
        return
    
    if limit:
        trajectory_files = trajectory_files[:limit]
    
    rich.print(f"[green]Found {len(trajectory_files)} trajectory files[/green]")
    rich.print(f"[cyan]Processing with {max_workers} parallel workers and k={k} judge calls per trajectory[/cyan]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Thread-safe accumulators
    lock = threading.Lock()
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_cost = 0.0
    total_agent_prompt_wasted = 0
    total_agent_completion_wasted = 0
    total_agent_tokens_wasted = 0
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        
        task = progress.add_task(f"Processing trajectories", total=len(trajectory_files))
        
        # Process trajectories in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_trajectory, traj_file, output_dir, model, api_key, k): traj_file
                for traj_file in trajectory_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                traj_file = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result:
                        # Thread-safe accumulation
                        with lock:
                            results.append(result)
                            
                            # Accumulate tokens and costs
                            judge_usage = result.get("judge_token_usage", {})
                            judge_cost = result.get("judge_cost_usd", {})
                            
                            total_input_tokens += judge_usage.get("input_tokens", 0)
                            total_output_tokens += judge_usage.get("output_tokens", 0)
                            total_tokens += judge_usage.get("total_tokens", 0)
                            total_input_cost += judge_cost.get("input_cost", 0.0)
                            total_output_cost += judge_cost.get("output_cost", 0.0)
                            total_cost += judge_cost.get("total_cost", 0.0)
                            
                            # Accumulate agent wastage
                            wastage_totals = result.get("agent_token_wastage_totals", {})
                            total_agent_prompt_wasted += wastage_totals.get("prompt_wasted", 0)
                            total_agent_completion_wasted += wastage_totals.get("completion_wasted", 0)
                            total_agent_tokens_wasted += wastage_totals.get("tokens_wasted", 0)
                    
                except Exception as e:
                    rich.print(f"[red]Error processing {traj_file.name}: {e}[/red]")
                
                progress.update(task, advance=1)
    
    # Generate summary statistics
    successful_results = [r for r in results if "error" not in r.get("judge_response", {})]
    failed_results = [r for r in results if "error" in r.get("judge_response", {})]
    
    # Aggregate statistics from successful results
    total_redundant_steps = 0
    category_counts = {}
    action_type_totals = {"search": 0, "read": 0, "write": 0, "execute": 0, "env": 0, "other": 0}
    efficiency_scores = []
    
    for result in successful_results:
        judge_resp = result.get("judge_response", {})
        summary = judge_resp.get("summary", {})
        
        if summary:
            total_redundant_steps += summary.get("redundant_steps", 0)
            
            if "efficiency_score" in summary:
                efficiency_scores.append(summary["efficiency_score"])
            
            for inefficiency in summary.get("main_inefficiencies", []):
                category_counts[inefficiency] = category_counts.get(inefficiency, 0) + 1
            
            action_dist = summary.get("action_type_distribution", {})
            for action_type, count in action_dist.items():
                if action_type in action_type_totals:
                    action_type_totals[action_type] += count
    
    summary = {
        "total_trajectories": len(results),
        "successful_analyses": len(successful_results),
        "failed_analyses": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "k_calls_per_trajectory": k,
        "total_judge_calls": len(results) * k,
        "total_redundant_steps": total_redundant_steps,
        "average_efficiency_score": sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0,
        "top_inefficiency_categories": sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "action_type_distribution": action_type_totals,        
        "agent_token_wastage_totals": {
            "prompt_wasted": total_agent_prompt_wasted,
            "completion_wasted": total_agent_completion_wasted,
            "tokens_wasted": total_agent_tokens_wasted
        },
        "token_usage_totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        },
        "cost_totals_usd": {
            "input_cost": round(total_input_cost, 6),
            "output_cost": round(total_output_cost, 6),
            "total_cost": round(total_cost, 6)
        },
        "pricing_per_1M_tokens_usd": {
            "input": INPUT_PRICE_PER_MTOK,
            "output": OUTPUT_PRICE_PER_MTOK
        }
    }
    
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    rich.print(f"\n[green]✅ Analysis complete![/green]")
    rich.print(f"  Total trajectories: {summary['total_trajectories']}")
    rich.print(f"  K calls per trajectory: {k}")
    rich.print(f"  Total judge API calls: {summary['total_judge_calls']}")
    rich.print(f"  Successful analyses: {summary['successful_analyses']}")
    rich.print(f"  Failed analyses: {summary['failed_analyses']}")
    rich.print(f"  Success rate: {summary['success_rate']:.1%}")
    rich.print(f"  Total redundant steps identified: {summary['total_redundant_steps']}")
    
    if efficiency_scores:
        rich.print(f"  Average efficiency score: {summary['average_efficiency_score']:.2f}")
    
    if summary['top_inefficiency_categories']:
        rich.print(f"\n[yellow]Top inefficiency categories:[/yellow]")
        for category, count in summary['top_inefficiency_categories'][:5]:
            rich.print(f"  - {category}: {count} occurrences")
    
    if any(summary['action_type_distribution'].values()):
        rich.print(f"\n[cyan]Action type distribution:[/cyan]")
        for action_type, count in summary['action_type_distribution'].items():
            if count > 0:
                rich.print(f"  - {action_type}: {count}")
    
    rich.print(f"\n[magenta]Agent token wastage (redundant steps):[/magenta]")
    rich.print(f"  Prompt wasted: {total_agent_prompt_wasted}")
    rich.print(f"  Completion wasted: {total_agent_completion_wasted}")
    rich.print(f"  Total wasted: {total_agent_tokens_wasted}")

    rich.print(f"\n[magenta]Judge token usage:[/magenta]")
    rich.print(f"  Input tokens: {total_input_tokens}")
    rich.print(f"  Output tokens: {total_output_tokens}")
    rich.print(f"  Total tokens: {total_tokens}")

    rich.print(f"\n[magenta]Estimated cost (USD):[/magenta]")
    rich.print(f"  Input cost: ${total_input_cost:.6f}")
    rich.print(f"  Output cost: ${total_output_cost:.6f}")
    rich.print(f"  Total cost: ${total_cost:.6f}")

    rich.print(f"\n[blue]Results saved to: {output_dir}[/blue]")
    rich.print(f"[blue]Judge prompts saved to: {output_dir}/judge_prompts/[/blue]")
    rich.print(f"[blue]Raw judge responses saved to: {output_dir}/raw_judge_responses/[/blue]")


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on mini SWE-agent trajectories with majority voting!")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/usr0/home/srgandhi/tool-overuse/results/base_dev",
        help="Directory containing trajectory subdirectories (default: results_sample)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        required=True,
        help="Directory to save judge results"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of trajectories to process (for testing)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model to use for analysis (default: gpt-5-mini)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or use OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of judge calls per trajectory for majority voting (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        rich.print("[red]Error: API key required (--api-key or OPENAI_API_KEY env var)[/red]")
        return 1
    
    # Validate directories
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        rich.print(f"[red]Results directory not found: {results_dir}[/red]")
        return 1
    
    output_dir = Path(args.output_dir)
    
    # Run analysis
    rich.print("[green]Starting mini SWE-agent trajectory analysis with majority voting...[/green]")
    rich.print(f"  Results directory: {results_dir}")
    rich.print(f"  Output directory: {output_dir}")
    rich.print(f"  Model: {args.model}")
    rich.print(f"  K (judge calls per trajectory): {args.k}")
    rich.print(f"  Limit: {args.limit or 'None'}")
    rich.print(f"  Max workers: {args.max_workers}")
    
    try:
        process_trajectories(
            results_dir=results_dir,
            output_dir=output_dir,
            limit=args.limit,
            model=args.model,
            api_key=api_key,
            max_workers=args.max_workers,
            k=args.k
        )
        return 0
    except Exception as e:
        rich.print(f"[red]Error during processing: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())