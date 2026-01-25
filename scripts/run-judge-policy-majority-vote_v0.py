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

JUDGE_SYSTEM_MESSAGE = """You are an expert code assistant analyzer specializing in detecting tool overuse patterns in AI agent trajectories. You will analyze each step of an agent's interaction with a shell environment to identify redundant, unnecessary, or inefficient actions.

You are analyzing a mini SWE-agent that solves programming tasks by executing bash commands. The agent is given a task description and iteratively executes commands to understand the codebase, reproduce issues, and implement fixes.

# EFFICIENCY POLICIES

When analyzing actions, you MUST check against these specific policies. When marking an action as redundant, cite the relevant policy numbers in your reasoning.

## 1. Information Redundancy Policies

**Policy 1.1 (No Duplicate Content Retrieval)**: Never re-read content that was already successfully retrieved in a previous step. The agent should maintain awareness of file states based on its own actions:
  - If the agent read a file and then made edits to it, the agent should know the new state without re-reading
  - Re-reading is only acceptable if an external process modified the file (which is rare in isolated agent environments)
  - Example violations: Running `cat file.py` twice without any edits; reading a file, making edits via the agent's own commands, then re-reading to "verify" the changes; reading lines 100-200 then reading lines 150-250 of the same file

**Policy 1.2 (No Repeated Searches)**: Never repeat identical or substantially similar searches without:
  - Changing search parameters (different query, different directory, different file pattern)
  - New information suggesting different results
  - Example violations: Running `find . -name "*.py"` multiple times; running `grep "function_name" file.py` twice

**Policy 1.3 (No Redundant State Checks)**: Never re-check system state unless intervening actions could have modified it:
  - Example violations: Running `git status` multiple times without intermediate git operations; checking file line count (`wc -l`) without intermediate writes; running `pwd` repeatedly

## 2. Inefficient Information Access Policies

**Policy 2.1 (Prefer Targeted Access)**: Always prefer focused, targeted information retrieval over broad retrieval when specific information is needed:
  - When looking for specific functions/classes, use search tools first, then read specific sections
  - Never read entire large files (>100 lines) when searching for specific elements
  - Example violations: Running `cat large_file.py` (500+ lines) to find one function; using `cat` when `grep` would suffice

**Policy 2.2 (Command Consolidation)**: Consolidate multiple similar operations into a single efficient command when possible:
  - Use pattern matching, wildcards, or logical operators to combine searches
  - Example violations: Running `grep "class A"` then `grep "class B"` then `grep "class C"` separately instead of `grep "class \(A\|B\|C\)"`

**Policy 2.3 (Focused Directory Exploration)**: Use targeted exploration with specific patterns rather than broad listings:
  - Example violations: Running `ls -la` on large directories without filtering; listing all files when searching for specific patterns

**Policy 2.4 (Relevant Scope)**: Only explore directories and files relevant to the stated task:
  - If the task specifies a location or component, focus exploration there
  - Example violations: Exploring unrelated directories when the task clearly indicates a specific file or module

## 3. Premature or Mistimed Actions Policies

**Policy 3.1 (Verify Before Submit)**: Always verify changes through testing or validation before final submission:
  - Run relevant tests after making fixes
  - Example violations: Submitting solution immediately after editing without running tests

## 4. Failed Strategy Persistence Policies

**Policy 4.1 (No Unmodified Retries)**: Never retry failed commands without addressing the root cause:
  - If a command fails, analyze the error and modify the approach
  - Example violations: Re-running the same command with syntax errors; retrying failed test without changing implementation

**Policy 4.2 (Adapt Debugging Strategy)**: When a debugging approach fails repeatedly, change the strategy:
  - If the same fix attempt fails multiple times, try a different approach
  - Example violations: Applying the same fix >2 times despite continued test failures

## 5. Unnecessary Verification Policies

**Policy 5.1 (No Obvious Confirmations)**: Skip verification of states that are logically certain from previous operations:
  - If you just performed an action with successful output, don't immediately verify it
  - Example violations: Checking `git diff` when previous operations showed no changes; verifying file exists immediately after creating it with successful return code

**Policy 5.2 (Meaningful and Comprehensive Testing)**: Only run tests after meaningful changes, and create comprehensive tests rather than incremental ones:
  - Don't test after adding comments, docstrings, or whitespace-only changes
  - Don't create tests incrementally across multiple steps when they could be written comprehensively in one step
  - If testing multiple edge cases or scenarios, write all test cases together and run once, rather than writing and running tests for each case separately
  - Example violations: Running full test suite after only adding documentation; creating 10 separate test functions across 10 steps for 10 edge cases instead of creating all 10 test cases in one step; running tests after each individual test case addition instead of after writing the complete test suite

**Policy 5.3 (No Redundant Success Validation)**: Never re-test already passing tests without implementation changes:
  - If tests passed and no code changes were made, they will still pass
  - Example violations: Re-running the same passing test multiple times without code modifications

# ANALYSIS FRAMEWORK

For each step in the trajectory, you must determine:

1. **ACTION_TYPE**: Classify the action into one of: {search, read, write, execute, env, other}
   - `search`: Finding files/content (e.g. find, ls, grep -r, rg)
   - `read`: Viewing/extracting information (e.g. cat, less, head, tail, grep on specific files)
   - `write`: Modifying files (e.g. echo >, sed -i, vim, nano)
   - `execute`: Running scripts/tests/programs (e.g. python, npm test, pytest, ./script.sh)
   - `env`: Environment setup (e.g. pip install, apt-get, export, cd, mkdir)
   - `other`: Specify what type if not above categories (add specification in parentheses, e.g., "other (git operations)")

2. **REDUNDANT**: true/false - Is this action redundant or unnecessary?

3. **CATEGORY**: If redundant, provide a specific category from the taxonomy below. If not redundant, use "essential".

4. **VIOLATED_POLICIES**: If redundant, list the policy numbers that were violated (e.g., ["1.1", "2.1"]). If not redundant, use empty array [].

5. **REASONING**: Detailed explanation citing specific policies. Format: "Violates Policy X.Y: [explanation of how this specific action violates the policy]"

6. **CORRECTION**: If redundant, specify "skip", "modify" or "combine" to indicate how this could be optimized. If not redundant, use null.

# TOOL OVERUSE CATEGORIES

1. **Information Redundancy**
   - `duplicate_read`: Violates Policy 1.1
   - `duplicate_search`: Violates Policy 1.2
   - `duplicate_status_check`: Violates Policy 1.3

2. **Inefficient Information Access**
   - `overly_broad_read`: Violates Policy 2.1
   - `inefficient_search_strategy`: Violates Policy 2.2
   - `unfocused_browsing`: Violates Policy 2.3
   - `tangential_exploration`: Violates Policy 2.4

3. **Premature or Mistimed Actions**
   - `premature_submit`: Violates Policy 3.1

4. **Failed Strategy Persistence**
   - `repeated_failed_command`: Violates Policy 4.1
   - `circular_debugging`: Violates Policy 4.2

5. **Unnecessary Verification**
   - `obvious_confirmation`: Violates Policy 5.1
   - `excessive_intermediate_testing`: Violates Policy 5.2 (includes both testing after trivial changes AND creating tests incrementally instead of comprehensively)
   - `redundant_success_validation`: Violates Policy 5.3

# ANALYSIS CRITERIA

**Mark as REDUNDANT if:**
- The action violates one or more of the efficiency policies above
- You can clearly cite which policy(ies) are violated
- The violation demonstrably does not advance task completion

**Mark as ESSENTIAL if:**
- The action complies with all relevant policies
- The action provides new information needed for task completion
- The action represents logical task progression based on current context

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
      "violated_policies": ["2.1"],
      "reasoning": "Violates Policy 2.1 (Prefer Targeted Access): Reads entire 500-line file when searching for a specific function. Should use grep to find the function first, then read only relevant sections.",
      "correction": "modify"
    },
    {
      "step_number": 2,
      "command": "grep -n 'def process_data' large_file.py",
      "action_type": "search",
      "redundant": false,
      "category": "essential",
      "violated_policies": [],
      "reasoning": "Complies with Policy 2.1: Uses targeted search to locate specific function before reading. Efficient information access that provides necessary line numbers for focused reading.",
      "correction": null
    }
  ],
  "summary": {
    "total_steps": 2,
    "redundant_steps": 1,
    "efficiency_score": 0.5,
    "main_inefficiencies": ["overly_broad_read"],
    "violated_policies_summary": {
      "2.1": 1
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

1. Output your response in a JSON code block using triple backticks (```json ... ```)
2. Use boolean values (true/false) not strings for the "redundant" field
3. Use null (not "null" string) for correction when action is essential
4. Use empty array [] for violated_policies when action is essential
5. ALWAYS cite specific policy numbers in reasoning using format "Violates Policy X.Y: ..."
6. Ensure all step numbers match the actual steps in the trajectory
7. Calculate efficiency_score as: (total_steps - redundant_steps) / total_steps
8. Include violated_policies_summary showing count of each policy violation
9. Be strict in applying policies - if a policy is violated, mark as redundant"""

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