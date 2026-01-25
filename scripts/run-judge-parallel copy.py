#!/usr/bin/env python3
"""
Judge script for analyzing tool overuse in mini SWE-agent trajectories.
Performs step-wise analysis of each action to identify redundant tool usage.
NOW WITH PARALLEL PROCESSING!
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import rich
from litellm import completion
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# --- Pricing (per 1M tokens) ---
INPUT_PRICE_PER_MTOK = 0.250   # $ per 1M input tokens
OUTPUT_PRICE_PER_MTOK = 2.000  # $ per 1M output tokens

JUDGE_SYSTEM_MESSAGE = """You are an expert code assistant analyzer specializing in detecting tool overuse patterns in AI agent trajectories. You will analyze each step of an agent's interaction with a shell environment to identify redundant, unnecessary, or inefficient actions.

You are analyzing a mini SWE-agent that solves programming tasks by executing bash commands. The agent is given a task description and iteratively executes commands to understand the codebase, reproduce issues, and implement fixes.

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

4. **REASONING**: Detailed explanation of why this action is redundant or essential.

5. **CORRECTION**: If redundant, specify "skip", "modify" or "combine" to indicate how this could be optimized. "skip" means skipping the step entirely, "modify" means modifying the step and "combine" means combining multiple suboptimal steps. If not redundant, use null.

# TOOL OVERUSE CATEGORIES

1. **Information Redundancy**
Actions that re-acquire already available information without justification.
* `duplicate_read`: Reading the same file/content without intermediate external modifications. This also includes viewing specific parts of a file if that part has been read before even if the agent explicitly made changes to the file. This also includes re-reading overlapping parts of the same file without any modifications having been made.
* `duplicate_search`: Repeating identical searches or listings
* `duplicate_status_check`: Re-checking unchanged system state (e.g. git status, pwd, etc.)

2. **Inefficient Information Access**
Actions that gather information sub-optimally, even if not strictly redundant.
* `overly_broad_read`: Reading entire files when specific sections needed (e.g., reading 500 lines for one function)
* `inefficient_search_strategy`: Using less effective search methods or Using multiple commands when one would suffice (e.g., multiple finds instead of grep -r)
* `unfocused_browsing`: Exploring without clear purpose or hypothesis
* `tangential_exploration`: Investigating areas unrelated to the task

3. **Premature or Mistimed Actions**
Actions performed at inappropriate points in the workflow.
* `premature_submit`: Submitting changes before verification

4. **Failed Strategy Persistence**
Continuing approaches despite clear failure signals.
* `repeated_failed_command`: Re-running commands that failed without addressing the cause
* `circular_debugging`: Repeating same debug cycle without progress

5. **Unnecessary Verification**
Validation actions that don't add value given the context.
* `obvious_confirmation`: Verifying trivially true conditions
* `excessive_intermediate_testing`: Over-testing during implementation
* `redundant_success_validation`: Re-confirming already validated successes

# ANALYSIS CRITERIA

**Mark as REDUNDANT if:**
- The action repeats information gathering without new insights
- The command output was already available from previous steps
- The action doesn't advance task completion
- A more efficient alternative was clearly available
- The action ignores context from previous steps
- The action explores blindly without a hypothesis

**Mark as ESSENTIAL if:**
- The action provides new information needed for task completion
- The action represents logical task progression
- The context has changed since similar previous actions
- The action is part of a debugging or verification process with clear purpose
- The action is targeted based on previous findings

# OUTPUT FORMAT

You MUST respond with your analysis in a JSON code block using triple backticks. Provide the JSON in the following exact structure:

```json
{
  "trajectory_id": "instance_id_here",
  "steps": [
    {
      "step_number": 1,
      "command": "find . -name '*.py' | grep -E '(test_|_test\\.py)' | head -20",
      "action_type": "search",
      "redundant": false,
      "category": "essential",
      "reasoning": "Targeted search for test files is necessary to understand testing structure and locate relevant tests for the issue.",
      "correction": null
    },
    {
      "step_number": 2,
      "command": "ls -la",
      "action_type": "search",
      "redundant": true,
      "category": "unfocused_browsing",
      "reasoning": "Broad directory listing after already finding specific test files. This explores the entire directory without a clear hypothesis, potentially overwhelming context. Should instead focus on the specific test files found.",
      "correction": "skip"
    }
  ],
  "summary": {
    "total_steps": 2,
    "redundant_steps": 1,
    "efficiency_score": 0.5,
    "main_inefficiencies": ["unfocused_browsing"],
    "action_type_distribution": {
      "search": 2,
      "read": 0,
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
4. Ensure all step numbers match the actual steps in the trajectory
5. Calculate efficiency_score as: (total_steps - redundant_steps) / total_steps
6. List only the categories that actually appear in redundant steps in main_inefficiencies
7. Ensure action_type_distribution sums match the actual number of steps analyzed"""

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
            api_key=api_key,
            reasoning_effort='minimal'
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


def process_single_trajectory(traj_file: Path, output_dir: Path, model: str, api_key: str) -> Dict[str, Any]:
    """Process a single trajectory file. Returns result dict or None on error."""
    # Load trajectory
    trajectory = load_trajectory(traj_file)
    if not trajectory:
        return None
    
    # Extract steps
    steps = extract_steps_from_trajectory(trajectory)
    if not steps:
        return None
    
    # Format for judge
    trajectory_content = format_trajectory_for_judge(trajectory, steps)
    
    # Call judge
    judge = call_judge(trajectory_content, model, api_key)
    judge_response = judge.get("analysis", {})
    judge_usage = judge.get("usage", {})
    judge_cost = judge.get("cost_usd", {})
    raw_judge_response = judge.get("raw_judge_response")
    judge_prompt = judge.get("judge_prompt")

    # --- Compute agent token wastage for redundant steps (snowball rule) ---
    usage_by_step = {
        s["step_number"]: (s.get("agent_token_usage") or {})
        for s in steps
    }

    wastage_records = []
    total_steps = len(steps)
    
    for step_report in (judge_response.get("steps") or []):
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
        "instance_id": trajectory.get('instance_id', traj_file.stem),
        "num_steps": len(steps),
        "judge_response": judge_response,
        "judge_token_usage": judge_usage,
        "judge_cost_usd": judge_cost,
        "raw_judge_response": raw_judge_response,
        "judge_prompt": judge_prompt,
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
    result_file = output_dir / f"{result['instance_id']}_judge_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def process_trajectories(results_dir: Path, output_dir: Path, limit: Optional[int] = None, 
                        model: str = "gpt-5-mini", api_key: str = None, max_workers: int = 5):
    """Process all trajectories in parallel."""
    
    # Find all trajectory files
    trajectory_files = list(results_dir.glob("*/*.traj.json"))
    
    if not trajectory_files:
        rich.print(f"[red]No trajectory files found in {results_dir}[/red]")
        return
    
    if limit:
        trajectory_files = trajectory_files[:limit]
    
    rich.print(f"[green]Found {len(trajectory_files)} trajectory files[/green]")
    rich.print(f"[cyan]Processing with {max_workers} parallel workers[/cyan]")
    
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
                executor.submit(process_single_trajectory, traj_file, output_dir, model, api_key): traj_file
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
    
    # Save all results
    all_results_file = output_dir / "all_judge_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
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


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on mini SWE-agent trajectories (with parallelism!)")
    
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
        default=5,
        help="Maximum number of parallel workers (default: 5)"
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
    rich.print("[green]Starting mini SWE-agent trajectory analysis (parallel mode)...[/green]")
    rich.print(f"  Results directory: {results_dir}")
    rich.print(f"  Output directory: {output_dir}")
    rich.print(f"  Model: {args.model}")
    rich.print(f"  Limit: {args.limit or 'None'}")
    rich.print(f"  Max workers: {args.max_workers}")
    
    try:
        process_trajectories(
            results_dir=results_dir,
            output_dir=output_dir,
            limit=args.limit,
            model=args.model,
            api_key=api_key,
            max_workers=args.max_workers
        )
        return 0
    except Exception as e:
        rich.print(f"[red]Error during processing: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())