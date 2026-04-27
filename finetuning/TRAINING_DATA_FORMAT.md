# PRM SFT Training Data Format

## Overview

Training data for distilling the Claude Opus PRM (Process Reward Model / supervisor)
into Qwen3-8B. Each sample teaches the model to produce structured supervisor feedback
given an agent's trajectory.

**Script:** `prepare_prm_sft_data_opus_distill_full_feedback_history.py`

**Source data:** Trajectory files from PRM-enhanced SWE-bench runs where Claude Opus
acted as the supervisor, stored in
`results_singularity/singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6/`.

## Sample Structure

Each training sample is a 3-message conversation in LlamaFactory sharegpt format:

```
{
  "messages": [
    {"role": "system",    "content": "<PRM system prompt>"},
    {"role": "user",      "content": "<trajectory context>"},
    {"role": "assistant", "content": "<PRM feedback — training target>"}
  ]
}
```

### Message 1: System (~1,050 tokens)

The PRM's system prompt — identical across all samples. Defines the 12 error
categories (Specification, Reasoning, Coordination errors) and the structured
response format (DETECTED/EVIDENCE/RECOVERY_ACTION per category, then
TASK_STATUS and OVERALL_GUIDANCE).

### Message 2: User (variable, bulk of tokens)

The user message contains the full context the PRM needs to evaluate. It is
composed of tagged blocks in this order:

```
[USER/ENVIRONMENT]: ## Original Task Given to Agent: ...     ← task context (always kept)

[...N earlier trajectory step(s) omitted...]                 ← only if truncated

[AGENT RESPONSE]: I need to fix the function...              ← agent's action/reasoning
[USER/ENVIRONMENT]: <returncode>0</returncode> <output>...   ← environment response
[AGENT RESPONSE]: The test failed because...                 ← next agent step
[USER/ENVIRONMENT]: <returncode>1</returncode> <output>...   ← next env response
...                                                          ← more trajectory steps

[...M earlier supervisor feedback(s) omitted...]             ← only if truncated

[PREVIOUS_FEEDBACK #N-2]: SPECIFICATION ERRORS: ...          ← 3rd most recent feedback
[PREVIOUS_FEEDBACK #N-1]: SPECIFICATION ERRORS: ...          ← 2nd most recent feedback
[PREVIOUS_FEEDBACK #N]:   SPECIFICATION ERRORS: ...          ← most recent feedback

[USER/ENVIRONMENT]: Please provide your supervisor feedback now based on the trajectory above.
```

**Block categories:**
- `task` — The original SWE-bench issue description (always preserved in full)
- `trajectory` — Alternating `[AGENT RESPONSE]` and `[USER/ENVIRONMENT]` blocks
  representing the coding agent's actions and environment outputs
- `feedback` — `[PREVIOUS_FEEDBACK #N]` blocks containing the PRM's own prior
  feedback from earlier invocations (ordered oldest → newest)
- `prompt` — Final instruction asking for supervisor feedback (always preserved)

### Message 3: Assistant (training target, ~500-2,000 tokens)

The PRM's structured feedback. Always preserved in full. Example:

```
SPECIFICATION ERRORS:
1. Task Specification Violations: DETECTED: No
2. Role Specification Violations: DETECTED: No
3. Step Repetition: DETECTED: Yes
EVIDENCE: The agent has gone through the same cycle three times...
RECOVERY_ACTION: Stop repeating the same sed approach...
4. Termination Condition Unawareness: DETECTED: No

REASONING ERRORS:
5. Problem Misidentification: DETECTED: No
6. Tool Selection Errors: DETECTED: Yes
EVIDENCE: Agent used sed for multi-line replacements...
RECOVERY_ACTION: Use a Python script for complex file edits...
7. Hallucinations: DETECTED: No
8. Information Processing Failures: DETECTED: No

COORDINATION ERRORS:
9. Task Derailment: DETECTED: No
10. Goal Deviation: DETECTED: No
11. Context Handling Failures: DETECTED: No
12. Verification Failures: DETECTED: Yes
EVIDENCE: Agent did not verify the file contents after each edit...
RECOVERY_ACTION: Always cat the modified file after editing...

TASK_STATUS: Needs correction
OVERALL_GUIDANCE: You are stuck in a loop repeating the same failing approach...
```

## Truncation Strategy

When a sample exceeds `--max_tokens` (e.g., 32768 for 32k training), smart
truncation is applied with these principles:

### What is NEVER truncated
- System prompt (message 1)
- Task context (first `[USER/ENVIRONMENT]` block)
- Final prompt ("Please provide your supervisor feedback now...")
- PRM response (message 3 — training target)

### Truncation rules
1. **Feedback cap (`--max_feedbacks`):** Before any token budgeting, past
   feedbacks are capped to the N most recent. E.g., `--max_feedbacks 5` keeps
   only feedbacks #(K-4) through #K, discarding older ones entirely. This
   prevents feedback history from crowding out trajectory context.

2. **Budget split (`--feedback_budget_ratio`):** The remaining token budget
   (after fixed portions) is split between trajectory and feedback. Default is
   30% feedback / 70% trajectory.

3. **Drop oldest first:** Both trajectory steps and feedbacks are dropped from
   the OLDEST end. Most recent steps and feedbacks are always prioritized.

4. **Never mid-cut:** Entire blocks are kept or dropped — no block is ever
   cut in the middle. If adding the next block would exceed budget, it is
   skipped entirely.

5. **Surplus reallocation:** If one category (trajectory or feedback) is under
   budget, its surplus is given to the other category.

### Example: before and after truncation

**Before** (99,839 tokens, 14 feedbacks, 150 trajectory blocks):
```
[task]           1,500 tokens
[trajectory x150] 85,000 tokens total
[feedback x14]   12,000 tokens total
[prompt]            18 tokens
[assistant]       1,321 tokens
```

**After** (`--max_tokens 32768 --max_feedbacks 5 --feedback_budget_ratio 0.3`):
```
[task]            1,500 tokens    ← preserved
[...90 earlier trajectory step(s) omitted...]
[trajectory x16] ~21,000 tokens  ← 16 most recent steps kept (70% budget)
[...9 earlier supervisor feedback(s) omitted...]
[feedback x5]    ~8,500 tokens   ← 5 most recent feedbacks (30% budget)
[prompt]             18 tokens    ← preserved
[assistant]       1,321 tokens    ← preserved (training target)
TOTAL:           ~32,339 tokens   ← within budget
```

## Data Generation

### Invocation pattern

The PRM is invoked every `prm_interval` agent steps (default: 5). So:
- Invocation 1 (after step 5): sees 5 trajectory steps, 0 past feedbacks
- Invocation 2 (after step 10): sees 10 trajectory steps, 1 past feedback
- Invocation N (after step 5N): sees 5N trajectory steps, N-1 past feedbacks

### Erroneous sample filtering

The `is_valid_prm_feedback()` function filters out samples where the "Current
Feedback" section of a supervisor message erroneously contained an agent
response instead of actual PRM feedback. Valid PRM feedback must contain
structural markers like `DETECTED:`, `SPECIFICATION ERRORS`, `TASK_STATUS:`,
etc.

### Usage

```bash
# Generate 32k training data with max 5 past feedbacks, 30% feedback budget
conda run -n tool-overuse python3 finetuning/prepare_prm_sft_data_opus_distill_full_feedback_history.py \
    --max_tokens 32768 \
    --max_feedbacks 5 \
    --feedback_budget_ratio 0.3 \
    --output_dir finetuning/prm_sft_data_opus_distill_full_feedback_history_32k

# Generate without truncation (for analysis)
conda run -n tool-overuse python3 finetuning/prepare_prm_sft_data_opus_distill_full_feedback_history.py \
    --output_dir finetuning/prm_sft_data_opus_distill_full_feedback_history_raw
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_tokens` | 0 (no limit) | Token budget per sample |
| `--max_feedbacks` | 0 (no cap) | Max past feedbacks to include |
| `--feedback_budget_ratio` | 0.3 | Fraction of variable budget for feedback |
| `--tokenizer` | Qwen/Qwen3-8B | Tokenizer for token counting |
| `--train_ratio` | 0.9 | Train/val split ratio |
