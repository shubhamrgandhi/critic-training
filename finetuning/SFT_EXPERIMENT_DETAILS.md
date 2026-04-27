# Details for SFT Experiments

## Source of Data

- **Agent model:** CWM (facebook/cwm)
- **Critic model:** Claude Opus 4.6 (`us.anthropic.claude-opus-4-6-v1`)
- **Benchmark:** SWE-Bench Verified (500 instances)
- **Feedback type:** Issue resolution feedback (`prm_issue_res`), invoked every 5 agent steps
- **Max agent steps:** 75
- **Result:** 305 / 500 instances resolved with CWM + Claude Opus critic

## Data Processing Pipeline

| Stage | Samples | Notes |
|-------|---------|-------|
| Raw feedback samples | 2573 | From all 500 runs |
| After denoising | 2393 | Filtered malformed samples (agent responses in feedback slot) |
| Rejection sampling | 1303 | Only from 305 resolved instances |

**Denoising:** Some feedback samples had the agent's response erroneously placed in the critic's "Current Feedback" section instead of actual structured PRM output. These were filtered using structural markers (presence of `DETECTED:`, `SPECIFICATION ERRORS`, `TASK_STATUS:`, etc.).

**Rejection sampling:** Only feedback samples from instances where the agent successfully resolved the issue were retained, under the assumption that feedback from successful trajectories is higher quality.

## Sample Format

Each training sample is a 3-message conversation (LlamaFactory sharegpt format):

1. **System** (~1,050 tokens): PRM system prompt defining 12 error categories (Specification, Reasoning, Coordination errors) and structured response format
2. **User** (variable): Concatenation of:
   - Original task/issue description
   - Agent trajectory (alternating agent actions and environment responses)
   - Up to 5 most recent previous PRM feedbacks
   - Final prompt requesting supervisor feedback
3. **Assistant** (training target, ~500-2,000 tokens): Structured feedback with DETECTED/EVIDENCE/RECOVERY_ACTION per error category, TASK_STATUS, and OVERALL_GUIDANCE

## Truncation Strategy (32k token budget)

- Never truncates: system prompt, task context, final prompt, PRM response
- Caps past feedbacks to 5 most recent
- Budget split: 70% trajectory / 30% feedback history
- Drops oldest trajectory steps and oldest feedbacks first (prioritizes recency)
- Never cuts mid-block (entire steps kept or dropped)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-8B |
| Fine-tuning type | Full (all 8.19B params) |
| Precision | pure bf16 |
| Max sequence length | 32,768 tokens |
| Batch size | 1 per device |
| GPUs | 8x NVIDIA L40S (46GB each) |
| Distributed strategy | FSDP (FULL_SHARD) |
| Learning rate | 5e-6 |
| LR scheduler | Cosine with 10% warmup |
| Epochs | 3 |
| Optimizer | AdamW (fused) |
| Gradient checkpointing | Enabled |
| Liger kernel | Enabled |
| Template | qwen3_nothink (thinking disabled) |
| Framework | LlamaFactory |

## Training Runs

| Run | Data | Samples | Steps/epoch | Total steps |
|-----|------|---------|-------------|-------------|
| `_clean` | All denoised | 2393 | ~300 | ~900 |
| `_rejection-sample` | Resolved only | 1303 | ~163 | ~489 |
