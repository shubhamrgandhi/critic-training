# finetuning/

Critic SFT pipeline. Qwen3-8B base, full-parameter SFT on 8x L40S via LlamaFactory + accelerate FSDP. Training data is per-step critiques of agent trajectories from a strong teacher (Opus 4.6).

> The trained model is referred to as the "critic" in this guide. Files, output dirs, dataset dirs, and YAML keys still contain `prm`/`PRM` (process reward model — the same concept). `prm`-named paths are not renamed since recipients may already have references to those paths.

## Files

### Training (resumable — recommended)

| File | Purpose |
|---|---|
| `train_sbatch_resumable.sh` | sbatch wrapper (47:30h walltime, self-resubmits via `afterany`, runs training in tmux, pushes to HF Hub on clean exit) |
| `run_qwen3_8b_critic_full_sft_l40s_resumable.sh` | Inner launcher. Activates `critic-training` conda env, auto-detects latest `checkpoint-*` and resumes via `--resume_from_checkpoint`, calls `python -m accelerate.commands.launch`. |
| `qwen3_8b_critic_full_sft_l40s_train_multiturn_resumable.yaml` | LlamaFactory recipe. `save_strategy: steps`, `save_steps: 200`, `save_total_limit: 2`, `save_only_model: false` (full optimizer state for resume). |
| `fsdp_full_sft_config.yaml` | Accelerate FSDP config (FULL_SHARD, BACKWARD_PRE prefetch, bf16). |

### Training (legacy)

| File | Purpose |
|---|---|
| `run_qwen3_8b_prm_full_sft_l40s.sh` | Original launcher. `FORMAT=multiturn` (default) or `FORMAT=flattened`. Expects you to submit from an already-activated conda shell. |
| `run_qwen3_8b_prm_lora_sft.sh` | LoRA variant for resource-constrained runs (single-GPU, 4-bit QLoRA). |
| `train_sbatch.sh` | Sbatch wrapper for the original launcher (12h walltime, no resume). |
| `qwen3_8b_prm_full_sft_l40s_train_multiturn.yaml` | LlamaFactory recipe — multiturn, `mask_history: true`, `qwen3_nothink` template. |
| `qwen3_8b_prm_full_sft_l40s_train.yaml` | LlamaFactory recipe — flattened (single-turn). |
| `qwen3_8b_prm_lora_sft_train.yaml` | LlamaFactory recipe — LoRA (single-turn). |

### Data prep

| File | Purpose |
|---|---|
| `prepare_prm_sft_data_opus_distill_full_feedback_history.py` | Takes a single raw agent trajectory dir, runs Opus 4.6 as judge for per-step critiques, writes a LlamaFactory-format JSONL. For combined datasets (mixed agent rollouts), run twice and concatenate the resulting `prm_sft_train.jsonl` files. |
| `debug_data_quality.py` | Sanity checks on a generated dataset (length distributions, format compliance, echo patterns). |

### Eval

The eval flow is two stages: generate responses with each model, then score them against the teacher's ground truth.

| File | Purpose |
|---|---|
| `eval_generate_responses.py` | Step 1: generate critic responses for a model via vLLM (OpenAI-compatible). Resumable; saves to `eval_results/<model-label>/responses.jsonl`. |
| `eval_compute_scores.py` | Step 2: BLEU-1/2/4 and ROUGE-1/2/L for two models' responses against the teacher ground truth. |
| `eval_error_classification.py` | Per-class precision/recall/F1 over critic error categories, plus plots. |
| `eval_intrusiveness.py` | Format compliance, alignment, and intervention-intensity analysis vs. ground truth. Produces plots + REPORT.md. |
| `eval_length_analysis.py` | Response-length distribution and length-vs-judge-win-rate analysis. |

All eval scripts use the OpenAI client (`from openai import OpenAI` + `--api-base`). No AWS/Bedrock dependencies — point them at any OpenAI-compatible endpoint (vLLM on Babel, the CMU LiteLLM gateway, etc.).

`eval_results/` holds the post-training generations and scored outputs for each variant we trained.

## Training data dirs

All three are also published on Hugging Face — fetch with `huggingface_hub.snapshot_download(...)` if you don't have Babel filesystem access.

| Local dir | HF dataset | Description |
|---|---|---|
| `prm_sft_r2egym_swebench_k5_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-k5-opus-distill-32k-multiturn) | k=5 intervention frequency, **detailed** prompt (no `instructions` style), R2E-Gym + SWE-bench Verified CWM rollouts critiqued by Opus 4.6. 3,135 samples. |
| `prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn) | k=5, **concise** ("instructions") prompt. CWM-only rollouts. 4,532 samples. **Headline single-agent critic training data.** |
| `prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-cwm-plus-qwen-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-cwm-plus-qwen-k5-opus-distill-32k-multiturn) | k=5, concise prompt, mixed teacher trajectories (CWM + Qwen3-Next-80B-Instruct agents). 6,447 samples (4,532 CWM + 1,915 Qwen3-Next-80B). |

Each contains `prm_sft_train.jsonl`, `dataset_info.json`, and a `metadata.json` with provenance. To use one for training, point `DATA_DIR` at it before running the launcher.

To fetch from HF instead:
```bash
huggingface-cli download --repo-type dataset \
    shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn \
    --local-dir prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn
```

## Naming conventions

- `instructions` = concise critic prompt that constrains critique to a short rubric.
- absence of `instructions` = detailed (open-ended) critic prompt.
- `k5` / `k10` = critic intervention frequency in agent steps.
- `multiturn` = each agent step is its own assistant turn (with `mask_history: true`); the model sees full prior context but is only trained on the current critique.
- `flattened` = single-turn format with the full trajectory in the user input.
- `32k` = max sequence length used during data prep.
- `r2egym_swebench` = trajectories sourced from both R2E-Gym and SWE-bench Verified.

## Quickstart

### Resumable (recommended)

```bash
cd finetuning/

# Pick training data
export DATA_DIR=$PWD/prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn

# Override paths if your home / data layout differs
export HF_HOME=/data/user_data/$USER/huggingface_cache
export SAVEDIR=/data/user_data/$USER/saves
export LLAMAFACTORY_DIR=$HOME/LlamaFactory
export OUTPUT_DIR=$SAVEDIR/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn

# (Optional) override the HF repo to push to on completion
# (default: shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn)
# HF caps repo_id at 96 chars total; keep your override under that.
export HF_REPO_ID=<your-username>/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn

# Submit. The wrapper pre-emptively chains the next sbatch with afterany dependency
# so a walltime hit doesn't cost you progress; the chain stops once
# TRAINING_COMPLETE is touched.
sbatch train_sbatch_resumable.sh
```

`train_sbatch_resumable.sh` requests 8x L40S, 32 cpus, 400G ram, 47:30h. It:
1. Activates `critic-training` conda env on the compute node.
2. Pre-emptively submits the next chained job with `afterany:$SLURM_JOB_ID` so you never lose a reservation.
3. Bails out (cancelling the chain) if `$OUTPUT_DIR/TRAINING_COMPLETE` already exists.
4. Runs training inside a tmux session named `prm_sft_<JOBID>` so you can `ssh <node> -t 'tmux attach -t prm_sft_<JOBID>'` to watch live.
5. On clean exit: touches `TRAINING_COMPLETE`, calls `huggingface_hub.HfApi().upload_folder()` to push the final model (excluding intermediate `checkpoint-*/`), touches `HF_PUSHED`, scancels the chained job.
6. On unclean exit: chained job picks up via `--resume_from_checkpoint` from the latest `checkpoint-*` in `$OUTPUT_DIR`.

### Legacy

```bash
cd finetuning/
export DATA_DIR=$PWD/prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn
export FORMAT=multiturn
sbatch train_sbatch.sh
```

`train_sbatch.sh` requests 8x L40S, 32 cpus, 400G ram, 12h. The launcher resolves the YAML recipe via `envsubst` to inject `$DATA_DIR` / `$OUTPUT_DIR`, then calls `accelerate launch src/train.py`.

## Eval quickstart

```bash
# Spin up vLLM serving the trained critic on Babel
sbatch babel-server/vllm_server_babel_qwen3-8b-opus-distill.sh
# Note the compute node from `squeue -u $USER`

# Step 1: generate critic responses (uses the vLLM you just started)
python eval_generate_responses.py \
    --model-name qwen3-8b-base \
    --model-label qwen3-8b-base \
    --api-base http://<vllm-node>:8071/v1
python eval_generate_responses.py \
    --model-name <your trained critic served name> \
    --model-label cwm-plus-qwen-critic \
    --api-base http://<vllm-node>:8071/v1

# Step 2: score one model against another (or against the teacher ground truth)
python eval_compute_scores.py --model-a qwen3-8b-base --model-b cwm-plus-qwen-critic

# Targeted analyses
python eval_error_classification.py
python eval_intrusiveness.py
python eval_length_analysis.py
```

If you don't have GPUs handy, use the CMU LiteLLM gateway as the api_base instead — see [HANDOVER.md §5](../HANDOVER.md#5-run-an-eval).
