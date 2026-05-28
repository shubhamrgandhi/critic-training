# Tool-Overuse Handover

This is a handover of the `tool-overuse` codebase. The original work used a mix of Babel (CMU SLURM cluster) and AWS (Bedrock + EC2 vLLM). The recipient has Babel access only. AWS-specific scripts have been excluded from this repo by design.

## What's where

### In this repo
- `finetuning/` — everything for SFT of the PRM (data prep, training configs, eval). Recipient should start here.
- `mini-swe-agent/` — the agent harness used for SWE-bench / R2E-Gym runs. Forked from princeton-nlp/mini-swe-agent with PRM-aware additions.
- `babel-server/` — selected vLLM launcher scripts for Babel (CWM agent model + qwen3-8b/opus-distill PRM variants + slurm reservation helpers). Used to be at `/home/srgandhi/babel-server/` (separate dir); a curated subset has been copied in here.
- `scripts/` — eval orchestration, stats, plotting, run launchers.
- `iaa/`, `human_annotation_csvs/` — annotation artifacts.

### Outside this repo (on Babel filesystem)
- `/data/user_data/srgandhi/saves/` — SFT checkpoints (the trained PRMs). Permissions opened for other Babel users. Subdirs:
  - `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn/` — main multiturn checkpoint
  - `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened/` — flattened-format variant
  - `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean/`, `..._rejection-sample/`, `..._think` variants — earlier ablations
- `/data/user_data/srgandhi/tool-overuse/` — eval result trees (the `results*` symlinks in the repo root point here):
  - `results/`, `results_singularity/`, `results_singularity_max_75_steps/`, `results_singularity_max_150_steps/`, `results_singularity_max_150_steps_prefix/` — main SWE-bench Verified runs
  - `results_r2egym_subset/`, `results_r2egym_swebench/` — R2E-Gym runs
- `/data/user_data/srgandhi/huggingface_cache/` — HF cache used by training (`HF_HOME` in `finetuning/run_qwen3_8b_prm_full_sft_l40s.sh`)
- `/home/srgandhi/LlamaFactory/` — LlamaFactory checkout used by the SFT training script (`LLAMAFACTORY_DIR`)

### Excluded from this repo (kept locally)
- `.venv/`, `.claude/` — local environments
- `logs/`, `sbatch_logs/`, `babel-server/{sbatch,vllm}_logs/` — run logs
- `paper`, `paper-context/` — paper drafts (in a separate repo)
- `openhands-critic/`, `manual_analysis/` — bulky / stale exploration dirs (kept on `/home/srgandhi/`)
- `scripts/aws_*.sh`, `scripts/test_bedrock_opus.py`, `scripts/test_richard_bedrock.py`, `scripts/test_sagemaker_prm.py`, `scripts/run_qwen_bedrock.sh`, `scripts/delete_richard_key.sh` — AWS-only, replace with Babel equivalents
- `babel-server/vllm_server_*` for Devstral, qwen2.5-coder, SWE-agent variants — only the CWM and qwen3-8b/opus-distill variants are included; the rest are in `/home/srgandhi/babel-server/` if needed
- Large `prm_sft_train.jsonl` / `prm_sft_val.jsonl` files (GitHub 100MB limit). The small `dataset_info.json` and `metadata.json` siblings are committed so the directory structure is preserved. Regenerate the jsonl with `finetuning/prepare_prm_sft_data_opus_distill_full_feedback_history.py` (see below).

## Reproduction paths

### Re-running SFT
```bash
cd finetuning/
# 1. Regenerate training data (if missing) — reads agent trajectories under /data/user_data/srgandhi/tool-overuse/...
python prepare_prm_sft_data_opus_distill_full_feedback_history.py

# 2. Launch SFT (8x L40S, FSDP via LlamaFactory)
FORMAT=multiturn bash run_qwen3_8b_prm_full_sft_l40s.sh
# Outputs to /data/user_data/srgandhi/saves/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn/
```
Key knobs in `run_qwen3_8b_prm_full_sft_l40s.sh`: `FORMAT` (`multiturn` or `flattened`), `SAVEDIR`, `DATA_DIR`, `LLAMAFACTORY_DIR`, `HF_HOME`. The recipient will likely want to override `SAVEDIR` to point to their own `/data/user_data/<their_user>/saves/`.

### Serving a PRM checkpoint with vLLM on Babel
```bash
cd babel-server/
# Reserve GPUs first
bash reserve_gpus.sh         # generic
bash reserve_a100_gpus.sh    # A100
# Then launch the right vLLM server
bash vllm_server_babel_qwen3-8b-opus-distill.sh   # serves the SFT'd PRM
bash vllm_server_babel_cwm.sh                     # serves CWM (agent model)
```
Each script reads a checkpoint path; update those paths to point at your own `saves/` if needed. Watch with `bash watch_babel.sh`.

### Running an eval (no AWS path)
The original setup had a Babel agent model + AWS Bedrock PRM. To run entirely on Babel, both need to be served from Babel.

1. Spin up the agent model server (e.g. `vllm_server_babel_cwm.sh`) — note the `babel-XX-YY:port` it exposes.
2. Spin up the PRM server (e.g. `vllm_server_babel_qwen3-8b-opus-distill.sh`).
3. Pick an agent config under `mini-swe-agent/configs/` — the relevant ones for the PRM-augmented SWE-bench runs are named like `swebench_singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k{5,10}_0_cwm_max150.yaml`.
4. Edit the config to point `model.api_base` and `prm_model.api_base` at the Babel servers from steps 1–2.
5. Launch via the relevant `scripts/run_prm_*.sh`. **Note**: scripts named `aws_*.sh` and orchestrators that invoke them (e.g. `orchestrate_d_trained_chain.sh` calls `aws_eval.sh`) are excluded — write Babel equivalents that submit via `sbatch` against the Babel servers.

A reference command for the headline run (k=5, swebench-instructions multiturn PRM, first 400 problems) is saved in memory; ask if you need it written out.

## Naming conventions to know

- **Detailed vs. concise**: in scripts/configs, `detailed` = the non-instructions PRM prompt; `concise` = the instructions PRM prompt. (Script names use `detailed/concise`; configs and most paper terminology use `instructions/non-instructions`.)
- **k=5 vs. k=10**: number of agent steps between PRM critiques (intervention frequency).
- **`_prefix`**: result trees built off the run-0 prefix, used to control for early-step variance. All `_prefix` runs use the run-0 base directory as `--prefix-dir`.
- **`_max150`**: max 150 agent steps (the headline setting). `_max75`, `_max100` are ablations.
- **`step_aware`**: PRM is told its current step index in addition to the trajectory.
- **`issue_res`**: PRM sees the issue resolution (gold patch context) — used only in some ablations.

## What I'd do first as the recipient

1. Clone this repo, set up a venv from `pyproject.toml` / `uv.lock`.
2. Verify read access to `/data/user_data/srgandhi/saves/` and `/data/user_data/srgandhi/tool-overuse/` (read perms have been opened; if blocked, contact Shubham).
3. Pick one of the served PRM checkpoints in `saves/`, launch its `babel-server/vllm_server_babel_*.sh`, and run a small (mini50) eval against an existing agent config to confirm the pipeline works end-to-end.
4. Then move to re-running SFT or building new ablations.

## Contact

Original author: Shubham Gandhi (srgandhi@andrew.cmu.edu). Memory of design decisions and gotchas is captured in `~/.claude/projects/-home-srgandhi-tool-overuse/memory/` (Shubham's local Claude memory).
