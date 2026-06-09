# critic-training — handover

End-to-end guide for two things:

1. SFT training the **critic** (Qwen3-8B, trained via SFT on critiques from a strong teacher) on Babel.
2. Running mini-swe-agent on SWE-bench Verified with a served critic.

The repo assumes you have:
- Babel access (CMU SLURM cluster, 8x L40S nodes available)
- A working `critic-training` conda env (Python 3.12)
- Either a vLLM server you can start on Babel, or access to a model gateway (the CMU LiteLLM gateway works out of the box)

**No AWS / Bedrock account required.** The reproduction path uses the CMU LiteLLM gateway for any externally-served models. See [§5 — Run an eval](#5-run-an-eval) for the LiteLLM template config.

**Released artifacts:** the trained critic and three SFT training datasets are grouped on Hugging Face under the [Critic Training for Code Agents](https://huggingface.co/collections/shubhamrgandhi/critic-training-for-code-agents-6a27adf94c9409f0db710fee) collection.

> Naming note: the trained model is referred to as the "critic" throughout this guide. Many filenames, output directories, and YAML keys still contain `prm`/`PRM` (process reward model — the same concept). Saved-checkpoint and dataset directory names are not renamed since recipients may already have references to those paths. New scripts/orchestration use `critic`.

---

## 1. Repo layout

```
critic-training/
├── finetuning/                     # critic SFT: data prep, training, eval
│   ├── run_qwen3_8b_prm_full_sft_l40s.sh             # original launcher (full SFT, 8x L40S)
│   ├── run_qwen3_8b_critic_full_sft_l40s_resumable.sh   # NEW: same launcher with auto-resume from latest checkpoint
│   ├── run_qwen3_8b_prm_lora_sft.sh                  # LoRA variant (single-GPU)
│   ├── train_sbatch.sh                               # original sbatch wrapper (12h walltime)
│   ├── train_sbatch_resumable.sh                     # NEW: 47:30h walltime + self-resubmitting + tmux + HF Hub auto-upload
│   ├── qwen3_8b_prm_full_sft_l40s_train_multiturn.yaml          # LlamaFactory recipe (multiturn, headline)
│   ├── qwen3_8b_critic_full_sft_l40s_train_multiturn_resumable.yaml # NEW: step-based ckpting, save_total_limit=2, save_only_model=false
│   ├── qwen3_8b_prm_full_sft_l40s_train.yaml                    # LlamaFactory recipe (flattened)
│   ├── qwen3_8b_prm_lora_sft_train.yaml                         # LlamaFactory recipe (LoRA)
│   ├── fsdp_full_sft_config.yaml                     # accelerate FSDP config
│   ├── prepare_prm_sft_data_opus_distill_full_feedback_history.py   # data generation (single source dir at a time)
│   ├── prm_sft_r2egym_swebench_k5_opus_distill_32k_multiturn/                       # training data (k=5, base prompt)
│   ├── prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn/          # training data (k=5, concise prompt)
│   ├── prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn/   # mixed CWM + Qwen3-Next-80B-Instruct teacher trajectories
│   ├── eval_*.py                    # 5-stage eval pipeline (see §7)
│   └── eval_results/               # post-training eval predictions
│
├── babel-server/                   # vLLM launchers + GPU reservation helpers (Babel SLURM)
│   ├── vllm_server_babel_cwm.sh                    # serve facebook/cwm as the agent model
│   ├── vllm_server_babel_qwen3-8b-opus-distill.sh  # serve any Qwen3-8B critic checkpoint
│   ├── reserve_a100_gpus.sh / reserve_gpus.sh / reserve_cpus.sh
│   └── watch_babel.sh
│
├── mini-swe-agent/                 # forked agent harness (PRM hooks + diff-cleanup added)
│   ├── src/minisweagent/...
│   └── configs/
│       ├── swebench_singularity_edit_obs_final_only_0_<agent>_max150.yaml          # base no-critic configs (cwm, qwen32b, qwen3-80b, claude-opus, etc.)
│       ├── swebench_singularity_edit_obs_final_only_prm_issue_res_*_max150.yaml    # critic-equipped variants
│       ├── swebench_singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_litellm-haiku_max150.yaml  # NEW: LiteLLM-only template (no AWS)
│       └── litellm_model_registry.json
│
├── scripts/                        # eval launchers, stats, plotting
│   ├── run_critic_max150.sh        # parametric critic-eval launcher (full 500)
│   ├── run_critic_max150_mini.sh   # parametric critic-eval launcher (slice :50)
│   ├── run_critic.sh               # base parametric launcher (variable step limit)
│   ├── run_qwen32b_full500.sh      # qwen3-32b base, full 500
│   ├── run_qwen3_80b_full500.sh    # qwen3-next-80b base, full 500
│   ├── run-mini-swe-agent-with-server.sh        # combined vLLM + agent in one sbatch (generic)
│   ├── run-mini-swe-agent-with-server_cwm.sh    # combined vLLM + agent in one sbatch (cwm)
│   ├── get_stats_full500_prefix.py              # main stats: resolved %, cost, steps, localization, stuck-in-loop
│   ├── get_stats_mini50.py                      # same for 50-instance subset
│   ├── make_pareto_full500.py                   # pareto plotting: resolved % vs cost / steps
│   ├── make_main_plots.py                       # paper figures
│   ├── plot_resolved_vs_steps_full500.py        # step-vs-resolved curves
│   ├── critic_feedback_distribution.py          # critic-intervention distribution analysis
│   ├── compute_extra_metrics.py                 # thin wrapper around get_stats_full500_prefix helpers
│   └── data/                                    # SWE-bench image-id manifests
│
└── pyproject.toml / uv.lock        # python deps
```

### Data on Babel filesystem (not in repo)

| Path | Contents |
|---|---|
| `/data/user_data/$USER/saves/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn/` | Main multiturn critic checkpoint (read access opened) |
| `/data/user_data/$USER/saves/qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn/` | Critic trained on instructions-style prompts (k=5) |
| `/data/user_data/$USER/saves/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn/` | Critic trained on mixed CWM + Qwen3-Next-80B trajectories (current run) |
| `/data/user_data/$USER/critic-training/results_singularity_max_150_steps_prefix/` | Headline eval result trees (the `results_singularity_max_150_steps_prefix` symlink in the repo root points here) |
| `/data/user_data/$USER/critic-training/sif_cache/` | Pre-pulled SWE-bench apptainer/singularity images (`SWEBENCH_SIF_CACHE`) |
| `$HOME/LlamaFactory/` | LlamaFactory checkout the SFT script invokes (`LLAMAFACTORY_DIR`) |

If you don't have read access to any of the above, ping me. Override paths by exporting `HF_HOME`, `SAVEDIR`, `LLAMAFACTORY_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `SWEBENCH_SIF_CACHE`, `SAVES_DIR` before running the launcher scripts.

---

## 2. Environment setup

You need **two** conda envs (one for this repo, one for vLLM serving):

### a. `critic-training` — finetuning, inference clients, stats

```bash
conda create -n critic-training python=3.12 -y
conda activate critic-training
pip install uv
uv sync                            # installs from pyproject.toml + uv.lock
                                   # (also editable-installs the forked mini-swe-agent
                                   #  via [tool.uv.sources] in pyproject.toml)
```

After `uv sync`, you should be able to:
- `python -c "import torch, transformers, accelerate, datasets, openai, litellm, minisweagent"` — no errors.
- `mini-extra swebench --help` — prints help.

### b. `vllm` — model serving

The vLLM server runs in a separate env (vLLM has aggressive deps that conflict with general ML stacks). Create it once:

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

The `babel-server/` launchers expect a conda env called `vllm`. If yours is named differently, set `CONDA_ENV=<your-env>` before submitting.

### c. LlamaFactory (separate checkout, used at training time)

```bash
git clone https://github.com/hiyouga/LLaMA-Factory ~/LlamaFactory
cd ~/LlamaFactory && pip install -e ".[torch,metrics]"
```

Install into the `critic-training` env (it's invoked by the SFT launcher's `python -m accelerate.commands.launch` call). Override the location via `LLAMAFACTORY_DIR`.

---

## 3. Train the critic (SFT)

Two launchers exist, pick one:

- **Original** ([finetuning/train_sbatch.sh](finetuning/train_sbatch.sh) → [run_qwen3_8b_prm_full_sft_l40s.sh](finetuning/run_qwen3_8b_prm_full_sft_l40s.sh)) — 12h walltime, no resume, expects you to submit from an already-activated conda shell.
- **Resumable** ([finetuning/train_sbatch_resumable.sh](finetuning/train_sbatch_resumable.sh) → [run_qwen3_8b_critic_full_sft_l40s_resumable.sh](finetuning/run_qwen3_8b_critic_full_sft_l40s_resumable.sh)) — 47:30h walltime, auto-detects latest `checkpoint-*` and resumes via `--resume_from_checkpoint`, **self-resubmits** a chained sbatch on `afterany` so chains survive walltime hits, runs inside a tmux session on the compute node, and pushes to HF Hub on clean exit.

Both run LlamaFactory full-parameter SFT on 8x L40S via accelerate + FSDP.

### Resumable launcher (recommended)

```bash
cd finetuning/

# (Optional) override defaults
export HF_HOME=/data/user_data/$USER/huggingface_cache
export SAVEDIR=/data/user_data/$USER/saves
export LLAMAFACTORY_DIR=$HOME/LlamaFactory
export DATA_DIR=$PWD/prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn
export OUTPUT_DIR=$SAVEDIR/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn

# (Optional) override the HF repo to push to (default matches the OUTPUT_DIR name above)
export HF_REPO_ID=<your-username>/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn

# Submit. The script will pre-emptively chain a follow-up job with afterany dependency
# so that if the first job hits walltime mid-step, the next job picks up from the latest
# checkpoint. The chain stops once TRAINING_COMPLETE is touched.
sbatch train_sbatch_resumable.sh
```

The yaml ([qwen3_8b_critic_full_sft_l40s_train_multiturn_resumable.yaml](finetuning/qwen3_8b_critic_full_sft_l40s_train_multiturn_resumable.yaml)) uses:
- `save_strategy: steps`, `save_steps: 200`
- `save_total_limit: 2` (only the 2 most recent checkpoints kept on disk to save space)
- `save_only_model: false` (full optimizer/scheduler state saved so resume works)
- `resume_from_checkpoint` is injected via CLI when a checkpoint is detected

### Monitoring the run

```bash
squeue -u $USER                                                                      # SLURM state
ssh <compute-node> -t 'tmux attach -t prm_sft_<JOBID>'                               # live training output
tail -f /data/user_data/$USER/saves/logs/<output-dir-basename>_train.log            # log tail (no SSH needed)
```

### HF Hub upload

After clean exit, the wrapper uses `huggingface_hub.HfApi().upload_folder()` to push the final model to `HF_REPO_ID`. It excludes intermediate `checkpoint-*/` directories — only the merged final model is pushed. Auth uses `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (already in env if you've used `huggingface-cli login`). Idempotent — writes a `HF_PUSHED` sentinel on success; failed uploads retry on the next chained run.

### Original launcher (legacy)

```bash
cd finetuning/
export DATA_DIR=$PWD/prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn
export FORMAT=multiturn          # or flattened
sbatch train_sbatch.sh
```

`FORMAT=multiturn` selects the multiturn yaml (each agent step is its own assistant turn with `mask_history: true`); `FORMAT=flattened` uses single-turn full-trajectory yaml. Multiturn is what the headline runs use.

### Available training data

All three are also published on Hugging Face:

| Local dir | HF dataset | Samples | Notes |
|---|---|---|---|
| `prm_sft_r2egym_swebench_k5_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-k5-opus-distill-32k-multiturn) | 3,135 | k=5, **detailed** prompt (no `instructions`), R2E-Gym + SWE-bench Verified CWM rollouts critiqued by Opus 4.6 |
| `prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn) | 4,532 | k=5, **concise** ("instructions") prompt. **Headline single-agent critic training data.** |
| `prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn/` | [`shubhamrgandhi/critic-sft-r2egym-swebench-cwm-plus-qwen-k5-opus-distill-32k-multiturn`](https://huggingface.co/datasets/shubhamrgandhi/critic-sft-r2egym-swebench-cwm-plus-qwen-k5-opus-distill-32k-multiturn) | 6,447 | k=5, concise prompt, mixed teacher trajectories (CWM + Qwen3-Next-80B-Instruct, 4,532 + 1,915) |

If you don't have Babel filesystem access, fetch from HF instead:
```bash
huggingface-cli download --repo-type dataset \
    shubhamrgandhi/critic-sft-r2egym-swebench-instructions-k5-opus-distill-32k-multiturn \
    --local-dir finetuning/prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn
```

To regenerate data from scratch from raw trajectory dirs, see `prepare_prm_sft_data_opus_distill_full_feedback_history.py` (one source dir per run; for combined datasets like the cwm-plus-qwen one, run twice and concatenate `prm_sft_train.jsonl` files).

---

## 4. Serve the critic

Reserve 1x L40S (the 8B critic fits comfortably) and launch the vLLM server.

```bash
cd babel-server/

# Override the checkpoint if needed (default points at the main multiturn critic)
export MODEL=/data/user_data/$USER/saves/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn
export PORT=8071

sbatch vllm_server_babel_qwen3-8b-opus-distill.sh
```

The server script is generic: any Qwen3-8B-class checkpoint works. Set `MODEL`, `SERVED_NAME`, `PORT` as needed.

For the agent model (CWM):
```bash
sbatch vllm_server_babel_cwm.sh                       # 8x L40S, port 8070
```

Once started, note the compute node hostname (`squeue -u $USER`); the eval launcher needs it as `--prm-node babel-XX-YY:8071`.

---

## 5. Run an eval

Two parametric launchers:

- [scripts/run_critic_max150.sh](scripts/run_critic_max150.sh) — full 500 instances of SWE-bench Verified, 150-step budget.
- [scripts/run_critic_max150_mini.sh](scripts/run_critic_max150_mini.sh) — same, sliced to first 50.

```bash
# settings: e.g. singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware
# k = critic intervention frequency (every k agent steps)
# agent: qwen32b | qwen3-80b | cwm | claude-opus | <any agent config you've built>
# prm:   the served critic name (matches SERVED_NAME on the vLLM server)
# prm-node: hostname:port from squeue

bash scripts/run_critic_max150.sh \
  prm_issue_res_instructions_step_aware \
  5 \
  0 \
  cwm \
  --prm qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn \
  --prm-node babel-3-9:8071 \
  --prefix-dir /data/user_data/$USER/critic-training/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_cwm
```

Conventions:
- `instructions` in a config name = concise critic prompt; absence = detailed prompt.
- `step_aware` = critic is given the current agent step number as part of its input.
- `k5` / `k10` = intervention every 5 or 10 agent steps.
- `_prefix` = the run uses a base trajectory prefix from a previous run as `--prefix-dir`. **Always** set `--prefix-dir` to the run-0 (no-critic) base trajectory dir — that's what `_prefix` settings are for.
- `0_cwm`, `0_qwen32b`, `0_qwen3-80b` = run index 0, agent model name. Run index just lets you do multi-seed runs.

Reference command for the headline `(cwm + r2egym-swebench instructions k=5 critic)` config:

```bash
bash scripts/run_critic_max150.sh \
  prm_issue_res_instructions_step_aware \
  5 0 cwm \
  --prm qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn \
  --prm-node <node:port> \
  --slice :400 \
  --prefix-dir /data/user_data/$USER/critic-training/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_cwm
```

The mini-swe-agent batch runner skips instances already in `preds.json`, so you can re-run safely to fill gaps.

### Running without AWS — the LiteLLM gateway path

If you don't want to use Bedrock (or don't have access), the CMU LiteLLM gateway exposes Anthropic, OpenAI, Gemini, and Llama models behind an OpenAI-compatible endpoint. Start the local strip-proxy:

```bash
# Brings up the proxy at 127.0.0.1:8765 (auto-runs the watchdog)
# Requires ~/.litellm_capstone_key to contain your CMU LiteLLM API key.
# (The proxy strips Anthropic-specific quirks and forwards to ai-gateway.andrew.cmu.edu.)
bash _archive/scripts/ensure_claude_litellm_proxy.sh

# Verify it's up
curl -s http://127.0.0.1:8765/healthz                 # → "ok"
curl -s http://127.0.0.1:8765/v1/models | jq '.data[].id'   # list available models
```

Use the template config [mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_litellm-haiku_max150.yaml](mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_litellm-haiku_max150.yaml) — both the agent and the critic point at `http://127.0.0.1:8765/v1` with `custom_llm_provider: openai`. Swap the `model_name` to whatever non-AWS model you want (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`, `us.anthropic.claude-sonnet-4-6`, `gpt-5-mini`, `gemini-2.5-flash`).

If you add a new gateway model, register it in [mini-swe-agent/configs/litellm_model_registry.json](mini-swe-agent/configs/litellm_model_registry.json) so LiteLLM's cost-calculator doesn't crash on the response. Existing entries (haiku-4.5, sonnet-4.6) are examples.

To smoke test inference end-to-end on a single instance (no GPU server, no Bedrock):
```bash
# Pre-pulled SIF needed under $SWEBENCH_SIF_CACHE/<instance_id>.sif
mini-extra swebench \
    --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_litellm-haiku_max150.yaml \
    --subset verified --split test \
    --filter '^astropy__astropy-12907$' \
    --workers 1 \
    --output /data/user_data/$USER/critic-training/results_smoke_test/litellm_haiku_test
```
Expected: ~18 min, ~$0.78 LiteLLM-quota cost, `LimitsExceeded` exit at the configured `step_limit` (the template caps at 30 for verification — bump to 150 for real runs).

### Base (no-critic) runs

For the run-0 base trees that critic runs use as `--prefix-dir`:

```bash
sbatch scripts/run_qwen32b_full500.sh                    # qwen3-32b
sbatch scripts/run_qwen3_80b_full500.sh                  # qwen3-next-80b
sbatch scripts/run-mini-swe-agent-with-server_cwm.sh     # cwm (serves vLLM + runs agent in one sbatch)
```

Outputs go to `results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_<agent>/`.

---

## 6. Stats & plotting

```bash
# Resolved %, cost, step counts, localization, stuck-in-loop across a sweep of result dirs
python scripts/get_stats_full500_prefix.py

# Pareto front: resolved % vs cost / steps
RESULTS_ROOT=/data/user_data/$USER/critic-training/results_singularity_max_150_steps_prefix \
  python scripts/make_pareto_full500.py
```

The stats script writes two CSVs into the result dir's parent:
- `full500_stats.csv` — resolved %, submitted %, avg steps, avg model/critic/total cost, localization rate, stuck-in-loop rate
- `full500_exit_statuses.csv` — submitted vs limits-exceeded vs context-window-exceeded vs other

For runs that combine a critic-equipped run with a base-cwm fallback (the headline reporting style), each critic row has a "+ base fallback" sibling row showing the union — instances the critic submitted are kept; instances the critic didn't submit fall back to the base run's solution.

---

## 7. Eval the trained critic

The post-training eval pipeline (in [finetuning/](finetuning/)) compares a candidate critic against the teacher's ground-truth critiques on a held-out set:

```bash
# Step 1: serve the critic on a vLLM (any Babel node)
sbatch babel-server/vllm_server_babel_qwen3-8b-opus-distill.sh

# Step 2: generate critic responses
python finetuning/eval_generate_responses.py \
    --model-name <served-name> \
    --model-label cwm-plus-qwen-critic \
    --api-base http://<vllm-node>:8071/v1

# Step 3: score against ground truth + per-class analysis + length analysis
python finetuning/eval_compute_scores.py --model-a qwen3-8b-base --model-b cwm-plus-qwen-critic
python finetuning/eval_error_classification.py
python finetuning/eval_intrusiveness.py
python finetuning/eval_length_analysis.py
```

All eval scripts use OpenAI-compatible API (`from openai import OpenAI` + `--api-base`). They have **zero** Bedrock/AWS dependencies; you can point them at the LiteLLM proxy or any vLLM server.

Results land in `finetuning/eval_results/<model-label>/`.

---

## 8. Naming cheatsheet

| Term | Meaning |
|---|---|
| critic / PRM | The trained reviewer model that intervenes every k agent steps. "PRM" survives in path/filename references; "critic" is the preferred term going forward. |
| k=5 / k=10 | Critic intervention frequency in agent steps |
| instructions / concise | Critic prompt that constrains the critique to a short rubric |
| detailed / non-instructions | Critic prompt with more open-ended critique format |
| step_aware | Critic input includes the current step index |
| issue_res | "issue resolution" prompt branch in mini-swe-agent configs |
| edit_obs_final_only | Agent observation policy: keep only final state of edits in history |
| prefix / `_prefix` | Run uses a previous trajectory as a starting prefix; critic kicks in from there |
| max150 | 150-step agent budget per instance |

---

## 9. Things to know

- The `results_singularity_max_150_steps_prefix` and similar dirs at the repo root are symlinks into `/data/user_data/$USER/critic-training/`. If you re-clone elsewhere, recreate the symlinks or set `RESULTS_ROOT` explicitly.
- `SWEBENCH_SIF_CACHE` should point at a pre-pulled cache of SWE-bench apptainer images. Pulling 500 of these on demand is slow.
- The critic is trained and served **without** thinking-mode tokens. Don't enable `enable_thinking` or the chat template's thinking branch when serving or constructing critic inputs.
- 8 workers is the default for vLLM-backed runs; bump to 20 for LiteLLM/API-backed runs.
- The vLLM server logs go to `babel-server/vllm_logs/`; agent logs go to `agent_logs/`. Both are gitignored.
- `_archive/` at the repo root holds older variants of scripts and configs. Gitignored. Useful if you need to dig up a specific past variant; not part of the reproducible path.

---

## 10. Contact

Ping Shubham (srgandhi@andrew.cmu.edu) for filesystem access, missing checkpoints, or anything that's broken.
