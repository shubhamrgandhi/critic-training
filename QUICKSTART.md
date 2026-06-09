# Quickstart

Five steps to either (A) run inference with the trained critic on SWE-bench Verified, or (B) retrain the critic from scratch.

For full repo layout, naming conventions, and the LiteLLM (no-AWS) reproduction path, see [HANDOVER.md](HANDOVER.md).

---

## 0. Prereqs

- CMU Babel access (or any SLURM cluster with L40S nodes; tweak `--gres=gpu:L40S:N` in `babel-server/*.sh` for other GPUs).
- `uv` installed: `pip install uv` (or your distro's package manager).

## 1. Clone + install

```bash
git clone <this-repo> critic-training && cd critic-training

# critic-training env: training, inference clients, stats
conda create -n critic-training python=3.12 -y && conda activate critic-training
pip install uv
uv sync                                # installs from pyproject.toml + uv.lock
                                       # (auto-installs the forked mini-swe-agent)

# vllm env: model serving (separate; vLLM deps conflict with the rest)
conda create -n vllm python=3.12 -y && conda activate vllm && pip install vllm
```

Sanity check: `conda activate critic-training && mini-extra swebench --help` → prints help.

## 2. Pick your path

### A. Run inference with the trained critic

The critic is on Hugging Face: [`shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn`](https://huggingface.co/shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn).

1. **Serve the critic** on Babel (1x L40S):
   ```bash
   sbatch babel-server/vllm_server_babel_qwen3-8b-opus-distill.sh
   # Note the compute node from `squeue -u $USER`
   ```
2. **Serve the agent** (CWM, 8x L40S):
   ```bash
   sbatch babel-server/vllm_server_babel_cwm.sh
   ```
3. **Pre-pull SWE-bench SIF images** (once; takes a while):
   ```bash
   export SWEBENCH_SIF_CACHE=/data/user_data/$USER/swebench_sifs
   # see HANDOVER §5 for the prepull command — or just let the run pull on demand
   ```
4. **Run the headline eval**:
   ```bash
   conda activate critic-training
   bash scripts/run_critic_max150.sh \
       prm_issue_res_instructions_step_aware 5 0 cwm \
       --prm qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn \
       --prm-node <critic-vllm-node>:8071 \
       --agent-node <cwm-vllm-node> \
       --slice :400 \
       --prefix-dir /data/user_data/$USER/critic-training/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_cwm
   ```
5. **Compute headline numbers**:
   ```bash
   python scripts/get_stats_full500_prefix.py
   # writes full500_stats.csv + full500_exit_statuses.csv
   ```

**No-AWS path**: skip steps 1–2 above and use the LiteLLM template config — see [HANDOVER §5 — Running without AWS](HANDOVER.md#running-without-aws--the-litellm-gateway-path).

### B. Retrain the critic from scratch

Training data is in `finetuning/prm_sft_*/`. The headline run uses `prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn/` (6,447 samples).

```bash
conda activate critic-training

# (One-time) clone LlamaFactory
git clone https://github.com/hiyouga/LLaMA-Factory ~/LlamaFactory
cd ~/LlamaFactory && pip install -e ".[torch,metrics]" && cd -

# Submit. Auto-resumes on walltime, pushes to HF on completion.
cd finetuning/
export DATA_DIR=$PWD/prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn
export OUTPUT_DIR=/data/user_data/$USER/saves/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn
export HF_REPO_ID=<your-username>/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn

sbatch train_sbatch_resumable.sh
```

Watch progress:
```bash
ssh <compute-node> -t 'tmux attach -t prm_sft_<JOBID>'
# or
tail -f /data/user_data/$USER/saves/logs/$(basename $OUTPUT_DIR)_train.log
```

Training runs ~37h on 8x L40S, FSDP full-shard, bsz=8, 3 epochs over 6,447 samples → 2,418 steps. Final loss should land around 0.18 from a starting loss around 0.95. On clean exit, the wrapper auto-pushes to `$HF_REPO_ID`.

---

## Where things live

| Thing | Path |
|---|---|
| Inference launchers | `scripts/run_critic_*.sh`, `scripts/run-mini-swe-agent-with-server*.sh` |
| Training launcher | `finetuning/train_sbatch_resumable.sh` |
| Stats / plotting | `scripts/get_stats_full500_prefix.py`, `scripts/make_pareto_full500.py` |
| Eval (post-train scoring) | `finetuning/eval_*.py` |
| vLLM launchers | `babel-server/vllm_server_babel_*.sh` |
| Mini-swe-agent fork (PRM hooks) | `mini-swe-agent/src/minisweagent/agents/default_prm.py` |
| Configs | `mini-swe-agent/configs/*.yaml` |
| Env / lock | `pyproject.toml`, `uv.lock`, `.python-version` |

For anything not covered here, [HANDOVER.md](HANDOVER.md) is the longer guide.
