#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda env so we use the right python/transformers/accelerate.
# ~/.local/bin precedes the env in PATH on this system, which would otherwise
# shadow the env's accelerate CLI. Force CONDA_PREFIX/bin to the front.
CONDA_ENV="${CONDA_ENV:-critic-training}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV" ]]; then
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi
echo "Using python: $(which python)"
echo "Using accelerate: $(which accelerate)"

# Configurable paths (override via environment variables before submitting).
export HF_HOME="${HF_HOME:-/data/user_data/$USER/huggingface_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
SAVEDIR="${SAVEDIR:-/data/user_data/$USER/saves}"
LOGDIR="$SAVEDIR/logs"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$HOME/LlamaFactory}"
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/prm_sft_r2egym_swebench_instructions_k5_cwm_plus_qwen_opus_distill_32k_multiturn}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SAVEDIR/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn}"
# Avoid NFS lock contention for Triton autotune cache
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

TRAIN_YAML="$SCRIPT_DIR/qwen3_8b_critic_full_sft_l40s_train_multiturn_resumable.yaml"

# Disable P2P to avoid NCCL hangs on PCIe-only cross-NUMA topology
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Auto-detect existing checkpoint to resume from
RESUME_ARGS=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [[ -n "$LATEST_CKPT" && -d "$LATEST_CKPT" ]]; then
    echo "=== Resuming from checkpoint: $LATEST_CKPT ==="
    RESUME_ARGS="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "=== Starting fresh training run (no checkpoint found in $OUTPUT_DIR) ==="
fi

echo "=== Qwen3-8B Full SFT for PRM (FSDP, 8x L40S) ==="
echo "Training config: $TRAIN_YAML"
echo "Training data: $DATA_DIR/"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$LLAMAFACTORY_DIR"

# Resolve template YAML with envsubst
RESOLVED_YAML=$(mktemp /tmp/prm_full_sft_l40s_config_XXXXXX.yaml)
envsubst < "$TRAIN_YAML" > "$RESOLVED_YAML"
trap "rm -f $RESOLVED_YAML" EXIT

LOG_FILE="$LOGDIR/$(basename "$OUTPUT_DIR")_train.log"

# Train with accelerate + FSDP. Use python -m to force the conda env's python
# instead of any stray accelerate binary on PATH.
echo "=== Training (log: $LOG_FILE) ==="
python -m accelerate.commands.launch \
    --config_file "$SCRIPT_DIR/fsdp_full_sft_config.yaml" \
    src/train.py "$RESOLVED_YAML" $RESUME_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT_DIR"
