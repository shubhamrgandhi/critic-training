#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable paths (override via environment variables)
export HF_HOME="${HF_HOME:-/data/user_data/srgandhi/huggingface_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
SAVEDIR="${SAVEDIR:-/data/user_data/srgandhi/saves}"
LOGDIR="$SAVEDIR/logs"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-/home/srgandhi/LlamaFactory}"
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/prm_sft_data_opus_distill_full_feedback_history_32k}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SAVEDIR/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6}"
# Avoid NFS lock contention for Triton autotune cache
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# Disable P2P to avoid NCCL hangs on PCIe-only cross-NUMA topology
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Qwen3-8B Full SFT for PRM (FSDP, 8x L40S) ==="
echo "Training data: $DATA_DIR/"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$LLAMAFACTORY_DIR"

# Resolve template YAML with envsubst
RESOLVED_YAML=$(mktemp /tmp/prm_full_sft_l40s_config_XXXXXX.yaml)
envsubst < "$SCRIPT_DIR/qwen3_8b_prm_full_sft_l40s_train.yaml" > "$RESOLVED_YAML"
trap "rm -f $RESOLVED_YAML" EXIT

# Train with accelerate + FSDP
echo "=== Training ==="
accelerate launch \
    --config_file "$SCRIPT_DIR/fsdp_full_sft_config.yaml" \
    src/train.py "$RESOLVED_YAML" \
    2>&1 | tee "$LOGDIR/$(basename "$OUTPUT_DIR")_train.log"

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT_DIR"
