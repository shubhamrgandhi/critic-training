#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Format: "flattened" or "multiturn" (controls training config and model name)
FORMAT="${FORMAT:-flattened}"
if [[ "$FORMAT" != "flattened" && "$FORMAT" != "multiturn" ]]; then
    echo "ERROR: FORMAT must be 'flattened' or 'multiturn', got '$FORMAT'" >&2
    exit 1
fi

# Configurable paths (override via environment variables before submitting).
# Defaults assume Babel-style /data/user_data/$USER layout and a LlamaFactory
# checkout at $HOME/LlamaFactory; override any of these if your layout differs.
export HF_HOME="${HF_HOME:-/data/user_data/$USER/huggingface_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
SAVEDIR="${SAVEDIR:-/data/user_data/$USER/saves}"
LOGDIR="$SAVEDIR/logs"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$HOME/LlamaFactory}"
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/prm_sft_r2egym_swebench_instructions_k5_opus_distill_32k_multiturn}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SAVEDIR/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-${FORMAT}}"
# Avoid NFS lock contention for Triton autotune cache
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# Select training config based on format
if [[ "$FORMAT" == "multiturn" ]]; then
    TRAIN_YAML="$SCRIPT_DIR/qwen3_8b_prm_full_sft_l40s_train_multiturn.yaml"
else
    TRAIN_YAML="$SCRIPT_DIR/qwen3_8b_prm_full_sft_l40s_train.yaml"
fi

# Disable P2P to avoid NCCL hangs on PCIe-only cross-NUMA topology
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=== Qwen3-8B Full SFT for PRM (FSDP, 8x L40S) ==="
echo "Format: $FORMAT"
echo "Training config: $TRAIN_YAML"
echo "Training data: $DATA_DIR/"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$LLAMAFACTORY_DIR"

# Resolve template YAML with envsubst
RESOLVED_YAML=$(mktemp /tmp/prm_full_sft_l40s_config_XXXXXX.yaml)
envsubst < "$TRAIN_YAML" > "$RESOLVED_YAML"
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
