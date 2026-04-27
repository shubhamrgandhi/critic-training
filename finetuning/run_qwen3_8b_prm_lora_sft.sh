#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable paths (override via environment variables)
export HF_HOME="${HF_HOME:-/data/user_data/srgandhi/huggingface_cache}"
SAVEDIR="${SAVEDIR:-/data/user_data/srgandhi/saves}"
LOGDIR="$SAVEDIR/logs"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-/home/srgandhi/LlamaFactory}"
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/prm_sft_data_opus_distill_full_feedback_history}"
export OUTPUT_DIR="$SAVEDIR/qwen3-8b-lora-sft-prm-opus-distill"
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

echo "=== Qwen3-8B QLoRA SFT for PRM ==="
echo "Training data: $DATA_DIR/"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$LLAMAFACTORY_DIR"

# Resolve template YAML with envsubst
RESOLVED_YAML=$(mktemp /tmp/prm_train_config_XXXXXX.yaml)
envsubst < "$SCRIPT_DIR/qwen3_8b_prm_lora_sft_train.yaml" > "$RESOLVED_YAML"
trap "rm -f $RESOLVED_YAML" EXIT

# Train
echo "=== Training ==="
llamafactory-cli train "$RESOLVED_YAML" \
    2>&1 | tee "$LOGDIR/qwen3_8b_prm_lora_sft_train.log"

echo ""
echo "=== Training complete ==="
echo "LoRA adapter saved to: $OUTPUT_DIR"
