#!/bin/bash
# run_critic_max150.sh
#
# Wrapper around run_critic.sh logic with configurable step limit.
# Uses config with _max150 suffix as base, overrides step_limit via --step-limit.
# Output dir reflects actual step limit: results_singularity_max_<N>_steps.
#
# Usage: same as run_critic.sh
#   ./run_critic_max150.sh <setting> <prm_interval> <run> <agent_model> [options]

set -eo pipefail

# Suppress Singularity/Apptainer mount warnings from system config
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

# Use /scratch (or /data/user_data/$USER) for tmp to avoid filling up root filesystem
export TMPDIR=/scratch
mkdir -p "$TMPDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETTING=$1
PRM_INTERVAL=$2
RUN=$3
AGENT_MODEL=$4
shift 4 2>/dev/null

if [ -z "$SETTING" ] || [ -z "$PRM_INTERVAL" ] || [ -z "$RUN" ] || [ -z "$AGENT_MODEL" ]; then
    echo "Usage: ./run_critic_max150.sh <setting> <prm_interval> <run> <agent_model> [options]"
    exit 1
fi

# Defaults
PRM_NAME="claude-opus-4-6"
PRM_NODE=""
AGENT_NODE="localhost"
WORKERS=8
SUBSET=verified
SPLIT=test
SLICE=":50"
STEP_LIMIT=""
API_BASE=""
PRM_DEDUP=""
PRM_DEDUP_THRESHOLD=""

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --prm)          PRM_NAME="$2";    shift 2 ;;
    --prm-node)     PRM_NODE="$2";    shift 2 ;;
    --agent-node)   AGENT_NODE="$2";  shift 2 ;;
    --workers)      WORKERS="$2";     shift 2 ;;
    --subset)       SUBSET="$2";      shift 2 ;;
    --split)        SPLIT="$2";       shift 2 ;;
    --slice)        SLICE="$2";       shift 2 ;;
    --api-base)     API_BASE="$2";    shift 2 ;;
    --prm-api-base) PRM_API_BASE_OVERRIDE="$2"; shift 2 ;;
    --step-limit)   STEP_LIMIT="$2";  shift 2 ;;
    --prm-dedup)    PRM_DEDUP="true"; shift ;;
    --prm-dedup-threshold) PRM_DEDUP_THRESHOLD="$2"; shift 2 ;;
    *)              echo "Unknown option: $1"; shift ;;
  esac
done

# Build config and output paths — base config always uses _max150; step_limit overridden if needed
FULL_SETTING="singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}"
CONFIG="${SCRIPT_DIR}/../mini-swe-agent/configs/swebench_${FULL_SETTING}_${RUN}_${AGENT_MODEL}_max150.yaml"
EFFECTIVE_STEPS="${STEP_LIMIT:-150}"
OUTPUT="${SCRIPT_DIR}/../results_singularity_max_${EFFECTIVE_STEPS}_steps/${FULL_SETTING}_${RUN}_${AGENT_MODEL}_prm_${PRM_NAME}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Determine agent api_base
if [ -n "$API_BASE" ]; then
    AGENT_API_BASE="$API_BASE"
    if [[ "$AGENT_API_BASE" != http* ]]; then
        AGENT_API_BASE="http://${AGENT_API_BASE}/v1"
    elif [[ "$AGENT_API_BASE" != */v1 ]]; then
        AGENT_API_BASE="${AGENT_API_BASE}/v1"
    fi
else
    AGENT_API_BASE="http://${AGENT_NODE}:8070/v1"
fi

# --- PRM handling ---
if [ "$PRM_NAME" = "claude-opus-4-6" ]; then
    if [ "$AGENT_NODE" != "localhost" ] || [ -n "$API_BASE" ] || [ -n "$STEP_LIMIT" ] || [ -n "$PRM_DEDUP" ]; then
        TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
        trap "rm -f $TEMP_CONFIG" EXIT
        python3 - "$CONFIG" "$TEMP_CONFIG" "$AGENT_API_BASE" "$STEP_LIMIT" "$PRM_DEDUP" "$PRM_DEDUP_THRESHOLD" <<'PYEOF'
import yaml, sys
base_config, temp_config, agent_api_base, step_limit, prm_dedup, prm_dedup_threshold = sys.argv[1:]
with open(base_config) as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_kwargs']['api_base'] = agent_api_base
if step_limit:
    cfg.setdefault('agent', {})['step_limit'] = int(step_limit)
if prm_dedup == 'true':
    cfg.setdefault('agent', {})['prm_dedup'] = True
if prm_dedup_threshold:
    cfg.setdefault('agent', {})['prm_dedup_threshold'] = float(prm_dedup_threshold)
with open(temp_config, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF
        CONFIG="$TEMP_CONFIG"
    fi
else
    if [ -z "$PRM_NODE" ] && [ -z "$PRM_API_BASE_OVERRIDE" ]; then
        echo "ERROR: --prm-node or --prm-api-base is required for vLLM PRM '$PRM_NAME'"
        exit 1
    fi

    case "$PRM_NAME" in
        sweagent7b)
            PRM_MODEL_NAME="SWE-bench/SWE-agent-LM-7B"
            PRM_PORT=8071
            ;;
        qwen3-8b)
            PRM_MODEL_NAME="Qwen/Qwen3-8B"
            PRM_PORT=8071
            ;;
        qwen3-8b-opus-distill)
            PRM_MODEL_NAME="qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"
            PRM_PORT=8071
            ;;
        qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample)
            PRM_MODEL_NAME="shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample"
            PRM_PORT=8071
            ;;
        qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think)
            PRM_MODEL_NAME="shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think"
            PRM_PORT=8071
            ;;
        qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean)
            PRM_MODEL_NAME="shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"
            PRM_PORT=8071
            ;;
        qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean_think)
            PRM_MODEL_NAME="shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean_think"
            PRM_PORT=8071
            ;;
        qwen25coder7b)
            PRM_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
            PRM_PORT=8071
            ;;
        cwm)
            PRM_MODEL_NAME="facebook/cwm"
            PRM_PORT=8070
            ;;
        *)
            PRM_MODEL_NAME="$PRM_NAME"
            PRM_PORT=8071
            ;;
    esac

    # Only enable thinking if the PRM name contains "_think"
    if [[ "$PRM_NAME" == *_think* ]]; then
        DISABLE_THINKING="false"
    else
        DISABLE_THINKING="true"
    fi

    if [ -n "$PRM_API_BASE_OVERRIDE" ]; then
        PRM_API_BASE="$PRM_API_BASE_OVERRIDE"
        if [[ "$PRM_API_BASE" != http* ]]; then
            PRM_API_BASE="http://${PRM_API_BASE}/v1"
        elif [[ "$PRM_API_BASE" != */v1 ]]; then
            PRM_API_BASE="${PRM_API_BASE}/v1"
        fi
    else
        PRM_API_BASE="http://${PRM_NODE}:${PRM_PORT}/v1"
    fi

    TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
    trap "rm -f $TEMP_CONFIG" EXIT

    python3 - "$CONFIG" "$TEMP_CONFIG" \
        "$AGENT_API_BASE" "$PRM_MODEL_NAME" "$PRM_API_BASE" "" "openai" "$DISABLE_THINKING" "$STEP_LIMIT" "$PRM_DEDUP" "$PRM_DEDUP_THRESHOLD" <<'PYEOF'
import yaml, sys

base_config, temp_config, agent_api_base, prm_model_name, prm_api_base, prm_api_key, prm_provider, disable_thinking, step_limit, prm_dedup, prm_dedup_threshold = sys.argv[1:]

with open(base_config) as f:
    cfg = yaml.safe_load(f)

# Override step_limit if provided
if step_limit:
    cfg.setdefault('agent', {})['step_limit'] = int(step_limit)
if prm_dedup == 'true':
    cfg.setdefault('agent', {})['prm_dedup'] = True
if prm_dedup_threshold:
    cfg.setdefault('agent', {})['prm_dedup_threshold'] = float(prm_dedup_threshold)

# Override agent model api_base
cfg['model']['model_kwargs']['api_base'] = agent_api_base

# Override prm_model section
prm_kwargs = {
    'custom_llm_provider': prm_provider,
    'api_base': prm_api_base,
    'temperature': 0.6 if disable_thinking != 'true' else 0.0,
    'n': 1,
    'max_completion_tokens': 4096,
    'drop_params': True,
}
if prm_api_key:
    prm_kwargs['api_key'] = prm_api_key
prm_kwargs['extra_body'] = {
    'chat_template_kwargs': {'enable_thinking': disable_thinking != 'true'},
}

cfg['prm_model'] = {
    'model_name': prm_model_name,
    'model_kwargs': prm_kwargs,
}

with open(temp_config, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF

    CONFIG="$TEMP_CONFIG"
fi

echo "=============================================="
echo "Setting:     ${SETTING} k=${PRM_INTERVAL} run=${RUN} (MAX ${EFFECTIVE_STEPS} STEPS)"
echo "Agent model: ${AGENT_MODEL} @ ${AGENT_API_BASE}"
echo "PRM model:   ${PRM_NAME}"
[ "$PRM_NAME" != "claude-opus-4-6" ] && echo "PRM server:  ${PRM_API_BASE}"
[ -n "$PRM_DEDUP" ] && echo "PRM dedup:   threshold=${PRM_DEDUP_THRESHOLD:-0.93}"
echo "Config:      ${CONFIG}"
echo "Output:      ${OUTPUT}"
echo "Slice:       ${SLICE}  Workers: ${WORKERS}"
echo "=============================================="

SLICE_ARGS=""
if [ -n "$SLICE" ]; then
    SLICE_ARGS="--slice $SLICE"
fi

mini-extra swebench \
    --config "$CONFIG" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --workers "$WORKERS" \
    --shuffle \
    $SLICE_ARGS \
    --output "$OUTPUT"
