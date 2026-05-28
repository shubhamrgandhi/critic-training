#!/bin/bash
# run_r2egym_prm.sh
#
# Run PRM experiments on R2E-Gym-Subset, analogous to run_prm_max150_mini.sh
# for SWE-bench. Results go to ../results_r2egym_subset/.
#
# Usage:
#   ./run_r2egym_prm.sh <setting> <prm_interval> <run> <agent_model> [options]
#
# Examples:
#   # Claude PRM with instructions prompt, k=10
#   ./run_r2egym_prm.sh prm_issue_res_instructions 10 0 cwm --prm claude-opus-4-6
#
#   # vLLM PRM
#   ./run_r2egym_prm.sh prm_issue_res 5 0 cwm --prm qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean --prm-api-base localhost:8071
#
#   # Base run (no PRM)
#   ./run_r2egym_prm.sh base 0 0 cwm

set -eo pipefail

export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"
export TMPDIR=/scratch
mkdir -p "$TMPDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETTING=${1:-prm_issue_res_instructions}
PRM_INTERVAL=${2:-10}
RUN=${3:-0}
AGENT_MODEL=${4:-cwm}
shift 4 2>/dev/null || true

# Defaults
PRM_NAME="claude-opus-4-6"
PRM_NODE=""
AGENT_NODE="localhost"
WORKERS=8
SUBSET="r2egym-subset"
SPLIT="train"
SLICE=":500"
STEP_LIMIT=""
API_BASE=""
PRM_API_BASE_OVERRIDE=""
PREFIX_DIR=""

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
    --prefix-dir)   PREFIX_DIR="$2";  shift 2 ;;
    *)              echo "Unknown option: $1"; shift ;;
  esac
done

EFFECTIVE_STEPS="${STEP_LIMIT:-150}"
RESULTS_DIR="${SCRIPT_DIR}/../results_r2egym_subset"

# Build config path and output path
if [ "$SETTING" = "base" ]; then
    FULL_SETTING="singularity_edit_obs_final_only"
    CONFIG="${SCRIPT_DIR}/../mini-swe-agent/configs/r2egym_${FULL_SETTING}_${RUN}_${AGENT_MODEL}_max150.yaml"
    OUTPUT="${RESULTS_DIR}/${FULL_SETTING}_${RUN}_${AGENT_MODEL}"
else
    FULL_SETTING="singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}"
    CONFIG="${SCRIPT_DIR}/../mini-swe-agent/configs/r2egym_${FULL_SETTING}_${RUN}_${AGENT_MODEL}_max150.yaml"
    OUTPUT="${RESULTS_DIR}/${FULL_SETTING}_${RUN}_${AGENT_MODEL}_prm_${PRM_NAME}"
fi

# Default prefix dir to base run 0 directory
if [ -z "$PREFIX_DIR" ]; then
    PREFIX_DIR="${RESULTS_DIR}/singularity_edit_obs_final_only_0_${AGENT_MODEL}"
fi

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
if [ "$SETTING" = "base" ]; then
    # Base run — no PRM, just optionally override agent api_base and prefix
    TEMP_CONFIG=$(mktemp /tmp/r2egym_config_XXXXXX.yaml)
    trap "rm -f $TEMP_CONFIG" EXIT
    python3 - "$CONFIG" "$TEMP_CONFIG" "$AGENT_API_BASE" "$STEP_LIMIT" "$PREFIX_DIR" <<'PYEOF'
import yaml, sys
base_config, temp_config, agent_api_base, step_limit, prefix_dir = sys.argv[1:]
with open(base_config) as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_kwargs']['api_base'] = agent_api_base
if step_limit:
    cfg.setdefault('agent', {})['step_limit'] = int(step_limit)
if prefix_dir:
    cfg.setdefault('agent', {})['prefix_trajectory_dir'] = prefix_dir
with open(temp_config, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF
    CONFIG="$TEMP_CONFIG"

elif [ "$PRM_NAME" = "claude-opus-4-6" ]; then
    TEMP_CONFIG=$(mktemp /tmp/r2egym_config_XXXXXX.yaml)
    trap "rm -f $TEMP_CONFIG" EXIT
    python3 - "$CONFIG" "$TEMP_CONFIG" "$AGENT_API_BASE" "$STEP_LIMIT" "$PREFIX_DIR" <<'PYEOF'
import yaml, sys
base_config, temp_config, agent_api_base, step_limit, prefix_dir = sys.argv[1:]
with open(base_config) as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_kwargs']['api_base'] = agent_api_base
if step_limit:
    cfg.setdefault('agent', {})['step_limit'] = int(step_limit)
if prefix_dir:
    cfg.setdefault('agent', {})['prefix_trajectory_dir'] = prefix_dir
with open(temp_config, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF
    CONFIG="$TEMP_CONFIG"

else
    if [ -z "$PRM_NODE" ] && [ -z "$PRM_API_BASE_OVERRIDE" ]; then
        echo "ERROR: --prm-node or --prm-api-base is required for vLLM PRM '$PRM_NAME'"
        exit 1
    fi

    case "$PRM_NAME" in
        qwen3-8b-opus-distill)
            PRM_MODEL_NAME="qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"
            PRM_PORT=8071
            ;;
        qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean)
            PRM_MODEL_NAME="shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean"
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

    TEMP_CONFIG=$(mktemp /tmp/r2egym_config_XXXXXX.yaml)
    trap "rm -f $TEMP_CONFIG" EXIT

    python3 - "$CONFIG" "$TEMP_CONFIG" \
        "$AGENT_API_BASE" "$PRM_MODEL_NAME" "$PRM_API_BASE" "" "openai" "$DISABLE_THINKING" "$STEP_LIMIT" "$PREFIX_DIR" <<'PYEOF'
import yaml, sys

base_config, temp_config, agent_api_base, prm_model_name, prm_api_base, prm_api_key, prm_provider, disable_thinking, step_limit, prefix_dir = sys.argv[1:]

with open(base_config) as f:
    cfg = yaml.safe_load(f)

if step_limit:
    cfg.setdefault('agent', {})['step_limit'] = int(step_limit)
if prefix_dir:
    cfg.setdefault('agent', {})['prefix_trajectory_dir'] = prefix_dir

cfg['model']['model_kwargs']['api_base'] = agent_api_base

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
echo "R2E-Gym Subset Run"
echo "Setting:     ${SETTING} k=${PRM_INTERVAL} run=${RUN} (MAX ${EFFECTIVE_STEPS} STEPS)"
echo "Agent model: ${AGENT_MODEL} @ ${AGENT_API_BASE}"
if [ "$SETTING" != "base" ]; then
    echo "PRM model:   ${PRM_NAME}"
    [ "$PRM_NAME" != "claude-opus-4-6" ] && echo "PRM server:  ${PRM_API_BASE}"
fi
echo "Prefix dir:  ${PREFIX_DIR}"
echo "Config:      ${CONFIG}"
echo "Output:      ${OUTPUT}"
echo "Slice:       ${SLICE:-all}  Workers: ${WORKERS}"
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
