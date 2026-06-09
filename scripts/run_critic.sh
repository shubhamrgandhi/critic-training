#!/bin/bash
# run_critic.sh
#
# Unified script for running mini-swe-agent with any PRM setting and PRM model.
# The agent vLLM server must already be running before calling this.
#
# Usage:
#   ./run_critic.sh <setting> <prm_interval> <run> <agent_model> [options]
#
# Settings:
#   prm_tool, prm_issue_res, prm_tool_issue_res
#   (also supports v0/v1 variants: prm_tool_v0, prm_tool_v1, etc.)
#
# Options:
#   --prm <name>          PRM model (default: claude-opus-4-6)
#                         Built-in vLLM models: cwm, sweagent7b, qwen3-8b, qwen3-8b-opus-distill, qwen25coder7b
#   --prm-node <host>     Node running PRM vLLM server (required for vLLM PRMs)
#   --agent-node <host>   Node running agent vLLM server (default: localhost)
#   --workers N           Number of workers (default: 8)
#   --slice S             Slice spec (default: :50)
#   --api-base <url>      Override agent api_base directly (alternative to --agent-node)
#
# Examples:
#   # Claude Opus PRM (default) — no --prm-node needed
#   ./run_critic.sh prm_tool 5 0 cwm --slice :500
#   ./run_critic.sh prm_issue_res 5 0 cwm --slice :500
#   ./run_critic.sh prm_tool_issue_res 5 0 cwm --slice :500
#
#   # vLLM PRM on a remote node
#   ./run_critic.sh prm_issue_res 5 0 cwm --prm sweagent7b --prm-node babel-1-23
#   ./run_critic.sh prm_tool 5 0 cwm --prm qwen3-8b --prm-node babel-1-23 --agent-node babel-s5-24
#
#   # v0/v1 variants
#   ./run_critic.sh prm_tool_v0 5 0 cwm --slice :500
#   ./run_critic.sh prm_tool_v1_issue_res 3 0 cwm --prm cwm --prm-node babel-5-32

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
    echo "Usage: ./run_critic.sh <setting> <prm_interval> <run> <agent_model> [options]"
    echo ""
    echo "Settings: prm_tool, prm_issue_res, prm_tool_issue_res"
    echo "          (also: prm_tool_v0, prm_tool_v1, prm_tool_v0_issue_res, etc.)"
    echo ""
    echo "Options:"
    echo "  --prm <name>         PRM model (default: claude-opus-4-6)"
    echo "                       vLLM models: cwm, sweagent7b, qwen3-8b, qwen3-8b-opus-distill, qwen25coder7b"
    echo "  --prm-node <host>    Node running PRM vLLM server (required for vLLM PRMs)"
    echo "  --agent-node <host>  Node running agent vLLM server (default: localhost)"
    echo "  --workers N          Number of workers (default: 8)"
    echo "  --slice S            Slice spec (default: :50)"
    echo "  --api-base <url>     Override agent api_base directly"
    echo ""
    echo "Examples:"
    echo "  ./run_critic.sh prm_tool 5 0 cwm --slice :500"
    echo "  ./run_critic.sh prm_issue_res 5 0 cwm --prm sweagent7b --prm-node babel-1-23"
    exit 1
fi

# Defaults
PRM_NAME="claude-opus-4-6"
PRM_NODE=""
AGENT_NODE="localhost"
WORKERS=12
SUBSET=verified
SPLIT=test
SLICE=":50"
API_BASE=""

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
    *)              echo "Unknown option: $1"; shift ;;
  esac
done

# Build config and output paths
FULL_SETTING="singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}"
CONFIG="${SCRIPT_DIR}/../mini-swe-agent/configs/swebench_${FULL_SETTING}_${RUN}_${AGENT_MODEL}.yaml"
OUTPUT="${SCRIPT_DIR}/../results_singularity/${FULL_SETTING}_${RUN}_${AGENT_MODEL}_prm_${PRM_NAME}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Determine agent api_base
if [ -n "$API_BASE" ]; then
    AGENT_API_BASE="$API_BASE"
    if [[ "$AGENT_API_BASE" != http* ]]; then
        AGENT_API_BASE="http://${AGENT_API_BASE}/v1"
    fi
else
    AGENT_API_BASE="http://${AGENT_NODE}:8070/v1"
fi

# --- PRM handling ---
if [ "$PRM_NAME" = "claude-opus-4-6" ]; then
    # Claude Opus: use the YAML config as-is (already has Bedrock prm_model)
    # Only patch agent api_base if overridden
    if [ "$AGENT_NODE" != "localhost" ] || [ -n "$API_BASE" ]; then
        TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
        trap "rm -f $TEMP_CONFIG" EXIT
        sed "s|api_base:.*|api_base: \"${AGENT_API_BASE}\"|" "$CONFIG" > "$TEMP_CONFIG"
        CONFIG="$TEMP_CONFIG"
    fi
else
    # vLLM PRM: need --prm-node to know where the server is
    if [ -z "$PRM_NODE" ]; then
        echo "ERROR: --prm-node is required for vLLM PRM '$PRM_NAME'"
        exit 1
    fi

    # Map prm_name -> model ID, port, extra flags
    DISABLE_THINKING="false"
    case "$PRM_NAME" in
        sweagent7b)
            PRM_MODEL_NAME="SWE-bench/SWE-agent-LM-7B"
            PRM_PORT=8071
            ;;
        qwen3-8b)
            PRM_MODEL_NAME="Qwen/Qwen3-8B"
            PRM_PORT=8071
            DISABLE_THINKING="true"
            ;;
        qwen3-8b-opus-distill)
            PRM_MODEL_NAME="qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6"
            PRM_PORT=8071
            DISABLE_THINKING="true"
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
            # Passthrough: treat PRM_NAME as a literal served model name
            PRM_MODEL_NAME="$PRM_NAME"
            PRM_PORT=8071
            DISABLE_THINKING="true"
            ;;
    esac

    PRM_API_BASE="http://${PRM_NODE}:${PRM_PORT}/v1"

    # Patch config: override prm_model section and agent api_base
    TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
    trap "rm -f $TEMP_CONFIG" EXIT

    python3 - "$CONFIG" "$TEMP_CONFIG" \
        "$AGENT_API_BASE" "$PRM_MODEL_NAME" "$PRM_API_BASE" "" "openai" "$DISABLE_THINKING" <<'PYEOF'
import yaml, sys

base_config, temp_config, agent_api_base, prm_model_name, prm_api_base, prm_api_key, prm_provider, disable_thinking = sys.argv[1:]

with open(base_config) as f:
    cfg = yaml.safe_load(f)

# Override agent model api_base
cfg['model']['model_kwargs']['api_base'] = agent_api_base

# Override prm_model section
prm_kwargs = {
    'custom_llm_provider': prm_provider,
    'api_base': prm_api_base,
    'temperature': 0.0,
    'n': 1,
    'max_completion_tokens': 4096,
    'drop_params': True,
}
if prm_api_key:
    prm_kwargs['api_key'] = prm_api_key
if disable_thinking == 'true':
    prm_kwargs['extra_body'] = {'enable_thinking': False}

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
echo "Setting:     ${SETTING} k=${PRM_INTERVAL} run=${RUN}"
echo "Agent model: ${AGENT_MODEL} @ ${AGENT_API_BASE}"
echo "PRM model:   ${PRM_NAME}"
[ "$PRM_NAME" != "claude-opus-4-6" ] && echo "PRM server:  ${PRM_API_BASE}"
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
