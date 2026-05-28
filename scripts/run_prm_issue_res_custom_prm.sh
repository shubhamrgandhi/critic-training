#!/bin/bash
# run_prm_issue_res_custom_prm.sh
#
# Run cwm agent + configurable PRM model for prm_issue_res setting.
# The agent vLLM server must already be running before calling this.
#
# Usage:
#   ./run_prm_issue_res_custom_prm.sh <prm_interval> <run> <agent_model> \
#       --prm-name      <label>       # used for output dir (e.g. cwm, qwen3-8b)
#       --prm-model-name <model_id>   # e.g. "facebook/cwm"
#       --prm-api-base  <url>         # e.g. "http://babel-s5-24:8070/v1"
#       [--prm-api-key  <key>]        # e.g. HF token
#       [--prm-provider openai]       # litellm provider (default: openai)
#       [--agent-api-base <url>]      # default: http://localhost:8070/v1
#       [--workers N]                 # default: 8
#       [--slice S]                   # default: :50
#
# Examples:
#   ./run_prm_issue_res_custom_prm.sh 5 0 cwm \
#       --prm-name cwm --prm-model-name "facebook/cwm" \
#       --prm-api-base "http://babel-s5-24:8070/v1" \
#       --agent-api-base "http://babel-s5-24:8070/v1"
#
#   ./run_prm_issue_res_custom_prm.sh 5 0 cwm \
#       --prm-name qwen3-8b --prm-model-name "Qwen/Qwen3-8B" \
#       --prm-api-base "https://api-inference.huggingface.co/models/Qwen/Qwen3-8B/v1" \
#       --prm-api-key "$HF_TOKEN" \
#       --agent-api-base "http://babel-s5-24:8070/v1"

export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"
export TMPDIR=/scratch
mkdir -p "$TMPDIR"

PRM_INTERVAL=$1
RUN=$2
AGENT_MODEL=$3
shift 3 2>/dev/null

if [ -z "$PRM_INTERVAL" ] || [ -z "$RUN" ] || [ -z "$AGENT_MODEL" ]; then
    echo "Usage: ./run_prm_issue_res_custom_prm.sh <prm_interval> <run> <agent_model> [options]"
    exit 1
fi

WORKERS=12
SLICE=":500"
AGENT_API_BASE="http://localhost:8070/v1"
PRM_NAME=""
PRM_MODEL_NAME=""
PRM_API_BASE=""
PRM_API_KEY=""
PRM_PROVIDER="openai"
DISABLE_THINKING="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    --prm-name)          PRM_NAME="$2";        shift 2 ;;
    --prm-model-name)    PRM_MODEL_NAME="$2";  shift 2 ;;
    --prm-api-base)      PRM_API_BASE="$2";    shift 2 ;;
    --prm-api-key)       PRM_API_KEY="$2";     shift 2 ;;
    --prm-provider)      PRM_PROVIDER="$2";    shift 2 ;;
    --agent-api-base)    AGENT_API_BASE="$2";  shift 2 ;;
    --workers)           WORKERS="$2";         shift 2 ;;
    --slice)             SLICE="$2";           shift 2 ;;
    --disable-thinking)  DISABLE_THINKING="true"; shift ;;
    *) echo "Unknown option: $1"; shift ;;
  esac
done

if [[ -z "$PRM_NAME" || -z "$PRM_MODEL_NAME" || -z "$PRM_API_BASE" ]]; then
    echo "ERROR: --prm-name, --prm-model-name, and --prm-api-base are all required"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG="${SCRIPT_DIR}/../mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_prm_issue_res_k${PRM_INTERVAL}_${RUN}_${AGENT_MODEL}.yaml"
OUTPUT="${SCRIPT_DIR}/../results_singularity/singularity_edit_obs_final_only_prm_issue_res_k${PRM_INTERVAL}_${RUN}_${AGENT_MODEL}_prm_${PRM_NAME}"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found: $BASE_CONFIG"
    exit 1
fi

TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
trap "rm -f $TEMP_CONFIG" EXIT

# Patch prm_model section and agent api_base using Python (safer than sed for nested YAML)
python3 - "$BASE_CONFIG" "$TEMP_CONFIG" \
    "$AGENT_API_BASE" "$PRM_MODEL_NAME" "$PRM_API_BASE" "$PRM_API_KEY" "$PRM_PROVIDER" "$DISABLE_THINKING" <<'PYEOF'
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

echo "=============================="
echo "Setting:     prm_issue_res k=${PRM_INTERVAL} run=${RUN}"
echo "Agent model: ${AGENT_MODEL} @ ${AGENT_API_BASE}"
echo "PRM model:   ${PRM_MODEL_NAME} @ ${PRM_API_BASE}"
echo "Output:      ${OUTPUT}"
echo "Slice:       ${SLICE}  Workers: ${WORKERS}"
echo "=============================="

mini-extra swebench \
    --config "$TEMP_CONFIG" \
    --subset verified \
    --split test \
    --workers "$WORKERS" \
    --shuffle \
    --slice "$SLICE" \
    --output "$OUTPUT"
