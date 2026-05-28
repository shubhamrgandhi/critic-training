#!/bin/bash
# run_mini-swe-agent_prm_issue_res.sh
#
# Lightweight script for running CWM + PRM (issue resolution) when vLLM is
# already serving elsewhere.  No server startup -- just the agent run.
#
# Usage:
#   ./run_mini-swe-agent_prm_issue_res.sh <prm_interval> <run> <model>
#   ./run_mini-swe-agent_prm_issue_res.sh 5 0 cwm
#   ./run_mini-swe-agent_prm_issue_res.sh 3 0 cwm --slice 10:20
#   ./run_mini-swe-agent_prm_issue_res.sh 5 0 cwm --workers 4 --slice :10
#   ./run_mini-swe-agent_prm_issue_res.sh 5 0 cwm --api-base babel-s5-24:8070

# Suppress Singularity/Apptainer mount warnings from system config
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

# Use /data/user_data/srgandhi for tmp to avoid filling up root filesystem
export TMPDIR=/data/user_data/srgandhi/tmp
mkdir -p "$TMPDIR"

PRM_INTERVAL=$1
RUN=$2
MODEL=$3
shift 3 2>/dev/null

if [ -z "$PRM_INTERVAL" ] || [ -z "$RUN" ] || [ -z "$MODEL" ]; then
    echo "Usage: ./run_mini-swe-agent_prm_issue_res.sh <prm_interval> <run> <model> [extra args]"
    echo "Example: ./run_mini-swe-agent_prm_issue_res.sh 5 0 cwm"
    echo "Example: ./run_mini-swe-agent_prm_issue_res.sh 3 0 cwm --slice :10 --workers 4"
    exit 1
fi

# Defaults (overridable via extra args)
WORKERS=4
SUBSET=verified
SPLIT=test
SLICE=":500"
API_BASE=""

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)   WORKERS="$2";   shift 2 ;;
    --subset)    SUBSET="$2";    shift 2 ;;
    --split)     SPLIT="$2";     shift 2 ;;
    --slice)     SLICE="$2";     shift 2 ;;
    --api-base)  API_BASE="$2";  shift 2 ;;
    *)           echo "Unknown option: $1"; shift ;;
  esac
done

SETTING="singularity_edit_obs_final_only_prm_issue_res_k${PRM_INTERVAL}"
CONFIG="../mini-swe-agent/configs/swebench_${SETTING}_${RUN}_${MODEL}.yaml"
OUTPUT="../results_singularity/${SETTING}_${RUN}_${MODEL}_prm_claude-opus-4-6"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# If --api-base is provided, create a temp config with the overridden api_base
# Accepts either a full URL (http://host:port/v1) or just host:port
if [ -n "$API_BASE" ]; then
    # If it doesn't start with http, wrap it
    if [[ "$API_BASE" != http* ]]; then
        API_BASE="http://${API_BASE}/v1"
    fi
    TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
    sed "s|api_base:.*|api_base: \"${API_BASE}\"|" "$CONFIG" > "$TEMP_CONFIG"
    CONFIG="$TEMP_CONFIG"
    trap "rm -f $TEMP_CONFIG" EXIT
    echo "API base: $API_BASE (overridden)"
fi

echo "Config:  $CONFIG"
echo "Output:  $OUTPUT"
echo "Slice:   $SLICE"
echo "Workers: $WORKERS"

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
