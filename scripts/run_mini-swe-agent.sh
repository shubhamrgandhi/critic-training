#!/bin/bash
# run_mini-swe-agent.sh
#
# Usage:
#   ./run_mini-swe-agent.sh <setting> <run> <model>
#   ./run_mini-swe-agent.sh singularity_edit_obs_final_only 0 cwm --api-base babel-s5-24:8070
#   ./run_mini-swe-agent.sh base 1 cwm --slice :10 --workers 4

# Suppress Singularity/Apptainer mount warnings from system config
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

export TMPDIR=/scratch
mkdir -p "$TMPDIR"

SETTING=$1
RUN=$2
MODEL=$3
shift 3 2>/dev/null

if [ -z "$SETTING" ] || [ -z "$RUN" ] || [ -z "$MODEL" ]; then
    echo "Usage: ./run_mini-swe-agent.sh <setting> <run> <model> [extra args]"
    echo "Example: ./run_mini-swe-agent.sh singularity_edit_obs_final_only 0 cwm --api-base babel-s5-24:8070"
    exit 1
fi

# Defaults (overridable via extra args)
WORKERS=8
SUBSET=verified
SPLIT=test
SLICE=":500"
API_BASE=""
OUTPUT=""

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)   WORKERS="$2";   shift 2 ;;
    --subset)    SUBSET="$2";    shift 2 ;;
    --split)     SPLIT="$2";     shift 2 ;;
    --slice)     SLICE="$2";     shift 2 ;;
    --api-base)  API_BASE="$2";  shift 2 ;;
    --output)    OUTPUT="$2";    shift 2 ;;
    *)           echo "Unknown option: $1"; shift ;;
  esac
done

CONFIG="../mini-swe-agent/configs/swebench_${SETTING}_${RUN}_${MODEL}.yaml"

# Use results_singularity/ for singularity configs, results/ otherwise (if --output not given)
if [ -z "$OUTPUT" ]; then
    if [[ "$SETTING" == singularity* ]]; then
        OUTPUT="../results_singularity/${SETTING}_${RUN}_${MODEL}"
    else
        OUTPUT="../results/${SETTING}_${RUN}_${MODEL}"
    fi
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# If --api-base is provided, create a temp config with the overridden api_base
if [ -n "$API_BASE" ]; then
    if [[ "$API_BASE" != http* ]]; then
        API_BASE="http://${API_BASE}/v1"
    fi
    TEMP_CONFIG=$(mktemp /tmp/swebench_config_XXXXXX.yaml)
    sed "s|api_base:.*|api_base: \"${API_BASE}\"|" "$CONFIG" > "$TEMP_CONFIG"
    CONFIG="$TEMP_CONFIG"
    trap "rm -f $TEMP_CONFIG" EXIT
    echo "API base: $API_BASE (overridden)"
fi

echo "Config:   $CONFIG"
echo "Output:   $OUTPUT"
echo "Slice:    $SLICE"
echo "Workers:  $WORKERS"

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