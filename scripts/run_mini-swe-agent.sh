#!/bin/bash
# run_mini-swe-agent.sh

SETTING=$1
RUN=$2
MODEL=$3

if [ -z "$SETTING" ] || [ -z "$RUN" ] || [ -z "$MODEL" ]; then
    echo "Usage: ./run_mini-swe-agent.sh <setting> <run> <model>"
    echo "Example: ./run_mini-swe-agent.sh base 1 cwm"
    exit 1
fi

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_${SETTING}_${RUN}_${MODEL}.yaml" \
    --subset verified \
    --split test \
    --workers 8 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/${SETTING}_${RUN}_${MODEL}"