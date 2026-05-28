#!/bin/bash
# run_prm_issue_res_node.sh
#
# Run mini-swe-agent prm_issue_res (k=5, run=0, cwm agent) pointing to
# a separately-running PRM vllm server on another node.
#
# Both the cwm agent server and the PRM vllm server must be running before calling this.
#
# Usage:
#   ./run_prm_issue_res_node.sh <prm_name> <prm_node> [cwm_node]
#
# Arguments:
#   prm_name   PRM model to use:
#                sweagent7b    -> SWE-bench/SWE-agent-LM-7B      (port 8071)
#                qwen3-8b      -> Qwen/Qwen3-8B                  (port 8071, thinking disabled)
#                qwen25coder7b -> Qwen/Qwen2.5-Coder-7B-Instruct (port 8071)
#                cwm           -> facebook/cwm                   (port 8070)
#   prm_node   Hostname of node running the PRM vllm server (e.g. babel-1-23)
#   cwm_node   Hostname of node running the cwm agent server (default: localhost)
#
# Examples:
#   ./run_prm_issue_res_node.sh sweagent7b    babel-1-23
#   ./run_prm_issue_res_node.sh qwen3-8b      babel-1-23
#   ./run_prm_issue_res_node.sh qwen25coder7b babel-1-23
#   ./run_prm_issue_res_node.sh cwm           babel-5-32

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRM_NAME="${1:-cwm}"
PRM_NODE="${2:-localhost}"
CWM_NODE="${3:-localhost}"

if [[ -z "$PRM_NAME" || -z "$PRM_NODE" ]]; then
    echo "Usage: $0 <prm_name> <prm_node> [cwm_node]"
    echo "  prm_name:  sweagent7b | qwen3-8b | qwen25coder7b | cwm"
    echo "  prm_node:  hostname of node running PRM vllm server (e.g. babel-aa-bb)"
    echo "  cwm_node:  hostname of node running cwm agent server (default: localhost)"
    exit 1
fi

# Map prm_name -> model ID, port, extra flags
case "$PRM_NAME" in
    sweagent7b)
        PRM_MODEL_NAME="SWE-bench/SWE-agent-LM-7B"
        PRM_PORT=8071
        EXTRA_ARGS=()
        ;;
    qwen3-8b)
        PRM_MODEL_NAME="Qwen/Qwen3-8B"
        PRM_PORT=8071
        EXTRA_ARGS=(--disable-thinking)
        ;;
    qwen25coder7b)
        PRM_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
        PRM_PORT=8071
        EXTRA_ARGS=()
        ;;
    cwm)
        PRM_MODEL_NAME="facebook/cwm"
        PRM_PORT=8070
        EXTRA_ARGS=()
        ;;
    *)
        echo "ERROR: Unknown prm_name '$PRM_NAME'."
        echo "  Valid: sweagent7b | qwen3-8b | qwen25coder7b | cwm"
        exit 1
        ;;
esac

AGENT_API_BASE="http://${CWM_NODE}:8070/v1"
PRM_API_BASE="http://${PRM_NODE}:${PRM_PORT}/v1"

echo "=============================="
echo "PRM:    $PRM_NAME ($PRM_MODEL_NAME)"
echo "PRM:    $PRM_API_BASE"
echo "Agent:  $AGENT_API_BASE"
echo "=============================="

# Activate tool-overuse env if mini-extra not on PATH
if ! command -v mini-extra &>/dev/null; then
    source ~/.bashrc
    conda activate tool-overuse
fi

"${SCRIPT_DIR}/run_prm_issue_res_custom_prm.sh" 5 0 cwm \
    --prm-name       "$PRM_NAME" \
    --prm-model-name "$PRM_MODEL_NAME" \
    --prm-api-base   "$PRM_API_BASE" \
    --agent-api-base "$AGENT_API_BASE" \
    "${EXTRA_ARGS[@]}"
