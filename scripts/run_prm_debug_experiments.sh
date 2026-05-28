#!/bin/bash
# run_prm_debug_experiments.sh
#
# Master orchestration script: submits one debug partition job per PRM setting,
# waits for each to finish before starting the next, then runs eval + stats.
#
# The cwm vllm server must already be running (your existing sbatch job).
# Each debug job starts its own 7B PRM server + runs the eval, then exits.
# cwm-as-PRM reuses the existing cwm server directly (no extra server needed).
#
# Usage (run from tmux on the login node):
#   export CWM_NODE=babel-s5-24
#   ./scripts/run_prm_debug_experiments.sh
#
# Required env vars:
#   CWM_NODE  - hostname of the compute node running the cwm vllm server

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CWM_NODE="${CWM_NODE:?ERROR: Set CWM_NODE (e.g. export CWM_NODE=babel-s5-24)}"

echo "=== PRM Debug Experiments ==="
echo "CWM node: $CWM_NODE"
echo "Time:     $(date)"
echo ""

# Verify cwm server is reachable before starting
echo "Checking cwm server at http://${CWM_NODE}:8070/health ..."
if ! curl -s --max-time 10 "http://${CWM_NODE}:8070/health" > /dev/null 2>&1; then
    echo "ERROR: cwm server not reachable at ${CWM_NODE}:8070. Is the sbatch job running?"
    exit 1
fi
echo "cwm server OK."
echo ""

submit_and_wait() {
    local prm_name="$1"
    local prm_model="$2"
    local disable_thinking="${3:-false}"

    echo "============================================================"
    echo "Submitting debug job: prm_name=${prm_name}"
    echo "  model:            ${prm_model}"
    echo "  disable_thinking: ${disable_thinking}"
    echo "============================================================"

    sbatch --wait \
        --job-name="prm_${prm_name}" \
        --export="ALL,PRM_MODEL_NAME=${prm_model},PRM_NAME=${prm_name},CWM_NODE=${CWM_NODE},DISABLE_THINKING=${disable_thinking}" \
        "${SCRIPT_DIR}/slurm_prm_debug_job.sh"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "--- DONE: ${prm_name} ---"
    else
        echo "--- FAILED: ${prm_name} (exit code ${exit_code}) ---"
    fi
    echo ""
}

# ---- 1. SWE-agent-LM-7B as PRM ----
submit_and_wait "sweagent7b" "SWE-bench/SWE-agent-LM-7B" "false"

# ---- 2. Qwen2.5-Coder-7B-Instruct as PRM ----
submit_and_wait "qwen25coder7b" "Qwen/Qwen2.5-Coder-7B-Instruct" "false"

# ---- 3. Qwen3-8B as PRM (thinking disabled) ----
submit_and_wait "qwen3-8b" "Qwen/Qwen3-8B" "true"

# ---- 4. cwm as PRM (reuses existing cwm server, no local server) ----
submit_and_wait "cwm" "facebook/cwm" "false"

# ---- Eval + Stats ----
echo "============================================================"
echo "RUNNING EVAL"
echo "============================================================"
"${SCRIPT_DIR}/eval.sh"

echo ""
echo "============================================================"
echo "COMPUTING STATS"
echo "============================================================"
python3 "${SCRIPT_DIR}/get_stats_table.py"

echo ""
echo "=== All done. $(date) ==="
