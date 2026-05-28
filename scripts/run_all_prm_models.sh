#!/bin/bash
# run_all_prm_models.sh
#
# Runs all 4 PRM model variants for prm_issue_res k=5 run=0,
# then evaluates and computes stats.
#
# cwm server must already be running (the existing sbatch job stays as-is).
# Qwen/SWE-agent PRM models use HuggingFace Inference API (free, no GPU needed).
#
# Usage:
#   CWM_NODE=babel-s5-24 HF_TOKEN=hf_xxxx ./scripts/run_all_prm_models.sh
#
# Required env vars:
#   CWM_NODE   - compute node running the cwm vllm server (e.g. babel-s5-24)
#   HF_TOKEN   - HuggingFace token for Inference API access

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CWM_NODE="${CWM_NODE:?ERROR: Set CWM_NODE to the compute node running cwm (e.g. export CWM_NODE=babel-s5-24)}"
HF_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN to your HuggingFace token (e.g. export HF_TOKEN=hf_xxxx)}"

AGENT_API_BASE="http://${CWM_NODE}:8070/v1"
HF_BASE="https://router.huggingface.co/hf-inference/models"

run_setting() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo "STARTING: ${name}"
    echo "============================================================"
    if "$@"; then
        echo "--- DONE: ${name} ---"
    else
        echo "--- FAILED: ${name} (continuing to next setting) ---"
    fi
}

# ---- 1. cwm as PRM (reuses existing cwm server, no extra resources needed) ----
run_setting "cwm as PRM" \
    "${SCRIPT_DIR}/run_prm_issue_res_custom_prm.sh" 5 0 cwm \
        --prm-name cwm \
        --prm-model-name "facebook/cwm" \
        --prm-api-base "${AGENT_API_BASE}" \
        --agent-api-base "${AGENT_API_BASE}"

# ---- 2. Qwen2.5-Coder-7B-Instruct as PRM (HF Inference API) ----
run_setting "Qwen2.5-Coder-7B-Instruct as PRM" \
    "${SCRIPT_DIR}/run_prm_issue_res_custom_prm.sh" 5 0 cwm \
        --prm-name qwen25coder7b \
        --prm-model-name "Qwen/Qwen2.5-Coder-7B-Instruct" \
        --prm-api-base "${HF_BASE}/Qwen/Qwen2.5-Coder-7B-Instruct/v1" \
        --prm-api-key "${HF_TOKEN}" \
        --agent-api-base "${AGENT_API_BASE}"

# ---- 3. Qwen3-8B as PRM (HF Inference API, thinking disabled) ----
run_setting "Qwen3-8B as PRM" \
    "${SCRIPT_DIR}/run_prm_issue_res_custom_prm.sh" 5 0 cwm \
        --prm-name qwen3-8b \
        --prm-model-name "Qwen/Qwen3-8B" \
        --prm-api-base "${HF_BASE}/Qwen/Qwen3-8B/v1" \
        --prm-api-key "${HF_TOKEN}" \
        --disable-thinking \
        --agent-api-base "${AGENT_API_BASE}"

# ---- 4. SWE-agent-LM-7B as PRM (HF Inference API) ----
# Note: This model may not be available on HF Serverless Inference.
# If it fails, see the README comment below for alternatives.
run_setting "SWE-agent-LM-7B as PRM" \
    "${SCRIPT_DIR}/run_prm_issue_res_custom_prm.sh" 5 0 cwm \
        --prm-name sweagent7b \
        --prm-model-name "SWE-bench/SWE-agent-LM-7B" \
        --prm-api-base "${HF_BASE}/SWE-bench/SWE-agent-LM-7B/v1" \
        --prm-api-key "${HF_TOKEN}" \
        --agent-api-base "${AGENT_API_BASE}"

# ---- 5. Evaluate all runs ----
echo ""
echo "============================================================"
echo "RUNNING EVAL (eval.sh)"
echo "============================================================"
"${SCRIPT_DIR}/eval.sh"

# ---- 6. Compute stats ----
echo ""
echo "============================================================"
echo "COMPUTING STATS (get_stats_table.py)"
echo "============================================================"
python3 "${SCRIPT_DIR}/get_stats_table.py"

echo ""
echo "All done! Check results_singularity/ for outputs."
