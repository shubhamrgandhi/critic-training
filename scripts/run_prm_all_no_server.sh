#!/bin/bash
#SBATCH --job-name=prm_all_agent
#SBATCH --output=../babel-server/sbatch_logs/prm_all_agent_%j.out
#SBATCH --error=../babel-server/sbatch_logs/prm_all_agent_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=preempt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu

# =============================================================================
# Runs three PRM settings sequentially against an EXISTING vLLM server
# No GPU needed -- just CPU for the agent + Bedrock PRM calls
# =============================================================================

set -e

API_BASE="${API_BASE:-babel-p5-32:8070}"
PRM_INTERVAL="${PRM_INTERVAL:-5}"

source ~/.bashrc
conda activate tool-overuse

cd /home/srgandhi/tool-overuse/scripts

mkdir -p ../babel-server/sbatch_logs

echo "=============================================="
echo "PRM All Settings (no server)"
echo "API base: $API_BASE"
echo "PRM interval: $PRM_INTERVAL"
echo "=============================================="

./run_mini-swe-agent_prm_issue_res.sh "$PRM_INTERVAL" 0 cwm --api-base "$API_BASE"
./run_mini-swe-agent_prm_tool_issue_res.sh "$PRM_INTERVAL" 0 cwm --api-base "$API_BASE"
./run_mini-swe-agent_prm_tool.sh "$PRM_INTERVAL" 0 cwm --api-base "$API_BASE"

echo "All three settings complete."
