#!/bin/bash
#SBATCH --job-name=cwm_prm_all
#SBATCH --output=../babel-server/sbatch_logs/cwm_prm_all_%j.out
#SBATCH --error=../babel-server/sbatch_logs/cwm_prm_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=256G
#SBATCH --partition=preempt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n9-28,babel-n9-32,babel-q9-24"

# =============================================================================
# Combined vLLM Server + Mini-SWE-Agent: runs three PRM settings sequentially
# =============================================================================
# Agent model:  facebook/cwm (self-hosted via vLLM, 8x L40S, TP=8)
# PRM model:    Claude Opus 4.6 (via AWS Bedrock)
# Environment:  singularity_edit_obs_final_only
#
# Settings (run sequentially):
#   1. prm_tool          -- tool overuse mitigation guidance
#   2. prm_issue_res     -- general issue resolution efficiency guidance
#   3. prm_tool_issue_res -- combined tool overuse + issue resolution guidance
#
# Usage:
#   sbatch scripts/run-mini-swe-agent-with-server_cwm_prm_all.sh
#   sbatch scripts/run-mini-swe-agent-with-server_cwm_prm_all.sh --prm-interval 3
#   sbatch scripts/run-mini-swe-agent-with-server_cwm_prm_all.sh --prm-interval 10 --slice 10:20
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL="facebook/cwm"
MODEL_NAME="cwm"
PORT="${PORT:-8070}"
WORKERS="${WORKERS:-8}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-:50}"
PRM_INTERVAL="${PRM_INTERVAL:-5}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --subset)
      SUBSET="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --slice)
      SLICE="$2"
      shift 2
      ;;
    --prm-interval)
      PRM_INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Setup environment
# -----------------------------------------------------------------------------
source ~/.bashrc
conda activate vllm

export HF_HOME=/data/user_data/srgandhi/cache
export TRANSFORMERS_CACHE=/data/user_data/srgandhi/cache
export HF_HUB_CACHE=/data/user_data/srgandhi/cache/hub
mkdir -p /data/user_data/srgandhi/cache

# Use a temp directory with more space for Singularity builds
export TMPDIR=/data/user_data/srgandhi/tmp
export SINGULARITY_TMPDIR=/data/user_data/srgandhi/tmp
export APPTAINER_TMPDIR=/data/user_data/srgandhi/tmp
mkdir -p $TMPDIR

# Navigate to project directory
cd /home/srgandhi/tool-overuse/scripts

# Create log directories
mkdir -p ../babel-server/sbatch_logs
mkdir -p ../babel-server/vllm_logs
mkdir -p ../agent_logs

# Get the compute node hostname
COMPUTE_NODE=$(hostname)

# Define the three settings to run
SETTINGS=("prm_tool" "prm_issue_res" "prm_tool_issue_res")

CONFIGS_DIR="/home/srgandhi/tool-overuse/mini-swe-agent/configs"
RESULTS_DIR="/home/srgandhi/tool-overuse/results_singularity"

echo "=============================================="
echo "CWM + Claude Opus PRM (All Three Settings)"
echo "=============================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Compute node:   $COMPUTE_NODE"
echo "Agent model:    $MODEL"
echo "PRM model:      Claude Opus 4.6 (Bedrock)"
echo "PRM interval:   every $PRM_INTERVAL steps"
echo "Port:           $PORT"
echo "Workers:        $WORKERS"
echo "Subset:         $SUBSET"
echo "Split:          $SPLIT"
echo "Slice:          $SLICE"
echo "Settings:       ${SETTINGS[*]}"
echo "=============================================="

# Validate all configs exist before starting the server
for SETTING in "${SETTINGS[@]}"; do
    CONFIG="${CONFIGS_DIR}/swebench_singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}_0_${MODEL_NAME}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config file not found: $CONFIG"
        echo "Available configs:"
        ls -1 "${CONFIGS_DIR}/swebench_singularity_edit_obs_final_only_${SETTING}_k*_0_${MODEL_NAME}.yaml" 2>/dev/null || echo "  (none found for ${SETTING})"
        exit 1
    fi
    echo "  OK: $CONFIG"
done

# -----------------------------------------------------------------------------
# vLLM Server Setup
# -----------------------------------------------------------------------------
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

VLLM_LOG_FILE="../babel-server/vllm_logs/server_${MODEL_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[1/3] Starting vLLM server..."
echo "Log file: $VLLM_LOG_FILE"

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --max-num-batched-tokens 131072 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --max-num-seqs 16 \
  --enforce-eager \
  --seed $((40 + PORT % 10)) \
  > $VLLM_LOG_FILE 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID"

# -----------------------------------------------------------------------------
# Cleanup function
# -----------------------------------------------------------------------------
CLEANUP_DONE=0
cleanup() {
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1

    echo ""
    echo "Cleaning up..."

    # Kill the vLLM server and all its child processes
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID: $SERVER_PID) and child processes..."
        kill -TERM -$SERVER_PID 2>/dev/null || true
        pkill -TERM -P $SERVER_PID 2>/dev/null || true
        kill -TERM $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 -$SERVER_PID 2>/dev/null || true
        pkill -9 -P $SERVER_PID 2>/dev/null || true
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi

    # Kill any orphaned singularity/apptainer processes
    echo "Killing any orphaned Singularity processes..."
    pkill -9 -u $USER -f "singularity exec" 2>/dev/null || true
    pkill -9 -u $USER -f "apptainer exec" 2>/dev/null || true
    pkill -9 -u $USER -f "singularity build" 2>/dev/null || true
    pkill -9 -u $USER -f "apptainer build" 2>/dev/null || true

    # Clean up temp sandbox directories
    echo "Cleaning up temp directories..."
    rm -rf /tmp/minisweagent-* 2>/dev/null || true
    rm -rf /tmp/build-temp-* 2>/dev/null || true
    rm -rf $TMPDIR/minisweagent-* 2>/dev/null || true
    rm -rf $TMPDIR/build-temp-* 2>/dev/null || true

    echo "Cleanup complete."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Wait for server to be ready
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Waiting for vLLM server to initialize..."

check_server_ready() {
    local max_attempts=600  # 50 minutes max wait
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi

        # Check if server process is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process died unexpectedly"
            echo "Last 50 lines of vLLM log:"
            tail -50 $VLLM_LOG_FILE
            return 1
        fi

        echo "Waiting for server... (attempt $((attempt+1))/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    echo "Server failed to start within expected time"
    return 1
}

if ! check_server_ready; then
    echo "ERROR: vLLM server failed to start. Check logs: $VLLM_LOG_FILE"
    exit 1
fi

echo ""
echo "=== Server Ready ==="
echo "Endpoint: http://localhost:$PORT"
echo "===================="

# -----------------------------------------------------------------------------
# Run the mini-swe-agent for each setting
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Running mini-swe-agent with PRM (all settings)..."

# Activate the mini-swe-agent environment
conda activate tool-overuse

# Suppress Singularity/Apptainer mount warnings
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

SLICE_ARGS=""
if [ -n "$SLICE" ]; then
    SLICE_ARGS="--slice $SLICE"
fi

OVERALL_EXIT=0
RUN_NUM=0

for SETTING in "${SETTINGS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))

    CONFIG="${CONFIGS_DIR}/swebench_singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}_0_${MODEL_NAME}.yaml"
    OUTPUT="${RESULTS_DIR}/singularity_edit_obs_final_only_${SETTING}_k${PRM_INTERVAL}_0_${MODEL_NAME}"
    AGENT_LOG_FILE="../agent_logs/agent_${SETTING}_k${PRM_INTERVAL}_${MODEL_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "=============================================="
    echo "  Setting ${RUN_NUM}/3: ${SETTING}"
    echo "=============================================="
    echo "Config:     $CONFIG"
    echo "Output:     $OUTPUT"
    echo "Agent log:  $AGENT_LOG_FILE"
    echo "Slice:      $SLICE"
    echo "----------------------------------------------"

    mini-extra swebench \
        --config "$CONFIG" \
        --subset "$SUBSET" \
        --split "$SPLIT" \
        --workers "$WORKERS" \
        --shuffle \
        $SLICE_ARGS \
        --output "$OUTPUT" 2>&1 | tee "$AGENT_LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    echo ""
    echo "----------------------------------------------"
    echo "  ${SETTING} completed (exit code: $EXIT_CODE)"
    echo "  Results: $OUTPUT"
    echo "----------------------------------------------"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "WARNING: ${SETTING} exited with code $EXIT_CODE, continuing to next setting..."
        OVERALL_EXIT=1
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed."
echo "  PRM interval:  k=${PRM_INTERVAL}"
echo "  Agent model:   ${MODEL_NAME}"
echo "  PRM model:     Claude Opus 4.6 (Bedrock)"
echo "  Settings run:  ${SETTINGS[*]}"
echo "  vLLM log:      $VLLM_LOG_FILE"
echo "=============================================="

# Explicit cleanup before exit
cleanup

exit $OVERALL_EXIT
