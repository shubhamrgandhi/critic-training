#!/bin/bash
#SBATCH --job-name=cwm_swe_agent
#SBATCH --output=../babel-server/sbatch_logs/cwm_base_%j.out
#SBATCH --error=../babel-server/sbatch_logs/cwm_base_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-o5-28 babel-p9-20"


# =============================================================================
# Combined vLLM Server + Mini-SWE-Agent (CWM base, no PRM)
# =============================================================================
# Usage:
#   sbatch scripts/run-mini-swe-agent-with-server_cwm.sh
#   sbatch scripts/run-mini-swe-agent-with-server_cwm.sh --slice :10
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)     PORT="$2";     shift 2 ;;
    --workers)  WORKERS="$2";  shift 2 ;;
    --subset)   SUBSET="$2";   shift 2 ;;
    --split)    SPLIT="$2";    shift 2 ;;
    --slice)    SLICE="$2";    shift 2 ;;
    *)          echo "Unknown option: $1"; shift ;;
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

export TMPDIR=/data/user_data/srgandhi/tmp
export SINGULARITY_TMPDIR=/data/user_data/srgandhi/tmp
export APPTAINER_TMPDIR=/data/user_data/srgandhi/tmp
mkdir -p $TMPDIR

cd /home/srgandhi/tool-overuse/scripts

mkdir -p ../babel-server/sbatch_logs
mkdir -p ../babel-server/vllm_logs

COMPUTE_NODE=$(hostname)

CONFIGS_DIR="/home/srgandhi/tool-overuse/mini-swe-agent/configs"
CONFIG="${CONFIGS_DIR}/swebench_singularity_edit_obs_final_only_0_${MODEL_NAME}.yaml"
RESULTS_DIR="/home/srgandhi/tool-overuse/results_singularity"
OUTPUT="${RESULTS_DIR}/singularity_edit_obs_final_only_0_${MODEL_NAME}"

echo "=============================================="
echo "CWM Base (no PRM)"
echo "=============================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Compute node:   $COMPUTE_NODE"
echo "Model:          $MODEL"
echo "Port:           $PORT"
echo "Workers:        $WORKERS"
echo "Slice:          $SLICE"
echo "Config:         $CONFIG"
echo "Output:         $OUTPUT"
echo "=============================================="

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

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
    if [ "$CLEANUP_DONE" -eq 1 ]; then return; fi
    CLEANUP_DONE=1
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill -TERM -$SERVER_PID 2>/dev/null || true
        pkill -TERM -P $SERVER_PID 2>/dev/null || true
        kill -TERM $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 -$SERVER_PID 2>/dev/null || true
        pkill -9 -P $SERVER_PID 2>/dev/null || true
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    pkill -9 -u $USER -f "singularity exec" 2>/dev/null || true
    pkill -9 -u $USER -f "apptainer exec" 2>/dev/null || true
    pkill -9 -u $USER -f "singularity build" 2>/dev/null || true
    pkill -9 -u $USER -f "apptainer build" 2>/dev/null || true
    rm -rf /tmp/minisweagent-* 2>/dev/null || true
    rm -rf $TMPDIR/minisweagent-* 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Wait for server to be ready
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Waiting for vLLM server to initialize..."

check_server_ready() {
    local max_attempts=600
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
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
# Run the mini-swe-agent
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Running mini-swe-agent..."

conda activate tool-overuse

mkdir -p ../agent_logs

export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

AGENT_LOG_FILE="../agent_logs/agent_base_${MODEL_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Config:     $CONFIG"
echo "Output:     $OUTPUT"
echo "Agent log:  $AGENT_LOG_FILE"

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
    --output "$OUTPUT" 2>&1 | tee "$AGENT_LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=============================================="
echo "Completed. Exit code: $EXIT_CODE"
echo "Results:   $OUTPUT"
echo "Agent log: $AGENT_LOG_FILE"
echo "vLLM log:  $VLLM_LOG_FILE"
echo "=============================================="

cleanup
exit $EXIT_CODE
