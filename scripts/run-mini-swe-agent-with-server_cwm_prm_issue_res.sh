#!/bin/bash
#SBATCH --job-name=cwm_vllm_server
#SBATCH --output=../babel-server/sbatch_logs/vllm_server_%j.out
#SBATCH --error=../babel-server/sbatch_logs/vllm_server_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-p9-28 babel-n9-32 babel-n5-28"

# =============================================================================
# Combined vLLM Server + PRM Issue Res (CWM model as both agent and PRM)
# =============================================================================

set -e

MODEL="facebook/cwm"
PORT=8070

# -----------------------------------------------------------------------------
# Setup environment (vllm for server)
# -----------------------------------------------------------------------------
source ~/.bashrc
conda activate vllm

export HF_HOME=/data/user_data/srgandhi/cache
export TRANSFORMERS_CACHE=/data/user_data/srgandhi/cache
export HF_HUB_CACHE=/data/user_data/srgandhi/cache/hub
mkdir -p /data/user_data/srgandhi/cache

mkdir -p /home/srgandhi/babel-server/vllm_logs
mkdir -p /home/srgandhi/babel-server/sbatch_logs

COMPUTE_NODE=$(hostname)
echo "=== vLLM Server Starting ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Compute node: $COMPUTE_NODE"
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "============================"

# -----------------------------------------------------------------------------
# Start vLLM server
# -----------------------------------------------------------------------------
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

VLLM_LOG_FILE="/home/srgandhi/babel-server/vllm_logs/server_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $VLLM_LOG_FILE"

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --enforce-eager \
  --seed $((40 + PORT % 10)) \
  2>&1 | tee $VLLM_LOG_FILE &

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
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# Wait for server to be ready
# -----------------------------------------------------------------------------
echo ""
echo "Waiting for vLLM server to initialize..."

check_server_ready() {
    local max_attempts=6000
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
        echo "Waiting for server to be ready... (attempt $((attempt+1))/$max_attempts)"
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
echo "Direct endpoint: http://$COMPUTE_NODE:$PORT"
echo "Health check:    http://$COMPUTE_NODE:$PORT/health"
echo "===================="

# -----------------------------------------------------------------------------
# Run prm_issue_res
# -----------------------------------------------------------------------------
echo ""
echo "Running run_prm_issue_res_node.sh cwm..."

cd /home/srgandhi/tool-overuse/scripts
conda activate tool-overuse

./run_prm_issue_res_node.sh cwm
EXIT_CODE=$?

echo ""
echo "run_prm_issue_res_node.sh completed. Exit code: $EXIT_CODE"
echo "vLLM log: $VLLM_LOG_FILE"

cleanup
exit $EXIT_CODE
