#!/bin/bash
#SBATCH --job-name=cwm_vllm_server
#SBATCH --output=sbatch_logs/vllm_server_%j.out
#SBATCH --error=sbatch_logs/vllm_server_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-q5-20, babel-q5-16"


# Default values
MODEL="facebook/cwm"
PORT=8070

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Load environment
source ~/.bashrc
conda activate vllm

export HF_HOME=/data/user_data/srgandhi/cache
export TRANSFORMERS_CACHE=/data/user_data/srgandhi/cache
export HF_HUB_CACHE=/data/user_data/srgandhi/cache/hub
mkdir -p /data/user_data/srgandhi/cache

# Navigate to script directory
cd /home/srgandhi/babel-server

# Create log directories
mkdir -p vllm_logs
mkdir -p sbatch_logs

# Get the compute node hostname
COMPUTE_NODE=$(hostname)
echo "=== vLLM Server Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $COMPUTE_NODE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Direct endpoint: http://$COMPUTE_NODE:$PORT"
echo "=========================="

# Export environment variables for vLLM
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

# Start vLLM server with proper logging
LOG_FILE="vllm_logs/server_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting vLLM server on $COMPUTE_NODE:$PORT"
echo "Log file: $LOG_FILE"

# Auto-restart wrapper: if vllm exits for any reason, restart it after 10s.
# This protects against vllm crashes, OOMs, and clean exits (which the
# previous version did not recover from). Trapped EXIT/INT/TERM still cleans up.
(
  while true; do
    vllm serve $MODEL \
      --host 0.0.0.0 \
      --port $PORT \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.9 \
      --dtype bfloat16 \
      --max-num-batched-tokens 262144 \
      --max-num-seqs 8 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --swap-space 16 \
      --enforce-eager \
      --seed $((40 + PORT % 10)) \
      2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    echo "[$(date)] vllm serve exited (rc=$rc); restarting in 10s..." | tee -a "$LOG_FILE"
    sleep 10
  done
) &

SERVER_PID=$!

# Function to check if server is ready
check_server_ready() {
    local max_attempts=6000
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Waiting for server to be ready... (attempt $((attempt+1))/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    echo "Server failed to start within expected time"
    return 1
}

# Wait for server to be ready
echo "Waiting for vLLM server to initialize..."
if check_server_ready; then
    echo "=== Server Ready ==="
    echo "Direct endpoint: http://$COMPUTE_NODE:$PORT"
    echo "Health check: http://$COMPUTE_NODE:$PORT/health"
    echo "API docs: http://$COMPUTE_NODE:$PORT/docs"
    echo "OpenAI-compatible endpoint: http://$COMPUTE_NODE:$PORT/v1/chat/completions"
    echo ""
    echo "To access from other babel nodes, use:"
    echo "  curl http://$COMPUTE_NODE:$PORT/health"
    echo "  or use the hostname in your client code"
    echo "=================="
else
    echo "Server startup failed. Check logs: $LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Keep the job running and monitor the server
echo "Server is running with PID: $SERVER_PID"
echo "Monitoring server process..."

# Wait for the server process to finish
wait $SERVER_PID
