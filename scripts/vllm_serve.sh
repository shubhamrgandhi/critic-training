#!/bin/bash

# Default values
MODEL="SWE-bench/SWE-agent-LM-32B"
PORT=8085

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
      echo "Unknown option: $1"
      echo "Usage: $0 [--model MODEL_NAME] [--port PORT]"
      exit 1
      ;;
  esac
done

# Load environment
source ~/.bashrc
conda activate vllm

# Set up cache directories in home directory (where you have permissions)
export HF_HOME=~/hf_cache
export TRANSFORMERS_CACHE=~/hf_cache
export HF_HUB_CACHE=~/hf_cache/hub
mkdir -p ~/hf_cache

# Navigate to script directory
cd ~/tool-overuse/scripts

# Create log directory
mkdir -p vllm_logs

# Get hostname
HOSTNAME=$(hostname)
echo "=== vLLM Server Starting on Local VM ==="
echo "Hostname: $HOSTNAME"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Cache directory: ~/hf_cache"
echo "Endpoint: http://localhost:$PORT"
echo "External: http://$HOSTNAME:$PORT"
echo "========================================"

# Export environment variables for vLLM
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

# Create log file
LOG_FILE="vllm_logs/server_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"

# Start vLLM server
# For 32B model on 2x RTX A4500 (20GB each), you'll need:
# - Quantization (AWQ or GPTQ) OR use a smaller model
# - Tensor parallelism across both GPUs
# - Conservative memory settings

echo "Starting vLLM server..."
echo "Note: 32B model requires quantized version (AWQ/GPTQ) for these GPUs"

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --max-num-seqs 2 \
  2>&1 | tee $LOG_FILE &

SERVER_PID=$!

# Function to check if server is ready
check_server_ready() {
    local max_attempts=300
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
    echo "Health check: http://localhost:$PORT/health"
    echo "API docs: http://localhost:$PORT/docs"
    echo "OpenAI-compatible endpoint: http://localhost:$PORT/v1/chat/completions"
    echo "===================="
else
    echo "Server startup failed. Check logs: $LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Keep the script running and monitor the server
echo "Server is running with PID: $SERVER_PID"
echo "Press Ctrl+C to stop the server"

# Wait for the server process to finish
wait $SERVER_PID