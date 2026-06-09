#!/bin/bash
#SBATCH --job-name=qwen3_8b_prm_vllm
#SBATCH --output=sbatch_logs/vllm_server_%j.out
#SBATCH --error=sbatch_logs/vllm_server_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=preempt
#SBATCH --mem=200G
# Add your own --mail-user / --mail-type / --exclude lines here if you want
# SLURM email notifications or want to exclude flaky nodes.

# Path to your trained PRM checkpoint. Override via --model or by exporting MODEL.
SAVES_DIR="${SAVES_DIR:-/data/user_data/$USER/saves}"
MODEL="${MODEL:-$SAVES_DIR/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
PORT="${PORT:-8071}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --port)  PORT="$2";  shift 2 ;;
    *) shift ;;
  esac
done

source ~/.bashrc
conda activate vllm

CACHE_DIR="${CACHE_DIR:-/data/user_data/$USER/cache}"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR/hub"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1
mkdir -p "$CACHE_DIR"

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p vllm_logs sbatch_logs

COMPUTE_NODE=$(hostname)
LOG_FILE="vllm_logs/server_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "=== vLLM Server Starting (Finetuned PRM) ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Compute node:  $COMPUTE_NODE"
echo "Model path:    $MODEL"
echo "Served as:     $SERVED_NAME"
echo "Port:          $PORT"
echo "Endpoint:      http://$COMPUTE_NODE:$PORT"
echo "============================================="

vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 131072 \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --swap-space 32 \
  --max-num-seqs 16 \
  --hf-overrides '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!

check_server_ready() {
    local attempt=0
    while [ $attempt -lt 6000 ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Waiting for server... (attempt $((attempt+1)))"
        sleep 5
        ((attempt++))
    done
    echo "Server failed to start within expected time."
    return 1
}

echo "Waiting for vLLM server to initialize..."
if check_server_ready; then
    echo "=== Server Ready ==="
    echo "Endpoint: http://$COMPUTE_NODE:$PORT/v1"
    echo "===================="
else
    echo "Server startup failed. Check: $LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}
trap cleanup EXIT INT TERM

echo "Server running (PID: $SERVER_PID). Monitoring..."
wait $SERVER_PID
