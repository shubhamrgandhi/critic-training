#!/bin/bash
# vllm_server_runpod.sh
#
# vLLM server for RunPod with 2x H100 NVL + reverse SSH tunnel to babel.
# Run inside tmux so it survives browser/terminal close.
#
# Usage:
#   tmux new -s vllm './vllm_server_runpod.sh [options]'
#
# Options:
#   --model <name>          Model to serve (default: facebook/cwm)
#   --port <port>           Local vLLM port (default: 8070)
#   --remote-port <port>    Port on remote host (default: same as --port)
#   --remote-host <host>    SSH tunnel target (default: login.babel.cs.cmu.edu)
#   --remote-user <user>    SSH user (default: srgandhi)
#   --no-tunnel             Skip SSH tunnel setup
#   --seed <n>              Random seed for reproducibility (default: 42)
#   --num-gpus <n>          Number of GPUs (default: 2)
#   --compute-node <host>   Babel compute node to forward to (default: babel-p5-20)

set -eo pipefail

# Defaults
MODEL="facebook/cwm"
PORT=8078
REMOTE_PORT=""
REMOTE_HOST="login.babel.cs.cmu.edu"
REMOTE_USER="srgandhi"
COMPUTE_TARGET="babel-p5-20"
NO_TUNNEL=false
SEED=42
NUM_GPUS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL="$2";          shift 2 ;;
    --port)           PORT="$2";           shift 2 ;;
    --remote-port)    REMOTE_PORT="$2";    shift 2 ;;
    --remote-host)    REMOTE_HOST="$2";    shift 2 ;;
    --remote-user)    REMOTE_USER="$2";    shift 2 ;;
    --compute-node)   COMPUTE_TARGET="$2"; shift 2 ;;
    --no-tunnel)      NO_TUNNEL=true;      shift ;;
    --seed)           SEED="$2";           shift 2 ;;
    --num-gpus)       NUM_GPUS="$2";       shift 2 ;;
    *)                echo "Unknown option: $1"; shift ;;
  esac
done

[ -z "$REMOTE_PORT" ] && REMOTE_PORT=$PORT

# --- Environment setup (RunPod) ---
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
export TQDM_DISABLE=1
mkdir -p /workspace/hf_cache/hub

# Fix tqdm compatibility issue with some vLLM versions
python3 << 'PATCH_EOF'
import glob
for path in glob.glob("/usr/local/lib/python*/dist-packages/vllm/model_executor/model_loader/weight_utils.py"):
    try:
        with open(path, 'r') as f:
            content = f.read()
        old = "        super().__init__(*args, **kwargs, disable=True)"
        new = '        kwargs["disable"] = True\n        super().__init__(*args, **kwargs)'
        if old in content:
            with open(path, 'w') as f:
                f.write(content.replace(old, new))
            print(f"Patched {path}")
        else:
            print(f"Already patched or different version: {path}")
    except Exception as e:
        print(f"Warning: Could not patch {path}: {e}")
PATCH_EOF

# Load HuggingFace token
for tokenfile in /workspace/.cache/huggingface/token ~/.cache/huggingface/token ~/.huggingface/token; do
    if [ -f "$tokenfile" ]; then
        export HF_TOKEN=$(cat "$tokenfile")
        echo "HF token loaded from $tokenfile"
        break
    fi
done
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HuggingFace token found. Run 'huggingface-cli login' if accessing gated models."
fi

# Log directory
mkdir -p /workspace/vllm_logs

COMPUTE_NODE=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/workspace/vllm_logs/server_${TIMESTAMP}.log"

echo "=== vLLM Server Starting (RunPod 2x H100 NVL) ==="
echo "Timestamp:  $TIMESTAMP"
echo "Host:       $COMPUTE_NODE"
echo "Model:      $MODEL"
echo "Port:       $PORT"
echo "GPUs:       $NUM_GPUS"
echo "Seed:       $SEED"
echo "Log:        $LOG_FILE"
echo "==================================================="

# --- Determinism / low-variance settings ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# H100 NVL has NVLink — P2P and custom all-reduce work great
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- Start vLLM server ---
echo "Starting vLLM server..."

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$NUM_GPUS" \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --enforce-eager \
  --seed "$SEED" \
  2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!

# --- Wait for server readiness ---
check_server_ready() {
    local max_attempts=600
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 5
        ((attempt++))
        if (( attempt % 12 == 0 )); then
            echo "Still waiting... ($(( attempt * 5 ))s elapsed)"
        fi
    done
    echo "Server failed to start within 50 minutes"
    return 1
}

echo "Waiting for vLLM server to initialize..."
if ! check_server_ready; then
    echo "Server startup failed. Check logs: $LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "=== Server Ready ==="
echo "Local:   http://localhost:$PORT/v1"
echo "Health:  http://localhost:$PORT/health"
echo "====================="

# --- Reverse SSH tunnel (ProxyJump: RunPod -> login node -> compute node) ---
# Single SSH connection using -J to jump through login node, landing directly
# on the compute node with a reverse tunnel.
#
# Result: compute node localhost:REMOTE_PORT -> RunPod :PORT

if [ "$NO_TUNNEL" = false ]; then
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")
    echo ""
    echo "RunPod external IP: $EXTERNAL_IP"
    echo "Setting up reverse SSH tunnel via ProxyJump..."
    echo "  RunPod :${PORT} --(${REMOTE_HOST})--> ${COMPUTE_TARGET} localhost:${REMOTE_PORT}"

    ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -J "${REMOTE_USER}@${REMOTE_HOST}" \
        -R "${REMOTE_PORT}:localhost:${PORT}" \
        "${REMOTE_USER}@${COMPUTE_TARGET}" \
        -N -f

    if [ $? -eq 0 ]; then
        echo "=== Tunnel Active ==="
        echo "From ${COMPUTE_TARGET}: http://localhost:${REMOTE_PORT}/v1"
        echo "======================"
    else
        echo "WARNING: SSH tunnel failed to start."
        echo "Set up keys: ssh-keygen -t ed25519 && ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    fi
fi

# --- Cleanup ---
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Kill local SSH tunnel processes
    pkill -f "ssh.*${REMOTE_PORT}.*${COMPUTE_TARGET}" 2>/dev/null
    pkill -f "ssh.*${REMOTE_PORT}.*${REMOTE_HOST}" 2>/dev/null
    # Kill the remote sshd listener on the compute node (it becomes orphaned otherwise)
    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        -c aes128-ctr,aes192-ctr,aes256-ctr \
        -o MACs=hmac-sha2-256,hmac-sha2-512,hmac-sha1 \
        "${REMOTE_USER}@${REMOTE_HOST}" \
        "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${COMPUTE_TARGET} 'pkill -u ${REMOTE_USER} -f \"sshd.*${REMOTE_PORT}\"'" 2>/dev/null || true
    # Kill vLLM server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server and tunnel stopped."
}
trap cleanup EXIT INT TERM

echo ""
echo "Server running (PID: $SERVER_PID). Press Ctrl+C to stop."
wait $SERVER_PID
