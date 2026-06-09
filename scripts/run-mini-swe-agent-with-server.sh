#!/bin/bash
#SBATCH --job-name=devstral_mini_swe_agent
#SBATCH --output=../../babel-server/sbatch_logs/swe_agent_%j.out
#SBATCH --error=../../babel-server/sbatch_logs/swe_agent_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=256G
#SBATCH --exclude="babel-n9-28,babel-n9-32"

# =============================================================================
# Combined vLLM Server + Mini-SWE-Agent Script
# =============================================================================
# This script:
# 1. Starts a vLLM server in the background
# 2. Waits for the server to be ready
# 3. Runs the mini-swe-agent with the correct endpoint
# 4. Cleans up on exit
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration - modify these as needed
# -----------------------------------------------------------------------------
MODEL="${MODEL:-mistralai/Devstral-Small-2507}"
PORT="${PORT:-8081}"
WORKERS="${WORKERS:-8}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-}"

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

export HF_HOME=/data/user_data/$USER/cache
export TRANSFORMERS_CACHE=/data/user_data/$USER/cache
export HF_HUB_CACHE=/data/user_data/$USER/cache/hub
mkdir -p /data/user_data/$USER/cache

# Use a temp directory with more space for Singularity builds
export TMPDIR=/data/user_data/$USER/tmp
export SINGULARITY_TMPDIR=/data/user_data/$USER/tmp
export APPTAINER_TMPDIR=/data/user_data/$USER/tmp
mkdir -p $TMPDIR

# Navigate to project directory
cd $REPO_ROOT/scripts

# Create log directories
mkdir -p ../../babel-server/sbatch_logs
mkdir -p ../../babel-server/vllm_logs

# Get the compute node hostname (for logging purposes)
COMPUTE_NODE=$(hostname)

echo "=============================================="
echo "Combined vLLM Server + Mini-SWE-Agent"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $COMPUTE_NODE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Server endpoint: http://localhost:$PORT"
echo "Workers: $WORKERS"
echo "=============================================="

# -----------------------------------------------------------------------------
# vLLM Server Setup
# -----------------------------------------------------------------------------
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_P2P_DISABLE=1

VLLM_LOG_FILE="../../babel-server/vllm_logs/server_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[1/3] Starting vLLM server..."
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
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --tool-call-parser mistral \
  > $VLLM_LOG_FILE 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID"

# -----------------------------------------------------------------------------
# Cleanup function
# -----------------------------------------------------------------------------
CLEANUP_DONE=0
cleanup() {
    # Prevent running cleanup twice
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1

    echo ""
    echo "Cleaning up..."

    # Kill the vLLM server and all its child processes
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID: $SERVER_PID) and child processes..."
        # Kill the process group (negative PID kills all processes in the group)
        kill -TERM -$SERVER_PID 2>/dev/null || true
        # Also try killing by parent PID in case process group didn't work
        pkill -TERM -P $SERVER_PID 2>/dev/null || true
        # Kill the main process
        kill -TERM $SERVER_PID 2>/dev/null || true
        # Wait for processes to terminate
        sleep 2
        # Force kill if still running
        kill -9 -$SERVER_PID 2>/dev/null || true
        pkill -9 -P $SERVER_PID 2>/dev/null || true
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi

    # Kill any orphaned singularity/apptainer processes from this user
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

# Activate the mini-swe-agent environment
conda activate "${CONDA_ENV:-critic-training}"

# Create agent log directory
mkdir -p ../agent_logs

# Suppress Singularity/Apptainer mount warnings from system config
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

# -----------------------------------------------------------------------------
# Define experiments to run: "SETTING RUN"
# Comment/uncomment lines as needed
# -----------------------------------------------------------------------------
EXPERIMENTS=(
    "singularity 0"
    # "prompt_efficient_edit_obs_final_only 1"
    # "prompt_efficient_edit_obs_diff 1"
)

MODEL_NAME="${MODEL_NAME:-Devstral-Small-2507}"
CONFIGS_DIR="$REPO_ROOT/mini-swe-agent/configs"
RESULTS_DIR="$REPO_ROOT/results_singularity"

OVERALL_EXIT_CODE=0

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    read -r SETTING RUN <<< "$EXPERIMENT"

    CONFIG="${CONFIGS_DIR}/swebench_${SETTING}_${RUN}_${MODEL_NAME}.yaml"
    OUTPUT="${RESULTS_DIR}/${SETTING}_${RUN}_${MODEL_NAME}"
    AGENT_LOG_FILE="../agent_logs/agent_${SETTING}_${RUN}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "----------------------------------------------"
    echo "Running experiment: SETTING=${SETTING} RUN=${RUN}"
    echo "Config: $CONFIG"
    echo "Output: $OUTPUT"
    echo "Agent log: $AGENT_LOG_FILE"
    echo "----------------------------------------------"

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config file not found: $CONFIG"
        OVERALL_EXIT_CODE=1
        continue
    fi

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
    echo "Experiment ${SETTING}_${RUN} completed with exit code: $EXIT_CODE"
    echo "Results saved to: $OUTPUT"

    if [ $EXIT_CODE -ne 0 ]; then
        OVERALL_EXIT_CODE=$EXIT_CODE
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed. Overall exit code: $OVERALL_EXIT_CODE"
echo "=============================================="

# Explicit cleanup before exit (trap should handle this, but just in case)
cleanup

exit $OVERALL_EXIT_CODE
