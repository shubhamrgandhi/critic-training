#!/bin/bash
#SBATCH --job-name=prm_qwen3_80b_k5_full500
#SBATCH --output=sbatch_logs/prm_qwen3_80b_k5_full500_%j.out
#SBATCH --error=sbatch_logs/prm_qwen3_80b_k5_full500_%j.err
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Full SWE-bench Verified (500) qwen3-next-80b-a3b agent + concise/instructions
# step-aware k=5 PRM (trained qwen3-8b-full-sft-prm-r2egym-swebench-
# instructions-k5-opus-distill-32k-lr5e6-multiturn served via SSH tunnel to EC2),
# prefixed from the existing 0_qwen3-80b base run.
#
# Submit:
#   sbatch scripts/run_prm_qwen3_80b_k5_full500_concise_step_aware_k5.sh
#
# Attach:
#   ./connect_job.sh <jobid> prm_qwen3_80b_k5_full

set -o pipefail

source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true

cd /home/srgandhi/tool-overuse
mkdir -p sbatch_logs

export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

export TMPDIR=/scratch
mkdir -p "$TMPDIR"

export SWEBENCH_SIF_CACHE="${SWEBENCH_SIF_CACHE:-/data/user_data/srgandhi/tool-overuse/sif_cache}"

echo "=== qwen3-80b agent + concise step_aware k=5 PRM (trained r2egym-swebench-instructions-k5) full 500 ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "===================================================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/prm_qwen3_80b_k5_full500_${SLURM_JOB_ID:-manual}_inner.sh"
cat > "$INNER_SCRIPT" <<'INNER'
#!/bin/bash
set -o pipefail
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true
cd /home/srgandhi/tool-overuse/scripts
export TMPDIR=/scratch
mkdir -p "$TMPDIR"
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"
export SWEBENCH_SIF_CACHE="/data/user_data/srgandhi/tool-overuse/sif_cache"

# ── Local SSH tunnel to EC2 vLLM (PRM critic) at port 8071 ──
# Each compute node maintains its own tunnel; no dependency on babel-x5-24.
EC2_HOST="ubuntu@ec2-52-55-36-179.compute-1.amazonaws.com"
EC2_KEY="/home/srgandhi/.ssh/ec2_red_dev"
LOCAL_PRM_PORT=8071

# Auto-restart loop in background. Keeps tunnel alive if it drops.
keep_tunnel() {
    while true; do
        ssh -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes \
            -i "$EC2_KEY" \
            -N -L "${LOCAL_PRM_PORT}:localhost:8071" \
            "$EC2_HOST"
        echo "[tunnel] dropped at $(date), retrying in 5s..."
        sleep 5
    done
}
keep_tunnel >> /home/srgandhi/tool-overuse/sbatch_logs/prm_qwen3_80b_k5_full500_tunnel_$(hostname).log 2>&1 &
TUNNEL_PID=$!
trap "kill $TUNNEL_PID 2>/dev/null || true" EXIT

# Wait for tunnel to come up (check chat-completions endpoint, not just /v1/models).
echo "[tunnel] waiting for PRM readiness on localhost:${LOCAL_PRM_PORT}..."
for i in $(seq 1 30); do
    if curl -s --max-time 3 "http://localhost:${LOCAL_PRM_PORT}/v1/models" | grep -q "shubhamrgandhi"; then
        echo "[tunnel] ready (attempt $i)"
        break
    fi
    sleep 2
done

curl -s --max-time 5 "http://localhost:${LOCAL_PRM_PORT}/v1/models" | head -c 200
echo

./run_prm_max150_mini.sh prm_issue_res_instructions_step_aware 5 0 qwen3-80b \
  --prm qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn \
  --prm-api-base localhost:8071 \
  --no-agent-api-base \
  --workers 20 \
  --subset princeton-nlp/SWE-bench_Verified \
  --split test \
  --slice ":500"

rc=$?
kill $TUNNEL_PID 2>/dev/null || true
echo
echo "=== prm_qwen3_80b_k5_full500 done at $(date) (rc=$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s prm_qwen3_80b_k5_full "$INNER_SCRIPT"

echo "=== Launched tmux session 'prm_qwen3_80b_k5_full' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} prm_qwen3_80b_k5_full"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
