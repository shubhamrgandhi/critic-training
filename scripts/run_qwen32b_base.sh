#!/bin/bash
#SBATCH --job-name=qwen32b_base
#SBATCH --output=sbatch_logs/qwen32b_base_%j.out
#SBATCH --error=sbatch_logs/qwen32b_base_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Run mini-swe-agent base run (no critic) using bedrock/qwen.qwen3-32b-v1:0 as
# the agent. CPU-only because all model inference happens at AWS Bedrock.
#
# Submit:
#   sbatch scripts/run_qwen32b_base.sh
#
# Attach:
#   ./connect_job.sh <jobid> qwen32b
#
# A long-lived tmux server runs inside the job step so the rich progress
# dashboard is visible from any reattach.

# Don't use 'set -e' at the top level — we want the slurm allocation to stay
# alive even if a child fails, so the user can re-launch from inside tmux.
set -o pipefail

# Activate user env (HF_TOKEN, AWS creds, Docker creds, conda)
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true

cd /home/srgandhi/tool-overuse
mkdir -p sbatch_logs

# Suppress mount warnings for Singularity
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

export TMPDIR=/scratch
mkdir -p "$TMPDIR"

echo "=== qwen32b base run starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "AWS_REGION:    ${AWS_REGION:-unset}"
echo "TMPDIR:        $TMPDIR"
echo "Started:       $(date)"
echo "=================================="

# Inner script that the tmux session will run.
INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen32b_base_${SLURM_JOB_ID:-manual}_inner.sh"
cat > "$INNER_SCRIPT" <<'INNER'
#!/bin/bash
set -o pipefail
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true
cd /home/srgandhi/tool-overuse
export TMPDIR=/scratch
mkdir -p "$TMPDIR"
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

mini-extra swebench \
  --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_0_qwen32b_max150.yaml \
  --subset MariusHobbhahn/swe-bench-verified-mini \
  --split test \
  --workers 20 \
  --shuffle \
  --output /home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_qwen32b

rc=$?
echo
echo "=== qwen32b base run done at $(date) (rc=$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

# Start tmux server (creates a placeholder so server stays up)
tmux new-session -d -s __keepalive 2>/dev/null || true
# Launch the run inside its own tmux session
tmux new-session -d -s qwen32b "$INNER_SCRIPT"

echo "=== Launched tmux session 'qwen32b' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} qwen32b"

# Hold the slurm allocation while the tmux server is alive.
trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
