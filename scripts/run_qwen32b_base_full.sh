#!/bin/bash
#SBATCH --job-name=qwen32b_base_full
#SBATCH --output=sbatch_logs/qwen32b_base_full_%j.out
#SBATCH --error=sbatch_logs/qwen32b_base_full_%j.err
#SBATCH --partition=cpu
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Run mini-swe-agent base run (no critic) on FULL SWE-bench Verified using
# bedrock/qwen.qwen3-32b-v1:0. CPU-only because all model inference is at
# AWS Bedrock; container ops use prepulled SIFs when available.
#
# Submit:
#   sbatch scripts/run_qwen32b_base_full.sh
#
# Attach:
#   ./connect_job.sh <jobid> qwen32b_full

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

# Use the prepulled SIF cache when available (falls back to docker:// pull)
export SWEBENCH_SIF_CACHE="${SWEBENCH_SIF_CACHE:-/data/user_data/srgandhi/tool-overuse/sif_cache}"

echo "=== qwen32b base run (full verified) starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "AWS_REGION:    ${AWS_REGION:-unset}"
echo "TMPDIR:        $TMPDIR"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Started:       $(date)"
echo "================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen32b_base_full_${SLURM_JOB_ID:-manual}_inner.sh"
cat > "$INNER_SCRIPT" <<INNER
#!/bin/bash
set -o pipefail
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true
cd /home/srgandhi/tool-overuse
export TMPDIR=/scratch
mkdir -p "\$TMPDIR"
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"
export SWEBENCH_SIF_CACHE="$SWEBENCH_SIF_CACHE"

mini-extra swebench \\
  --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_0_qwen32b_max150.yaml \\
  --subset princeton-nlp/SWE-bench_Verified \\
  --split test \\
  --workers 20 \\
  --shuffle \\
  --output /home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_0_qwen32b

rc=\$?
echo
echo "=== qwen32b base full run done at \$(date) (rc=\$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s qwen32b_full "$INNER_SCRIPT"

echo "=== Launched tmux session 'qwen32b_full' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} qwen32b_full"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
