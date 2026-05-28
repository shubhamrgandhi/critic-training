#!/bin/bash
#SBATCH --job-name=qwen32b_full500
#SBATCH --output=sbatch_logs/qwen32b_full500_%j.out
#SBATCH --error=sbatch_logs/qwen32b_full500_%j.err
#SBATCH --partition=cpu
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#SBATCH --dependency=afterany:8036050
#
# Full SWE-bench Verified (500) qwen3-32b base run, extending the existing
# 0_qwen32b output dir (which already contains 50 mini results).
# The mini-swe-agent batch runner skips instances already in preds.json,
# so this only runs the 450 non-mini instances.
#
# Submit:
#   sbatch scripts/run_qwen32b_full500.sh
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

export SWEBENCH_SIF_CACHE="${SWEBENCH_SIF_CACHE:-/data/user_data/srgandhi/tool-overuse/sif_cache}"

echo "=== qwen32b base run (full 500) starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "============================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen32b_full500_${SLURM_JOB_ID:-manual}_inner.sh"
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
echo "=== qwen32b full500 done at \$(date) (rc=\$rc) ==="
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
