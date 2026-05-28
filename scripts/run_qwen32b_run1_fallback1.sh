#!/bin/bash
#SBATCH --job-name=qwen32b_run1_fb1
#SBATCH --output=sbatch_logs/qwen32b_run1_fb1_%j.out
#SBATCH --error=sbatch_logs/qwen32b_run1_fb1_%j.err
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Fix-up: prior qwen32b_run1_fb118 job (8051879) completed 117/118 fallback
# instances before it was killed; this picks up the one remaining instance
# (matplotlib__matplotlib-14623).

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

echo "=== qwen32b run-1 fix-up (1 instance) on $(hostname) at $(date) ==="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen32b_run1_fb1_${SLURM_JOB_ID:-manual}_inner.sh"
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
  --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_1_qwen32b_max150.yaml \\
  --subset princeton-nlp/SWE-bench_Verified \\
  --split test \\
  --workers 1 \\
  --filter '^matplotlib__matplotlib-14623$' \\
  --output /home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_1_qwen32b

rc=\$?
echo
echo "=== qwen32b run1 fb1 done at \$(date) (rc=\$rc) ==="
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s qwen32b_run1_fb1 "$INNER_SCRIPT"

echo "=== Launched tmux session 'qwen32b_run1_fb1' on $(hostname) ==="
trap 'tmux kill-server 2>/dev/null || true; exit 0' INT TERM
while true; do
    sleep 600
done
