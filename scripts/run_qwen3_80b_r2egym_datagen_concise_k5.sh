#!/bin/bash
#SBATCH --job-name=q380b_r2egym_datagen
#SBATCH --output=sbatch_logs/q380b_r2egym_datagen_%j.out
#SBATCH --error=sbatch_logs/q380b_r2egym_datagen_%j.err
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# qwen3-next-80b-a3b agent + concise k=5 critic (claude-opus-4-6 PRM) data-gen
# run on the R2E-Gym v1 minus subset (SWE-bench union). Mirror of the CWM
# datagen run at:
#   results_r2egym_swebench/singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_claude-opus-4-6
# but with qwen3-80b as the agent. Slice :500 only for now. No step-aware.
#
# Submit:
#   sbatch scripts/run_qwen3_80b_r2egym_datagen_concise_k5.sh
#
# Attach:
#   ./connect_job.sh <jobid> q380b_r2egym_datagen

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

echo "=== qwen3-80b r2egym datagen (concise k=5, slice :500) starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "=========================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/q380b_r2egym_datagen_${SLURM_JOB_ID:-manual}_inner.sh"
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
  --config mini-swe-agent/configs/r2egym_v1minussubset_singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_qwen3-80b_max150.yaml \\
  --subset /data/user_data/srgandhi/tool-overuse/r2egym_v1_minus_subset_parquet \\
  --split train \\
  --workers 20 \\
  --shuffle \\
  --slice ":500" \\
  --output /home/srgandhi/tool-overuse/results_r2egym_swebench/singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_qwen3-80b_prm_claude-opus-4-6

rc=\$?
echo
echo "=== q380b_r2egym_datagen done at \$(date) (rc=\$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s q380b_r2egym_datagen "$INNER_SCRIPT"

echo "=== Launched tmux session 'q380b_r2egym_datagen' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} q380b_r2egym_datagen"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
