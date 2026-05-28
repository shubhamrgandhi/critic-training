#!/bin/bash
#SBATCH --job-name=prm_qwen32b_concise_k10_full500
#SBATCH --output=sbatch_logs/prm_qwen32b_concise_k10_full500_%j.out
#SBATCH --error=sbatch_logs/prm_qwen32b_concise_k10_full500_%j.err
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Full SWE-bench Verified (500) qwen3-32b agent + concise/instructions
# k=10 PRM (claude-opus-4-6, Bedrock — no tunnel required), prefixed from
# the existing 0_qwen32b base run. Mirror of the cwm reference run at:
#   results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_prm_issue_res_instructions_k10_0_cwm_prm_claude-opus-4-6
#
# Submit:
#   sbatch scripts/run_prm_qwen32b_full500_concise_k10.sh
#
# Attach:
#   ./connect_job.sh <jobid> prm_qwen32b_concise_k10_full

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

echo "=== qwen32b agent + concise k=10 PRM (claude-opus-4-6) full 500 ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "===================================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/prm_qwen32b_concise_k10_full500_${SLURM_JOB_ID:-manual}_inner.sh"
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

./run_prm_max150_mini.sh prm_issue_res_instructions 10 0 qwen32b \
  --no-agent-api-base \
  --workers 20 \
  --subset princeton-nlp/SWE-bench_Verified \
  --split test \
  --slice ":500"

rc=$?
echo
echo "=== prm_qwen32b_concise_k10_full500 done at $(date) (rc=$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s prm_qwen32b_concise_k10_full "$INNER_SCRIPT"

echo "=== Launched tmux session 'prm_qwen32b_concise_k10_full' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} prm_qwen32b_concise_k10_full"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
