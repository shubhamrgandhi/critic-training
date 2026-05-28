#!/bin/bash
#SBATCH --job-name=qwen3_80b_run1_fb35
#SBATCH --output=sbatch_logs/qwen3_80b_run1_fb35_%j.out
#SBATCH --error=sbatch_logs/qwen3_80b_run1_fb35_%j.err
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# qwen3-next-80b-a3b run-1 covering ONLY the 35 SWE-bench Verified instances
# where base run-0 (singularity_edit_obs_final_only_0_qwen3-80b) did not
# submit a patch (exit_status != "Submitted"). These are the instances for
# which run-0 cannot serve as the base for a critic-run fallback, so we need
# a run-1 result to cover them.
#
# The output directory already contains 50 instances from the earlier mini50
# expansion. The runner skips existing entries in preds.json, so this only
# extends the directory with the 35 new instances.
#
# Submit:
#   sbatch scripts/run_qwen3_80b_run1_fallback35.sh
#
# Attach:
#   ./connect_job.sh <jobid> qwen3_80b_run1_fb35

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

echo "=== qwen3-80b run-1 (35 fallback instances) starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "=========================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen3_80b_run1_fb35_${SLURM_JOB_ID:-manual}_inner.sh"
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

# Anchored alternation over the 35 instance_ids that base run-0 did not submit.
FALLBACK_REGEX='^(django__django-10554|django__django-11133|django__django-11728|django__django-13344|django__django-13964|django__django-14349|django__django-14725|django__django-15128|django__django-15382|django__django-15554|django__django-16315|matplotlib__matplotlib-22871|matplotlib__matplotlib-26208|pytest-dev__pytest-10081|scikit-learn__scikit-learn-13142|scikit-learn__scikit-learn-13439|scikit-learn__scikit-learn-13496|scikit-learn__scikit-learn-14983|scikit-learn__scikit-learn-25102|sphinx-doc__sphinx-9673|sympy__sympy-12481|sympy__sympy-13877|sympy__sympy-14976|sympy__sympy-16766|sympy__sympy-16792|sympy__sympy-19040|sympy__sympy-19495|sympy__sympy-20428|sympy__sympy-21379|sympy__sympy-21612|sympy__sympy-22456|sympy__sympy-23262|sympy__sympy-23534|sympy__sympy-23824|sympy__sympy-24443)$'

mini-extra swebench \\
  --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_1_qwen3-80b_max150.yaml \\
  --subset princeton-nlp/SWE-bench_Verified \\
  --split test \\
  --workers 20 \\
  --filter "\$FALLBACK_REGEX" \\
  --output /home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_1_qwen3-80b

rc=\$?
echo
echo "=== qwen3_80b run1 fb35 done at \$(date) (rc=\$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s qwen3_80b_run1_fb35 "$INNER_SCRIPT"

echo "=== Launched tmux session 'qwen3_80b_run1_fb35' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} qwen3_80b_run1_fb35"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
