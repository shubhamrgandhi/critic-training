#!/bin/bash
#SBATCH --job-name=qwen32b_run1_fb118
#SBATCH --output=sbatch_logs/qwen32b_run1_fb118_%j.out
#SBATCH --error=sbatch_logs/qwen32b_run1_fb118_%j.err
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# qwen3-32b run-1 covering ONLY the 118 SWE-bench Verified instances where
# base run-0 (singularity_edit_obs_final_only_0_qwen32b) did not submit a
# patch (exit_status != "Submitted"; mostly ContextWindowExceededError or
# LimitsExceeded). These are the instances for which run-0 cannot serve as
# the base for a critic-run fallback.
#
# The output directory already contains 50 instances from the earlier mini50
# expansion. The runner skips existing entries in preds.json, so this only
# extends the directory with the 118 new instances.
#
# Submit:
#   sbatch scripts/run_qwen32b_run1_fallback118.sh
#
# Attach:
#   ./connect_job.sh <jobid> qwen32b_run1_fb118

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

echo "=== qwen3-32b run-1 (118 fallback instances) starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SWEBENCH_SIF_CACHE"
echo "Cached SIFs:   $(ls $SWEBENCH_SIF_CACHE 2>/dev/null | wc -l)"
echo "Started:       $(date)"
echo "==========================================================="

INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/qwen32b_run1_fb118_${SLURM_JOB_ID:-manual}_inner.sh"
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

# Anchored alternation over the 118 instance_ids that base run-0 did not submit.
FALLBACK_REGEX='^(astropy__astropy-13453|astropy__astropy-14365|astropy__astropy-14539|astropy__astropy-7606|astropy__astropy-8707|django__django-10097|django__django-11099|django__django-11163|django__django-11211|django__django-11265|django__django-11477|django__django-11490|django__django-11532|django__django-11555|django__django-11728|django__django-11734|django__django-11740|django__django-12663|django__django-12754|django__django-12858|django__django-12965|django__django-13023|django__django-13028|django__django-13033|django__django-13109|django__django-13112|django__django-13343|django__django-13346|django__django-13406|django__django-13449|django__django-13569|django__django-13964|django__django-14017|django__django-14034|django__django-14140|django__django-14155|django__django-14238|django__django-14493|django__django-14787|django__django-14915|django__django-14999|django__django-15104|django__django-15127|django__django-15128|django__django-15252|django__django-15280|django__django-15380|django__django-15503|django__django-15554|django__django-15563|django__django-15629|django__django-15741|django__django-15851|django__django-15930|django__django-15957|django__django-16032|django__django-16145|django__django-16263|django__django-16429|django__django-16454|django__django-16485|django__django-16642|django__django-16661|django__django-16901|django__django-16938|django__django-16950|django__django-17087|matplotlib__matplotlib-14623|matplotlib__matplotlib-20488|matplotlib__matplotlib-20826|matplotlib__matplotlib-22865|matplotlib__matplotlib-23299|matplotlib__matplotlib-23314|matplotlib__matplotlib-26208|psf__requests-1724|psf__requests-1921|psf__requests-2931|pydata__xarray-3151|pydata__xarray-6744|pylint-dev__pylint-4661|pylint-dev__pylint-6386|pylint-dev__pylint-6528|pylint-dev__pylint-7080|pylint-dev__pylint-8898|pytest-dev__pytest-10356|pytest-dev__pytest-5840|pytest-dev__pytest-6197|scikit-learn__scikit-learn-13124|scikit-learn__scikit-learn-13496|scikit-learn__scikit-learn-25973|sphinx-doc__sphinx-10614|sphinx-doc__sphinx-11445|sphinx-doc__sphinx-8593|sympy__sympy-12481|sympy__sympy-13974|sympy__sympy-14248|sympy__sympy-14531|sympy__sympy-15349|sympy__sympy-15599|sympy__sympy-16450|sympy__sympy-16597|sympy__sympy-16792|sympy__sympy-18763|sympy__sympy-19346|sympy__sympy-19495|sympy__sympy-20590|sympy__sympy-20801|sympy__sympy-20916|sympy__sympy-21379|sympy__sympy-22456|sympy__sympy-22714|sympy__sympy-22914|sympy__sympy-23262|sympy__sympy-23413|sympy__sympy-23534|sympy__sympy-24066|sympy__sympy-24213|sympy__sympy-24562)$'

mini-extra swebench \\
  --config mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_1_qwen32b_max150.yaml \\
  --subset princeton-nlp/SWE-bench_Verified \\
  --split test \\
  --workers 20 \\
  --filter "\$FALLBACK_REGEX" \\
  --output /home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix/singularity_edit_obs_final_only_1_qwen32b

rc=\$?
echo
echo "=== qwen32b run1 fb118 done at \$(date) (rc=\$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain so you can rerun."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s qwen32b_run1_fb118 "$INNER_SCRIPT"

echo "=== Launched tmux session 'qwen32b_run1_fb118' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} qwen32b_run1_fb118"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
