#!/bin/bash
#SBATCH --job-name=prepull_swebench
#SBATCH --output=sbatch_logs/prepull_swebench_%j.out
#SBATCH --error=sbatch_logs/prepull_swebench_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32,babel-x5-32"
#
# Pre-pull SWE-bench Verified mini docker images to a persistent SIF cache.
#
# Submit:
#   sbatch scripts/run_prepull_swebench_sifs.sh
#
# Attach (live progress in tmux):
#   ./connect_job.sh <jobid> prepull

set -o pipefail

source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true

cd /home/srgandhi/tool-overuse
mkdir -p sbatch_logs

# Use local NVMe for staging during the pull, NFS for final landing
export SIF_CACHE="${SIF_CACHE:-/data/user_data/srgandhi/tool-overuse/sif_cache}"
export STAGING_DIR="${STAGING_DIR:-/scratch/srgandhi_swebench_sif_staging}"
export PARALLEL="${PARALLEL:-32}"
export TARGETS_TSV="${TARGETS_TSV:-/home/srgandhi/tool-overuse/scripts/data/swebench_verified_full_images.tsv}"

mkdir -p "$SIF_CACHE" "$STAGING_DIR"

echo "=== prepull starting ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $(hostname)"
echo "SIF_CACHE:     $SIF_CACHE"
echo "STAGING_DIR:   $STAGING_DIR"
echo "PARALLEL:      $PARALLEL"
echo "TARGETS_TSV:   $TARGETS_TSV"
echo "Started:       $(date)"
echo "========================"

# Inner script run inside tmux for live attach
INNER_SCRIPT="/home/srgandhi/tool-overuse/sbatch_logs/prepull_swebench_${SLURM_JOB_ID:-manual}_inner.sh"
cat > "$INNER_SCRIPT" <<INNER
#!/bin/bash
set -o pipefail
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true
cd /home/srgandhi/tool-overuse

export SIF_CACHE="$SIF_CACHE"
export STAGING_DIR="$STAGING_DIR"
export PARALLEL="$PARALLEL"
export TARGETS_TSV="$TARGETS_TSV"

bash /home/srgandhi/tool-overuse/scripts/prepull_swebench_sifs.sh
rc=\$?
echo
echo "=== prepull done at \$(date) (rc=\$rc) ==="
echo "Press any key or detach with Ctrl-b d. Tmux session will remain."
exec bash
INNER
chmod +x "$INNER_SCRIPT"

tmux new-session -d -s __keepalive 2>/dev/null || true
tmux new-session -d -s prepull "$INNER_SCRIPT"

echo "=== Launched tmux session 'prepull' on $(hostname) ==="
echo "Inner script: $INNER_SCRIPT"
echo "To attach:    ./connect_job.sh ${SLURM_JOB_ID:-<jobid>} prepull"

trap 'tmux kill-server 2>/dev/null || true; echo "Reservation released."; exit 0' INT TERM
while true; do
    sleep 600
done
