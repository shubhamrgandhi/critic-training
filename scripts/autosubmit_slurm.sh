#!/bin/bash
#SBATCH --job-name=autosubmit
#SBATCH --output=sbatch_logs/autosubmit_%j.out
#SBATCH --error=sbatch_logs/autosubmit_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srgandhi@andrew.cmu.edu
#SBATCH --exclude="babel-n5-20,babel-x5-28,babel-w5-32, babel-x5-32"
#
# Run the autosubmit replay over a results dir on a CPU node.
#
# Submit:
#   sbatch --export=ALL,RESULTS_DIR=<absolute path>,WORKERS=8 autosubmit_slurm.sh
#
# This:
#   - Builds Singularity sandboxes per non-Submitted instance
#   - Replays bash commands then runs the canonical agent submit
#   - Writes <RESULTS_DIR>/preds-autosubmit.json (NEW file; never touches preds.json)
#   - Logs progress every few completions
#
# READ-ONLY guarantees: never modifies preds.json, traj.json, or report.json.

set -eo pipefail

if [ -z "${RESULTS_DIR:-}" ]; then
    echo "ERROR: RESULTS_DIR not set. Submit with --export=ALL,RESULTS_DIR=..."
    exit 1
fi
WORKERS="${WORKERS:-8}"

# Suppress mount warnings the existing run scripts also suppress
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

# Sandbox directory on local NVMe (matches existing scripts)
export TMPDIR=/scratch
mkdir -p "$TMPDIR"

SCRIPT_DIR="/home/srgandhi/tool-overuse/scripts"
cd /home/srgandhi/tool-overuse
mkdir -p sbatch_logs

COMPUTE_NODE="$(hostname)"

echo "=== Autosubmit replay started ==="
echo "Job ID:        ${SLURM_JOB_ID:-(none)}"
echo "Compute node:  $COMPUTE_NODE"
echo "Results dir:   $RESULTS_DIR"
echo "Workers:       $WORKERS"
echo "Started:       $(date)"
echo "==============================="

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: results dir does not exist: $RESULTS_DIR"
    exit 1
fi
if [ ! -f "$RESULTS_DIR/preds.json" ]; then
    echo "WARNING: $RESULTS_DIR has no preds.json — Submitted entries will use info.submission as fallback"
fi

OUT="$RESULTS_DIR/preds-autosubmit.json"
PROGRESS="$RESULTS_DIR/preds-autosubmit-progress.json"

if [ -f "$OUT" ]; then
    echo "NOTE: $OUT already exists. The replay script writes to a temp file and atomically replaces, so this run will overwrite. (preds.json itself is safe.)"
fi

# Activate conda env so 'singularity' is on PATH (this is a /usr/bin binary anyway)
source ~/.bashrc 2>/dev/null || true
conda activate tool-overuse 2>/dev/null || true

RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then
    RESUME_ARG="--resume"
fi

python3 "$SCRIPT_DIR/autosubmit_replay.py" \
    --results-dir "$RESULTS_DIR" \
    --output "$OUT" \
    --workers "$WORKERS" \
    --progress-log "$PROGRESS" \
    $RESUME_ARG

echo "=== Done at $(date) ==="
echo "Output: $OUT"
