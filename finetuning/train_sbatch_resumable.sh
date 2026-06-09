#!/bin/bash
#SBATCH --job-name=prm_sft_train_cwm_qwen
#SBATCH --output=sbatch_logs/prm_sft_train_%j.out
#SBATCH --error=sbatch_logs/prm_sft_train_%j.err
#SBATCH --time=47:30:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40S:8
#SBATCH --mem=400G

set -uo pipefail

# SLURM copies the sbatch script to a scratch dir, so BASH_SOURCE points there.
# Use SLURM_SUBMIT_DIR (set by sbatch to the original working directory) instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

mkdir -p sbatch_logs

# Activate conda env (override CONDA_ENV via env if your env name differs)
CONDA_ENV="${CONDA_ENV:-critic-training}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        echo "ERROR: failed to activate conda env '$CONDA_ENV' from $CONDA_BASE" >&2
        exit 1
    }
    echo "Activated conda env: $CONDA_ENV ($(which python))"
else
    echo "WARNING: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh; using system python" >&2
fi

SAVEDIR="${SAVEDIR:-/data/user_data/$USER/saves}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAVEDIR/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn}"
DONE_SENTINEL="$OUTPUT_DIR/TRAINING_COMPLETE"
HF_PUSHED_SENTINEL="$OUTPUT_DIR/HF_PUSHED"
# Default HF repo: same name as the local OUTPUT_DIR, matching prior pushed-model
# convention (qwen3-8b-full-sft-prm-...). 96-char HF limit on org+name.
HF_REPO_ID="${HF_REPO_ID:-shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn}"

export OUTPUT_DIR

# Try to push the final checkpoint to HF Hub. Idempotent: skips if already pushed
# this run (HF_PUSHED_SENTINEL). Failure is logged but does not fail the job.
# Uses huggingface_hub.HfApi via the conda env python (more robust than relying
# on whichever huggingface-cli is found on PATH).
push_to_hf() {
    if [[ -f "$HF_PUSHED_SENTINEL" ]]; then
        echo "=== HF push: already pushed (sentinel: $HF_PUSHED_SENTINEL); skipping ==="
        return 0
    fi
    if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        echo "WARNING: HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set; skipping HF push" >&2
        return 1
    fi
    echo "=== HF push: $OUTPUT_DIR  ->  $HF_REPO_ID ==="
    python - <<'PYEOF' "$HF_REPO_ID" "$OUTPUT_DIR" "$SLURM_JOB_ID"
import os, sys, traceback
from huggingface_hub import HfApi, create_repo
repo_id, output_dir, job_id = sys.argv[1], sys.argv[2], sys.argv[3]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
try:
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_dir,
        commit_message=f"SFT checkpoint from job {job_id}",
        ignore_patterns=[
            "checkpoint-*/**",     # only push the final merged model, not intermediate ckpts
            "TRAINING_COMPLETE",
            "HF_PUSHED",
            ".tmux_train_rc.*",
            "logs/**",
        ],
        token=token,
    )
    print("HF upload OK")
    sys.exit(0)
except Exception as e:
    print(f"HF upload FAILED: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
PYEOF
    rc=$?
    if [[ $rc -eq 0 ]]; then
        touch "$HF_PUSHED_SENTINEL"
        echo "=== HF push: success ==="
    else
        echo "=== HF push: FAILED (rc=$rc); will retry on next chained job ===" >&2
    fi
    return $rc
}

TMUX_SESSION="prm_sft_${SLURM_JOB_ID}"

echo "=== PRM SFT Training (resumable, self-resubmitting) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Time limit: $(squeue -j $SLURM_JOB_ID -h -o %l 2>/dev/null || echo unknown)"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "=== Live training output ==="
echo "  Tmux:  ssh $(hostname) -t 'tmux attach -t ${TMUX_SESSION}'"
echo "  Tail:  tail -f $SAVEDIR/logs/$(basename "$OUTPUT_DIR")_train.log"
echo ""

# Pre-emptively chain the next job so that if we hit the walltime mid-step,
# the next reservation is already queued. The next job will be a no-op
# if TRAINING_COMPLETE sentinel exists.
NEXT_JOB_ID=""
if [[ -z "${PRM_SFT_NO_CHAIN:-}" ]]; then
    echo "=== Submitting next job in chain (afterany dependency) ==="
    NEXT_JOB_ID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SCRIPT_DIR/train_sbatch_resumable.sh" 2>&1) || {
        echo "WARNING: failed to submit next job in chain: $NEXT_JOB_ID"
        NEXT_JOB_ID=""
    }
    echo "Next job: ${NEXT_JOB_ID:-<none>}"
    echo ""
fi

# Bail out (and free reservation) if training is already complete.
# But before cancelling the chain, try the HF push first — if a prior job
# finished training but failed the upload, this gives us another chance.
if [[ -f "$DONE_SENTINEL" ]]; then
    echo "=== TRAINING_COMPLETE sentinel found: $DONE_SENTINEL ==="
    push_to_hf || true
    echo "=== Cancelling remaining chained job(s) ==="
    if [[ -n "$NEXT_JOB_ID" ]]; then
        scancel "$NEXT_JOB_ID" 2>/dev/null || true
    fi
    exit 0
fi

# Run training inside a tmux session so the user can `tmux attach` on the
# compute node to see live output. Pipe tmux output to a log file too.
RC_FILE="$OUTPUT_DIR/.tmux_train_rc.${SLURM_JOB_ID}"
rm -f "$RC_FILE"

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 50 \
        "bash '$SCRIPT_DIR/run_qwen3_8b_critic_full_sft_l40s_resumable.sh'; \
         echo \$? > '$RC_FILE'; \
         echo '=== training exited (rc=\$(cat $RC_FILE)). Session will close in 60s ==='; \
         sleep 60"

    # Poll until tmux session ends (training finished or job killed)
    while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
        sleep 30
    done

    if [[ -f "$RC_FILE" ]]; then
        TRAIN_RC=$(cat "$RC_FILE")
    else
        TRAIN_RC=1
    fi
else
    echo "WARNING: tmux not found, running training directly (no live attach)" >&2
    bash "$SCRIPT_DIR/run_qwen3_8b_critic_full_sft_l40s_resumable.sh"
    TRAIN_RC=$?
fi

echo ""
echo "=== Training exit code: $TRAIN_RC ==="

if [[ $TRAIN_RC -eq 0 ]]; then
    # Drop sentinel only if accelerate exited cleanly (training fully done)
    touch "$DONE_SENTINEL"
    echo "=== Marked TRAINING_COMPLETE ==="
    push_to_hf || true
    echo "=== Cancelling next chained job ${NEXT_JOB_ID:-<none>} ==="
    if [[ -n "$NEXT_JOB_ID" ]]; then
        scancel "$NEXT_JOB_ID" 2>/dev/null || true
    fi
else
    echo "=== Training did not finish cleanly; chained job ${NEXT_JOB_ID:-<none>} will resume ==="
fi

exit $TRAIN_RC
