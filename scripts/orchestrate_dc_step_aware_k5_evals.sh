#!/usr/bin/env bash
# Chain orchestrator for D/C step-aware k=5 qwen runs.
#
# Behavior:
#   1. Watches qwen32b k=5 (jobid passed as arg, default 8134453). When it
#      hits 500 preds: scancel SLURM, then queue aws_eval.sh sequentially
#      (waits for any other aws_eval.sh / orchestrator eval to finish first).
#   2. Submits qwen3-80b k=5 sbatch, captures its jobid.
#   3. Watches qwen3-80b k=5. When it hits 500 preds: scancel and queue
#      aws_eval.sh sequentially.
#
# Sequential mutex: before firing an eval, waits until no other
# aws_eval.sh process is running (either from this orchestrator or from
# the prior orchestrate_dc_step_aware_evals.sh).
#
# Run inside tmux on the login node:
#   tmux new -s dc_sa_k5_evals
#   cd /home/srgandhi/tool-overuse
#   ./scripts/orchestrate_dc_step_aware_k5_evals.sh

set -uo pipefail
cd /home/srgandhi/tool-overuse

ROOT="/home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix"
LOG_DIR="/home/srgandhi/tool-overuse/sbatch_logs"
LOG="$LOG_DIR/dc_sa_k5_evals_$(date +%Y%m%d_%H%M%S).log"

QWEN32B_JOB="${1:-8134453}"
QWEN32B_RUN="singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k5_0_qwen32b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
QWEN380B_RUN="singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k5_0_qwen3-80b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
QWEN380B_SBATCH="scripts/run_prm_qwen3_80b_full500_d_trained_concise_step_aware_k5.sh"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_preds() {
    local f="$1"
    python3 -c "import json,sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)" "$f" 2>/dev/null || echo 0
}

job_alive() {
    local jobid="$1"
    local state
    state=$(squeue -j "$jobid" -h -o "%T" 2>/dev/null || echo "")
    [ -n "$state" ]
}

# Sequential mutex: wait until no other aws_eval.sh is running.
wait_for_eval_slot() {
    local rb="$1"
    while true; do
        local n
        n=$(pgrep -af 'aws_eval.sh' | grep -v "$$" | grep -v "$0" | wc -l || echo 0)
        if [ "$n" -le 0 ]; then
            return 0
        fi
        log "  [eval-mutex] $rb waiting; $n other aws_eval.sh process(es) running"
        sleep 60
    done
}

run_eval_blocking() {
    local rb="$1"
    wait_for_eval_slot "$rb"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local elog="$LOG_DIR/aws_eval_${rb}_${stamp}.log"
    log "  [eval] firing aws_eval.sh for $rb (foreground; log: $elog)"
    bash -c "cd /home/srgandhi/tool-overuse && ./scripts/aws_eval.sh '$rb'" \
        > "$elog" 2>&1
    local rc=$?
    log "  [eval] aws_eval.sh for $rb exited rc=$rc"
}

watch_until_done() {
    local jid="$1"
    local rd="$2"
    while true; do
        local n=0
        if [ -f "$ROOT/$rd/preds.json" ]; then
            n=$(count_preds "$ROOT/$rd/preds.json")
        fi
        if [ "$n" -ge 500 ]; then
            log "[$rd] preds=500/500. cancelling job $jid."
            scancel "$jid" 2>/dev/null || true
            sleep 5
            return 0
        elif ! job_alive "$jid"; then
            log "[$rd] job $jid no longer in queue (preds=$n/500). marking infer-done."
            return 0
        else
            log "[$rd] preds=$n/500 (job $jid alive)"
            sleep 60
        fi
    done
}

log "starting dc_sa_k5 chain orchestrator. log: $LOG"
log "phase1: watching qwen32b k=5 (job $QWEN32B_JOB)"
watch_until_done "$QWEN32B_JOB" "$QWEN32B_RUN"

log "phase1: queueing qwen32b k=5 eval (waits for sequential slot)"
run_eval_blocking "$QWEN32B_RUN"

log "phase2: submitting qwen3-80b k=5 sbatch"
SUBMIT_OUT=$(sbatch "$QWEN380B_SBATCH" 2>&1) || {
    log "ERROR: sbatch for qwen3-80b k=5 failed: $SUBMIT_OUT"
    exit 1
}
log "phase2: $SUBMIT_OUT"
QWEN380B_JOB=$(echo "$SUBMIT_OUT" | awk '{print $NF}')
log "phase2: qwen3-80b k=5 jobid=$QWEN380B_JOB"

log "phase3: watching qwen3-80b k=5 (job $QWEN380B_JOB)"
watch_until_done "$QWEN380B_JOB" "$QWEN380B_RUN"

log "phase3: queueing qwen3-80b k=5 eval (waits for sequential slot)"
run_eval_blocking "$QWEN380B_RUN"

log "ALL DONE. dc_sa_k5 chain orchestrator exiting at $(date)"
