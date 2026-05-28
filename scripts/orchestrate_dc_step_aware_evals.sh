#!/usr/bin/env bash
# Watch the two D/C step-aware k=10 qwen runs (job 8133350 = qwen3-80b,
# 8133351 = qwen32b). When each hits 500 preds, scancel the SLURM job and
# queue an aws_eval.sh run. Evals are run SEQUENTIALLY (one at a time) to
# avoid clogging the eval host.
#
# Run inside tmux on the login node:
#   tmux new -s dc_sa_evals
#   cd /home/srgandhi/tool-overuse
#   ./scripts/orchestrate_dc_step_aware_evals.sh

set -uo pipefail
cd /home/srgandhi/tool-overuse

ROOT="/home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix"
LOG_DIR="/home/srgandhi/tool-overuse/sbatch_logs"
LOG="$LOG_DIR/dc_sa_evals_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Two parallel runs: jobid|run_dir_basename
declare -a RUNS=(
  "8133350|singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_qwen3-80b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
  "8133351|singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_qwen32b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
)

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

# Track which runs are still pending (not yet evaluated)
declare -A DONE_INFER=()   # run_basename -> 1 once preds=500 (or job died)
declare -A DONE_EVAL=()    # run_basename -> 1 once aws_eval.sh fired

# Returns 0 if any run is still doing inference (job alive AND not yet at 500)
any_pending_infer() {
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r jid rd <<< "$entry"
        if [ -z "${DONE_INFER[$rd]:-}" ]; then
            return 0
        fi
    done
    return 1
}

run_eval_blocking() {
    local rb="$1"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local elog="$LOG_DIR/aws_eval_${rb}_${stamp}.log"
    log "  [eval] firing aws_eval.sh for $rb (foreground; log: $elog)"
    bash -c "cd /home/srgandhi/tool-overuse && ./scripts/aws_eval.sh '$rb'" \
        > "$elog" 2>&1
    local rc=$?
    log "  [eval] aws_eval.sh for $rb exited rc=$rc"
    DONE_EVAL[$rb]=1
}

log "starting dc_sa eval orchestrator. log: $LOG"
log "watching ${#RUNS[@]} runs"

# Phase 1: poll inference progress; queue completed runs for eval
while any_pending_infer; do
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r jid rd <<< "$entry"
        [ -n "${DONE_INFER[$rd]:-}" ] && continue

        local_n=0
        if [ -f "$ROOT/$rd/preds.json" ]; then
            local_n=$(count_preds "$ROOT/$rd/preds.json")
        fi

        if [ "$local_n" -ge 500 ]; then
            log "[$rd] preds=500/500. cancelling job $jid."
            scancel "$jid" 2>/dev/null || true
            sleep 5
            DONE_INFER[$rd]=1
            log "[$rd] queued for sequential eval."
            run_eval_blocking "$rd"
        elif ! job_alive "$jid"; then
            log "[$rd] job $jid no longer in queue (preds=$local_n/500). marking infer-done; will eval whatever is there."
            DONE_INFER[$rd]=1
            run_eval_blocking "$rd"
        else
            log "[$rd] preds=$local_n/500 (job $jid alive)"
        fi
    done
    any_pending_infer && sleep 60
done

# Phase 2: any runs that were marked infer-done but somehow skipped eval
for entry in "${RUNS[@]}"; do
    IFS='|' read -r jid rd <<< "$entry"
    if [ -n "${DONE_INFER[$rd]:-}" ] && [ -z "${DONE_EVAL[$rd]:-}" ]; then
        log "[fallback-eval] $rd"
        run_eval_blocking "$rd"
    fi
done

log "ALL DONE. orchestrator exiting at $(date)"
