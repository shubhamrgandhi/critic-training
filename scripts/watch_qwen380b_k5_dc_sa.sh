#!/usr/bin/env bash
# Watch qwen3-80b k=5 D/C step-aware inference (job passed as $1, default
# 8140352). When preds.json hits 500: scancel SLURM, then wait for any
# running aws_eval.sh to finish (sequential mutex), then fire aws_eval.sh.
#
# Run detached:
#   nohup ./scripts/watch_qwen380b_k5_dc_sa.sh 8140352 \
#     >> sbatch_logs/watch_qwen380b_k5_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown

set -uo pipefail
cd /home/srgandhi/tool-overuse

ROOT="/home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix"
LOG_DIR="/home/srgandhi/tool-overuse/sbatch_logs"
LOG="$LOG_DIR/watch_qwen380b_k5_dc_sa_$(date +%Y%m%d_%H%M%S).log"

JOB="${1:-8140352}"
RD="singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k5_0_qwen3-80b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_preds() {
    python3 -c "import json,sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)" "$1" 2>/dev/null
}

job_alive() {
    [ -n "$(squeue -j "$1" -h -o "%T" 2>/dev/null)" ]
}

# Sequential mutex: count other aws_eval.sh procs (not our pid, not our shell tree)
n_other_evals() {
    pgrep -af 'aws_eval.sh' 2>/dev/null \
        | grep -v -E "(pgrep|grep|$$)" \
        | wc -l \
        | tr -d ' \n'
}

wait_for_eval_slot() {
    while true; do
        local n
        n=$(n_other_evals)
        # n must be a single integer
        if [ "${n:-1}" = "0" ]; then
            return 0
        fi
        log "  [eval-mutex] waiting; $n other aws_eval.sh process(es) running"
        sleep 60
    done
}

log "starting qwen3-80b k=5 D/C step-aware watcher. log: $LOG  job: $JOB"
log "watching $RD"

# Phase 1: poll until 500 preds or job dies
while true; do
    n=0
    if [ -f "$ROOT/$RD/preds.json" ]; then
        n=$(count_preds "$ROOT/$RD/preds.json")
    fi
    if [ "$n" -ge 500 ]; then
        log "preds=500/500. cancelling job $JOB."
        scancel "$JOB" 2>/dev/null || true
        sleep 5
        break
    elif ! job_alive "$JOB"; then
        log "job $JOB no longer in queue (preds=$n/500). proceeding to eval."
        break
    else
        log "preds=$n/500 (job $JOB alive)"
        sleep 60
    fi
done

# Phase 2: wait for any other aws_eval to finish, then fire ours
log "queueing eval for $RD (waits for sequential slot)"
wait_for_eval_slot

stamp=$(date +%Y%m%d_%H%M%S)
elog="$LOG_DIR/aws_eval_${RD}_${stamp}.log"
log "firing aws_eval.sh (foreground; log: $elog)"
bash -c "cd /home/srgandhi/tool-overuse && ./scripts/aws_eval.sh '$RD'" \
    > "$elog" 2>&1
rc=$?
log "aws_eval.sh exited rc=$rc"

log "ALL DONE."
