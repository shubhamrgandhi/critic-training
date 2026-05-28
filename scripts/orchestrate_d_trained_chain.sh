#!/usr/bin/env bash
# Orchestrate the 4 D-trained PRM runs sequentially.
#
# Watches the currently-running job's preds.json for 500 entries, then:
#   1. Cancels the SLURM job (releases tunnel + node)
#   2. Fires sb-cli eval in background on the login node (no SLURM dep)
#   3. Submits the next sbatch
# Repeats until queue is empty, fires final eval, and exits.
#
# Run inside a tmux session on the login node:
#   tmux new -s d_trained_chain
#   cd /home/srgandhi/tool-overuse
#   ./scripts/orchestrate_d_trained_chain.sh
#   (Ctrl-b d to detach)

set -uo pipefail
cd /home/srgandhi/tool-overuse

ROOT="/home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix"
LOG_DIR="/home/srgandhi/tool-overuse/sbatch_logs"
LOG="$LOG_DIR/d_trained_chain_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Chain entries: jobid_or_NEXT|next_sbatch_or_empty|run_dir_basename
# The first entry's jobid is the currently-running job. Subsequent _NEXT_ values
# are filled in from the prior submission's sbatch output.
declare -a CHAIN=(
  "8130218|scripts/run_prm_qwen3_80b_full500_d_trained_detailed_k10.sh|singularity_edit_obs_final_only_prm_issue_res_k5_0_qwen32b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
  "_NEXT_|scripts/run_prm_qwen32b_full500_d_trained_detailed_k10.sh|singularity_edit_obs_final_only_prm_issue_res_k10_0_qwen3-80b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
  "_NEXT_||singularity_edit_obs_final_only_prm_issue_res_k10_0_qwen32b_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
)

count_preds() {
    local f="$1"
    python3 -c "import json,sys
try:
    print(len(json.load(open(sys.argv[1]))))
except Exception:
    print(0)" "$f" 2>/dev/null || echo 0
}

wait_until_500() {
    local run_basename="$1"
    local jobid="$2"
    local run_dir="$ROOT/$run_basename"
    log "  watching $run_dir/preds.json (job $jobid)"
    local stagnant_loops=0
    local last_n=-1
    while true; do
        # If the SLURM job is no longer running, bail out (let outer caller handle)
        local state
        state=$(squeue -j "$jobid" -h -o "%T" 2>/dev/null || echo "")
        if [ -z "$state" ]; then
            log "  job $jobid no longer in queue (likely finished). proceeding."
            return 0
        fi

        local n=0
        if [ -f "$run_dir/preds.json" ]; then
            n=$(count_preds "$run_dir/preds.json")
        fi
        if [ "$n" != "$last_n" ]; then
            log "  preds=$n/500 (job state=$state)"
            last_n="$n"
            stagnant_loops=0
        else
            stagnant_loops=$((stagnant_loops+1))
            if (( stagnant_loops % 10 == 0 )); then
                log "  preds=$n/500 (no change for ${stagnant_loops}m, job state=$state)"
            fi
        fi
        if [ "$n" -ge 500 ]; then
            return 0
        fi
        sleep 60
    done
}

fire_eval_bg() {
    local run_basename="$1"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local eval_log="$LOG_DIR/aws_eval_${run_basename}_${stamp}.log"
    log "  firing aws_eval.sh for $run_basename (log: $eval_log)"
    nohup bash -c "cd /home/srgandhi/tool-overuse && ./scripts/aws_eval.sh '$run_basename'" \
        > "$eval_log" 2>&1 &
    log "  eval pid: $!"
}

# ── main ──
log "starting orchestrator. log: $LOG"
log "chain length: ${#CHAIN[@]}"

prev_jobid=""
i=0
n=${#CHAIN[@]}
for entry in "${CHAIN[@]}"; do
    i=$((i+1))
    IFS='|' read -r jobid next_sbatch run_basename <<< "$entry"
    if [ "$jobid" = "_NEXT_" ]; then
        jobid="$prev_jobid"
    fi
    log "──────────────────────────────────────────────"
    log "[$i/$n] tracking job=$jobid run=$run_basename"

    wait_until_500 "$run_basename" "$jobid"

    log "[$i/$n] preds reached 500 (or job exited). cancelling job $jobid"
    scancel "$jobid" 2>/dev/null || log "  (scancel returned non-zero)"
    sleep 5

    fire_eval_bg "$run_basename"

    if [ -n "$next_sbatch" ]; then
        log "[$i/$n] submitting next job: $next_sbatch"
        out=$(sbatch "$next_sbatch")
        log "  $out"
        new_jobid=$(echo "$out" | awk '{print $NF}')
        if [[ "$new_jobid" =~ ^[0-9]+$ ]]; then
            prev_jobid="$new_jobid"
            log "  next job id: $prev_jobid"
        else
            log "  ERROR: could not parse jobid from sbatch output. aborting."
            exit 1
        fi
    else
        log "[$i/$n] no more jobs to submit. chain complete."
    fi
done

# Extra: eval the cwm prm_issue_res k=10 D-trained run that already has 500
# preds but no report.json yet. Fire after the chain so evals don't pile up.
EXTRA_RUN="singularity_edit_obs_final_only_prm_issue_res_k10_0_cwm_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn"
if [ -f "$ROOT/$EXTRA_RUN/preds.json" ] && [ ! -f "$ROOT/$EXTRA_RUN/report.json" ]; then
    log "──────────────────────────────────────────────"
    log "[extra] firing eval for $EXTRA_RUN"
    fire_eval_bg "$EXTRA_RUN"
else
    log "[extra] skipping $EXTRA_RUN (preds missing or report exists)"
fi

log "ALL DONE. orchestrator exiting at $(date)"
