#!/bin/bash
# Watch Qwen Bedrock runs and trigger AWS Docker eval as each finishes.
# Each run is "done" when its preds.json has 50 entries (full mini set).
# Once done: scancel the slurm job, run aws_eval.sh on its output dir.
#
# Usage: bash scripts/watch_qwen_evals.sh
#
# Edit JOBS array below to add/remove runs to watch.

set -uo pipefail

# Format: "jobid:run_dir_basename"
JOBS=(
    "8035863:singularity_edit_obs_final_only_2_qwen32b"
    "8035864:singularity_edit_obs_final_only_0_qwen3-235b"
    "8035865:singularity_edit_obs_final_only_0_qwen3-80b"
    "8036046:singularity_edit_obs_final_only_1_qwen3-80b"
    "8036047:singularity_edit_obs_final_only_2_qwen3-80b"
)

RESULTS_ROOT="/home/srgandhi/tool-overuse/results_singularity_max_150_steps_prefix"
EXPECTED=50  # mini set size
LOG="/home/srgandhi/tool-overuse/logs/watch_qwen_evals_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

count_preds() {
    local f="$1/preds.json"
    [ -f "$f" ] || { echo 0; return; }
    python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo 0
}

job_alive() {
    local jid="$1"
    squeue -u srgandhi -j "$jid" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|COMPLETING"
}

declare -A DONE
for j in "${JOBS[@]}"; do DONE["$j"]=0; done

echo "=== watcher started at $(date) ==="
echo "Watching ${#JOBS[@]} jobs."

while true; do
    all_done=1
    for j in "${JOBS[@]}"; do
        if [ "${DONE[$j]}" -eq 1 ]; then continue; fi
        all_done=0
        jid="${j%%:*}"
        rd="${j##*:}"
        full="$RESULTS_ROOT/$rd"
        n=$(count_preds "$full")
        echo "[$(date +%H:%M:%S)] $rd (job $jid): preds=$n/$EXPECTED"

        if [ "$n" -ge "$EXPECTED" ]; then
            echo "  -> done. cancelling job $jid"
            scancel "$jid" 2>&1 || true
            sleep 5
            echo "  -> running aws_eval.sh for $rd"
            if bash /home/srgandhi/tool-overuse/scripts/aws_eval.sh --workers=8 "$rd"; then
                resolved=$(python3 -c "import json; print(len(json.load(open('$full/report.json')).get('resolved_ids',[])))" 2>/dev/null || echo "?")
                echo "  ✓ $rd: $resolved/50 resolved"
            else
                echo "  ✗ aws_eval failed for $rd (will not retry; rerun manually)"
            fi
            DONE["$j"]=1
        elif ! job_alive "$jid"; then
            echo "  ! job $jid no longer alive but only $n/$EXPECTED preds — marking watched, you may need to investigate"
            DONE["$j"]=1
        fi
    done

    if [ "$all_done" -eq 1 ]; then
        echo "=== all watched runs evaluated at $(date) ==="
        break
    fi
    sleep 120
done
