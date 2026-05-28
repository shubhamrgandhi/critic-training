#!/usr/bin/env bash
# Evaluate preds-autosubmit.json files via sb-cli.
# Writes report-autosubmit.json next to each preds-autosubmit.json.
# Never modifies the original preds.json or report.json.
#
# Usage:
#   ./scripts/eval_autosubmit.sh                       # eval all preds-autosubmit.json (skip if report-autosubmit.json exists)
#   ./scripts/eval_autosubmit.sh --force               # re-run even if report-autosubmit.json exists
#   ./scripts/eval_autosubmit.sh run_name1 run_name2   # only evaluate specific runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/../results_singularity_max_150_steps_prefix"

FORCE=0
ONLY_RUNS=()
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE=1
    else
        ONLY_RUNS+=("$arg")
    fi
done

if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "Error: results dir not found: $RESULTS_ROOT"
    exit 1
fi

echo "Scanning $RESULTS_ROOT for preds-autosubmit.json files..."
[[ ${#ONLY_RUNS[@]} -gt 0 ]] && echo "Filtering to ${#ONLY_RUNS[@]} specified run(s)."
echo ""

ran=0
skipped=0
failed=0

while IFS= read -r preds_path; do
    run_dir="$(dirname "$preds_path")"
    run_name="$(basename "$run_dir")"
    report_path="$run_dir/report-autosubmit.json"

    if [[ ${#ONLY_RUNS[@]} -gt 0 ]]; then
        match=0
        for r in "${ONLY_RUNS[@]}"; do
            [[ "$run_name" == "$r" ]] && match=1 && break
        done
        [[ "$match" -eq 0 ]] && continue
    fi

    if [[ -f "$report_path" && "$FORCE" -eq 0 ]]; then
        echo "  [done]  ${run_name}  (report-autosubmit.json exists)"
        ((skipped++)) || true
        continue
    fi

    results_basename="$(basename "$RESULTS_ROOT")"
    step_tag="${results_basename#results_singularity_}"
    run_id="${step_tag}__${run_name}_autosubmit_$(date +%Y%m%d)"
    echo "  [eval]  ${run_name}  -> report-autosubmit.json  (run_id: ${run_id})"

    if sb-cli submit swe-bench_verified test \
        --predictions_path "$preds_path" \
        --output_dir "$run_dir" \
        --run_id "$run_id"; then
        sb_output="${run_dir}/swe-bench_verified__test__${run_id}.json"
        if [[ -f "$sb_output" ]]; then
            mv "$sb_output" "$report_path"
            echo "         -> saved $report_path"
            ((ran++)) || true
        else
            echo "  [warn]  expected output not found: $sb_output"
            ((failed++)) || true
        fi
    else
        echo "  [FAIL]  $run_name"
        ((failed++)) || true
    fi
    echo ""
done < <(find -L "$RESULTS_ROOT" -maxdepth 2 -name preds-autosubmit.json | sort)

echo ""
echo "Done. Ran $ran eval(s), skipped $skipped, failed $failed."
