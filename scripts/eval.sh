#!/usr/bin/env bash
# Evaluate results_singularity runs that have a preds.json but no report.json yet.
# Uses sb-cli to submit predictions and renames the output to report.json.
#
# Usage:
#   ./scripts/eval.sh                        # skip already-evaluated runs
#   ./scripts/eval.sh --force                # re-run even if report.json exists
#   ./scripts/eval.sh --repeat=2             # run eval twice: report.json + report_1.json
#   ./scripts/eval.sh run_name1 run_name2    # only evaluate specific runs
#   ./scripts/eval.sh --force --repeat=2 run_name1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/../results_singularity_max_150_steps_prefix"

FORCE=0
REPEAT=1
ONLY_RUNS=()
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE=1
    elif [[ "$arg" =~ ^--repeat=([0-9]+)$ ]]; then
        REPEAT="${BASH_REMATCH[1]}"
    else
        ONLY_RUNS+=("$arg")
    fi
done

if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "Error: results directory not found: ${RESULTS_ROOT}"
    exit 1
fi

echo "Scanning ${RESULTS_ROOT} for preds.json files..."
if [[ ${#ONLY_RUNS[@]} -gt 0 ]]; then
    echo "Filtering to ${#ONLY_RUNS[@]} specified run(s)."
fi
echo ""

ran=0
skipped=0
failed=0

while IFS= read -r preds_path; do
    run_dir="$(dirname "$preds_path")"
    run_name="$(basename "$run_dir")"
    report_path="${run_dir}/report.json"

    # If specific runs were requested, skip everything else
    if [[ ${#ONLY_RUNS[@]} -gt 0 ]]; then
        match=0
        for r in "${ONLY_RUNS[@]}"; do
            if [[ "$run_name" == "$r" ]]; then match=1; break; fi
        done
        if [[ "$match" -eq 0 ]]; then
            continue
        fi
    fi

    results_basename="$(basename "$RESULTS_ROOT")"
    step_tag="${results_basename#results_singularity_}"

    for (( i=0; i<REPEAT; i++ )); do
        if (( i == 0 )); then
            out_report="${run_dir}/report.json"
        else
            out_report="${run_dir}/report_${i}.json"
        fi

        if [[ -f "$out_report" && "$FORCE" -eq 0 ]]; then
            echo "  [done]  ${run_name}  ($(basename "$out_report") exists)"
            (( skipped++ )) || true
            continue
        fi

        if (( REPEAT > 1 )); then
            run_id="${step_tag}__${run_name}_$(date +%Y%m%d_%H)_r${i}"
        else
            run_id="${step_tag}__${run_name}_$(date +%Y%m%d)"
        fi
        echo "  [eval]  ${run_name}  -> $(basename "$out_report")  (run_id: ${run_id})"

        if sb-cli submit swe-bench_verified test \
            --predictions_path "$preds_path" \
            --output_dir "$run_dir" \
            --run_id "$run_id"; then

            sb_output="${run_dir}/swe-bench_verified__test__${run_id}.json"
            if [[ -f "$sb_output" ]]; then
                mv "$sb_output" "$out_report"
                echo "         -> saved ${out_report}"
            else
                echo "  [warn]  expected output not found: ${sb_output}"
            fi

            (( ran++ )) || true
        else
            echo "  [FAIL]  ${run_name} (repeat ${i})"
            (( failed++ )) || true
        fi
    done

    echo ""
done < <(find -L "$RESULTS_ROOT" -maxdepth 2 -name preds.json | sort)

echo ""
echo "Done. Ran ${ran} eval(s), skipped ${skipped}, failed ${failed}."
