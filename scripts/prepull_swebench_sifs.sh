#!/bin/bash
# Pre-pull SWE-bench Verified docker images to persistent .sif files.
# After this, agent runs read from local .sif and never hit Docker Hub.
#
# Output: $SIF_CACHE/<instance_id>.sif (NFS-shared across compute nodes)
#
# Strategy:
#   1. PARALLEL pulls in parallel
#   2. Rotate Docker Hub credentials per worker (4 accounts x 100 pulls/6h)
#   3. Pull to node-local /scratch first, then mv to NFS (faster unpack)
#   4. Skip if target .sif already exists (re-runnable)
#
# Usage:
#   bash scripts/prepull_swebench_sifs.sh
#
# Env vars:
#   SIF_CACHE     default /data/user_data/srgandhi/tool-overuse/sif_cache
#   PARALLEL      default 16
#   STAGING_DIR   default /scratch/srgandhi_swebench_sif_staging
#   TARGETS_TSV   default scripts/data/swebench_verified_mini_images.tsv

set +e
source ~/.bashrc 2>/dev/null
set -eo pipefail

SIF_CACHE="${SIF_CACHE:-/data/user_data/srgandhi/tool-overuse/sif_cache}"
STAGING_DIR="${STAGING_DIR:-/scratch/srgandhi_swebench_sif_staging}"
PARALLEL="${PARALLEL:-16}"
TARGETS_TSV="${TARGETS_TSV:-/home/srgandhi/tool-overuse/scripts/data/swebench_verified_mini_images.tsv}"

mkdir -p "$SIF_CACHE" "$STAGING_DIR"

export SINGULARITY_TMPDIR="$STAGING_DIR"
export APPTAINER_TMPDIR="$STAGING_DIR"

build_target_list() {
    if [ ! -f "$TARGETS_TSV" ]; then
        echo "ERROR: target mapping file $TARGETS_TSV not found" >&2
        return 1
    fi
    cat "$TARGETS_TSV"
}

pull_one() {
    local iid="$1"
    local image="$2"
    local worker_idx="$3"
    local target="$SIF_CACHE/${iid}.sif"

    if [ -f "$target" ] && [ -s "$target" ]; then
        echo "  [SKIP] $iid (already cached)"
        return 0
    fi

    # Rotate creds: idx 1 -> slot 4, idx 2 -> slot 3, ... idx 5 -> slot 4 ...
    local cred_slot=$(( 4 - ((worker_idx - 1) % 4) ))
    local u_var="DOCKER_USERNAME_${cred_slot}"
    local p_var="DOCKER_PASSWORD_${cred_slot}"
    if [ -n "${!u_var:-}" ] && [ -n "${!p_var:-}" ]; then
        export SINGULARITY_DOCKER_USERNAME="${!u_var}"
        export SINGULARITY_DOCKER_PASSWORD="${!p_var}"
        export APPTAINER_DOCKER_USERNAME="${!u_var}"
        export APPTAINER_DOCKER_PASSWORD="${!p_var}"
    fi

    local staging="$STAGING_DIR/${iid}.sif.partial"
    # Cross-filesystem mv from /scratch to NFS is non-atomic (copy+unlink),
    # which means a reader could see a partial SIF mid-copy. To make the
    # destination appear atomically, we copy to a sibling tmp file on the
    # same NFS mount, then rename — same-fs rename is atomic.
    local tmp_target="${target}.tmp.$$"
    echo "  [pull] $iid (slot $cred_slot)"
    rm -f "$staging" "$tmp_target"
    if singularity pull --force "$staging" "docker://$image" 2>&1 | tail -3 | sed "s|^|    [$iid] |"; then
        if cp "$staging" "$tmp_target" && mv "$tmp_target" "$target"; then
            rm -f "$staging"
            echo "  [OK] $iid"
        else
            rm -f "$staging" "$tmp_target"
            echo "  [FAIL-COPY] $iid"
            return 1
        fi
    else
        rm -f "$staging" "$tmp_target"
        echo "  [FAIL] $iid"
        return 1
    fi
}

export -f pull_one
export SIF_CACHE STAGING_DIR

echo "=== Pre-pulling SWE-bench Verified SIF cache ==="
echo "  cache:    $SIF_CACHE"
echo "  staging:  $STAGING_DIR"
echo "  parallel: $PARALLEL"
echo "  targets:  $TARGETS_TSV"
echo ""

TARGETS_FILE=$(mktemp)
build_target_list > "$TARGETS_FILE"
TOTAL=$(wc -l < "$TARGETS_FILE")
echo "Targets: $TOTAL instances"
echo ""

nl -ba "$TARGETS_FILE" | xargs -P "$PARALLEL" -L 1 bash -c '
    idx="$1"
    iid="$2"
    image="$3"
    pull_one "$iid" "$image" "$idx" || true
' _

rm -f "$TARGETS_FILE"

echo ""
echo "=== Done. Cache contents: ==="
ls -lh "$SIF_CACHE" | head -10
echo "..."
TOTAL_SIFS=$(ls "$SIF_CACHE" | wc -l)
TOTAL_SIZE=$(du -sh "$SIF_CACHE" | awk '{print $1}')
echo "Total: $TOTAL_SIFS SIFs, $TOTAL_SIZE"
