#!/bin/bash
# Convert all iter_* torch distributed checkpoints under a training run directory to HuggingFace format.
#
# Usage:
#   bash scripts/convert_all_checkpoints.sh \
#     --input-dir /tmp/instance_storage/batch_0_200_step_filter \
#     --output-dir /tmp/hf_checkpoints \
#     --origin-hf-dir /path/to/original/hf/model \
#     [--extra-args "--chunk-size 5368709120 --vocab-size 152064"]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_SCRIPT="${SCRIPT_DIR}/../thirdparty/slime/tools/convert_torch_dist_to_hf.py"

INPUT_DIR=""
OUTPUT_DIR=""
ORIGIN_HF_DIR=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --origin-hf-dir)
            ORIGIN_HF_DIR="$2"; shift 2 ;;
        --extra-args)
            EXTRA_ARGS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --input-dir and --output-dir are required."
    echo "Usage: $0 --input-dir <dir> --output-dir <dir> [--origin-hf-dir <dir>] [--extra-args \"...\"]"
    exit 1
fi

if [[ -z "$ORIGIN_HF_DIR" ]]; then
    echo "Warning: --origin-hf-dir not provided. You must pass --model-name via --extra-args."
fi

# Find all iter_* directories, sorted numerically
ITER_DIRS=$(find "$INPUT_DIR" -maxdepth 1 -type d -name "iter_*" | sort)

if [[ -z "$ITER_DIRS" ]]; then
    echo "No iter_* directories found in $INPUT_DIR"
    exit 1
fi

NUM_ITERS=$(echo "$ITER_DIRS" | wc -l)
echo "Found $NUM_ITERS checkpoint(s) to convert."
echo ""

FAILED=0
SKIPPED=0
for ITER_PATH in $ITER_DIRS; do
    ITER_NAME=$(basename "$ITER_PATH")
    ITER_OUTPUT="$OUTPUT_DIR/$ITER_NAME"

    echo "=== Converting $ITER_NAME ==="
    echo "  Input:  $ITER_PATH"
    echo "  Output: $ITER_OUTPUT"

    if [[ -f "$ITER_OUTPUT/config.json" ]]; then
        echo "  Skipped: $ITER_NAME (output already exists at $ITER_OUTPUT)"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi

    CMD="python3 $CONVERT_SCRIPT --input-dir $ITER_PATH --output-dir $ITER_OUTPUT --force"
    if [[ -n "$ORIGIN_HF_DIR" ]]; then
        CMD="$CMD --origin-hf-dir $ORIGIN_HF_DIR"
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    if eval "$CMD"; then
        echo "  Done: $ITER_NAME"
    else
        echo "  FAILED: $ITER_NAME"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Summary ==="
echo "Total: $NUM_ITERS, Succeeded: $((NUM_ITERS - FAILED - SKIPPED)), Skipped: $SKIPPED, Failed: $FAILED"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
