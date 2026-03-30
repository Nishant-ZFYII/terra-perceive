#!/usr/bin/env bash
# smoke_test.sh — M6.2: One-command Phase 1 smoke test.
#
# Runs the full pipeline on bundled sample data and verifies
# exactly three output artifacts are produced:
#   output/bev_traversability.png
#   output/safety_events.csv
#   output/timing_report.txt
#
# Usage: bash scripts/smoke_test.sh
# Exit code: 0 = pass, 1 = fail

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/output"
PIPELINE_SH="$REPO_ROOT/scripts/run_pipeline.sh"

BIN_FILE="$REPO_ROOT/data/sample/lidar/000000.bin"
IMG_FILE="$REPO_ROOT/data/sample/camera/000000.jpg"

INTERNAL_OUTPUT="$OUTPUT_DIR/smoke_run"

# Canonical artifact paths (fixed names for hiring manager / Docker checkpoint)
BEV_PNG="$OUTPUT_DIR/bev_traversability.png"
SAFETY_CSV="$OUTPUT_DIR/safety_events.csv"
TIMING_TXT="$OUTPUT_DIR/timing_report.txt"

mkdir -p "$OUTPUT_DIR"

echo "Terra Perceive — Phase 1 Smoke Test"
echo ""

SMOKE_START=$(date +%s)

# Run the full pipeline into a dedicated subdirectory
bash "$PIPELINE_SH" "$BIN_FILE" "$IMG_FILE" "$INTERNAL_OUTPUT"

# Copy canonical artifacts to output/ root
cp "$INTERNAL_OUTPUT/bev_2d.png"         "$BEV_PNG"
cp "$INTERNAL_OUTPUT/safety_events.csv"  "$SAFETY_CSV"
cp "$INTERNAL_OUTPUT/timing_report.txt"  "$TIMING_TXT"

SMOKE_END=$(date +%s)
SMOKE_ELAPSED=$((SMOKE_END - SMOKE_START))

echo ""
echo "Artifacts:"

PASS=1
for artifact in "$BEV_PNG" "$SAFETY_CSV" "$TIMING_TXT"; do
    if [ -f "$artifact" ] && [ -s "$artifact" ]; then
        echo "  [OK]  $artifact"
    else
        echo "  [FAIL] $artifact — missing or empty"
        PASS=0
    fi
done

echo ""
echo "Total smoke test time: ${SMOKE_ELAPSED}s"

if [ "$SMOKE_ELAPSED" -gt 120 ]; then
    echo "  [WARN] Exceeded 2-minute target (${SMOKE_ELAPSED}s > 120s)"
fi

echo ""
if [ "$PASS" -eq 1 ]; then
    echo "  SMOKE TEST PASSED"
    exit 0
else
    echo "  SMOKE TEST FAILED"
    exit 1
fi
