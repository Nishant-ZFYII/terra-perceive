#!/usr/bin/env bash
# run_pipeline.sh — Phase 1 end-to-end pipeline (P1-M6.3)
# Usage: bash scripts/run_pipeline.sh [bin] [image] [output_dir]
#
# Defaults to data/sample frame 000000.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PIPELINE_RUNNER="$REPO_ROOT/build/construction_perception/pipeline_runner"
SAFETY_RUNNER="$REPO_ROOT/build/construction_perception/safety_runner"

DEFAULT_BIN="$REPO_ROOT/data/sample/lidar/000000.bin"
DEFAULT_IMG="$REPO_ROOT/data/sample/camera/000000.jpg"

BIN_FILE="${1:-$DEFAULT_BIN}"
IMG_FILE="${2:-$DEFAULT_IMG}"
OUTPUT_DIR="${3:-$REPO_ROOT/output/pipeline_$(date +%Y%m%d_%H%M%S)}"

GRID_BIN="$OUTPUT_DIR/traversability_grid.bin"
BEV_PREFIX="$OUTPUT_DIR/bev"
FUSION_PNG="$OUTPUT_DIR/bev_fusion.png"
CSV_FILE="$OUTPUT_DIR/safety_events.csv"
TIMING_REPORT="$OUTPUT_DIR/timing_report.txt"

mkdir -p "$OUTPUT_DIR"

TIMINGS=()
TOTAL_MS=0

info() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_stage() {
  local label="$1"; shift
  info "Starting: $label"
  local start_ms end_ms delta
  start_ms=$(date +%s%3N)
  "$@"
  end_ms=$(date +%s%3N)
  delta=$((end_ms - start_ms))
  TIMINGS+=("$label: ${delta}ms")
  TOTAL_MS=$((TOTAL_MS + delta))
  info "Done: $label in ${delta}ms"
}

# --- Preflight ---
[ -x "$PIPELINE_RUNNER" ] || { echo "ERROR: pipeline_runner not found. Run 'colcon build' first."; exit 1; }
[ -x "$SAFETY_RUNNER"   ] || { echo "ERROR: safety_runner not found. Run 'colcon build' first."; exit 1; }
[ -f "$BIN_FILE"        ] || { echo "ERROR: LiDAR file not found: $BIN_FILE"; exit 1; }
[ -f "$IMG_FILE"        ] || { echo "ERROR: Image file not found: $IMG_FILE"; exit 1; }

info "Terra Perceive — Phase 1 Pipeline"
info "LiDAR  : $BIN_FILE"
info "Image  : $IMG_FILE"
info "Output : $OUTPUT_DIR"
echo ""

# Step 1: Sector RANSAC + Traversability grid
run_stage "Sector RANSAC + Traversability" \
  "$PIPELINE_RUNNER" "$BIN_FILE" "$GRID_BIN"

# Step 2: BEV visualisation
run_stage "BEV Visualisation" \
  python3 "$REPO_ROOT/scripts/visualize_bev.py" --grid "$GRID_BIN" --out "$BEV_PREFIX"

# Step 3: Camera-LiDAR projection + SegFormer fusion
run_stage "Cam-LiDAR Fusion" \
  python3 "$REPO_ROOT/scripts/compare_bev_fusion.py" "$IMG_FILE" "$BIN_FILE" "$FUSION_PNG"

# Step 4: Safety supervisor (vehicle at 2 m/s, synthetic worker at -1.2 m/s approach)
run_stage "Safety Supervisor" \
  "$SAFETY_RUNNER" "$GRID_BIN" "2.0" "$CSV_FILE"

# Step 5: Timing report
info "Writing timing report..."
{
  echo "generated: $(date)"
  for entry in "${TIMINGS[@]}"; do
    echo "$entry"
  done
  echo "total: ${TOTAL_MS}ms"
} > "$TIMING_REPORT"

echo ""
info "Pipeline complete"
info "BEV image    : ${BEV_PREFIX}_2d.png"
info "Fusion BEV   : $FUSION_PNG"
info "Safety CSV   : $CSV_FILE"
info "Timing report: $TIMING_REPORT"
