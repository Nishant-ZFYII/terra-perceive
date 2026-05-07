#!/bin/bash
# start_dashboard_demo.sh — bring up the M3 dashboard demo for OBS recording.
#
# Single-pass through ALL 2847 frames at 5 Hz (no --loop):
#   per-frame BEV from results/m3/slam_ema_covg2o_perframe/
#   per-frame camera JPEGs from RELLIS pylon_camera_node/
# Run-time: ~9 min 30 sec from first PUB to publishers exiting.
#
# Workflow:
#   1. Open OBS Studio, configure Window Capture targeting "Terra Perceive".
#   2. Run this script.
#   3. When dashboard window appears, click Start Recording in OBS.
#   4. Wait until the publishers print "sequence exhausted, exiting" (~9.5 min).
#   5. Click Stop Recording in OBS.
#   6. Close the dashboard window. This script then tears down broker + procs.

set -u

REPO=/home/nishant/MS_Project/terra-perceive
PY=~/miniconda3/envs/terra-perceive/bin/python

CAMERA_DIR=$REPO/data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node
RUN_DIR=$REPO/results/m3/slam_ema_covg2o_perframe

cd "$REPO"

cleanup() {
    echo ""
    echo "[cleanup] killing publishers"
    pkill -f publish_synced_stream.py  2>/dev/null || true
    pkill -f publish_grid_stream.py    2>/dev/null || true
    pkill -f publish_camera_stream.py  2>/dev/null || true
    echo "[cleanup] stopping nats broker"
    docker compose -f docker/docker-compose.yml stop nats >/dev/null 2>&1 || true
    echo "[cleanup] done."
}
trap cleanup EXIT

# Pre-flight
[ -d "$RUN_DIR" ]    || { echo "missing $RUN_DIR"; exit 1; }
[ -d "$CAMERA_DIR" ] || { echo "missing $CAMERA_DIR"; exit 1; }

echo "[1/4] starting NATS broker"
docker compose -f docker/docker-compose.yml up -d nats >/dev/null
for i in 1 2 3 4 5; do
    if echo > /dev/tcp/127.0.0.1/4222 2>/dev/null; then
        echo "      broker ready"; break
    fi
    sleep 0.5
done

echo "[2/4] starting LOCKSTEP publisher (BEV + camera at same frame index, --hz 2.5)"
echo "      using publish_synced_stream.py — paces BEV and camera together so"
echo "      the dashboard never sees one subject racing ahead of the other."
PYTHONUNBUFFERED=1 "$PY" scripts/publish_synced_stream.py \
    --run "$RUN_DIR" --camera-dir "$CAMERA_DIR" --hz 2.5 \
    > /tmp/synced.log 2>&1 &
SYNC_PID=$!
echo "      synced publisher pid=$SYNC_PID  (tail -f /tmp/synced.log to watch)"

# Keep the variable names cleanup() expects so the trap kills the right thing.
BEV_PID=$SYNC_PID
CAM_PID=0

echo "[3/4] (no separate camera publisher — handled by the lockstep script above)"

# Let publishers warm up before the dashboard pops up.
sleep 4

echo "[4/4] starting dashboard (window: 'Terra Perceive')"
echo "      Start OBS recording NOW. Close the dashboard window when done."
echo "      Publisher exits naturally after frame 2846. At 2.5 Hz that's ~19 min."
echo ""

# Foreground: this script blocks here until the user closes the dashboard
# window (or hits Ctrl-C). The trap will then run cleanup().
"$PY" scripts/dashboard_subscriber.py
