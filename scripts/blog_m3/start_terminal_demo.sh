#!/bin/bash
# start_terminal_demo.sh — terminal-cast demo for OBS recording.
#
# Same publishers as start_dashboard_demo.sh but no GUI. The terminal
# running this script shows interleaved publisher logs (PUB lines for
# each subject as messages flow). Point OBS Window Capture at this
# terminal and record.
#
# Run-time: ~9 min 30 sec from first PUB to publishers exiting.
#
# Workflow:
#   1. Resize your terminal to 1280x720 or 1920x1080 (whatever you want
#      the recording aspect to be), pick a clean monospace font.
#   2. Open OBS Studio, configure Window Capture targeting this terminal.
#   3. Run this script.
#   4. Click Start Recording in OBS as soon as PUB lines appear.
#   5. Stop Recording when publishers print "sequence exhausted, exiting".
#   6. Ctrl-C this script to clean up.

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

[ -d "$RUN_DIR" ]    || { echo "missing $RUN_DIR"; exit 1; }
[ -d "$CAMERA_DIR" ] || { echo "missing $CAMERA_DIR"; exit 1; }

echo "[1/3] starting NATS broker"
docker compose -f docker/docker-compose.yml up -d nats >/dev/null
for i in 1 2 3 4 5; do
    echo > /dev/tcp/127.0.0.1/4222 2>/dev/null && { echo "      broker ready"; break; }
    sleep 0.5
done

echo "[2/3] starting LOCKSTEP publisher (single pass, --hz 2.5, BEV + camera at same frame)"
PYTHONUNBUFFERED=1 "$PY" scripts/publish_synced_stream.py \
    --run "$RUN_DIR" --camera-dir "$CAMERA_DIR" --hz 2.5 \
    > /tmp/synced.log 2>&1 &
SYNC_PID=$!
echo "      synced pid=$SYNC_PID"
sleep 1

echo ""
echo "[3/3] perception.> traffic on the broker (interleaved, live):"
echo "================================================================"
echo ""

# Single log file from the lockstep publisher. -q suppresses any
# file-header chatter; -n 0 skips back-history so the recording starts
# from the first newly-emitted line.
# DO NOT use `exec` here — `exec` replaces bash with tail, which drops the
# EXIT trap and leaves publishers + broker running after Ctrl-C.
tail -F -q -n 0 /tmp/synced.log
