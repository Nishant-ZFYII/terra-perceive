#!/bin/bash
# record_dashboard.sh — fully-automated screen recording of the M3 dashboard.
#
# Brings up the broker + 2 publishers + dashboard, records the dashboard
# window for DURATION seconds, then tears everything down. Output MP4
# lands in docs/assets/m3/hero_dashboard_demo.mp4.
#
# Usage:
#   bash scripts/blog_m3/record_dashboard.sh           # 60s default
#   DURATION=30 bash scripts/blog_m3/record_dashboard.sh
#
# Prerequisites:
#   - Docker daemon running, docker compose available
#   - ffmpeg with x11grab support (default on Ubuntu)
#   - xdotool installed (sudo apt install xdotool)
#   - python env with nats-py, protobuf, matplotlib, Pillow

set -u

REPO=/home/nishant/MS_Project/terra-perceive
PY=~/miniconda3/envs/terra-perceive/bin/python
DURATION=${DURATION:-60}
OUT=$REPO/docs/assets/m3/hero_dashboard_demo.mp4
LOG=/tmp/record_dashboard.log

CAMERA_DIR=$REPO/data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node
RUN_DIR=$REPO/results/m3/slam_ema_covg2o_perframe

cd "$REPO"

cleanup() {
    echo "[cleanup] killing publishers + dashboard"
    pkill -f publish_grid_stream.py || true
    pkill -f publish_camera_stream.py || true
    pkill -f dashboard_subscriber.py || true
    echo "[cleanup] stopping nats broker"
    docker compose -f docker/docker-compose.yml stop nats >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Pre-flight checks
for cmd in xdotool ffmpeg docker; do
    command -v "$cmd" >/dev/null || { echo "MISSING: $cmd"; exit 1; }
done
[ -d "$CAMERA_DIR" ] || { echo "missing camera dir: $CAMERA_DIR"; exit 1; }
[ -d "$RUN_DIR"    ] || { echo "missing run dir:    $RUN_DIR"; exit 1; }

echo "[1/5] starting NATS broker"
docker compose -f docker/docker-compose.yml up -d nats
# Poll until broker accepts connections
for i in 1 2 3 4 5 6 7 8 9 10; do
    if echo > /dev/tcp/127.0.0.1/4222 2>/dev/null; then
        echo "      broker ready"
        break
    fi
    sleep 0.5
done

echo "[2/5] starting BEV publisher"
PYTHONUNBUFFERED=1 $PY scripts/publish_grid_stream.py --run "$RUN_DIR" --hz 5 --loop \
    > "$LOG.bev"  2>&1 &
BEV_PID=$!

echo "[3/5] starting camera publisher"
PYTHONUNBUFFERED=1 $PY scripts/publish_camera_stream.py --camera-dir "$CAMERA_DIR" --hz 5 --loop \
    > "$LOG.cam" 2>&1 &
CAM_PID=$!

# Give publishers a head start so the dashboard sees data immediately
sleep 4

echo "[4/5] starting dashboard (DISPLAY=$DISPLAY)"
PYTHONUNBUFFERED=1 $PY scripts/dashboard_subscriber.py > "$LOG.dash" 2>&1 &
DASH_PID=$!

# Wait for the dashboard window to appear. Try xdotool first; fall back to
# wmctrl if installed. Search for the suptitle text "Terra Perceive" which
# Matplotlib's Tk backend exposes as the window title.
WIN_ID=""
for i in $(seq 1 60); do
    WIN_ID=$(xdotool search --name "Terra Perceive" 2>/dev/null | head -1)
    [ -n "$WIN_ID" ] && break
    if command -v wmctrl >/dev/null; then
        WID_HEX=$(wmctrl -l 2>/dev/null | awk '/Terra Perceive/ {print $1; exit}')
        if [ -n "$WID_HEX" ]; then
            WIN_ID=$((WID_HEX))
            break
        fi
    fi
    sleep 0.5
done

if [ -z "$WIN_ID" ]; then
    echo "FAIL: dashboard window never appeared after 30 s."
    echo "  dashboard log:"
    sed 's/^/    /' "$LOG.dash" | tail -40
    echo "  bev log:"
    sed 's/^/    /' "$LOG.bev" | tail -10
    echo "  cam log:"
    sed 's/^/    /' "$LOG.cam" | tail -10
    echo "  visible windows mentioning Terra/perceive:"
    xdotool search --name "" 2>/dev/null | xargs -I{} xdotool getwindowname {} 2>/dev/null | grep -iE "terra|perceive" || true
    exit 2
fi

# Get window geometry for ffmpeg crop
GEO=$(xdotool getwindowgeometry --shell "$WIN_ID")
eval "$GEO"   # sets X, Y, WIDTH, HEIGHT
echo "      dashboard window id=$WIN_ID at +${X},+${Y} size ${WIDTH}x${HEIGHT}"

# Let the dashboard fill the buffer + animation render once
sleep 3

mkdir -p "$(dirname "$OUT")"
echo "[5/5] recording ${DURATION}s -> $OUT"
ffmpeg -y -loglevel warning \
    -video_size "${WIDTH}x${HEIGHT}" -framerate 30 -f x11grab \
    -i ":0.0+${X},${Y}" -t "$DURATION" \
    -c:v libx264 -preset medium -crf 22 -pix_fmt yuv420p \
    "$OUT"

echo ""
echo "DONE: $OUT"
ls -lh "$OUT"
