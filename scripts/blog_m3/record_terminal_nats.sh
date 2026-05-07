#!/bin/bash
# record_terminal_nats.sh — terminal-cast recording of the NATS broker logs.
#
# Drives the same end-to-end demo as record_dashboard.sh but captures the
# broker terminal showing connection / subject / message-flow log lines
# instead of the visual dashboard. Useful for the blog's "the wire is
# alive" moment without needing the GUI.
#
# Output: docs/assets/m3/nats_demo_terminal.mp4 (or .gif if you prefer)
#
# Usage:
#   bash scripts/blog_m3/record_terminal_nats.sh           # 30s default
#   DURATION=20 bash scripts/blog_m3/record_terminal_nats.sh
#
# Prerequisites:
#   - asciinema (sudo apt install asciinema) for recording
#   - agg (https://github.com/asciinema/agg) for asciinema -> gif
#     OR ffmpeg + a tool like ttygif if you prefer

set -u

REPO=/home/nishant/MS_Project/terra-perceive
PY=~/miniconda3/envs/terra-perceive/bin/python
DURATION=${DURATION:-30}
CAST=/tmp/nats_demo.cast
OUT_GIF=$REPO/docs/assets/m3/nats_demo_terminal.gif
OUT_MP4=$REPO/docs/assets/m3/nats_demo_terminal.mp4

CAMERA_DIR=$REPO/data/RELLIS-3D/Rellis_3D_pylon_camera_node/Rellis-3D/00000/pylon_camera_node
RUN_DIR=$REPO/results/m3/slam_ema_covg2o_perframe

cd "$REPO"

cleanup() {
    pkill -f publish_grid_stream.py || true
    pkill -f publish_camera_stream.py || true
    docker compose -f docker/docker-compose.yml stop nats >/dev/null 2>&1 || true
}
trap cleanup EXIT

command -v asciinema >/dev/null || { echo "install asciinema: sudo apt install asciinema"; exit 1; }

echo "[1/3] starting NATS broker (logs visible in this terminal)"
docker compose -f docker/docker-compose.yml up -d nats
for i in 1 2 3 4 5; do
    echo > /dev/tcp/127.0.0.1/4222 2>/dev/null && break
    sleep 0.5
done

echo "[2/3] starting publishers"
$PY scripts/publish_grid_stream.py --run "$RUN_DIR" --hz 5 --loop \
    > /tmp/bev.log 2>&1 &
$PY scripts/publish_camera_stream.py --camera-dir "$CAMERA_DIR" --hz 5 --loop \
    > /tmp/cam.log 2>&1 &
sleep 3

echo "[3/3] recording broker logs for ${DURATION}s -> $CAST"
# asciinema records the terminal; we follow the broker's docker logs as the
# subject of the recording.
asciinema rec --overwrite --idle-time-limit 2 -t "Terra Perceive: NATS broker live" \
    "$CAST" -c "timeout ${DURATION} docker compose -f docker/docker-compose.yml logs -f nats 2>&1 | head -200"

mkdir -p "$(dirname "$OUT_GIF")"

# Convert to GIF if `agg` is available; otherwise just leave the cast.
if command -v agg >/dev/null; then
    echo "converting to GIF -> $OUT_GIF"
    agg --font-size 14 --speed 1 "$CAST" "$OUT_GIF"
fi

# Optional: convert to MP4 via ffmpeg + a frame-dump
if command -v ffmpeg >/dev/null && [ -f "$OUT_GIF" ]; then
    echo "converting GIF -> MP4 ($OUT_MP4)"
    ffmpeg -y -loglevel warning -i "$OUT_GIF" \
        -movflags faststart -pix_fmt yuv420p \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        "$OUT_MP4"
fi

echo ""
echo "DONE"
ls -lh "$CAST" "$OUT_GIF" "$OUT_MP4" 2>/dev/null
echo ""
echo "If GIF/MP4 are missing, install:  sudo apt install asciinema  &&  cargo install --git https://github.com/asciinema/agg agg"
