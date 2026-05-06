#!/bin/bash
# generate_protos.sh — generate Python protobuf bindings for local dev.
# Inside the Docker image this runs at build time (Dockerfile.perception:48).
# For host-side dev, run this once after cloning or after editing any .proto.
#
# Usage:  bash transport/generate_protos.sh

set -e
cd "$(dirname "$0")/.."

PROTOC=$(command -v protoc || true)
if [ -z "$PROTOC" ]; then
    echo "ERROR: protoc not found. Install with:"
    echo "    sudo apt install protobuf-compiler"
    exit 1
fi

OUT=transport/proto
# Python bindings (used by detection_node.py, tracker_bridge.py, replay_safety_events.py).
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/perception.proto"
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/safety.proto"     2>/dev/null || true
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/telemetry.proto"  2>/dev/null || true

# C++ bindings (used by ros2_nodes/tracker_node.cpp via #include "perception.pb.h").
$PROTOC -I="$OUT" --cpp_out="$OUT" "$OUT/perception.proto"
$PROTOC -I="$OUT" --cpp_out="$OUT" "$OUT/safety.proto"        2>/dev/null || true
$PROTOC -I="$OUT" --cpp_out="$OUT" "$OUT/telemetry.proto"     2>/dev/null || true

echo "Generated (Python):"
ls -la "$OUT"/*_pb2.py 2>/dev/null || echo "  (no _pb2.py files — check for proto syntax errors)"
echo "Generated (C++):"
ls -la "$OUT"/*.pb.h 2>/dev/null
ls -la "$OUT"/*.pb.cc 2>/dev/null
