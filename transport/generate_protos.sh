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
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/perception.proto"
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/safety.proto"     2>/dev/null || true
$PROTOC -I="$OUT" --python_out="$OUT" "$OUT/telemetry.proto"  2>/dev/null || true

echo "Generated:"
ls -la "$OUT"/*_pb2.py 2>/dev/null || echo "  (no _pb2.py files — check for proto syntax errors)"
