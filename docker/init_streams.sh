#!/bin/bash
# init_streams.sh — provision the project's durable JetStream streams.
#
# All actual logic lives in transport/init_streams.py so the configuration
# is kept in Python (single source of truth, importable by tests). This
# bash wrapper exists for the wondrous-crane plan A4 spec ("docker/init_streams.sh")
# and for symmetry with the rest of the docker/ directory.
#
# Usage:
#   bash docker/init_streams.sh
#   NATS_URL=nats://localhost:4222 bash docker/init_streams.sh   # override URL
#
# Idempotent: safe to run repeatedly. Each stream prints its action
# (created | updated | no_change) on stdout.

set -e

cd "$(dirname "$0")/.."

NATS_URL="${NATS_URL:-nats://localhost:4222}"
exec python -m transport.init_streams --nats-url "$NATS_URL"
