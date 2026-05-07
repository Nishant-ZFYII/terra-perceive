#!/usr/bin/env bash
# sync_from_hpc.sh — pull rendered ablation G outputs back from HPC.
#
# Only pulls the FINAL renders (PNG, MP4, GIF) plus metrics.json — the
# ~13 GB of intermediate per-frame CSVs stay on HPC scratch.
#
# Usage:
#   bash scripts/sync_from_hpc.sh np3129@torch-login-b-0 /scratch/np3129

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <user@host> <hpc_scratch_root>"
    exit 1
fi

REMOTE="$1"
SCRATCH="$2"
REPO="terra-perceive-p2m4"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${LOCAL_REPO}/results_m4/ablation_g"

echo "Pulling ablation G renders from HPC..."
rsync -avz --progress \
    "${REMOTE}:${SCRATCH}/${REPO}/results_m4/ablation_g/" \
    "${LOCAL_REPO}/results_m4/ablation_g/"

echo
echo "Pulled. Local files:"
ls -la "${LOCAL_REPO}/results_m4/ablation_g/" | head -20
