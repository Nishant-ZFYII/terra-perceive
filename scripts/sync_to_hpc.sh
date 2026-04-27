#!/usr/bin/env bash
# sync_to_hpc.sh — push code + RELLIS bags from laptop to NYU HPC scratch.
#
# Usage:
#   bash scripts/sync_to_hpc.sh np3129@torch-login-b-0 /scratch/np3129
#
# Two phases:
#   1. Code (small, fast) — push the worktree minus build/, results_m4/, data/.
#   2. Data (BIG, ~28 GB) — push only if not already there.
#
# Requires:
#   - SSH key set up: `ssh-copy-id <user>@<host>` once.
#   - rsync on both ends (Greene/Torch has it).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <user@host> <hpc_scratch_root>"
    echo "  e.g.  $0 np3129@torch-login-b-0 /scratch/np3129"
    exit 1
fi

REMOTE="$1"
SCRATCH="$2"
REPO="terra-perceive-p2m4"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_BAGS="${LOCAL_REPO}/data/RELLIS-3D"

echo "============================================================"
echo "Phase 1: code → ${REMOTE}:${SCRATCH}/${REPO}"
echo "============================================================"
rsync -avz --progress \
    --exclude 'build/' \
    --exclude 'install/' \
    --exclude 'log/' \
    --exclude 'results_m4/' \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.vscode/' \
    --exclude 'compile_commands.json' \
    "${LOCAL_REPO}/" \
    "${REMOTE}:${SCRATCH}/${REPO}/"

echo
echo "============================================================"
echo "Phase 2: RELLIS bags → ${REMOTE}:${SCRATCH}/data/RELLIS-3D"
echo "  --partial  : keeps interrupted bags resumable next run"
echo "  (no --ignore-existing — rsync's delta algorithm skips already-"
echo "   transferred bytes via size+checksum comparison, which is what we"
echo "   actually want for resume after a SIGHUP'd rsync)"
echo "============================================================"
ssh "${REMOTE}" "mkdir -p ${SCRATCH}/data/RELLIS-3D"
rsync -avz --progress --partial --append-verify \
    "${LOCAL_BAGS}/"*.bag \
    "${REMOTE}:${SCRATCH}/data/RELLIS-3D/"

echo
echo "============================================================"
echo "Phase 3: symlink data/RELLIS-3D into the repo on HPC"
echo "============================================================"
ssh "${REMOTE}" "
    cd ${SCRATCH}/${REPO}
    # data/ may be a stale file or broken symlink from a prior run — repair before mkdir.
    if [[ -e data && ! -d data ]] || { [[ -L data ]] && [[ ! -e data ]]; }; then
        rm -f data
    fi
    mkdir -p data
    if [[ -L data/RELLIS-3D && ! -e data/RELLIS-3D ]]; then
        rm -f data/RELLIS-3D
    fi
    if [[ ! -L data/RELLIS-3D ]]; then
        ln -s ${SCRATCH}/data/RELLIS-3D data/RELLIS-3D
    fi
    ls -la data/
"

echo
echo "============================================================"
echo "DONE — repo + bags pushed to HPC."
echo "Next step on HPC:"
echo "  ssh ${REMOTE}"
echo "  cd ${SCRATCH}/${REPO}"
echo "  apptainer build /scratch/\$USER/p2m4.sif apptainer/perception.def    # one-time, ~15 min"
echo "  mkdir -p /scratch/\$USER/p2m4_logs"
echo "  sbatch slurm/run_ablation_g.slurm"
echo "  squeue -u \$USER     # check status"
echo "============================================================"
