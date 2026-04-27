#!/bin/bash
#SBATCH --job-name=tp_m4_g
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=cpu_normal
# ↑ Verify these on first submission:
#   sacctmgr list user $USER format=Account%30 -p   # → --account
#   sinfo -o "%P %D %c %m" | head                   # → --partition

# =============================================================================
# Slurm job: full Ablation G pipeline on NYU Torch HPC.
#
# Environment: conda prefix env on $SCRATCH (no Apptainer).
# Mirrors the activation pattern from Nishant-ZFYII/ml_inference's train.slurm.
#
# Submit from $HOME/terra-perceive-p2m4 with:
#   sbatch slurm/run_ablation_g.slurm
#
# Wall-clock budget: 3 hours. Expected actual: ~30-45 min compute.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Module + conda activation (NYU HPC convention)
# -----------------------------------------------------------------------------
module purge
module load anaconda3/2025.06

ENV_PATH=$SCRATCH/conda_envs/terra_perceive_m4

source $(conda info --base)/etc/profile.d/conda.sh
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH
export PYTHONNOUSERSITE=True

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO=$HOME/terra-perceive-p2m4
DATA_OUT=$SCRATCH/m4_perframe
RELLIS_BAGS=$SCRATCH/data/RELLIS-3D

mkdir -p "${DATA_OUT}"
cd "${REPO}"

# -----------------------------------------------------------------------------
# Pre-flight: verify bags + symlink them into the repo
# -----------------------------------------------------------------------------
if [[ ! -d "${RELLIS_BAGS}" ]]; then
    echo "ERROR: RELLIS bags not found at ${RELLIS_BAGS}"
    echo "       run scripts/sync_to_hpc.sh from your laptop first."
    exit 1
fi
mkdir -p "${REPO}/data"
if [[ ! -L "${REPO}/data/RELLIS-3D" ]]; then
    ln -s "${RELLIS_BAGS}" "${REPO}/data/RELLIS-3D"
fi

# Tell run_ablation_g.sh to write heavy data under $SCRATCH instead of the
# laptop's /media/... default (works because we parameterised EXT_ROOT).
export TP_M4_EXT_ROOT="${DATA_OUT}"

# -----------------------------------------------------------------------------
# Build C++ binaries (cached if unchanged)
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[$(date)] colcon build"
echo "============================================================"
colcon build --packages-select construction_perception \
             --cmake-args -DCMAKE_BUILD_TYPE=Release \
             --merge-install || {
    echo "ERROR: colcon build failed"
    exit 1
}

# Verify the binaries are present.
for bin in obstacle_extractor dbscan_cli tracker_runner; do
    if [[ ! -x "build/construction_perception/${bin}" ]]; then
        echo "ERROR: build/construction_perception/${bin} missing"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Run the full pipeline
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[$(date)] Running ablation G pipeline"
echo "============================================================"
bash scripts/run_ablation_g.sh

echo "============================================================"
echo "[$(date)] DONE"
echo "============================================================"
echo "Final renders : ${REPO}/results_m4/ablation_g/"
echo "Heavy outputs : ${DATA_OUT}/"
echo
echo "Pull renders to laptop with:"
echo "  bash scripts/sync_from_hpc.sh torch /scratch/\$USER"
