#!/bin/bash
#SBATCH --job-name=tp_m13_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_742_general
#SBATCH --partition=l40s_public
# Partition + GRES match the working pattern from ml_pipeline/eval_corridor_v4.slurm.

# =============================================================================
# P3-M13: train the appearance encoder (8 → 64 → 32 → 32 MLP, batch-hard
# triplet loss) on RELLIS DBSCAN clusters.
#
# Inputs (pre-staged on HPC by sync_to_hpc.sh):
#   $SCRATCH/m4_perframe/clusters_sweetspot/   per-frame DBSCAN CSVs
#   $SCRATCH/m4_perframe/appearance/            corpus_stats.json + features
#                                               (produced by extract_features.py;
#                                               re-run on HPC before submit, see
#                                               below)
#   ~/MS_Project/terra-perceive-p2m4/python/appearance/labels.csv
#   ~/MS_Project/terra-perceive-p2m4/python/appearance/pair_candidates.csv
#
# Output:
#   $SCRATCH/m13_train/encoder.pt                 best-val checkpoint
#   $SCRATCH/m13_train/training_log.json          per-epoch loss + val accuracy
#
# Wall-clock budget: 2 hours. Expected actual on L40S: ~10-15 min training
# (20 epochs × 400 steps × ~30 ms/step).
#
# Submit from $HOME/MS_Project/terra-perceive-p2m4 with:
#   sbatch slurm/train_appearance.slurm
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Module + conda activation (mirrors run_ablation_g.slurm)
# -----------------------------------------------------------------------------
module purge
module load anaconda3/2025.06

# M13 uses a slim env (just numpy + torch + tqdm). The earlier
# terra_perceive_m4 env had ros/rosbags deps that aren't needed for
# training and bloated setup time.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ENV_PATH=$SCRATCH/conda_envs/terra_perceive_m13
source $(conda info --base)/etc/profile.d/conda.sh
source activate $ENV_PATH
export PATH=$ENV_PATH/bin:$PATH
export PYTHONNOUSERSITE=True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Repo is rsynced to $SCRATCH (not $HOME) by scripts/sync_to_hpc.sh —
# matches the path that run_ablation_g.slurm uses.
REPO=$SCRATCH/terra-perceive-p2m4
DATA_ROOT=$SCRATCH/m4_perframe
OUT_DIR=$SCRATCH/m13_train

mkdir -p "${OUT_DIR}"
cd "${REPO}"

# -----------------------------------------------------------------------------
# Pre-flight checks — fail fast if data isn't staged.
# -----------------------------------------------------------------------------
for required in \
    "${DATA_ROOT}/clusters_sweetspot" \
    "${REPO}/python/appearance/labels.csv" \
    "${REPO}/python/appearance/pair_candidates.csv" \
; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required input missing: ${required}"
        echo "       run scripts/sync_to_hpc.sh from your laptop first."
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Step 1: ensure feature corpus exists. extract_features.py is fast (~30 s
# even single-threaded), and the corpus_stats.json must be regenerated
# whenever new clusters land. Idempotent — overwrites prior outputs.
# -----------------------------------------------------------------------------
FEAT_DIR=${DATA_ROOT}/appearance
if [[ ! -f "${FEAT_DIR}/corpus_stats.json" ]]; then
    echo "============================================================"
    echo "[$(date)] extracting features (no corpus_stats.json yet)"
    echo "============================================================"
    python python/appearance/extract_features.py \
        --clusters-dir "${DATA_ROOT}/clusters_sweetspot" \
        --out-dir      "${FEAT_DIR}" \
        --workers      4
fi

# -----------------------------------------------------------------------------
# Step 2: train. tqdm progress bars in the .out file (Slurm captures them).
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[$(date)] training appearance encoder"
echo "============================================================"
python python/appearance/train.py \
    --features-dir "${FEAT_DIR}" \
    --clusters-dir "${DATA_ROOT}/clusters_sweetspot" \
    --labels-csv   python/appearance/labels.csv \
    --pair-csv     python/appearance/pair_candidates.csv \
    --out-dir      "${OUT_DIR}" \
    --epochs           20 \
    --steps-per-epoch  400 \
    --n-classes        32 \
    --k-per-class      4  \
    --margin           0.2 \
    --lr               1e-3 \
    --device           auto

echo "============================================================"
echo "[$(date)] DONE"
echo "============================================================"
echo "Checkpoint    : ${OUT_DIR}/encoder.pt"
echo "Training log  : ${OUT_DIR}/training_log.json"
echo
echo "Next step (back on laptop):"
echo "  rsync from HPC to local:"
echo "    rsync -av torch:${OUT_DIR}/ ./python/appearance/checkpoints/"
echo "  then run torch_to_eigen_check.py to dump weights into"
echo "  include/appearance_model_weights.hpp + tests/data/*.csv"
