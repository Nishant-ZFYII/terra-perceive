#!/usr/bin/env python3
"""train.py — train the M13 appearance encoder.

Architecture: 8 → 64 → 32 → 32 (L2-normalized) MLP. Matches
`include/appearance_encoder.hpp` exactly so the C++ inference path drops
in trained weights without code changes.

Loss: batch-hard triplet (Hermans, Beyer, Leibe — "In Defense of the
Triplet Loss for Person Re-Identification", arXiv 2017, §4). Within each
minibatch of N classes × K samples per class, for every anchor:
    - hardest positive  = farthest same-class sample
    - hardest negative  = closest different-class sample
    - loss              = max(0, d(a, p_hard) − d(a, n_hard) + margin)

Where a "class" = a unique (frame, cluster_id) tuple, and "K samples per
class" = K geometric augmentations of the cluster's point cloud
(point dropout + per-point jitter + small random in-plane rotation).
The augmentations test the encoder's invariance to DBSCAN segmentation
flicker (which produces slightly different point subsets across frames)
and to LiDAR scan jitter — exactly the failure modes that motivated
adding appearance to the M12 SORT cost matrix.

Validation: labels.csv from the M13 hand-labeling session, held strictly
out of training. For each labeled pair, encode both clusters, compute
embedding distance, threshold-classify. Best validation accuracy across
threshold sweep is the headline number. The hand-labeled set breaks the
training-data circularity flagged in Decision-D of the wondrous-crane plan.

Outputs:
    {out_dir}/encoder.pt          — best-val checkpoint
    {out_dir}/training_log.json   — per-epoch loss + val accuracy

Inputs (paths default to the project layout that runs locally):
    --features-dir   {ext_root}/m4_perframe/appearance      (from extract_features.py)
    --clusters-dir   {ext_root}/m4_perframe/clusters_sweetspot
    --labels-csv     python/appearance/labels.csv
    --pair-csv       python/appearance/pair_candidates.csv  (used to map
                                                              pair_id → which
                                                              clusters were
                                                              hand-labeled,
                                                              so we never leak
                                                              them into training)

Reading list:
    Wojke, Bewley, Paulus — "Simple Online and Realtime Tracking with a
        Deep Association Metric" (ICIP 2017). §3 explains how the
        embeddings produced here will be combined into the SORT cost
        matrix in the M13 day-9 integration.
    Hermans, Beyer, Leibe — "In Defense of the Triplet Loss for Person
        Re-Identification" (arXiv 2017). §4 is batch-hard mining.

Wall-clock target: ~10 min on NYU Torch L40S, ~1-2 hr on a laptop CPU
(with AMP off and batch 128, single thread).

Usage on HPC:
    sbatch slurm/train_appearance.slurm
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Local: reuse the SAME feature-extraction function that produced the
# offline features files — guarantees the on-the-fly augmentation pipeline
# computes features the same way as what the encoder was trained against.
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import cluster_features, FEATURE_NAMES, load_cluster_csv  # noqa: E402


# =============================================================================
# Augmentation — point-level geometric perturbation. Tests encoder's
# invariance to DBSCAN re-segmentation, scan jitter, and small ego-motion.
# =============================================================================

class Augmenter:
    """Apply random geometric augmentations to a (N, 3) cluster point cloud.

    Steps (each independently sampled per call):
        1. Dropout — keep a random fraction of points in [keep_min, 1.0].
           Models DBSCAN dropping a point or two between frames.
        2. Jitter — add per-point Gaussian noise of std jitter_sigma (m).
           Models LiDAR's per-point range noise (~0.03-0.05 m typical).
        3. In-plane rotation — small random yaw around z.
           Models brief ego heading changes within a frame's exposure.

    Returns: (N', 3) augmented cluster.
    """
    def __init__(self, jitter_sigma: float = 0.04,
                 keep_min: float = 0.7,
                 rot_deg: float = 4.0,
                 rng: np.random.Generator | None = None):
        self.jitter_sigma = jitter_sigma
        self.keep_min = keep_min
        self.rot_deg = rot_deg
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] < 2:
            return pts.copy()
        # 1. dropout
        keep = self.rng.uniform(self.keep_min, 1.0)
        n = pts.shape[0]
        n_keep = max(2, int(round(n * keep)))
        idx = self.rng.choice(n, size=n_keep, replace=False)
        out = pts[idx].copy()
        # 2. jitter
        out += self.rng.normal(0.0, self.jitter_sigma, size=out.shape).astype(out.dtype)
        # 3. small in-plane rotation around the cluster centroid
        if self.rot_deg > 0:
            theta = np.deg2rad(self.rng.uniform(-self.rot_deg, self.rot_deg))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=out.dtype)
            ctr = out.mean(axis=0, keepdims=True)
            out = (out - ctr) @ R.T + ctr
        return out


# =============================================================================
# Memory bank — load all training clusters' point clouds once, then
# regenerate augmented features per batch.
# =============================================================================

def build_memory_bank(clusters_dir: Path,
                      held_out: set[tuple[int, int]],
                      max_frames: int | None = None,
                      ) -> Dict[Tuple[int, int], np.ndarray]:
    """Return {(frame, cluster_id): (N, 3) points}.

    Skips any (frame, cluster_id) in `held_out` (the val-set clusters,
    so they never see a training gradient). Skips clusters with < 5
    points (degenerate features, no signal).
    """
    files = sorted(clusters_dir.glob("clusters_*.csv"))
    if max_frames is not None:
        files = files[:max_frames]
    bank: Dict[Tuple[int, int], np.ndarray] = {}
    for path in tqdm(files, desc="loading clusters"):
        frame = int(path.stem.split("_")[1])
        for cid, pts in load_cluster_csv(path).items():
            if (frame, cid) in held_out:
                continue
            if pts.shape[0] < 5:
                continue
            bank[(frame, cid)] = pts
    return bank


# =============================================================================
# MLP — must match include/appearance_encoder.hpp byte-for-byte at
# inference. Z-score normalization is INSIDE the model so a single
# forward() call mirrors C++ encode() Step 1 → Step 5.
# =============================================================================

class AppearanceMLP(nn.Module):
    """8 → 64 → 32 → 32 → L2-normalized embedding."""
    def __init__(self, feat_mean: np.ndarray, feat_std: np.ndarray):
        super().__init__()
        self.register_buffer("feat_mean",
                             torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std",
                             torch.tensor(feat_std,  dtype=torch.float32))
        self.l1 = nn.Linear(8,  64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Z-score (encoder Step 1)
        x = (x - self.feat_mean) / self.feat_std
        # Layers 1-2 (Steps 2-3): linear + ReLU
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Layer 3 (Step 4): linear, no activation
        x = self.l3(x)
        # L2 normalize (Step 5)
        return F.normalize(x, p=2, dim=-1)


# =============================================================================
# Batch-hard triplet loss — Hermans 2017 §4.
# =============================================================================

def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor,
                            margin: float = 0.2) -> torch.Tensor:
    """emb: (B, D) unit-norm embeddings. labels: (B,) integer class ids.

    Returns scalar loss. For each anchor in the batch, hardest-positive
    is the same-class sample farthest away in embedding space, hardest-
    negative is the different-class sample closest. Loss is mean over
    anchors of max(0, d(a, p_hard) − d(a, n_hard) + margin).
    """
    # Pairwise euclidean distances. emb is unit-norm so |a-b|² = 2(1 − a·b).
    pdist = torch.cdist(emb, emb, p=2)            # (B, B)

    same = labels.unsqueeze(0) == labels.unsqueeze(1)     # (B, B)
    diff = ~same
    same.fill_diagonal_(False)                            # exclude self

    # Hardest positive — max distance among same-class entries.
    # Where there's no positive (label is unique in this batch), masked
    # max becomes 0; the resulting triplet loss for that anchor is just
    # a clamp on (− d_neg + margin), which is non-positive most of the
    # time → contributes nothing. So unique-class anchors are no-ops.
    hp = (pdist * same.float()).max(dim=1).values         # (B,)

    # Hardest negative — min distance among different-class entries.
    # Mask same-class entries by adding a large constant so they lose
    # the min argmin.
    big = pdist.max().detach() + 1.0
    hn = (pdist + big * (~diff).float()).min(dim=1).values  # (B,)

    return F.relu(hp - hn + margin).mean()


# =============================================================================
# Training-batch sampler — N classes × K augmentations.
# =============================================================================

def sample_batch(bank: Dict[Tuple[int, int], np.ndarray],
                 keys: List[Tuple[int, int]],
                 n_classes: int, k_per_class: int,
                 aug: Augmenter,
                 rng: random.Random,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n_classes random clusters; augment each k_per_class ways;
    extract features. Returns (features (N*K, 8), class_ids (N*K,))."""
    chosen = rng.sample(keys, n_classes)
    feats = np.empty((n_classes * k_per_class, 8), dtype=np.float32)
    cls   = np.empty(n_classes * k_per_class, dtype=np.int64)
    for i, key in enumerate(chosen):
        pts = bank[key]
        for j in range(k_per_class):
            aug_pts = aug(pts)
            feats[i * k_per_class + j] = cluster_features(aug_pts)
            cls[i * k_per_class + j] = i
    return feats, cls


# =============================================================================
# Validation against held-out labels.csv.
# =============================================================================

def evaluate_on_labels(model: AppearanceMLP,
                       val_pairs: List[Tuple[int, int, int, int, str]],
                       clusters_cache: Dict[Tuple[int, int], np.ndarray],
                       device: torch.device,
                       ) -> Tuple[float, float, Dict[str, float]]:
    """Returns (best_threshold, best_accuracy, breakdown).

    val_pairs: list of (frame_a, cluster_a, frame_b, cluster_b, label)
               where label ∈ {"same", "different"} (skips already filtered out).
    """
    model.eval()
    pos_d, neg_d = [], []
    n_skipped = 0
    with torch.no_grad():
        for frame_a, cid_a, frame_b, cid_b, label in val_pairs:
            pts_a = clusters_cache.get((frame_a, cid_a))
            pts_b = clusters_cache.get((frame_b, cid_b))
            if pts_a is None or pts_b is None or pts_a.shape[0] < 2 or pts_b.shape[0] < 2:
                n_skipped += 1
                continue
            f_a = cluster_features(pts_a)
            f_b = cluster_features(pts_b)
            x = torch.tensor(np.stack([f_a, f_b]), dtype=torch.float32,
                             device=device)
            e = model(x)              # (2, 32) unit-norm
            d = float(torch.norm(e[0] - e[1], p=2).item())
            (pos_d if label == "same" else neg_d).append(d)

    if not pos_d or not neg_d:
        return 0.0, 0.0, {"n_pos": len(pos_d), "n_neg": len(neg_d),
                          "n_skipped": n_skipped}

    # Sweep threshold; "same" predicted if d <= τ.
    all_d = sorted(set(pos_d + neg_d))
    best_acc, best_τ = 0.0, 0.5
    n_total = len(pos_d) + len(neg_d)
    for τ in all_d:
        tp = sum(1 for d in pos_d if d <= τ)
        tn = sum(1 for d in neg_d if d > τ)
        acc = (tp + tn) / n_total
        if acc > best_acc:
            best_acc, best_τ = acc, τ

    breakdown = {
        "n_pos":       len(pos_d),
        "n_neg":       len(neg_d),
        "n_skipped":   n_skipped,
        "pos_d_mean":  float(np.mean(pos_d)),
        "pos_d_std":   float(np.std(pos_d)),
        "neg_d_mean":  float(np.mean(neg_d)),
        "neg_d_std":   float(np.std(neg_d)),
        "best_thresh": float(best_τ),
    }
    return best_τ, best_acc, breakdown


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True,
                   help="dir with corpus_stats.json (and per-frame features files)")
    p.add_argument("--clusters-dir", type=Path, required=True,
                   help="dir with clusters_NNNNNN.csv (raw points needed for augmentation)")
    p.add_argument("--labels-csv",   type=Path, required=True)
    p.add_argument("--pair-csv",     type=Path, required=True)
    p.add_argument("--out-dir",      type=Path, required=True)

    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--n-classes",     type=int,   default=32,
                   help="distinct classes per batch (N in N×K)")
    p.add_argument("--k-per-class",   type=int,   default=4,
                   help="augmented samples per class (K in N×K)")
    p.add_argument("--margin",        type=float, default=0.2)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max-frames",    type=int,   default=None,
                   help="cap clusters loaded — for fast smoke runs")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="auto = cuda if compatible, else cpu (default)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Device selection. Auto: prefer CUDA but probe a no-op tensor first;
    # fall back to CPU on capability mismatch (e.g., older Pascal cards
    # against a Hopper-built PyTorch). The HPC L40S will pass cleanly;
    # the laptop GTX 1060 falls through to CPU.
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        if torch.cuda.is_available():
            try:
                _probe = torch.zeros(1, device="cuda")
                _probe = _probe + 1.0     # exercises a kernel
                torch.cuda.synchronize()
                device = torch.device("cuda")
            except (RuntimeError, torch.AcceleratorError):
                print("[train] CUDA available but unusable on this device — "
                      "falling back to CPU", file=sys.stderr)
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    print(f"[train] device: {device}")

    # --- corpus stats (z-score) -----------------------------------------
    stats_path = args.features_dir / "corpus_stats.json"
    with stats_path.open() as f:
        stats = json.load(f)
    feat_mean = np.array(stats["mean"], dtype=np.float32)
    feat_std  = np.array(stats["std"],  dtype=np.float32)
    print(f"[train] corpus stats: {stats['n_clusters_total']} clusters across "
          f"{stats['n_frames']} frames")

    # --- val pairs (labels.csv) -----------------------------------------
    pair_lookup = {}
    with args.pair_csv.open() as f:
        for r in csv.DictReader(f):
            pair_lookup[int(r["pair_id"])] = (
                int(r["frame_a"]), int(r["cluster_a"]),
                int(r["frame_b"]), int(r["cluster_b"]),
            )

    val_pairs: List[Tuple[int, int, int, int, str]] = []
    held_out: set[tuple[int, int]] = set()
    with args.labels_csv.open() as f:
        for r in csv.DictReader(f):
            label = r["label"]
            if label == "skip":
                continue
            pid = int(r["pair_id"])
            if pid not in pair_lookup:
                continue
            fa, ca, fb, cb = pair_lookup[pid]
            val_pairs.append((fa, ca, fb, cb, label))
            held_out.add((fa, ca))
            held_out.add((fb, cb))
    print(f"[train] {len(val_pairs)} validation pairs, "
          f"{len(held_out)} clusters held out from training")

    # --- training memory bank ------------------------------------------
    print(f"[train] loading clusters_*.csv into memory ...")
    bank = build_memory_bank(args.clusters_dir, held_out, args.max_frames)
    keys = list(bank.keys())
    print(f"[train] {len(bank)} training clusters in memory bank")
    if len(bank) < args.n_classes:
        raise SystemExit(f"need at least {args.n_classes} clusters; "
                         f"got {len(bank)}")

    # --- val cluster cache (separate from training bank) ---------------
    val_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for frame_a, cid_a, frame_b, cid_b, _ in val_pairs:
        for fc in [(frame_a, cid_a), (frame_b, cid_b)]:
            if fc not in val_cache:
                cluster_csv = args.clusters_dir / f"clusters_{fc[0]:06d}.csv"
                clusters = load_cluster_csv(cluster_csv)
                val_cache[fc] = clusters.get(fc[1], np.empty((0, 3), dtype=np.float32))

    # --- model + optimizer ---------------------------------------------
    model = AppearanceMLP(feat_mean, feat_std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[train] model has {sum(p.numel() for p in model.parameters())} parameters")

    aug = Augmenter(rng=rng_np)

    # --- training loop --------------------------------------------------
    log = []
    best_val_acc = 0.0
    best_path = args.out_dir / "encoder.pt"
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(range(args.steps_per_epoch),
                   desc=f"epoch {epoch}/{args.epochs}")
        for _ in bar:
            feats_np, cls_np = sample_batch(
                bank, keys, args.n_classes, args.k_per_class, aug, rng_py)
            feats = torch.from_numpy(feats_np).to(device)
            labels = torch.from_numpy(cls_np).to(device)
            emb = model(feats)
            loss = batch_hard_triplet_loss(emb, labels, margin=args.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            avg=f"{running_loss / (bar.n + 1):.4f}")

        avg_loss = running_loss / args.steps_per_epoch
        τ, val_acc, breakdown = evaluate_on_labels(
            model, val_pairs, val_cache, device)
        log.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_threshold": τ,
            "breakdown": breakdown,
        })
        elapsed = time.time() - t0
        print(f"[train] epoch {epoch}: loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.4f} (τ={τ:.3f})  "
              f"pos_d={breakdown['pos_d_mean']:.3f}±{breakdown['pos_d_std']:.3f}  "
              f"neg_d={breakdown['neg_d_mean']:.3f}±{breakdown['neg_d_std']:.3f}  "
              f"elapsed={elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "feat_mean": feat_mean.tolist(),
                "feat_std":  feat_std.tolist(),
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_threshold": τ,
            }, best_path)
            print(f"[train]   ↳ checkpoint saved (best so far)")

    # --- log -----------------------------------------------------------
    with (args.out_dir / "training_log.json").open("w") as f:
        json.dump({
            "args": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
            "best_val_accuracy": best_val_acc,
            "log": log,
        }, f, indent=2)

    print()
    print(f"[train] DONE. best val acc = {best_val_acc:.4f}")
    print(f"[train] checkpoint:   {best_path}")
    print(f"[train] log:          {args.out_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
