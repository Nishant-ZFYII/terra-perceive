# Phase 3 — Tracking Robustness on Real Data

## Why this exists as its own phase (and not as M4 add-ons)

Phase 2's tracker (P2-M4) was scoped to "implement SORT from scratch and prove
algorithmic correctness via unit + integration tests." That ships. What
Phase 2 did NOT scope was: *make the tracker robust enough on real RELLIS
LiDAR data that a viewer can identify persistent objects across frames.*

This wasn't a blind oversight in M4 — `plan_p2m4.md:222` listed IMM, Deep
SORT, and KD-tree as "what I'd improve" footer items. But running the M4
tracker end-to-end on RELLIS revealed that those footer items are actually
load-bearing for portfolio-grade demonstration:

> **M4-on-RELLIS quantitative finding:** 979 distinct track IDs assigned
> over 285 seconds of driving. Mean track lifetime: 17.4 frames. 43% of
> tracks lived ≤5 frames. DBSCAN cluster count varied 3 → 26 per frame on
> stationary scenes.

The algorithm itself is correct (20/20 tests pass). The instability is
upstream and at the velocity-discontinuity edge case. Phase 3 closes that
gap with techniques the literature already canonicalised (IMM, multi-frame
accumulation, learned association). With Phase 3 in place, the same tracker
on the same data should produce on the order of 50–80 distinct IDs — close
to the count of physical objects in the scene.

Phase 3 is also where the blog post's "production-grade tracking" claim
becomes defensible. Phase 2 demonstrates *I can build SORT*; Phase 3
demonstrates *I can make it work on real data*.

---

## Bridge from Phase 2

What Phase 3 inherits:

- `tracker::SORTTracker` class with greedy + Munkres dispatchers (P2-M4).
- `tracker::dbscan` clusterer with brute-force O(N²) neighbor search (P2-M4).
- `obstacle_extractor` + `dbscan_cli` CLI binaries (P2-M4 / Ablation G).
- Per-frame `clusters_NNNNNN.csv` and `tracks.csv` artifacts on
  `/media/.../m4_perframe/` (Ablation G output, ~13 GB).
- `animate_tracker_vs_dbscan.py` visualisation harness with the
  side-by-side flicker→stable layout (P2-M4 closing-hero).
- The 979-track baseline for measuring improvement.

What Phase 3 does NOT depend on:

- Phase 2's M5 (YOLO + 3D-lift + safety + NATS) — Phase 3 can run in
  parallel with M5 because it operates on the LiDAR pipeline, not the
  camera pipeline.
- nuScenes (M7) — Phase 3 measures against RELLIS to keep the comparison
  fair to the M4 baseline. nuScenes evaluation lives in M8.

---

## Phase 3 milestones

### P3-M12 — IMM Kalman filter (Interacting Multiple Model)

**Goal:** Replace the single constant-velocity Kalman filter inside `Track`
with an IMM filter that runs constant-velocity (CV) and constant-position
(CP) models in parallel and switches between them per frame. Solves the
M4 stationary-segment flicker directly.

**Sub-tasks:**

- `include/imm_filter.hpp` + `src/imm_filter.cpp`: two-mode IMM with
  Markov mode-transition matrix, per-mode predict/update, mode probability
  update via Bayes' rule, mixed-output state estimate.
- `tests/cpp/test_imm.cpp`: at least 5 cases —
  `ConstantVelocityTrackedByCV`, `StationaryTrackedByCP`,
  `ModeSwitchOnDeceleration`, `MixingProbabilitiesSumToOne`,
  `BothModelsAgreeOnSteadyState`.
- Modify `Track` struct: replace `KalmanFilter2D kf` with
  `IMMFilter kf` (or feature-flag both for ablation).
- Re-run M4's RELLIS closing-hero pipeline with IMM. Measure new
  distinct-ID count; expect ≤200 (vs M4 baseline 979).

**Exit criterion:** RELLIS distinct-track-IDs reduced by ≥50% vs M4
baseline. Stationary-segment flicker (frames 1750–1830) shows the same
SORT track ID throughout for at least 5 visible trees.

**Reading:**
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking
  and Navigation* (2001), Ch. 11 — IMM derivation.
- Blackman, Popoli, *Design and Analysis of Modern Tracking Systems*
  (1999), Ch. 8 — multi-model approaches.

**Wall-clock:** 4–6 days.

---

### P3-M13 — Multi-frame point cloud accumulation with ego-motion compensation

**Goal:** Smooth DBSCAN cluster centroids by accumulating LiDAR over a short
sliding window, registered into a common frame using SLAM odometry from
P2-M2. Reduces cluster fragmentation from sensor angular aliasing.

**Sub-tasks:**

- New `include/cloud_accumulator.hpp` + `src/cloud_accumulator.cpp`:
  ring buffer of N most-recent clouds, SE(3) transforms via the same
  pose source used in M3 accumulator (SLAM by default).
- Pre-DBSCAN stage in `obstacle_extractor`: instead of clustering one
  frame's obstacle points, cluster the union of N=3 frames' points
  reprojected into the current frame.
- `tests/cpp/test_cloud_accumulator.cpp`: ring buffer correctness,
  transform consistency, cluster centroid stability test.
- Re-run M4's RELLIS pipeline with N ∈ {1, 3, 5} and measure:
  - DBSCAN per-frame cluster-count variance (M4 baseline: 4.55 stddev).
  - Mean cluster centroid drift between adjacent frames (currently 10–30 cm).
  - Distinct track IDs over the recording.

**Exit criterion:** DBSCAN per-frame cluster-count standard deviation
reduced by ≥40%. Mean cluster centroid drift reduced by ≥50%.

**Reading:**
- Yin, Du, *"Dynamic Point-Cloud Accumulation for Robust Object Detection"*
  (2020 IROS).
- Engelmann, Hess, *"3D Object Tracking with LiDAR Point-Cloud Accumulation"*
  (2021).

**Wall-clock:** 3–5 days.

---

### P3-M14 — KD-tree neighbor search for DBSCAN

**Goal:** Drop `region_query` from O(N) per call to O(log N) so DBSCAN
scales to ≥50k-point clouds (currently choking at 10k in Debug mode).

**Sub-tasks:**

- Pick implementation: nanoflann (single-header, BSD license) is the
  standard for LiDAR-domain projects. Vendor in `third_party/nanoflann`.
- `region_query` rewritten to query a `KDTreeEigenAdapter`. Build the tree
  ONCE per call to `dbscan()` (not per query).
- Re-run `test_dbscan` `LargeClusterPerformance` test at N = 50,000.
  Tighten the wall-clock assertion accordingly.
- Add a benchmark: brute-force vs KD-tree at N ∈ {1k, 10k, 50k, 100k}
  with a chart for the blog.

**Exit criterion:** DBSCAN on 50k-point cloud completes in <1 sec Release.
Brute-force baseline > 5 sec at the same N.

**Reading:**
- Friedman, Bentley, Finkel, *"An Algorithm for Finding Best Matches in
  Logarithmic Expected Time"* (TOMS 1977) — original KD-tree paper.
- nanoflann documentation.

**Wall-clock:** 2 days.

---

### P3-M15 — Cluster appearance features in matching cost

**Goal:** Augment SORT's Euclidean-distance cost matrix with cluster-size,
point-density, and bounding-box features. A 200-point tree shouldn't match
a 15-point fragment that happens to be 0.4m away.

**Sub-tasks:**

- Extend `Track` struct with running estimates of cluster size + bbox
  dimensions from past matches.
- Modify `SORTTracker::match()` cost matrix:
  `cost(i, j) = α * euclidean(i, j) + β * size_diff(i, j) + γ * density_diff(i, j)`
  with α, β, γ tuned on the RELLIS dataset.
- New ablation: sweep (α, β, γ) and measure ID stability + false-merge rate.
- Tests: feature normalization correctness, feature-cost-matrix shape,
  fallback when track has no size estimate yet.

**Exit criterion:** ID-stability metric (MOTA-like) improves by ≥10% vs
P3-M12 baseline (IMM-only, no appearance).

**Wall-clock:** 3–4 days.

---

### P3-M16 — Deep SORT-style learned appearance embeddings

**Goal:** Train a small neural network that embeds each cluster's point
distribution into a fixed-length appearance vector. Use cosine similarity
between embeddings as an additional term in the matching cost. This is the
production AV-grade technique.

**Sub-tasks:**

- Encoder: PointNet-style architecture (or DGCNN), embedding dim 64.
  Train on a contrastive loss over RELLIS cluster pairs (same physical
  object across frames = positive; different objects = negative).
- Generate the contrastive training set from M4's RELLIS tracker
  output: tracks with lifetime ≥30 frames are reliable positives.
- Cost matrix: weighted sum of (Euclidean position cost) + (cosine
  embedding cost), with the weight tuned by ablation.
- Compare to non-learned appearance (P3-M15) on the same RELLIS recording.

**Exit criterion:** Distinct-track-IDs reduced by an additional ≥30% vs
P3-M15 baseline. Re-identification across occlusion ≥10 frames working
on at least one demonstrable case in the recording.

**Reading:**
- Wojke, Bewley, Paulus, *"Simple Online and Realtime Tracking with a
  Deep Association Metric"* (ICIP 2017) — Deep SORT paper.
- Qi et al., *"PointNet"* (CVPR 2017) — point cloud encoder.
- Wang et al., *"Dynamic Graph CNN for Learning on Point Clouds"*
  (TOG 2019) — DGCNN; better for irregular-density LiDAR clusters.

**Wall-clock:** 8–14 days. The training data curation is the slow part,
not the model code.

---

### P3-M17 — End-to-end re-validation + numbers for the blog

**Goal:** Final re-run of the M4 closing-hero pipeline with all Phase 3
improvements stacked. Produce the before/after comparison figures that
ship in the M10 blog post AND a new Phase-3 blog post.

**Sub-tasks:**

- Re-render `sort_vs_dbscan.mp4` with the full Phase 3 stack. Same
  visual layout, but the right panel should show stable IDs throughout.
- A 4-panel comparison figure: M4 baseline | + IMM | + accumulation |
  + appearance.
- Quantitative table for the blog:

| Pipeline stage | Distinct track IDs | Mean lifetime (frames) | Stationary-segment flicker |
|---|---|---|---|
| M4 baseline | 979 | 17.4 | severe |
| + IMM | (target ≤200) | (target ≥80) | minimal |
| + accumulation | (target ≤120) | (target ≥150) | none |
| + learned appearance | (target ≤80) | (target ≥250) | none |

- Save the original M4 closing-hero animation as
  `results_m4/ablation_g/sort_vs_dbscan_baseline.mp4` for the comparison
  blog figures. (Don't overwrite — the baseline IS data.)

**Exit criterion:** A blog-ready figure demonstrating each Phase 3
improvement with numerical evidence. The story "I diagnosed the failure
mode on real data and fixed it stage by stage" is the headline.

**Wall-clock:** 2 days.

---

## Phase 3 exit criteria

- Distinct-track-IDs on RELLIS reduced from 979 (M4 baseline) to
  ≤80 (P3-M16 target).
- Mean track lifetime increased from 17.4 frames to ≥250 frames.
- Stationary segment 1750–1830 shows persistent IDs on visible trees
  throughout, no flicker.
- KD-tree DBSCAN handles 50k-point clouds in <1 sec Release.
- One Phase-3 blog post (`docs/phase3-tracking-robustness.md`) drafted
  with the 4-panel before/after figure.

---

## Why this is a portfolio-defining sequence

Phase 2 demonstrates *"I can implement classical algorithms from scratch and
prove they work via tests."* That's necessary but commoditised. LeetCode
covers it.

Phase 3 demonstrates *"I diagnosed a real-data failure mode that the unit
tests didn't catch, escalated through three production-grade fixes, and
measured the improvement at each stage."* That's the engineering signal
that distinguishes a portfolio piece from a homework submission.

The before/after figure from P3-M17 — flickering 979-track baseline next
to a stable 80-track Phase-3 result — is the single most defensible
demonstration in the whole project for an AV / robotics interview.

---

## References (consolidated)

1. Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking
   and Navigation* (Wiley 2001), Ch. 11.
2. Blackman, Popoli, *Design and Analysis of Modern Tracking Systems*
   (Artech House 1999), Ch. 8.
3. Wojke, Bewley, Paulus, *"Simple Online and Realtime Tracking with a
   Deep Association Metric"* (ICIP 2017).
4. Yin, Du, *"Dynamic Point-Cloud Accumulation for Robust Object
   Detection"* (IROS 2020).
5. Friedman, Bentley, Finkel, *"An Algorithm for Finding Best Matches in
   Logarithmic Expected Time"* (TOMS 1977).
6. Qi et al., *"PointNet: Deep Learning on Point Sets for 3D Classification
   and Segmentation"* (CVPR 2017).
7. Wang et al., *"Dynamic Graph CNN for Learning on Point Clouds"*
   (ACM TOG 2019).
8. nanoflann library — github.com/jlblancoc/nanoflann.

---

## Wall-clock summary

| Milestone | Days |
|---|---|
| P3-M12 IMM Kalman | 4–6 |
| P3-M13 Multi-frame accumulation | 3–5 |
| P3-M14 KD-tree DBSCAN | 2 |
| P3-M15 Appearance features (geometric) | 3–4 |
| P3-M16 Deep SORT (learned embeddings) | 8–14 |
| P3-M17 Re-validation + figures | 2 |
| **Total** | **22–33 days** |

This is roughly 4–6 weeks of work, runnable in parallel with Phase 2
M5+ on different worktrees.
