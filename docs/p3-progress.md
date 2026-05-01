# Phase 3 — Progress dashboard

Standing snapshot of what's in flight, what's blocked, and what can run in
parallel without me. Updated after each working session.

**Last update**: 2026-04-27 — **Phase-4 sweep complete. Structural ceiling confirmed; no single config is "the headline" — the curve is the story.** Six tracker variants run on RELLIS 2847 frames. Each trades false-merge tail vs distinct-ID count; none simultaneously hit the [150, 200] / [250, 320] target window because DBSCAN's per-frame cluster instability on forest geometry (centroid jitter 3 m, cluster-count delta 96% of frame pairs) is the structural floor. Best false-merge result: **K=3 eps=0.5 — 0 cascade revivals at >20m world drift** (vs K=1 Mahal-v2's 11) at the cost of cluster-fragmentation distinct-ID inflation (242 → 307). Best distinct-ID result: **K=1 Mahal-v2 — 242 distinct** with 11 surviving false-merges. **Decision: ship neither as a single headline; ship the SWEEP** as the M13.5 + Phase-4 production story. The curve is what survives interview scrutiny: "we measured the ceiling, here's the trade-off shape, here's why it sits where it does (DBSCAN cluster jitter on RELLIS forest is the floor), here's the next step (learned 3D detector)." Phase-4 status: complete — Phase-5 (learned detector) is the documented Phase-3 stretch. Animation render deferred to HPC. Currently running: K=3 eps=0.7 (recover distinct count) and min_hits=3 (already done — null result on distinct). Full sweep + ceiling diagnosis in `docs/m10-debug-log.md` "Phase-4 sweep — the curve and the wall".

---

## Test count

| Suite | Count | Status |
|---|---|---|
| test_kalman | 5 | ✅ |
| test_hungarian | 8 | ✅ |
| test_sort_tracker | 7 | ✅ |
| test_dbscan | 4 | ✅ |
| test_imm | 7 | ✅ (caveat: lock-in bias documented) |
| test_appearance_encoder | 4 | ✅ (4/4 — PyTorch ↔ Eigen match within 1e-5; ≥95% triplets satisfy d_pos < d_neg) |
| **Total** | **35** | **35 passing** |

## Latest RELLIS metrics (M12 IMM, identical detections)

### CV-baseline reproducibility — RESOLVED

The earlier "regression flag" was a misattribution by me, not a code
regression. Today's post-refactor code reproduces the M10 blog's 979
number exactly when run with the **correct** config:

| Config | CV distinct | CV mean lifetime | Rows |
|---|---|---|---|
| `min_hits=3, process_noise=0.5` (matches M10 blog) | **979** | **17.35** | 16,990 |
| `min_hits=1, process_noise=2.0` (current script default) | 2131 | 22.23 | 47,375 |
| `min_hits=1, process_noise=0.5` | 2021 | 23.44 | 47,375 |

**Finding:** The M10 blog body claims "tuning to `min_hits: 3 → 1` knocked
the count down to 979." That's incorrect — the 979 number is the
`min_hits=3` config; loosening to `min_hits=1` *increases* distinct IDs
(more one-frame ghost tracks get published instantly). Blog narrative is
wrong; numbers are real.

The Track refactor (`KalmanFilter2D kf` →
`std::unique_ptr<IFilter> filter`) is verified algorithmically clean.

### Headline IMM-vs-CV at script default (`min_hits=1, process_noise=2.0`)

| Metric | CV | IMM | Δ |
|---|---|---|---|
| Distinct IDs | 2131 | **1678** | **−21%** |
| Mean lifetime (frames) | 22.2 | **28.2** | **+27%** |
| Total publishable rows | 47,375 | 47,375 | identical |

### IMM lifetime distribution (script default config)

| Lifetime band | Count | % |
|---|---|---|
| exactly 1 frame (DBSCAN ghost) | 404 | 24.1% |
| 2–5 frames | 399 | 23.8% |
| 6–15 frames | 269 | 16.0% |
| 16–30 frames | 172 | 10.3% |
| 31–100 frames (3–10 s) | 295 | 17.6% |
| >100 frames (>10 s) | 139 | 8.3% |
| **Total** | **1678** | — |
| Max lifetime | — | 461 frames (46.1 s) |

**Reading**: 48% of tracks live ≤ 0.5 s — upstream DBSCAN noise; M13's appearance encoder cannot fix this band (the tracks die before any embedding update runs on them). 26% live > 3 s — these are the genuinely-tracked physical objects, plus the over-segmented or briefly-occluded fragments. **M13 attacks the 2–30 frame band** by re-associating fragments that DBSCAN over-segmented or that gating dropped after brief occlusion. Target: distinct 1678 → ≤ 250, mean lifetime ≥ 60.

### IMM at the M10-blog baseline config — **DONE**

| Config: `min_hits=3, process_noise=0.5` | CV | IMM | Δ |
|---|---|---|---|
| Distinct track IDs | 979 | **808** | **−17.5%** |
| Mean lifetime (frames) | 17.35 | **19.87** | **+14.5%** |
| Total publishable rows | 16,990 | 16,056 | −5.5% |

**Verdict: M12 ships.** IMM win is real and reproducible against the
M10 blog's published number. Hard floor `< 800` missed by 8 IDs (808).
Original ≤ 600 target not hit; that's a Phase-3.5 motivation for M13.

### M13 SHIPS — final config and ablation — **DONE**

**Final config (production):**
`IMM filter, λ=0.2, max_misses=300, min_hits=3, process_noise=0.5,
meas_noise=0.3, max_dist=5.0`

**Headline RELLIS numbers (2849 frames, ~285 sec drive):**

| Stack | Distinct IDs | Mean lifetime | Δ vs M4 baseline | Note |
|---|---|---|---|---|
| M4 (CV, max_misses=10) | 979 | 17.35 | — | |
| M12 (IMM, max_misses=10) | 808 | 19.87 | −17.5% / +14.5% | |
| M13 (IMM + app λ=0.2, max_misses=300) | 237 | 57.79 | −75.8% / +233% | |
| M13.5 (IMM + cascade, max_age=30, ego anchor) — Fix A defensive | 202 | ~80 | −79.4% / +361% | retired once Fix B shipped |
| M13.5 (IMM + cascade, max_age=300, WORLD anchor) — Fix B + audit caveat | 127 | 373.0 | −87.0% / +2050% | shipped but audit-incomplete: 95 cascade revivals across drive still have > 20 m world drift, leaking through `kLostPosGateScale=5.0` (25 m gate). Numbers PARTIALLY inflated by false-merges. |
| M13.5 (IMM + cascade, max_age=300, gate=10 m) — Fix C ATTEMPTED, NOT SHIPPED | 299 | 158.4 | −69.5% / +813% | Fixed-gate tightening 5.0 → 2.0. Killed all > 20 m drift false-merges (0 stationary-window > 10 m drift) BUT over-rejected DBSCAN-noisy legitimate revivals (5–15 m drift on partial-tree clusters). Worse than M13 cascade-off (237). Reverted; data showed fixed-distance gate can't separate "noisy stationary" from "different object". |
| M13.5 Mahalanobis-v1 (combined IMM cov, χ²=5.99) — TESTED, SUPERSEDED | 207 | 228.9 | −78.9% / +1219% | First Mahalanobis attempt. IMM combined covariance balloons via inter-mode spread term — admits 16 cascade revivals @ >20 m world drift on stationary ego segments. Combined cov is correct for estimation, wrong for gating. |
| M13.5 Mahalanobis-v2 (per-mode IMM cov, χ²=5.99) — knee of trade-off curve | 242 | 195.8 | −75.3% / +1027% | Picks the more confident sub-model's P_position for gating. 11 cascade revivals @ >20 m drift remain (vs 95 for Fix B) — 8.6× false-merge reduction. Sits at the trade-off knee on the *tracker* dimension. |
| M13.5 Mahalanobis-v3 (per-mode IMM cov, χ²=2.28) — TESTED, OVER-TIGHT | 384 | 123.4 | −60.8% / +611% | 2.6× tighter χ² (1σ ellipse instead of 95% confidence). Drops false-merges to 4 @ >20 m, BUT distinct IDs balloon to 384 (worse than M13 cascade-off) and lifetime collapses. Empirical proof that the gate-tuning knob has no sweet spot — DBSCAN noise dictates the ceiling. |
| **Phase-4: K=3 eps=0.5 + Mahal-v2** — knee of trade-off curve on detector dim. | **307** | **238.2** | **−68.6% / +1273%** | Multi-frame point cloud accumulation: compose K=3 obstacle clouds in world frame using SLAM ego pose, re-DBSCAN. **0 cascade revivals @ >20 m world drift (vs 11 for K=1 Mahal-v2). Halves >10 m drift bucket (248→130).** Cluster fragmentation pushes detections per frame 16.6→25.7 → distinct count rises. Lifetime up 22%. Cleaner per-cluster jitter (gap=5: 3.07 → 1.96 m, −35%). |
| Phase-4: K=3 eps=0.5 + min_hits=3 — TESTED, NULL RESULT | 307 | 145.4 | — | Stricter publication threshold. Did not reduce distinct count (the K=3 fragmented clusters are stable, not 1-frame ghosts). Strictly worse on lifetime than min_hits=1. Confirms cluster fragmentation is real, not transient. |
| **Phase-4: K=3 eps=0.7 + Mahal-v2** — combined knee | **272** | **194.8** | **−72.2% / +1023%** | Looser DBSCAN re-merges the K=3 accumulation fragmentation. **0 cascade revivals @ >20 m drift** preserved (vs K=1's 11), drive-wide >10 m count holds at 132 (vs K=3 eps=0.5's 130). Distinct count recovers from 307 → 272 (36% above target window's upper bound, vs K=3 eps=0.5's 53% above). Lifetime drops from 238 → 195 because looser DBSCAN merges fragments → fewer cluster instances → cascade has fewer revival anchors per physical object. **The "closest single point to target window" — but still NOT inside it.** No DBSCAN-paradigm config simultaneously hits [150-200] distinct + [250-320] lifetime + 0 false-merges. The structural ceiling is fully mapped. |
| ~~M13.5 max_age=300, ego anchor~~ | ~~99~~ | ~~163.66~~ | ~~−89.9%~~ | ARTIFACT: ~50% of long-gap revivals were false-merges of different physical objects sharing ego-relative position after ego motion (see `docs/m10-debug-log.md` "False revivals" STORY) |

`docs/m10-debug-log.md` carries the full ablation tables and the
"unsexy parameter mattered most" STORY entry.

### Ablation summary — what actually moved the needle

Two independent variables manipulated:

**(a) max_misses** (track-memory length): 10 → 300 → 500. Going from
M4's `max_misses=10` to `max_misses=300` alone moved distinct-IDs from
808 to ~244 (IMM-only). This was the dominant lever.

**(b) λ** (appearance weight in cost matrix): 0.0 → 0.2 → 0.8. At
`max_misses=10`: λ=0.1 gave −2% over IMM-only. At `max_misses=300`:
λ=0.2 gave −3% over IMM-only (244 → 237). Encoder-only contribution
is small but real. λ ≥ 0.5 regresses badly at any max_misses.

### Why max_misses=10 was the wrong default (rebirth-gap analysis)

For M12's 808 distinct IDs, found that **666 of them (82%)** had a
"parent track" within 2m of their birth position. Distribution of the
death-to-birth gap:

| gap band | count | % | regime |
|---|---|---|---|
| 1–10 frames | 54 | 8.1% | catchable by max_misses=10 |
| 11–60 frames | 175 | 26.3% | catchable by max_misses=60 |
| 61–300 frames | 310 | 46.6% | catchable by max_misses=300 |
| 301+ frames | 127 | 19.1% | needs full re-detection / cascade |

Conclusion: M4's `max_misses=10` (= 1-second patience) was throwing away
73% of the recoverable signal. The encoder couldn't help because the
relevant tracks were already pruned before re-detection arrived.

### Encoder ROI assessment

3 hours of hand-labeling + ~15 min HPC training + days of plumbing
produced an encoder that contributes ~1–3% additional distinct-ID
reduction beyond what `max_misses` alone delivers. The encoder is
**fully functional** (4/4 unit tests, < 1e-5 PyTorch ↔ Eigen agreement,
99% val acc on hand labels) but in this regime the IMM's
position-uncertainty growth + 5m gating already discriminates most
matches; the encoder's tiebreaker contribution is small.

**Honest interview framing:** built end-to-end Deep SORT-style pipeline
to spec; ran a proper ablation isolating max_misses from λ; found that
the upstream pipeline parameter dominated. The encoder's small win is
real but not headline-grade in this configuration.

### Phase-3.5 cascade matching — **DONE**

Implemented Deep SORT-style cascade matching: tracks transition Live → Lost
when misses > max_misses (instead of immediate erase), Lost tracks freeze
their position and stay in a retired pool for `max_age` frames, two-stage
match (Live first, then Lost on unmatched dets with relaxed position
gating). Lost track revives to Live on a successful match (filter re-init
at the new measurement, embedding preserved).

Final RELLIS result: **99 distinct IDs, 163.66 mean lifetime**.

| max_age | distinct | mean lifetime |
|---|---|---|
| 0 (cascade off) | 794 | 20.57 |
| 30 | 202 | 80.47 |
| 100 | 129 | 126.03 |
| 300 | 99 | 163.66 |
| 500 | 97 | 168.07 |
| 1000 | 95 | 172.35 |

Diminishing returns past max_age=300 (~30 sec). Default in
`scripts/run_tracker_on_rellis.sh` set to 300.

37/37 unit tests green (added two cascade tests:
`CascadeRevivesAfterLongOcclusion`, `CascadeRespectsMaxAgeBudget`).

### Phase-4 follow-ups (deferred, prioritized)

- ~~**Fix B — ego-motion compensation for cascade `lost_pos`.**~~
  **SHIPPED 2026-04-27.** Stored as `Track.lost_pos_world`; SLAM ego
  pose threaded through tracker via the `--ego-poses` CLI flag and a
  per-update `Eigen::Isometry2f T_world_ego` argument with default
  Identity (legacy / synthetic test compat). On Live→Lost the freeze
  position is mapped to world frame; on cascade match it's projected
  back into current-ego before gating. Result on RELLIS:
  **127 distinct IDs, 373-frame mean lifetime** at `max_age=300` —
  inside the predicted 120–150 window from the false-revivals
  analysis. Regression test: `CascadeRevivalSurvivesEgoMotion`
  (and the paired `CascadeRevivalWithoutEgoMotionAnchorsToDecoy` to
  pin the bug in place). 2 hr 15 min including doc updates and
  full-RELLIS validation, vs the 2-3 hr estimate.

- **Upstream DBSCAN tuning.** Even at the honest M13.5 ~202 distinct,
  ~24% are 1-frame ghost tracks. Stricter `min_samples` would directly
  reduce the upstream noise. KD-tree neighbor query for runtime.

- **Multi-frame point cloud accumulation.** Stabilizes detector
  inputs (less DBSCAN flicker → fewer Lost transitions to recover
  from). Needs SLAM odometry production-ready (which we have from
  M2 — same prerequisite as Fix B above).

---

## In flight

| What | Owner | Where | Blocking |
|---|---|---|---|
| ✅ M12 IMM RELLIS validation (808 vs 979 baseline at the M10-blog config) | done | results_m4/ablation_g/ | nothing |
| ✅ M13 hand-labeling (500 pairs, 99% prior agreement) | done | python/appearance/labels.csv | nothing |
| ✅ M13 training pipeline + HPC submission + checkpoint | done | python/appearance/checkpoints/encoder.pt | nothing |
| ✅ M13 weight export to C++ + reference test data | done | include/appearance_model_weights.hpp + tests/data/*.csv | nothing |
| **M13 cost-matrix integration in `sort_tracker.cpp:79-85`** | **user** | TBD | end-to-end appearance run + λ sweep |

---

## Parallelizable RIGHT NOW (no dependencies)

These can run in any terminal, in any order, without coordination.

### User-side

- **Hand-label ~50 cluster pairs.** Tk UI ready. ~1–2 hr.
  ```bash
  python python/appearance/build_pairs.py \
      --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \
      --num-pairs 100 --out python/appearance/pair_candidates.csv

  python python/appearance/label_pairs_cli.py \
      --pair-csv     python/appearance/pair_candidates.csv \
      --clusters-dir /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \
      --camera-dir   /media/nishant/SeeGayt2/terra_perceive/m4_perframe/extracted_frames_camera \
      --out          python/appearance/labels.csv
  ```
  Resumable, auto-skips already-labeled pairs.
  Goal: ≥ 50 non-skip labels. Triggers M13 training.

- **Read Bar-Shalom §11.6.1 + §11.6.6** (~30 pages). Was Tue 04-28's
  reading; can do anytime. Feeds m12-imm.md blog.

- **Read Wojke 2017 §3 + Hermans 2017 §4.** ~40 pages combined. Required
  reading before M13 day 2 (cost-matrix integration).

### My-side (queued, can scaffold whenever you ping)

- **`python/appearance/extract_features.py`** — per-cluster 8-dim feature
  extractor. Reads clusters_sweetspot/, writes features_NNNNNN.csv.
  ~80 lines of glue. Pure mine.
- **`python/appearance/train.py`** — PyTorch + batch-hard triplet loss.
  ~150 lines. Pure mine.
- **`slurm/train_appearance.slurm`** — HPC submission script. ~30 lines.
- **`python/appearance/torch_to_eigen_check.py`** — round-trip dump:
  trained weights → `appearance_model_weights.hpp` (overwrites placeholder)
  + `tests/data/appearance_reference.csv` + `tests/data/appearance_triplets.csv`.
- **`scripts/run_phase3_endtoend.sh`** — M14 wrapper.
- **`scripts/render_4panel_comparison.py`** — extends animate_tracker_vs_dbscan.py.

---

## Blocked / sequential

These have hard prerequisites — can't start until upstream lands.

| Item | Blocked on | Unblocks when |
|---|---|---|
| ~~Track-refactor CV regression~~ ✅ | — | RESOLVED — was a config attribution issue. |
| ~~IMM run at M10-blog baseline config~~ ✅ | — | DONE — IMM 808 / 19.87 vs CV 979 / 17.35. M12 ships. |
| M10 blog narrative correction | — | one paragraph fix in `m10-sort-tracker.md` re: which config produced 979 (the "min_hits 3→1" line is backwards). |
| `m12-imm.md` blog draft | M10 narrative fix | numbers are stable; can start drafting now in parallel. |
| HPC training submission | extract_features.py + train.py + labels.csv (≥ 50) | all three ready |
| Real `appearance_reference.csv` | training run completes | weights dumped on HPC |
| `EncoderMatchesPyTorchReference` test activation | reference CSV exists | torch_to_eigen_check.py runs |
| `EmbeddingDistanceMonotoneOnAugmentation` activation | triplets CSV exists | same |
| Cost-matrix integration in `sort_tracker.cpp:79-85` | encoder validated against PyTorch | reference test green |
| RELLIS appearance run (`--use-appearance`) | cost-matrix integration done | tests still green |
| M14 4-panel re-render | M12 + M13 ship | metrics tables ready |

---

## Recently shipped

| Date | Item |
|---|---|
| 2026-04-27 | **M13 trained encoder lands** — 99.0% val acc (epoch 18) on 499 hand-labeled held-out pairs. ~1 min wall-clock on L40S. C++ Eigen forward pass matches PyTorch < 1e-5. 35/35 tests green. |
| 2026-04-27 | M13 hand-labeling: 500 pairs, 99% agreement with prior heuristic, 1 skip. Re-reviewed 4 disagreements; kept as-is (defensible DBSCAN over-segmentation of long structures). |
| 2026-04-27 | M13 HPC submission: 5 corrections fixed in sequence (data symlink rsync, slurm partition `l40s_public`, REPO=$SCRATCH path, slim conda env on scratch, /tmp tmpfs → $TMPDIR=$SCRATCH/tmp). All logged in m10-debug-log.md. |
| 2026-04-27 | M13 pipeline written: extract_features.py (8 features/cluster, corpus stats), train.py (8→64→32→32 MLP + batch-hard triplet, tqdm progress), slurm/train_appearance.slurm, torch_to_eigen_check.py (weight export + reference CSVs). |
| 2026-04-27 | **M12 SHIPS:** IMM 808 / 19.87 vs CV 979 / 17.35 at M10-blog config (`min_hits=3, p_n=0.5`). −17.5% distinct IDs. Earlier "regression" alarm was a config attribution mistake. |
| 2026-04-27 | M12 day 5: Track refactor → `unique_ptr<IFilter>`, `--filter cv|imm` flag |
| 2026-04-27 | M13 day 1 scaffold: appearance_encoder + 4 test scaffolds + placeholder weights |
| 2026-04-27 | M13 labeling pipeline: build_pairs.py + label_pairs_cli.py |
| 2026-04-27 | docs/m10-debug-log.md M12 section appended (5 entries + open items) |
| 2026-04-27 | scripts/run_tracker_on_rellis.sh: `--filter` plumbed via TP_M4_FILTER env |
| 2026-04-26 | M12 day 1-4: IMM filter implementation, 7/7 tests, Occam's-razor diagnosis |
| 2026-04-26 | HPC `data/` symlink fix in sync_to_hpc.sh + docs/hpc-access.md |

---

## Maintenance rule

Update this file at the end of every working session. Keep it under ~150
lines so it loads fast. Recently-shipped older than ~10 days drops off.

If a row in the "blocked" table unblocks: move it to "in flight" or
"parallelizable" depending on who's the doer.

If something breaks: log to `docs/m10-debug-log.md` (existing log), don't
duplicate here. This file is forward-looking; the log is post-hoc.
