# Deferred TODO — blog-quality animations

When we have a final headline number we're confident in, render the
full animation suite below in one batch. The render is ~30 min per
config; doing them together overnight is cheaper than rendering
incrementally as each fix lands.

**Trigger condition for rendering — REVISED 2026-04-27.** The
"land in target window" trigger is no longer applicable: the
Phase-4 sweep showed no single config hits the original window,
and the blog narrative has been reframed as "ship the sweep,
not a number" (see `docs/m10-debug-log.md` "Phase-4 sweep — the
curve and the wall"). Trigger is now: **render after K=3 eps=0.7
lands** (in flight; this completes the sweep table). The render
batch is the side-by-side / 4-panel comparison that shows the
curve qualitatively, NOT a single hero animation pretending one
config is "the answer."

Render in batch on HPC to keep the laptop free.

Targets:
1. M4 baseline (979 distinct, flicker)
2. K=1 Mahal-v2 (242, tracker knee)
3. K=3 eps=0.5 (307, detector knee — zero teleportation false-merges)
4. K=3 eps=0.7 (TBD, included if it lands as a meaningful new point)

## Inputs to preserve

These are the files needed to regenerate every animation panel below.
Snapshotted now so they survive any future tracker iteration.

### Per-stage `tracks.csv` (already preserved as `_only` siblings)

| Variant | Path | Distinct | Lifetime |
|---|---|---|---|
| M4 baseline | `results_m4/ablation_g/sort_on_rellis/tracks.csv` (regenerate w/ `TP_M4_FILTER=cv TP_M4_MAX_AGE=0 TP_M4_EGO_POSES=`) | 979 | 17.4 |
| Fix B (world-anchor only, gate=25) | `tracks_fixB_only.csv` | 127 | 373.0 |
| Fix C (gate=10 fixed) | `tracks_fixC_only.csv` | 299 | 158.4 |
| Mahal-v1 (combined cov, χ²=5.99) | `tracks_mahal_v1_combined.csv` | 207 | 228.9 |
| Mahal-v2 (per-mode cov, χ²=5.99) — **current production** | `tracks_mahal_v2_per_mode.csv` | 242 | 195.8 |
| Mahal-v3 (per-mode cov, χ²=2.28) | `tracks_mahal_v3_chi228.csv` | 384 | 123.4 |
| Phase-4 K=3 accumulation | TBD (`tracks_phase4_k3.csv` once shipped) | TBD | TBD |

All preserved in `results_m4/ablation_g/sort_on_rellis/`. Do not delete.

### LiDAR + camera + cluster CSVs (read-only, on external disk)

- `data/extracted_frames_full/` — 2849 LiDAR PCDs (laptop disk)
- `${TP_M4_EXT_ROOT}/extracted_frames_camera/` — 2849 RGB JPEGs
  (default external disk `/media/nishant/SeeGayt2/...`)
- `${TP_M4_EXT_ROOT}/clusters_sweetspot/` — per-frame DBSCAN cluster
  CSVs (current `eps=0.5`, `min_samples=5` config)

After Phase-4 K=3 lands, a NEW cluster directory will exist
(`clusters_sweetspot_k3/` or similar). Both old and new cluster dirs
must be retained — the K=1 baseline is panel #2 of the comparison
animation.

### Pose + detections CSVs

- `data/poses_slam_full.csv` — SLAM ego pose, 2848 frames. Required
  for Fix B (world-frame anchor) and for Phase-4 (K-frame accumulation
  in world frame).
- `results_m4/ablation_g/rellis_detections.csv` — current detections.
  Re-generate with K=3 once Phase-4 lands; rename old to
  `rellis_detections_k1.csv` (no-silent-deletes rule).

## Animations to render (all in one batch)

Run with `TP_M4_RENDER_ANIM=1`. Each is one invocation of
`scripts/run_tracker_on_rellis.sh` with a different env-var prefix.

1. **3-panel closing-hero, Mahal-v2 (current)** — camera | DBSCAN | SORT.
   ~30 min on laptop. Output: `sort_vs_dbscan.mp4` + `.gif`.
   ```bash
   TP_M4_RENDER_ANIM=1 bash scripts/run_tracker_on_rellis.sh
   ```
2. **3-panel closing-hero, Phase-4 K=3** — same panels, new tracker.
   Run after Phase-4 ships. Output via `OUT_ROOT` override.
3. **4-panel comparison** — camera | M4 baseline | Mahal-v2 | Phase-4.
   The blog's killer asset. Requires `scripts/render_4panel_comparison.py`
   (already in repo per the plan; verify and bump if Mahal-v2 / Phase-4
   columns differ from the original M4/M12/M13 column set).

## Why we're holding the render

- **Each render = 30 min CPU + occupies the laptop.** Iterating on
  Phase-4 needs the same laptop free for tracker runs (~5 sec each).
- **Two false-final celebrations already this session.** 99 distinct
  was an artifact; 127 distinct was partially-inflated; 242 is the
  current shipped number but the user (correctly) flagged
  "verify before celebrating, render after." Rendering before
  ground-truth wastes the artifact.
- **Animation is a sealed asset** — once it's published in the blog
  it shouldn't be re-rendered casually. Get the numbers right first.

## When this gets unblocked

After Phase-4 audit lands and matches the expected window
(distinct in [150, 200], lifetime in [250, 320], 0 stationary
> 10 m drift, < 5 drive-wide > 20 m drift), come back here, render
the full batch, and update the blog draft. Until then this file
stays open.
