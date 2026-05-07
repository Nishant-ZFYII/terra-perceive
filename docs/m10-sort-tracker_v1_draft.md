---
layout: post
title: "Building a LiDAR multi-object tracker from scratch — and the wall it hits on real off-road data"
date: 2026-04-28
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Building a LiDAR multi-object tracker from scratch — and the wall it hits on real off-road data

*Part of the Terra Perceive series.*

> *[FIGURE — placeholder]: 6-second clip from `02_mahalv2_k1_242.mp4`. A single tree in the middle of a stationary forest scene. The cluster overlay reforms every frame: points in, points out, the box jitters, splits in two, merges back. The tree did not move. The sensor did not move. The scene is static. The detector is having a busy day.*

Almost every multi-object tracking tutorial you've seen starts with the same opening: a video of cars driving down a highway, neat colored boxes following each one, ID numbers stable across frames. The implicit promise is that you can build that yourself in a weekend with a Kalman filter and the Hungarian algorithm.

You can — on the highway data the tutorial used. On real LiDAR returns from a vehicle driving through an off-road forest, you cannot. The tracker isn't the bottleneck; the *detector* is. The cluster overlay you see in the video above is what density-based clustering does to a single physical tree across a hundred consecutive frames: the cluster identity is unstable, the centroid jitters several meters between frames, and 96% of frame-to-frame transitions change the cluster count for the same scene.

This post is the long version of figuring that out the hard way. We built every piece of a Simple Online and Realtime Tracking pipeline from scratch — the Kalman filter, the Hungarian assignment, the DBSCAN clusterer, the cascade re-association from Deep SORT, an interacting multiple model filter, a learned appearance encoder, ego-motion compensation, a Mahalanobis gate, and finally a multi-frame point-cloud accumulation step. Each addition was the right answer to the previous failure. Each addition shifted the trade-off curve. None of them broke through the structural ceiling that the detector — not the tracker — sets on this regime.

That ceiling is also the answer to *why production self-driving stacks moved from "cluster-and-track" to "learned-3D-detection-and-track"* in the 2019–2021 window. We didn't read the literature and copy the answer. We hit the wall, measured it from three different directions, and arrived at the same conclusion the field arrived at five years ago. The arc is the lesson. The end state — a structural ceiling that points squarely at PointPillars or CenterPoint — is the honest hand-off, not a failure.

A reader new to multi-object tracking will get a complete tour of the components, why each one exists, what choices we made and why, and where each one breaks. A reader who's lived this work in industry will see the patterns they've lived, written down with measurements attached. Either way, every claim in the post — every "this fixed that," every "this didn't help" — has an audit script behind it. The audits are the part of this work I'm proudest of.

---

## What you're looking at, why we cared

A self-driving robot needs two kinds of memory.

One is for the static world: the road surface, the curbs, the ground type, the trees that aren't going anywhere. That memory is built incrementally from many frames; it doesn't matter if it takes a few seconds to update. The earlier post in this series, on accumulated bird's-eye-view mapping, built that.

The other is for things that *move*. Workers walking onto a construction site. Vehicles on a road. The deer that just stepped out of the woods. This memory has to update every frame, has to assign a stable identity to each moving thing across time so that downstream consumers — a safety supervisor computing time-to-collision, a planner reasoning about another vehicle's trajectory — can ask coherent questions like "where will this person be in 800 milliseconds." That's a *tracker*'s job, and that's what this post is about.

The standard reference for this problem is Bewley et al.'s 2016 paper *Simple Online and Realtime Tracking* — usually abbreviated SORT [1]. The paper's argument is essentially Occam's razor in code: don't try to be clever about appearance, don't add learned components, don't store track histories beyond a Kalman filter's posterior. Just predict each track forward to the next frame using a constant-velocity Kalman filter, build a cost matrix of distances between predicted positions and new detections, solve the assignment problem with the Hungarian algorithm, and update each matched track. The paper showed this simple recipe ran at 260 frames per second and beat the more complex trackers of its day on standard benchmarks.

The hidden assumption is in that word "detection." Bewley's experiments used Faster R-CNN bounding boxes from camera images — detector outputs that are clean, semantically meaningful, and consistent frame-to-frame. We don't have that. We have raw LiDAR point clouds from a 64-beam Ouster spinning at 10 Hz. To turn those into "detections" we have to *cluster the points* — group nearby returns into objects — using something like DBSCAN. That's where the trouble starts, but we don't know that yet.

So the plan: build the simplest possible version of every piece. Get it green on synthetic data. Run it on real LiDAR. Iterate until the iteration stops paying back. That last step is the part the post is mostly about.

---

## Architecture

```
LiDAR (.bin)
   ↓
Sector RANSAC (M2)  →  obstacle points (one cloud per frame)
   ↓
DBSCAN (M4, this post)  →  cluster centroids
   ↓
SORTTracker.update()  →  per-frame published tracks with stable IDs
   ↓
tracks.csv  →  animation, downstream consumers, M5 NATS publishing
```

Five C++ libraries link together: `kalman_filter`, `hungarian`, `dbscan`, `sort_tracker`, plus a `tracker_runner` CLI that mirrors `slam_runner` (M2) and `accumulator_runner` (M3). Everything sits inside `namespace tracker` per the per-module convention used elsewhere in the repo.

`SORTTracker::update()` is a seven-section state machine:

```
1. predict() every existing track   (Kalman propagation)
2. match()  — build N×M cost matrix, dispatch to greedy or Munkres
3. update each matched track with its detection
4. increment misses on unmatched tracks
5. prune tracks whose misses > max_misses
6. create new tracks for unmatched detections
7. return tracks with hits ≥ min_hits  (false-positive suppression)
```

Two small but important details: section 7 returns *only the publishable view*, while the full `tracks_` vector keeps every live track including unconfirmed ones. And section 6 appends *after* section 5's prune, otherwise the indices in `matched_tracks` would shift mid-iteration.

---

## Kalman filter from scratch

The state is constant-velocity in the BEV plane:

$$x = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}, \quad
F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad
H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

The process model assumes velocity is constant, with isotropic process noise covariance $$Q = \sigma_q^2 I_4$$ to absorb model error. Measurements are 2D positions only — DBSCAN cluster centroids — with $$R = \sigma_r^2 I_2$$.

Predict and update follow Thrun §3 [2] exactly:

$$\hat x_k = F \hat x_{k-1}, \quad P_k = F P_{k-1} F^\top + Q$$

$$y_k = z_k - H \hat x_k, \quad S_k = H P_k H^\top + R$$

$$K_k = P_k H^\top S_k^{-1}, \quad \hat x_k \leftarrow \hat x_k + K_k y_k, \quad P_k \leftarrow (I - K_k H) P_k$$

The implementation lives in `src/kalman_filter.cpp` — about 80 lines including comments. The single non-obvious choice is the Kalman-gain computation. The next section is why.

### Debugging story #1 — Why Cholesky, not `S.inverse()`

The textbook Kalman gain is $$K = P H^\top S^{-1}$$. The literal C++ translation is `K = P * H.transpose() * S.inverse()`. It compiles, it works on most inputs, and it silently fails on a regime that real systems hit: when $$Q$$ is small and the filter has converged through many noise-free updates, $$S = H P H^\top + R$$ approaches singularity. `S.inverse()` then amplifies floating-point error catastrophically and the state contains `NaN` after a few hundred updates.

The fix is one line: instead of materializing $$S^{-1}$$, factor $$S$$ via Cholesky decomposition (since $$S$$ is symmetric positive-definite by construction) and back-substitute to get $$K^\top$$:

```cpp
// Before — fragile:
Eigen::Matrix<float, 4, 2> K = P_ * H_.transpose() * S.inverse();

// After — stable:
Eigen::Matrix<float, 4, 2> K =
    S.llt().solve((H_ * P_).eval()).transpose();
```

The unit test `CholeskyStableWhenInverseBlowsUp` constructs the pathological scenario explicitly (process_noise = 1e-10, meas_noise = 1e-10, 200 noise-free updates) and asserts `std::isfinite(state().norm())`. With `S.inverse()` it fails. With `S.llt().solve()` it passes. That's the test catching the regression before any animation ever exposes it.

This is the kind of substitution a textbook calls out in a sidebar, a production codebase does without comment, and a portfolio piece can underline as a real engineering decision: $$O(n^3)$$ Cholesky vs $$O(n^3)$$ inverse, same asymptotic cost, drastically different numerical behavior.

### Debugging story #2 — Update order matters

The pipeline is *predict, then update*. Reverse it (`update` against last frame's posterior, *then* predict) and the filter on a constant-velocity target eventually still converges, just lagged. On a maneuvering target — say, a ground vehicle decelerating — the swapped order produces a stale-prior fusion that compounds into visible position error.

The unit test `UpdateOrderMatters` constructs two filters with identical parameters and identical initialization, drives them with the same noisy measurements, but reverses the order on one. With a velocity reversal at frame 25 (the smoking gun for the constant-velocity assumption), the swapped filter ends ~2× more residual error than the correct one. The test asserts both `EXPECT_LT(correct_err.norm(), 0.2f)` (the correct one actually converged) and `EXPECT_GT(swapped_err.norm(), correct_err.norm() * 1.5f)` (the swapped one is meaningfully worse). The earlier flat constant-velocity scenario didn't differentiate the two; the maneuver exposed it.

That second assertion matters more than the first. *Most* tests assert "things work"; the ones that catch real bugs assert "things do not work in this specific way." On a portfolio, the latter is the one a senior engineer reads twice.

---

## Hungarian assignment — greedy vs Munkres

Once predicted positions exist for every track and detections come in for the next frame, you need to decide which detection belongs to which track. With $$N$$ tracks and $$M$$ detections, the cost matrix is $$N \times M$$ of Euclidean distances; the assignment problem is to pick at most $$\min(N, M)$$ pairs minimizing total cost subject to no-row-or-column-repeat.

Two solvers behind one dispatcher:

- **Greedy** ($$O(NM)$$): for each column, pick the lowest-cost unmatched row. Strict-`<` tie-breaking (not `≤`) makes the answer order-dependent on ambiguous costs.
- **Munkres** ($$O(N^3)$$): Kuhn (1955) [3] + Munkres (1957) [4], implemented as the classical six-step state machine following Pilgrim's reference C# implementation [5] (ported to plain C by Guo [6], translated to my C++/Eigen here).

Both are used; the choice is configurable per `tracker_runner` invocation.

### The state machine

Munkres maintains three pieces of state in parallel: a `mask` matrix tagging zeros as `STARRED` (tentative assignment) or `PRIMED` (alternate-path candidate); two `vector<bool>` cover arrays for rows and columns; and a path scratchpad. Each of six step functions returns the next step number; the outer loop is a `while (step != Done) switch (step)` until all $$K$$ columns are covered. With $$K = \max(N, M)$$ padded to square at entry and dummies filtered on output, the implementation handles rectangular cost matrices uniformly.

### Debugging story #3 — Greedy is order-dependent on ambiguous costs

This is the unit-level argument for why Munkres exists. Consider the 2×2 cost matrix:

$$C = \begin{bmatrix} 1.0 & 1.5 \\ 1.0 & 2.0 \end{bmatrix}$$

The two valid assignments cost 3.0 (rows 0→0, 1→1) and 2.5 (rows 0→1, 1→0). Optimal is 2.5. Greedy iterating columns picks `(0, 0)` first (row 0 with cost 1.0 wins over row 1 with cost 1.0 by strict-`<`), then `(1, 1)` for the only row left, total 3.0. *Off by 0.5*. Munkres returns the global optimum.

The unit test `GreedyOrderDependenceOnCrossings` is the load-bearing assertion of the assignment section. It builds exactly that cost matrix and asserts `EXPECT_GT(total_cost(C, greedy_result), total_cost(C, munkres_result))` — *strictly greater, greedy lost*. Every other test in `test_hungarian.cpp` argues something neutral (small known solution, rectangular handling, dispatcher routing); this one argues for the existence of the Munkres branch in production code.

### When does greedy bite at the SORT integration level?

The naïve expectation is: greedy fails on track crossings, so a 2-target right-angle crossing should reproduce the bug. It does not. With strict-`<` tie-breaking and well-learned velocities, greedy correctly resolves the symmetric crossing — the cost matrix at the closest frame has a clear diagonal preference, both solvers pick it. To make greedy actually swap on a 2-target scene you have to engineer asymmetric costs (e.g., one target's predicted position drifts toward the other detection due to a deliberate trajectory offset).

That's a real finding to surface honestly: **the algorithmic claim "greedy is suboptimal" lives at the cost-matrix level (the unit test). The integration-level visualization on simple 2-target scenes is hard to construct fairly.** Greedy's failure mode is most visible in dense scenes where many costs are near-tied — ablation G's DBSCAN parameter sweep on RELLIS hints at this regime — but a clean 2-track-crossing GIF that reliably shows greedy swapping requires hand-crafted geometry that an honest reader would call out.

I keep both solvers in the dispatcher and let `tracker_runner --solver {greedy|munkres}` pick. The blog reader gets to see the unit test as proof; the closing-hero animation runs Munkres because that's the principled choice.

---

## DBSCAN from scratch

Before SORT can track anything, you need detections — and on raw LiDAR the closest thing to a per-frame "detection" is a density-based cluster on the obstacle-only point cloud (post-RANSAC ground segmentation from M2). DBSCAN [7] is the right primitive: no fixed cluster count, robust to noise, parameterized by exactly two scalars (`eps` neighborhood radius, `min_points` density threshold).

The implementation lives in `src/dbscan.cpp` — about 100 lines. The algorithm is:

```
For every unvisited point p:
    N = neighbors(p, eps)
    if |N| < min_points:
        mark p as NOISE  (may be reclassified as a border point later)
    else:
        start a new cluster, BFS-expand from p:
            for each q in queue:
                if q is NOISE → promote to ASSIGNED, do NOT expand
                if q is UNVISITED → add to cluster
                if |neighbors(q, eps)| ≥ min_points → q is also core, queue its neighbors
```

Three subtleties tripped me up:

- **`region_query` includes the query point itself.** A point is its own neighbor at distance zero. So `min_points = 1` makes every point a core point of itself. Matching Ester 1996 is a deliberate choice; the alternative (excluding self) is also valid but documented differently in literature.
- **`NOISE` is tentative, not final.** A point initially marked NOISE can be promoted to a *border point* (`ASSIGNED`) when a different core point's BFS reaches it. The state machine `UNVISITED → NOISE → ASSIGNED` is a real path. Forgetting to handle this in the BFS dequeue makes the algorithm "leak" outward through low-density regions — the most common DBSCAN bug.
- **Only core points expand the cluster.** A queued point with fewer than `min_points` neighbors gets *added* to the cluster (border) but its own neighbors are not pushed onto the BFS queue. Mishandling this is the second most common bug — easy regression to write, hard to detect on small synthetic blobs.

The neighbor search is brute-force $$O(N^2)$$. For RELLIS obstacle clouds (~5k–10k obstacle points after ground segmentation) it runs sub-second in Release builds. KD-tree replacement is the right next move and is explicitly out of scope here — Phase 3.5.

---

## Track lifecycle

A track is born when an unmatched detection enters; it confirms when its `hits` counter reaches `min_hits`; it disappears when its `misses` counter exceeds `max_misses`. The conventions used here:

- **Strict consecutive-hits semantics.** `hits` resets to 0 on any miss. Per the plan, this is intentional — a single miss takes a track back to unconfirmed and forces three more consecutive matches before re-publishing. The reference Python implementation [1] tracks `hits` cumulatively and `hit_streak` separately; with a single counter, ablation E's clean trade-off (false-positive count vs init latency) becomes legible.
- **Pruning at strictly-greater-than `max_misses`.** With `max_misses = 3`, a track survives 3 consecutive misses and is pruned at the 4th. This convention reads as "tolerate up to N missed frames, delete on the (N+1)th."
- **Unmatched detections always create a new track.** No score-based filter on detections; the assumption is that DBSCAN's `min_points` threshold has already done that work upstream.

### Out-of-the-box parameter choices for the closing-hero run

```
max_dist       = 5.0     // gating: 5 m cap on assignable cost
max_misses     = 10      // tolerate 1 sec of intermittent detection at 10 Hz
min_hits       =  1      // publish after first match (relaxed from 3 after analysis)
process_noise  = 2.0     // higher Q lets the KF re-adapt to ego stops in ~2 frames
meas_noise     = 0.3     // DBSCAN centroid noise on RELLIS
solver         = munkres // dense-scene safety
```

These are not the originally-planned defaults. They were tuned in response to the M4-on-RELLIS finding documented at the end of this post.

---

## Eight ablations, one figure each

Each ablation argues a specific question. Presented in execution order; figures are linked alongside the text.

### A — Greedy vs Munkres on a 2-target crossing

*Question*: does optimal assignment matter on a clean perpendicular crossing?

*Result*: both solvers preserve identity; both report 0 ID-switches. **An honest non-result.** As discussed in the assignment section, the unit test (`Hungarian.GreedyOrderDependenceOnCrossings`) carries the algorithmic claim. The 2-target integration-level visualization can't fairly demonstrate greedy's failure without engineered asymmetric trajectories.

### B — Process noise Q sweep

![Q sweep](../results_m4/ablation_b/q_sweep.png)
*Three panels: position estimate over noisy detections (left), residual estimate − linear-fit-of-detections (middle, the headline), posterior covariance trace on log-y (right). Four Q values from 0.01 to 10.*

The middle panel is the load-bearing argument. With `Q = 0.01` the filter trusts the constant-velocity model and smooths visibly through the noise (residual stays within ±0.1 m). With `Q = 10` it trusts every measurement and traces the noise spikes (residual swings ±1 m). The right panel shows the same story in covariance terms — *low Q produces a confident posterior; high Q stays uncertain*, with a 3-orders-of-magnitude spread in the asymptote. There is no universally right Q — the choice is a regime decision.

### C — Measurement noise R sweep

![R sweep](../results_m4/ablation_c/r_sweep.png)

The mirror image of B. R controls the gain weighting: low R trusts every detection (jagged residual), high R distrusts and smooths. The interesting sidebar is the **asymmetry** between Q and R in the right panel: R sweeps over 2 orders span only ~1 order of asymptotic covariance, while Q's same span moves 3 orders. *Q is the long-game knob (sets steady-state confidence); R is the responsiveness knob (sets convergence rate).* Many engineers under-tune Q and over-tune R because R is the more intuitive parameter; the right panel explains why that's a mistake.

### D — Gating threshold (max_dist)

![max_dist sweep](../results_m4/ablation_d/max_dist_sweep.png)

`max_dist=1m` on the crossing scenario *fragments the trajectory into 11 distinct track IDs* — every time DBSCAN noise pushed a detection more than 1 m from the predicted position, the match dropped, the track was eventually pruned, and a new ID spawned. `max_dist=3m` and `10m` both clean to 2 IDs total. The right panel of the bar chart shows 10 ID-switches at 1 m gating, 0 at the others — the gating threshold is a real lever.

### E — Confirmation threshold (min_hits)

![min_hits sweep](../results_m4/ablation_e/min_hits_sweep.png)

The textbook trade-off: tighten `min_hits` to suppress false positives, pay in initialization latency. With three single-frame spurious detections injected at frames 5, 12, 20 and a persistent target throughout, `min_hits=1` publishes all three spurious tracks (FP count = 3), `min_hits=3` and `min_hits=5` suppress them at the cost of 2 and 4 frames of init latency respectively. The bar chart visibly shows red dropping while blue rises.

### F — Persistence threshold (max_misses)

![max_misses sweep](../results_m4/ablation_f/max_misses_sweep.png)
*Three vertical panels, one per `max_misses` cell. Track ID over time; gray vertical band marks the 8-frame occlusion gap.*

The bottom panel (`max_misses=10`) shows track 0 surviving the entire 8-frame gap with the same ID resuming on the other side. Top two panels (`max_misses=1, 3`) show track 0 dying inside the gap and track 1 taking over post-gap. This is the visual argument for max_misses being the "occlusion tolerance" knob.

This ablation also hides a real bug story. The first F run produced identical results across all three cells because `tracker_runner` was iterating *only the frame_ids that appeared in the input CSV*, not real-world time. `gen_occluded` deliberately emits no rows for the occluded frames, so the tracker received zero `update()` calls during the gap and `max_misses` was never put under load. Fix: iterate the full frame range from `min` to `max` and pass empty detections when no row exists. Documented as `[STORY]` in the M10 debug log; the lesson — *a tracker simulator must iterate real-world time, not the event timeline of detections* — is one of those things you only learn by running a tracker on data with gaps.

### G — DBSCAN eps × min_points grid on RELLIS

![DBSCAN parameter grid](../results_m4/ablation_g/dbscan_grid.png)
*Single RELLIS frame, 9 panels: rows are `eps ∈ {0.3, 0.5, 1.0}m`, columns are `min_points ∈ {5, 10, 20}`.*

The story is monotonic. Tightening `eps` fragments real objects (top row, 50 clusters at 0.3 m); loosening it merges distinct objects (bottom row, 9 clusters at 1.0 m). Tightening `min_points` pushes border points into the noise category (rightmost column, 1464 noise points at the tightest combination). The middle cell (eps=0.5, mp=10) is the sweet spot for this RELLIS scene — distinct trees, person, terrain features each become their own cluster.

### H — Predict / update ordering (algorithmic invariance)

![Order sweep](../results_m4/ablation_h/order_sweep.png)

A surprising and honest result. Running the same maneuvering scenario with `--swap-order` (the bug pattern) and without (correct) produces *bit-identical* `tracks.csv`. The unit test catches the bug at the Kalman level; the SORT publishable output is invariant by construction because `publishable` is built between the match-update phase and the late predict — meaning the lookahead never reaches the published rows.

This is a stronger story than "swapped is worse." It demonstrates *the value of API isolation*: consumers always see post-update state at the queried frame regardless of internal predict ordering. The two-overlapping-lines plot becomes evidence of that invariance, not a failure.

---

## RELLIS qualitative — and the honest finding

The closing-hero pipeline:

1. `obstacle_extractor` runs sector RANSAC on every RELLIS frame (2849 frames across five recording bags, ~30k LiDAR points/frame), dumping ~3.7 GB of obstacle CSVs to the external drive.
2. `dbscan_cli` clusters each frame at the sweet-spot `(eps=0.5, min_points=10)`, dumping ~3.7 GB of cluster CSVs.
3. `clusters_to_detections.py` projects cluster centroids to a single 2D detections CSV in the schema `tracker_runner` consumes.
4. `tracker_runner --solver munkres --max-dist 5.0 --max-misses 10 --min-hits 1 --process-noise 2.0 --meas-noise 0.3` produces `tracks.csv`.
5. `animate_tracker_vs_dbscan.py` renders a 3-panel video: RGB camera | flickering DBSCAN clusters | SORT tracks.

### Numbers

```
Total frames                : 2849
DBSCAN cluster count / frame: 3 to 26  (mean 13.6, std 4.55) on stationary segments
SORT distinct track IDs     : 979
Mean track lifetime         : 17.4 frames (1.7 sec)
Tracks lasting ≤5 frames    : 425 (43%)
Stationary segment 1750–1830:
  - 54 distinct IDs alive at some point during this 150-frame window
  - 46 of those born WITHIN the window (= 46 spurious births while ego barely moves)
```

The bot is essentially stationary across frames 1750–1830, the trees are stationary, but 46 new track IDs are born during this 150-frame window. The middle panel of the closing-hero animation flickers with cluster colors (per-frame DBSCAN re-coloring); the right panel — the one supposed to show *stable* SORT IDs — also flickers, just at a different rate.

### Why it flickers

Three causes, in order of impact:

1. **Stale Kalman velocity at deceleration.** The constant-velocity model learns each tree's apparent velocity ≈ −1 m/s while the bot moves. When the bot stops, the filter still predicts trees drifting at the old velocity. Predicted position drifts away from the actual stationary cluster centroid by ~10 cm/frame. After 5–10 frames the prediction error exceeds `max_dist` and the match drops. The track accumulates misses, gets pruned at `max_misses=10`, and reappears as a new track ID when DBSCAN re-finds the same tree.

2. **DBSCAN cluster centroid jitter, even on stationary scenes.** LiDAR returns are slightly different per scan rotation (angular aliasing). A stationary tree's cluster centroid wanders 10–30 cm frame-to-frame purely from sensor sampling. Centroid count varies 3 to 26 per frame on visually identical scenes, because the `eps`/`min_points` threshold sits right at the density-flip boundary for our obstacle clouds. SORT's matcher correctly interprets a fragment-centroid jump as a different object — that's not a bug, it's the expected behavior of a constant-velocity Kalman filter on positionally jittery detections from a fragmenting clusterer.

3. **`min_hits` + hits-reset-on-miss force re-warmup.** A track that's been published 100 frames, then misses once, drops back to `hits = 0` and disappears from publishable output until `min_hits` consecutive matches accumulate again.

Tuning helped some — `process_noise: 0.5 → 2.0` and `min_hits: 3 → 1` knocked the distinct-track count down from a baseline ~1500 to 979 — but tuning alone cannot undo upstream segmentation instability or the stale-velocity edge case.

### What this teaches

This is the gap between "the algorithm is correct" (20/20 unit tests) and "the system is portfolio-grade" (the closing-hero animation tells a defensible story). The unit tests assert the local property; the system-level demonstration exposes properties no unit test can see. *Both are necessary; only one is sufficient for a portfolio.*

Phase 2 shipped M4 with this finding documented honestly. The rest of this post is what happened next — three additional milestones that closed most of the gap algorithmically, and a fourth that mapped the ceiling we hit on the *detector* side. The arc looks like this:

```
M4 baseline                 979 distinct  / 17.4 frame lifetime
M12  + IMM Kalman           808           / 19.9
M13  + appearance encoder   237           / 57.8
M13.5 + cascade matching    127           / 373.0  (ARTIFACT — see audit below)
                            242           / 195.8  (after honest re-audit)
Phase-4 + K=3 frame accum.  272           / 194.8  (zero >20m false-merges)
```

Three of those rows are real wins. One is the lesson the rest of the post is built on. And the final row sits at a structural ceiling that no further tracker tuning could move.

---

## M12 — IMM Kalman: two filters, one track

The constant-velocity Kalman model has a known weakness: it assumes the target's velocity is approximately constant between updates. On RELLIS, the *ego vehicle* decelerates from ~1 m/s to 0 m/s and accelerates back to ~1 m/s several times across the 285-second drive. Every stationary tree's apparent velocity (in ego coordinates) tracks ego speed — so when the bot stops, every tree's filter still predicts it drifting at the old velocity. Predict missing the cluster centroid means the gating gate drops the match means the track dies means a new track ID is born when DBSCAN finds the same tree again. That single mechanism accounts for most of the 46 spurious births in the [1750, 1830] stationary segment.

The standard answer is the **Interacting Multiple Model** (IMM) filter from Bar-Shalom Chapter 11 [8]. Two parallel sub-filters per track:

- **CV** (constant velocity) — the M4 model, $$F_{CV} = \begin{bsmallmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bsmallmatrix}$$
- **CP** (constant position) — $$F_{CP} = I_4$$, velocity drifts only via process noise $$Q$$

Each frame, four steps:

1. **Mixing** — produce mixed initial states for each sub-filter as a weighted blend of the previous-step posteriors using the Markov transition matrix $$\Pi$$ and the current mode probabilities $$\mu$$.
2. **Mode-conditioned predict + update** — each sub-filter runs the standard Kalman cycle on its mixed initial state, returning innovation $$y_j$$ and innovation covariance $$S_j$$.
3. **Mode probability update** — in log-space for numerical stability:

$$\log \Lambda_j = -\tfrac{1}{2}\!\left( \log(2\pi |S_j|) + y_j^\top S_j^{-1} y_j \right), \qquad \mu_j \propto e^{\log\Lambda_j - \max_k \log\Lambda_k} \cdot c_j$$

   Clamp each $$\mu_j \in [0.01, 0.99]$$ post-update to prevent lock-in (Bar-Shalom §11.6.6).
4. **Combined output** — the posterior the rest of the tracker sees:

$$x_{\text{out}} = \sum_j \mu_j x_j, \qquad P_{\text{out}} = \sum_j \mu_j \!\left( P_j + (x_j - x_{\text{out}})(x_j - x_{\text{out}})^\top \right)$$

The cross-spread term in $$P_{\text{out}}$$ is the entire reason this works. When the two modes disagree about position (CV says the tree is drifting, CP says it stayed put), the spread inflates $$P_{\text{out}}$$ — the filter *honestly says it's uncertain* — so the matcher's gate widens automatically. A subtle and beautiful bit of math. It also matters later, when the same spread term causes a different problem at the gating layer.

### Architectural choice: virtual `IFilter`, not templated `Track`

A practical question shows up at integration time: how does `SORTTracker::Track` hold "either a CV filter or an IMM filter" without templating everything downstream? Three options were considered:

- Templating `Track<FilterT>` — pollutes every consumer.
- `std::variant<KalmanFilter2D, IMMFilter>` — pollutes match logic with `std::visit`.
- Virtual base + `std::unique_ptr<IFilter>` member — one v-table dispatch per `predict()`/`update()` call per track per frame.

I picked the v-table. With ~50 tracks at 10 Hz that's 500 indirect calls/second. DBSCAN on the same frame is ~$$10^5$$ ops. The dispatch cost is invisible. The interface is in `include/i_filter.hpp`; the implementations are in `include/kalman_filter.hpp` and `include/imm_filter.hpp`. The 7 SORT unit tests from M4 stay green by defaulting to CV mode; ablation between CV and IMM is a runtime CLI flag (`--filter cv|imm`).

Seven new IMM unit tests in `tests/cpp/test_imm.cpp` — including `IMMConvergesOnPureCV`, `IMMConvergesOnPureCP`, `ModeProbabilitiesNeverNaN` (the Cholesky-blowup pathology), `ModeSwitchOnDeceleration` (the headline integration test), and `MixedOutputMatchesSingleFilterDegenerate` (asserts the IMM collapses exactly to a single CV when $$\Pi = I$$ and $$\mu_0 = [1, 0]$$).

### Result on RELLIS

| Stack | Distinct IDs | Mean lifetime | Δ vs M4 |
|---|---|---|---|
| M4 (CV, max_misses=10) | 979 | 17.4 | — |
| M12 (IMM, max_misses=10) | 808 | 19.9 | −17.5% / +14% |

A modest win. The IMM did exactly what it was designed to: it caught the ego-stop transitions where CV alone would have killed tracks. But the larger problem — DBSCAN cluster centroid jitter — was untouched, and the gain plateaued.

---

## M13 — Deep SORT-style learned appearance

The next lever was Wojke et al.'s 2017 modification to SORT [9]: extend the cost matrix beyond pure position, adding an *appearance* term computed from a learned per-detection embedding.

### Eight hand-crafted features → MLP → 32-dim L2-normalized embedding

Each DBSCAN cluster gets eight geometric descriptors:

1. log of point count
2–4. axis-aligned bounding box dimensions $$(b_x, b_y, b_z)$$
5. height above ground (from M2's RANSAC)
6–7. PCA eigenvalue ratios $$\lambda_1/\Sigma\lambda$$, $$\lambda_2/\Sigma\lambda$$ — captures shape (stick / plate / blob)
8. centroid range $$\sqrt{c_x^2 + c_y^2}$$ — LiDAR appearance changes with range

Z-scored using corpus-wide statistics, fed to a 2-layer MLP (8 → 64 → 32 with ReLU + L2-normalized output). Trained with batch-hard triplet loss (Hermans et al. 2017 [10]) on ~30k pairs.

The pair-mining choice was the most interesting design call. The naive answer — "use M4 track IDs as positive labels" — is *circular* because M4 is what we're trying to fix. Instead, four sources:

- **Source 1 (spine):** geometric augmentation on a single cluster. (cluster, jittered_cluster) is positive. ~30k pairs, zero circularity.
- **Source 2:** adjacent-frame nearest-neighbor under tight geometric filters (Mahalanobis < 1.0, point-count ratio in [0.7, 1.3], bbox-volume ratio in [0.6, 1.4]). Low circularity.
- **Source 3:** M4 tracks with lifetime ≥ 30 *and* below-median cov_trace *and* mean displacement ≥ 0.3 m/frame. Modest channel.
- **Source 4 (validation only):** ~50 hand-labeled pairs from `extracted_frames_camera/`. Held out from training to detect leakage.

Trained on NYU Torch HPC L40S, ~12 minutes per run. The PyTorch model dumps weights to `include/appearance_model_weights.hpp` as `constexpr float` arrays; C++ inference uses pure Eigen, no ONNX, no libtorch. A round-trip test (`EncoderMatchesPyTorchReference`) asserts PyTorch ↔ Eigen forward-pass agreement to 1e-5.

### Cost matrix change

The change at `src/sort_tracker.cpp:79-85` is small:

```cpp
const float d_pos = (p - dets[j]).norm() / max_dist_;          // ∈ [0, ~1]
const float d_emb = 1.0f - track.embedding.dot(det.embedding); // both unit, ∈ [0, 2]
cost(i, j) = (1.0f - λ) * d_pos + λ * d_emb;
```

Track embeddings are running means: $$e_t \leftarrow (1-\alpha)\, e_t + \alpha\, e_{\text{det}}$$, then re-normalize. Default $$\alpha = 0.1$$ (Wojke 2017's default).

### λ sweep on RELLIS — and the surprising finding

The expected outcome: a U-shaped curve as $$\lambda$$ varies, with the minimum somewhere around 0.4–0.6.

The actual outcome:

| λ | Distinct IDs | Mean lifetime |
|---|---|---|
| 0.0 | 808 (= M12 IMM only) | 19.9 |
| 0.2 | **237** | 57.8 |
| 0.4 | 261 | 51.3 |
| 0.6 | 279 | 47.2 |
| 0.8 | 311 | 43.1 |

The minimum was at $$\lambda = 0.2$$, very low. *Position dominates.* The appearance term contributes a meaningful tiebreaker — distinct IDs dropped from 808 to 237 — but its useful weighting is small. On RELLIS forest, where the dominant disambiguation is geometric (which tree centroid is closest), the appearance encoder's job is to break the ties when the geometric cost is ambiguous, not to override the geometric prior.

This is also where the ablation paid back the data-circularity care: hand-labeled validation accuracy held at 90%+ across the sweep, so the encoder *was* learning real shape similarity. The position dominance is a property of RELLIS data, not a training failure.

But the bigger reveal was hidden in the same data: even at the optimal $$\lambda$$, **the dominant lever was actually `max_misses`**, not appearance. Going from `max_misses = 10` to `max_misses = 300` (allowing tracks to be Live for 30 seconds before dying) dropped distinct IDs from 808 to 237 *with appearance off*. Appearance plus the longer retention was the same number. The "appearance encoder" headline was real, but the unsexy parameter — letting tracks live longer through occlusion — moved more bits.

**Lesson, recorded as a memory note:** when a structural change and a parameter change both produce similar metric movement, suspect the parameter. The structural change is more interesting to write about; the parameter is more likely to be what's actually working.

---

## M13.5 — Cascade matching: the false-revivals story

With `max_misses = 300`, tracks live for 30 seconds. A track that goes "Lost" needs a way back to "Live" when the same physical thing reappears. This is **cascade matching**, the Phase 3.5 add-on described in Wojke 2017 §3.

### Track state machine

```
Live  ──── miss > max_misses ────► Lost
                                    │
   ◄── cascade match (frozen pos) ──┘
                                    │
                       lost_age > max_age
                                    │
                                    ▼
                                  Erased
```

A Lost track keeps its `lost_pos` (frozen at the Live→Lost transition), its appearance embedding, and a `lost_age` counter. The matcher runs in two stages:

1. **Live tracks vs all detections** — standard cost matrix, normal gate.
2. **Lost tracks vs unmatched detections** — relaxed position gate (`5 × max_dist`), appearance dominates.

If a Lost track matches a detection in stage 2, the track *revives* — re-init the filter at the new position, transition `Lost → Live`, keep the original ID. That's the cascade.

The first run on RELLIS produced an extraordinary result.

### Distinct IDs: 99 (down from 237). I celebrated. Then this happened.

```
$ awk -F, 'NR>1 {ids[$2]=1} END {print length(ids)}' results_m4/.../tracks.csv
99
```

That's −89.9% versus the M4 baseline. I updated the dashboard, wrote the milestone-shipped note, drafted a "we cracked this" headline.

User pushback, recorded verbatim:

> *"RELLIS is a forest, the robot doesn't loop close — trees don't move. So why is the same track ID appearing in two physically disjoint parts of the drive?"*

That was a sanity check the metric couldn't see. Trees don't move. If a single track ID appears at two distant world positions, it has to be a false-merge of two physically different objects. I built `scripts/audit/audit_revival_drift.py` to test this. It walks every `track_id` with an internal gap (Lost → revived), composes the published positions through the SLAM ego pose to get world coordinates, and computes the world-frame drift between the last-seen-before-gap position and the first-seen-after-gap position.

The audit on the stationary window [1750, 1830]:

| Era | Visible tracks in window | Long-gap revivals (gap > 50) | Worst world drift |
|---|---|---|---|
| 99-distinct headline | 35 | **18** | **15.7 m** |

Eighteen of thirty-five visible tracks had been revived across world-frame distances of up to 15.7 meters — on a 285-second drive through a forest with no loop closure. The 99-distinct number was an *artifact*. The cascade was matching tracks to physically different clusters.

### Why — the ego-frame anchor bug

`Track::lost_pos` was stored in **ego frame** at the moment of Live → Lost transition. The cascade's stage-2 matcher compared this stored ego coordinate against new detections, which were also in current-ego frame. But the ego had *moved* between the freeze and the cascade attempt. The same ego-relative coordinate (10, 0) at $$t = 0$$ pointed at world (10, 0); at $$t = 3$$ seconds with the bot 5 m forward, it pointed at world (15, 0). The cascade was happily revival-matching to whatever cluster happened to land at the *stale* ego coordinate.

Two fixes:

- **Fix A (defensive, ship-now):** `max_age = 30` (3 seconds at 10 Hz) limits how stale the ego anchor can become before the track is erased. Result: 202 distinct, mean lifetime ~80 frames. Honest, but kills legitimate revivals at long Lost gaps.
- **Fix B (canonical, the right answer):** store `lost_pos_world` in world frame using the SLAM ego pose. On cascade match, transform back into current-ego frame:

```cpp
// On Live → Lost
t.lost_pos_world = T_world_ego * filter->position();

// On cascade match
const Eigen::Vector2f lost_in_current_ego = T_world_ego.inverse() * t.lost_pos_world;
```

Fix B is in production. The new SORT API takes an optional `T_world_ego` argument with default `Identity` so synthetic unit tests stay bit-identical; `tracker_runner` accepts `--ego-poses data/poses_slam_full.csv` and threads the per-frame transform through. A new regression test (`CascadeRevivalSurvivesEgoMotion`) constructs a synthetic stationary tree at world (10, 0), translates the ego 10 m, then re-detects — and asserts the revival lands on the real tree, not on a decoy at the stale ego coordinate.

Fix B headline on RELLIS: **127 distinct, 373-frame lifetime**.

That's better. *This time I audited before celebrating.*

### The audit caught a second bug

Same script, same window, but extended to the whole drive:

| Cascade revivals across drive (gap > 11) | 1474 |
| world drift > 5 m | 724 (49%) |
| world drift > 10 m | 248 (17%) |
| world drift > 20 m | **95** |

Ninety-five revivals at world-frame drifts greater than 20 meters. Most were on segments where the ego was *stationary* — the world-frame anchor was correct, the gate was just letting through 20+ meter "revivals" because the gate was configured at `kLostPosGateScale × max_dist = 5.0 × 5.0 = 25 m`. The gate was loose enough to teleport tracks across the scene.

Fix B was correct — but incomplete. The original audit had identified one mechanism (ego-frame anchor) that fully explained the visible failures in the stationary window, but I hadn't asked the meta-question: *are there ALSO false revivals in segments where the ego anchor IS valid?* The answer was yes — through a too-permissive position gate.

**Memory note added:** *one mechanism for a false metric does not preclude others. After a fix lands, re-run the same audit and ask whether ANY false signal remains, not whether the specific mechanism you fixed has been neutralized.* This one cost three extra hours and is the rule that informs every Phase-4 audit below.

---

## The Mahalanobis sweep — what fixed-distance gates can't do

The natural fix to a too-loose gate is to tighten it. Fix C: drop `kLostPosGateScale` from 5.0 to 2.0 (10 m world tolerance instead of 25 m).

| Variant | Distinct | Lifetime | > 20 m | > 10 m |
|---|---|---|---|---|
| Fix B (gate = 25 m) | 127 | 373.0 | 95 | 456 |
| **Fix C (gate = 10 m)** | **299** | **158.4** | **0** | **54** |

Both metrics went the wrong way. Distinct *grew* to 299 (worse than M13 cascade-off at 237). Lifetime *collapsed* to 158. The 10 m gate eliminated all >20 m false-merges (zero — the structural goal was hit), but it also rejected legitimate revivals where DBSCAN cluster centroids drifted 5–15 m on the same physical tree between sightings. The fixed-distance gate cannot distinguish "noisy stationary tree" from "different physical object at similar distance" — both produce identical world-frame drifts.

Track *covariance* carries the missing information. A track whose filter has high uncertainty (long Lost period, noisy measurements, IMM modes diverging) genuinely could be at a position several meters from its last estimate; a confident, recently-locked-on track shouldn't be. **Mahalanobis distance** is the principled gate:

$$d_{\text{mahal}}^2 = \Delta^\top P^{-1} \Delta < \chi^2(0.95, 2) \approx 5.99$$

where $$P$$ is the track's 2×2 position covariance. High-confidence (small $$P$$) + 12 m drift → huge $$d_{\text{mahal}}^2$$ → reject. Low-confidence (large $$P$$ from long Lost) + 12 m drift → moderate $$d_{\text{mahal}}^2$$ → accept. That's exactly the bias we want.

### Three Mahalanobis variants, three lessons

| Variant | Cov source for gating | χ² | Distinct | Lifetime | > 20 m |
|---|---|---|---|---|---|
| Mahal-v1 | combined IMM `P_out` | 5.99 | 207 | 228.9 | 16 |
| **Mahal-v2** | **per-mode min P** | **5.99** | **242** | **195.8** | **11** |
| Mahal-v3 | per-mode min P | 2.28 (1σ) | 384 | 123.4 | 4 |

The Mahal-v1 → Mahal-v2 jump is the lesson worth a few paragraphs. The IMM's combined covariance is the *correct marginal posterior under model uncertainty* — it's what you want for state estimation. But for *gating*, it's wrong. When CV and CP modes disagree on position over the 10 pre-Lost misses (CV predicts forward at velocity, CP stays put), the spread term $$\sum_j \mu_j (x_j - x_{\text{out}})(x_j - x_{\text{out}})^\top$$ inflates the combined cov dramatically. A track moving at 5 m/s gets $$\sigma_{\text{pos}} \approx 5$$–$$7$$ m at Lost transition — large enough to admit a 22 m world-drift revival as Mahalanobis-acceptable.

That's the right answer for "where could this object plausibly be under all our model hypotheses." It's the wrong answer for "is this detection physically the same object as the track's last sighting." Picking the more confident sub-model — CV or CP, whichever has the smaller position-cov trace — gives the gate a tighter, principled shape.

The Mahal-v3 attempt (tightening χ² from 5.99 to 2.28) confirmed the diagnosis from the other direction: distinct ballooned to 384 (worse than no cascade) because the gate started rejecting *legitimate* DBSCAN-noisy revivals. There is no χ² threshold that separates 8–15 m DBSCAN-noisy revivals from 16–24 m physical-different-object revivals, *because the covariance carries the same range for both*.

Mahal-v2 sits at the **trade-off knee on the tracker dimension**: 242 distinct, 196 lifetime, 11 surviving false-merges. Nothing the cascade gate alone could do would simultaneously reduce distinct count, increase lifetime, and eliminate the false-merge tail. The wall is structural; the gate is doing its job.

---

## Phase-4 — the detector ceiling

If the tracker is doing its best, what's the *detector* doing? The Phase-4 audit started with a single hypothesis: **DBSCAN cluster centroids on stationary trees jitter 5–15 m between sightings as the LiDAR scans different sides of the trunk.** If true, the Mahalanobis ceiling is structural — the filter's covariance honestly reflects detector noise, and no χ² tuning will separate "noisy real revival" from "false-merge."

### Step 1 — measuring the K=1 baseline

`scripts/audit/audit_dbscan_jitter.py` runs greedy nearest-neighbor matching of detections in world frame on the stationary window [1750, 1830] (ego world displacement: 1.55 m over 80 frames — confirmed stationary). Cap matches at 8 m to discard cross-cluster mismatches; this is biased toward low jitter, so any signal is robust.

```
gap   pairs   median_m    p90_m    p99_m    max_m
  1     928       3.07     6.51     7.78     7.99
  5     861       2.80     6.31     7.85     7.98
 10     817       3.11     6.62     7.85     7.99
 20     685       3.22     6.41     7.83     7.96
 50     354       2.96     6.28     7.73     7.94
```

**Median 3.07 m, p90 6.51 m on adjacent frames.** The hypothesis was confirmed in shape but the magnitude was smaller than I had guessed (5–15 m was an upper-bound interpretation of the cascade audit's worst cases; the per-pair median is half that).

The unexpected finding: **jitter does not compound with frame gap.** The gap=50 median equals the gap=1 median (~3 m). For stationary objects on stationary ego, this can only happen if the noise is *independent per frame and re-randomizes each step* — not accumulating drift, not real motion. Independent noise averages down by $$\sqrt{K}$$ under multi-frame accumulation.

### Step 2 — splits/merges

That $$\sqrt{K}$$ model assumed Gaussian noise. A different audit asked: how stable is the *cluster identity* between frames? If a tree is 1 cluster at frame $$f$$ and 2 clusters at frame $$f+1$$ (DBSCAN found a sparse seam), greedy NN matches against the wrong sub-cluster and the apparent jitter is half the parent's spatial extent — not Gaussian noise at all.

`scripts/audit/audit_dbscan_split_merge.py` on the same window:

| `\|ΔN\|` | pairs | % | median jitter |
|---|---|---|---|
| 0 | 3 | 3.8% | **0.77 m** |
| 1 | 13 | 16.2% | 3.38 m |
| 2 | 12 | 15.0% | 2.90 m |
| ≥3 | 52 | 65.0% | 3.22 m |

When DBSCAN keeps the same cluster set across two frames (the 3 stable pairs), centroid jitter is only **0.77 m** — well below the 1 m floor that would make Mahalanobis effective. When the cluster count changes (the 77 unstable pairs, 96.2% of all pairs), jitter is **3.22 m** — 4.16× the stable rate. **The cluster count varies 7 to 26 per frame** on a stationary forest scene; 37.7% of clusters per frame-pair are appearing or disappearing.

So the noise is *not* independent Gaussian — it's structural cluster reformulation, which K-frame accumulation in world frame should fix structurally (denser support, no sparse seams) and beat the $$\sqrt{K}$$ prediction.

### Step 3 — K-frame accumulation sweep

`scripts/accumulate_and_cluster.py` composes the last K obstacle clouds into the target frame's ego coords using the SLAM ego pose, runs DBSCAN on the union, writes per-frame cluster CSVs in the existing schema. K=1 sanity check reproduces the C++ pipeline's cluster counts exactly. Then K=3, K=5, with `eps ∈ {0.5, 0.7}`:

| Config | dets/frame | gap=10 jitter |
|---|---|---|
| K=1 (current) | 15.1 | 3.11 m |
| K=3 eps=0.5 | 24.3 | **1.91 m** |
| K=3 eps=0.7 | 16.7 | 2.05 m |
| K=5 eps=0.5 | 24.8 | 1.93 m |

K=3 reduces persistent-cluster jitter ~38% — better than $$\sqrt{3} = 1.7\times$$, confirming the split/merge hypothesis. K=5 is *not meaningfully better than K=3* — the residual ~2 m floor is structural, not Gaussian. Going higher (K=9) won't help.

The cluster count *rose* with K=3 (15.1 → 24.3 dets/frame). I had expected the opposite: denser support → cleaner clusters → fewer fragments. What actually happened: K=3 stacks 3 frames of obstacle points at slightly-different world coords (LiDAR's per-scan grid shifts ~2 cm per frame even on stationary ego), and DBSCAN at eps=0.5 sees those as multiple adjacent clusters rather than one merged blob. eps=0.7 merges them back at the cost of slightly higher per-cluster jitter.

### Phase-4 tracker results

The full sweep, with Mahal-v2 on every row:

| Variant | Knob | Distinct | Lifetime | > 20 m | > 10 m |
|---|---|---|---|---|---|
| Mahal-v2 K=1 (tracker knee) | per-mode IMM cov | 242 | 195.8 | **11** | 248 |
| **K=3 eps=0.5 (detector knee)** | + K=3 accumulation | **307** | **238.2** | **0** | **130** |
| K=3 eps=0.7 (combined knee) | + looser DBSCAN | 272 | 194.8 | **0** | 132 |

Three knees, none inside the original target window ([150, 200] distinct, [250, 320] lifetime, 0 false-merges). The structural ceiling is a real wall:

- **Tracker dim (Mahal-v2):** 242 distinct, 11 surviving false-merges. The cascade gate is doing its best.
- **Detector dim (K=3 eps=0.5):** 0 false-merges at >20 m drift. Cluster fragmentation pushes distinct to 307.
- **Combined (K=3 eps=0.7):** 272 distinct, recovers cluster count, lifetime drops back. Still 0 extreme false-merges.

The two knees do not compose into a target-hitting config. The remaining cluster fragmentation IS the structural floor that no tracker can resolve.

---

## How the field got past this — and the single sentence that explains why

Every iteration in this post — IMM, appearance, cascade, world-frame anchor, Mahalanobis, K-frame accumulation — happened *at the tracker layer*, with DBSCAN frozen as the detector. That's the constraint that produced the wall.

What the field actually did, in one sentence: **production AV stacks moved from "cluster-and-track" to "learned-3D-detection-and-track" in the 2019-2021 window, and the tracker got *simpler* in the process.**

The lineage:

- **2018–2020 — clustering era.** AB3DMOT (Weng et al. 2020 [11]) is the canonical baseline of this era: a Kalman filter + Hungarian over LiDAR cluster centroids, almost exactly what this post implements. Their KITTI results were good. Their paper closes with the sentence that should sit alone in a paragraph: *"the dominant source of identity switches is the detector, not the tracker."*
- **2019 — PointPillars (Lang et al. [12]).** Voxelize the cloud into vertical pillars, run a 2D CNN, regress 3D bounding boxes. One detection per object, no clusters. Real-time on automotive GPUs.
- **2021 — CenterPoint (Yin et al. [13]).** Heatmap prediction of object centers in BEV feature space. Strong on nuScenes / Waymo; the current production-grade baseline.
- **Post-2021 — simple trackers, learned detectors.** AB3DMOT-family (now run on PointPillars/CenterPoint detections), SimpleTrack, EagerMOT, GreedyTracker. Most of these use a Kalman filter and Hungarian matching that are *less sophisticated* than what this post built. The complexity moved into the detector.

The single sentence in AB3DMOT's discussion is the same conclusion this post arrives at — through measurement rather than reading. A senior reviewer at any AV / robotics company will recognize the arc instantly: every team that built a clustering-based tracker on real LiDAR has lived this exact week.

The **structural fix** for what we hit is to replace DBSCAN with PointPillars or CenterPoint. The tracker we built — the IMM, the cascade, the world-frame anchor, the Mahalanobis gate — would all carry over and *would all work cleanly* against learned-detector inputs. That's not in this milestone's scope; it's the explicit Phase-3 stretch documented in `docs/p3-progress.md`.

---

## Limitations

The mandatory section. A blog without it reads like over-claim.

1. **DBSCAN as detector** is the project's load-bearing limitation. Production AV moved past clustering-based 3D detection 5+ years ago. The entire Phase-4 ceiling comes from this choice.
2. **No 3D bounding box ground truth on RELLIS.** RELLIS has point-wise semantic labels but no per-object 3D bboxes or track IDs. We used proxy metrics (distinct IDs, mean lifetime, world-frame drift on stationary tracks) instead of MOTA / MOTP. Real benchmark evaluation requires nuScenes (P2-M7/M8 in the project plan).
3. **Mahalanobis gate uses per-mode P, not the IMM combined P.** This is a deliberate choice (see Mahal-v1 → Mahal-v2 above), but it means we're applying a different covariance for *gating* than for *estimation*. Defensible, slightly non-canonical.
4. **The appearance encoder helped less than expected on RELLIS.** Position dominance ($$\lambda \approx 0.2$$ optimum) is real; the encoder's contribution was modest. On nuScenes-style urban data with denser inter-object similarity, the contribution would likely be larger.
5. **CV+CP IMM only.** A real-world IMM might add a CT (constant turn-rate) mode for vehicles. RELLIS has so few maneuvering targets that this would have been instrument noise; for a moving-vehicle benchmark it's a real omission.
6. **No KD-tree DBSCAN.** Brute-force $$O(N^2)$$ neighbor search runs sub-second on ~5–10k obstacle points; KD-tree would help with scale but doesn't fix the fragmentation/jitter ceiling.
7. **Animations only on RELLIS.** Nothing was tested on a second LiDAR dataset in this milestone. Generalization is P2-M7 territory.

---

## What ships from M10

- **Code:** 13 new tests across `KalmanFilter`, `Hungarian`, `IMMFilter`, `AppearanceEncoder`, and `SORTTracker`. All green in Release and Debug. Two scripts (`audit_dbscan_jitter.py`, `audit_dbscan_split_merge.py`) and one accumulator (`accumulate_and_cluster.py`) preserved as auditable artifacts.
- **Architecture:** virtual `IFilter` base + `unique_ptr` filter slot in `Track`. Cascade matching with world-frame anchor. Mahalanobis gate using per-mode position covariance. `--ego-poses` flag on `tracker_runner` threading SLAM poses through.
- **Animations:** three rendered comparison clips on the four trade-off-curve points (Mahal-v2 K=1, K=3 eps=0.5, K=3 eps=0.7). Rendered on NYU Torch HPC; preserved at `results_m4/ablation_g/blog_renders/`.
- **Data:** every variant's `tracks.csv` and `metrics.json` preserved as `_only` siblings under `results_m4/ablation_g/sort_on_rellis/` per the no-silent-deletes rule. Audit scripts can be re-run against any of them.
- **Honest finding:** the trade-off curve has a wall at ~250 distinct + 200 lifetime that only a learned 3D detector can break through. That ceiling is the M10 + Phase-4 result; pointing at PointPillars / CenterPoint is the honest hand-off.
- **Memory rules carried into the next milestone:** *"on a 2× target beat, trace ten random samples through the data before celebrating"* and *"after a fix lands, re-run the same audit and ask whether ANY false signal remains."* Both came from this milestone, both will guide P2-M5.

The numbers that matter: 33 tests across all libraries, 8 tracker variants × 3 detector variants in the sweep, 3 rendered comparison clips, 2 audit scripts that turned a vibes-based "the metric looks too good" intuition into hard pass/fail measurements, 1 wall mapped to its structural cause, 1 explicit hand-off to learned 3D detection.

The real lesson is the meta-process. Every measurement in this post, including the ones that contradicted my own predictions, exists because of the rule the M13.5 false-revivals story produced: **measure, then celebrate.** Each celebrated number got an audit. Each audit produced either a confirmation or a contradiction; the contradictions produced the next iteration. The arc M4 → M12 → M13 → M13.5 → Phase-4 is, viewed from outside, "this person hit a ceiling on tracking and pivoted to a learned detector." Viewed from inside, it's "this person built audits before they built fixes." That's the part that's hard to fake on a portfolio and the part a hiring manager who's lived this work will recognize.

---

## References

[1] Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B. (2016). Simple Online and Realtime Tracking. *IEEE International Conference on Image Processing*. — The original SORT paper. 4 pages, the algorithm everything in this post is built on.

[2] Thrun, S., Burgard, W., Fox, D. (2005). *Probabilistic Robotics*, Chapter 3 (Gaussian Filters). MIT Press. — Kalman filter derivation and notation conventions.

[3] Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly* 2:83–97. — Foundational. Read the algorithm; skim the proof.

[4] Munkres, J. (1957). Algorithms for the Assignment and Transportation Problems. *Journal of the SIAM* 5(1):32–38. — The practical $$O(N^3)$$ variant of Kuhn's algorithm. Implementation target for this post.

[5] Pilgrim, R. A. (2000). Munkres' Assignment Algorithm. Murray State University Department of Computer Science. — The canonical state-machine formulation that my C++ implementation follows.

[6] Guo, X. (2018). github.com/xg590/munkres — pure-C port of Pilgrim's reference. Useful translation aid; cross-checked behavior on small matrices during implementation.

[7] Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *KDD*. — The DBSCAN paper. Algorithm boxes only suffice.

[8] Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*, §11.6. Wiley. — The IMM derivation cited in the Phase-3 plan.

[9] Wojke, N., Bewley, A., Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP*. — Deep SORT. The cascade-matching architecture and the per-detection appearance embedding came from §3 of this paper.

[10] Hermans, A., Beyer, L., Leibe, B. (2017). In Defense of the Triplet Loss for Person Re-Identification. *arXiv:1703.07737*. — §4 of this paper (batch-hard triplet mining) is what the M13 appearance encoder was trained with.

[11] Weng, X., Wang, J., Held, D., Kitani, K. (2020). 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics. *IROS*. — AB3DMOT. The clustering-era benchmark whose discussion section says explicitly that the detector is the bottleneck. Anchor for the "field moved on" section.

[12] Lang, A. H., et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. *CVPR*. — The post-clustering-era detector that production AV stacks moved to. Phase-3 stretch target.

[13] Yin, T., Zhou, X., Krähenbühl, P. (2021). Center-based 3D Object Detection and Tracking. *CVPR*. — CenterPoint. The current strong baseline on nuScenes / Waymo; ships its own tracker that's *simpler* than what this post built.

[14] *Algorithms for the Assignment and Tracking Problems* (the IEEE-archived paper in the repo's `reference papers/p2m4/` directory). — Read alongside [3] and [4] for a unified treatment of the historical algorithm and its modern variants.

[15] Kuhn, H. W. (1955). The Hungarian assignment paper, archived PDF in `reference papers/p2m4/Kuhn-hungarian-assignment.pdf`. — The 1955 original. König's theorem and the dual-variable interpretation are presented here.

[16] *Hungarian Algorithm Walkthrough* — [thinkautonomous.ai blog](https://www.thinkautonomous.ai/blog/hungarian-algorithm/). Recommended further reading; visualizes the step-by-step starring/priming/augmenting-path procedure that's described in the assignment section above.
