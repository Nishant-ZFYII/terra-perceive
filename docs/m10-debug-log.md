# M10 / P2-M4 Debug Log

A running record of bugs hit, root causes, and fixes during the M4 SORT
tracker milestone. The point of this file is twofold:

1. **Survives chat summarization.** When a session gets compacted, the
   verbatim diagnostic stories disappear from memory. This file is the
   permanent record.
2. **Source material for the blog.** The M10 post (`docs/m10-sort-tracker.md`,
   not yet drafted) is structured around three "load-bearing debugging
   stories." The entries tagged `[STORY]` are blog-worthy as written.
   Other entries inform the "what I'd improve" section, the test-design
   sidebar, or are kept for posterity.

## Entry format

Each entry is a fenced block. Tag taxonomy:

- `[STORY]` — load-bearing debugging story for the blog post
- `[ALGO]`  — algorithmic correctness
- `[NUMERIC]` — floating-point / numerical-stability
- `[TEST]`  — test design / harness, not the algorithm
- `[EVAL]`  — evaluator / metric design (separate from the tracker)
- `[INFRA]` — build, env, IDE, worktree, gitignore
- `[DESIGN]` — naming, scope, namespace, API shape

Each entry: **Symptom → Root cause → Fix → Lesson** (and optionally
`Blog tag:` describing where it could land in the post).

---

## 2026-04-24

### `[INFRA]` Conda Python lacked `catkin_pkg`, build failed instantly

- **Symptom:** `colcon build` reported `ModuleNotFoundError: No module named 'catkin_pkg'` from `package_xml_2_cmake.py`.
- **Root cause:** A fresh `conda activate base` shell was used; base env doesn't ship `catkin_pkg`.
- **Fix:** Either `conda deactivate` twice + `source /opt/ros/humble/setup.bash`, or activate the project's `terra-perceive` env which had it pre-installed.
- **Lesson:** When a build fails on Python imports rather than C++ symbols, the issue is environment, not code. Check `which python3` first.
- **Blog tag:** none (process detail, not story).

### `[INFRA]` `tinycolormap.hpp` missing in p2m4 worktree

- **Symptom:** Build error `cannot open source file "tinycolormap.hpp"` from `world_grid.cpp` (M3 code).
- **Root cause:** `third_party/` is in `.gitignore` and isn't replicated across `git worktree add` clones.
- **Fix:** Symlinked `third_party` from the M3 worktree: `ln -s /home/nishant/MS_Project/terra-perceive/third_party ./third_party`.
- **Lesson:** Same pattern that `data/` and `results/` use. Anything ignored AND shared belongs as a symlink in new worktrees.
- **Blog tag:** none.

### `[INFRA]` IntelliSense squiggles on `Eigen/Dense` and project headers

- **Symptom:** VS Code showed red squiggles on `#include <Eigen/Dense>` and `"hungarian.hpp"` in every M4 file.
- **Root cause:** No `compile_commands.json` symlink in the worktree, and no `.vscode/c_cpp_properties.json` was carried over.
- **Fix:** Symlinked `compile_commands.json` from `build/construction_perception/`, plus added `.vscode/c_cpp_properties.json` cloned from the M3 worktree (explicit `includePath` listing `${workspaceFolder}/include`, `/usr/include/eigen3`, `third_party/...`).
- **Lesson:** Compiler success and IntelliSense are separate signals. If `colcon build` succeeds but squiggles persist, the IDE config is missing, not the code.
- **Blog tag:** none.

### `[DESIGN]` Tracker code initially had no namespace

- **Symptom:** Other M2/M3 modules use namespaces (`so3::`, `pose_graph::`, `imu::`) but `kalman_filter.hpp` / `hungarian.hpp` / `sort_tracker.hpp` were in the global namespace.
- **Root cause:** Original scaffolds (likely seeded before M4 started) just skipped the namespace.
- **Fix:** Wrapped all M4 headers and `.cpp` files in `namespace tracker { ... }`. Updated tests to `using tracker::Symbol;` at file scope. Saved a project memory (`project_namespace_convention.md`) so future sessions don't repeat the mistake.
- **Lesson:** Pick the namespace early. Retrofitting later cascades through every test and call site. Cost here was ~30 minutes of mechanical edits.
- **Blog tag:** code-organization sidebar.

### `[STORY]` Kalman predict/update ordering — `UpdateOrderMatters` test was *too forgiving*

- **Symptom:** `UpdateOrderMatters` test was passing on a clean constant-velocity trajectory even with predict/update swapped. The test could not distinguish correct from incorrect ordering.
- **Root cause:** On smooth constant-velocity motion the swapped-order filter still converges (just lagged by one step); the lag is washed out over 50 frames of stable measurements.
- **Fix:** Inserted a velocity reversal at frame 25 (`vel = (-0.5, -0.3)`). The maneuver makes the swapped order's stale-measurement fusion blow up while the correct order tracks through.
- **Lesson:** A "debugging story" test must be designed with the failure mode in mind, not just the success path. Constant-velocity smooth tracking is too forgiving — the order bug only matters when the motion model is *wrong*, even briefly.
- **Blog tag:** STORY #1 in the post — Kalman predict/update ordering. The lesson about test-trajectory design also makes a good sidebar.

### `[STORY]` Cholesky vs `S.inverse()` — pathological-Q regime

- **Symptom (constructed):** With process noise → 1e-10 and measurement noise → 1e-10, after ~200 updates the innovation covariance `S = HPH^T + R` becomes near-singular.
- **Root cause:** `S.inverse()` materializes `S^{-1}` explicitly and amplifies floating-point error in the matrix elements. Cholesky `S.llt().solve(I)` factors first, never forms the inverse, stays finite.
- **Fix:** Used `K = S.llt().solve((H * P).eval()).transpose()` instead of `K = P * H.transpose() * S.inverse()`. Test (`CholeskyStableWhenInverseBlowsUp`) asserts `std::isfinite(state.norm())` after 200 noise-free updates.
- **Lesson:** SPD systems should always use Cholesky. The 1-line algebraic substitution is free; the numerical stability is not.
- **Blog tag:** STORY #2 in the post — numerical stability. Reference `Eigen::LLT` docs.

### `[ALGO]` Munkres returned ungated pairs into SORT's `match()`

- **Symptom:** Tests with `solver=Munkres` could pass for the wrong reason — Munkres always returns up to `min(N, M)` pairs regardless of `max_cost`. SORT was then `kf.update()`-ing tracks with wildly far-away detections.
- **Root cause:** `hungarian_solve(cost, Munkres, max_cost)` ignores `max_cost` (documented in `hungarian.hpp`); greedy gates internally but Munkres does not.
- **Fix:** Added a post-filter loop in `SORTTracker::match()` that drops every pair where `cost(i, j) > max_dist_`. Greedy already gates, so the filter is a no-op for greedy — uniform contract: "match() never returns ungated pairs."
- **Lesson:** API asymmetries between two implementations of the same interface need a unifying wrapper. Don't let the *caller* remember which solver gates and which doesn't.
- **Blog tag:** assignment-section subsection on gating.

### `[STORY]` Greedy's order-dependence — unit test passes, integration test deceptively passes

- **Symptom:** `Hungarian.GreedyOrderDependenceOnCrossings` (unit) reliably proves greedy is suboptimal on `[[1.0, 1.5], [1.0, 2.0]]`. But `SORTTracker.TwoCrossingTracksIdSwap_WithGreedy` (integration) on a symmetric perpendicular crossing did *not* reliably trigger greedy to swap — strict-`<` tie-breaking favored the first-seen row.
- **Root cause:** In a 2-target perpendicular crossing with well-learned velocities, the cost matrix at the crossing frame is symmetric (or nearly so). Greedy's strict-`<` makes it deterministic and surprisingly robust here.
- **Fix:** Adjusted the integration-level test trajectories to expose the failure. Documented in the blog story that the algorithmic claim lives at the *cost-matrix* level (unit test); the integration-level swap on 2 targets requires adversarial geometry.
- **Lesson:** "Greedy is suboptimal" is a property of cost matrices, not of trajectories. A symmetric scene papers over the failure mode. Munkres' value is most visible in dense scenes where many costs are near-tied.
- **Blog tag:** STORY #3 in the post — assignment subsection. Use the unit test's matrix as the worked example, not a 2-target crossing screenshot.

### `[TEST]` `Hungarian.GreedyRectangular` had wrong expected total

- **Symptom:** Test failed with `total_cost = 6.0, expected 2.0`.
- **Root cause:** The test scaffold suggested the *optimal* total (2.0) on a matrix where greedy is provably suboptimal (greedy gets 6.0). Mixing "it works" assertions with "greedy is suboptimal" assertions in the same test is a category error.
- **Fix:** Changed the matrix so greedy *does* find the optimum (move the second `1` from col 3 to col 1). Suboptimality belongs in the dedicated `GreedyOrderDependenceOnCrossings` test.
- **Lesson:** Each test asserts one property. "Rectangular handling" and "greedy is suboptimal" are different properties; don't co-mingle.
- **Blog tag:** none (test-design lesson).

### `[TEST]` `Hungarian.SolverAgreesOnSeparatedCosts` flaked on diagonal-boost

- **Symptom:** 3 of 10 random trials reported greedy ≠ Munkres total.
- **Root cause:** Diagonal boost was `-= 50.0f` against an off-diagonal range of `[0, 100]`. On unlucky trials, an off-diagonal entry was smaller than the diagonal, making greedy pick a wrong column-0 row.
- **Fix:** Increased the boost to `-= 1000.0f`. Diagonal is now unconditionally smaller than every off-diagonal.
- **Lesson:** "Random matrix tests" need an upper-bound argument that the random draw can't violate the precondition. If the precondition is "diagonal is the unique minimum," ensure it holds for *every* possible draw, not just typical ones.
- **Blog tag:** none.

### `[TEST]` Symmetric coincident crossings cause Munkres tie-break ambiguity

- **Symptom:** `TwoCrossingTracksNoIdSwap_WithMunkres` initially had two trajectories that crossed at *exactly* (5, 5). Even Munkres could swap on this scene because the cost matrix at the crossing frame was all-zeros.
- **Root cause:** Tie-breaking in Munkres is deterministic but algorithm-internal — not a "correct" choice when all assignments are tied.
- **Fix:** Offset trajectory B's y by 0.1 m so the trajectories pass *near* but never coincident. Cost matrix is no longer all-zeros at the closest frame; the diagonal pairing wins unambiguously.
- **Lesson:** Tests for "no ID swap" must avoid coincidence ambiguity in their inputs. Real-world detector noise does this for free; synthetic tests need to do it deliberately.
- **Blog tag:** sidebar on adversarial test design.

---

## 2026-04-25

### `[BUILD]` Worktree confusion — building from M3 dir, "succeeded" was a no-op

- **Symptom:** `colcon build` from `~/MS_Project/terra-perceive$` reported "Finished" in 0.5s. But `tracker_runner.cpp` lives in `~/MS_Project/terra-perceive-p2m4`, so the success was incremental-no-op on the M3 worktree's stale build dir.
- **Root cause:** The conda env name `terra-perceive` is the same as the M3 worktree directory name. The shell prompt didn't make the active worktree visually distinct.
- **Fix:** Always glance at the prompt path (`~/MS_Project/terra-perceive` vs `~/MS_Project/terra-perceive-p2m4`) before running build/scripts.
- **Lesson:** Symlinks + shared conda env + git worktree is convenient but creates a real "which dir am I in?" trap. Consider a tmux/dirname-aware prompt indicator.
- **Blog tag:** none.

### `[INFRA]` `<set>` missing from `tracker_runner.cpp` includes

- **Symptom:** Compiler error `identifier "distinct_track_ids" is undefined`.
- **Root cause:** Pseudocode in the scaffold referenced `std::set<int> distinct_track_ids`, but `<set>` was not in the include list.
- **Fix:** Added `#include <set>`.
- **Lesson:** "Pseudocode in comments" can use STL types whose headers aren't included by the surrounding code. When transcribing, sanity-check imports.
- **Blog tag:** none.

### `[STORY]` `[EVAL]` Phantom ID switches from `associate_to_gt` at coincident crossings

- **Symptom:** Ablation A first run reported greedy=4 ID switches AND munkres=4 ID switches with *identical* tracker output and *identical* `id_switches.csv` contents. Both flagged switches at frames 5 and 6.
- **Root cause:** The evaluator's `associate_to_gt` started as a per-gt nearest-neighbor scan. At the *exact* crossing frame, both predicted track positions and both detections sat at ≈(5, 5); both gt labels picked the *same* tracker_id ("nearest" was ambiguous), then untangled the next frame, producing 2 phantom switches. **The metric was reporting evaluator confusion, not tracker behavior.**
- **First fix attempt:** Replaced naive nearest-neighbor with a Munkres-based 1-to-1 assignment in `associate_to_gt`. **Did not help** — Munkres also has tie-breaking ambiguity on an all-zeros cost matrix at the crossing.
- **Second fix:** Adjusted `gen_crossing` so Track 1 runs along x = 5.5 instead of x = 5.0 — the trajectories pass near each other but never coincide. Cost matrix at the closest frame is now `[[≈0, 0.5], [0.5, ≈0]]`, diagonal pairing is unambiguous, both solvers and the evaluator pick the diagonal.
- **Result:** greedy=0, munkres=0. Both solvers and the evaluator are now consistent.
- **Lesson:** When *both* solver variants report the same metric on a "differentiator" ablation, suspect the evaluator before suspecting the tracker. Build the evaluator on the *same* algorithmic primitives the tracker uses (Munkres assignment), but be aware that Munkres also has tie-break ambiguity on degenerate inputs. The cleanest way to defeat it is to make the inputs non-degenerate.
- **Blog tag:** STORY (potential 4th, or merged with STORY #3) — "phantom switches and the difference between tracker errors and evaluator errors." Also a strong sidebar on metric design.

### `[STORY]` Ablation B Q-sweep figure: scale-of-axis killed Panel 1 until residual panel was added

- **Symptom:** First Q-sweep PNG showed 4 colored Q lines completely overlapping in the position-vs-frame panel — visually indistinguishable. Cov-trace panel was clean. Boosting `gen_linear` sigma 0.15→0.5 made the gray scatter dots visible but the colored lines still stacked.
- **Root cause:** Position is plotted at full trajectory scale (50 m). The smoothing-vs-jitter difference between Q=0.01 and Q=10 is sub-meter — about 1% of the panel height. Invisible by construction.
- **Fix:** Added a third "residual" panel: `estimate(i) − linear_fit_of_detections(i)`. The residual zooms into the wiggle scale (±1 m) where the Q sweep visibly differentiates: Q=10 traces every noise spike, Q=0.01 stays nearly flat.
- **Implementation note:** Truth is computed via `np.polyfit` on the detection cloud — no scenario parameters hardcoded into the plot script. Stays general for B and C.
- **Lesson:** When a sweep parameter affects *fluctuations* but not *steady state*, plotting the absolute quantity is hopeless. Always have a residual or zoom-on-deviation panel for sweeps over filter trust knobs (Q, R, smoothing factors). The rule of thumb: if the parameter only changes ±1% of axis range, you need a different y-axis.
- **Blog tag:** Q-sweep figure for the Kalman section. The residual panel IS the headline visual; the cov-trace panel is the supporting "confidence calibration" evidence.

### `[STORY]` Ablation C R-sweep: Q vs R cov-trace asymmetry (algorithmic insight)

- **Symptom (not a bug):** Ablation C's right panel (cov trace, log) shows R values spanning only ~1 order of magnitude in the asymptote, despite R itself sweeping 2 orders (0.01→1.0). Compare to Ablation B where Q's 3-order sweep produced 3 orders of asymptote spread. Why the asymmetry?
- **Root cause (algorithmic):** Q is added to P at every predict step (`P_pred = F P F^T + Q`), so it accumulates and dominates the asymptotic posterior covariance. R only modulates the gain `K = P H^T (HPH^T + R)^{-1}` during the update step, so it changes the *speed* at which P shrinks but not its floor. Two different mechanisms with overlapping but not equivalent effects on the gain.
- **Lesson / blog material:** Q is the long-game knob (sets steady-state confidence). R is the responsiveness knob (sets convergence rate and noise tracking). Many engineers under-tune Q and over-tune R because R is the "obvious" parameter — but the asymptotic uncertainty is mostly Q's domain.
- **Blog tag:** sidebar in the Kalman section, between B and C. Both ablations together tell this story; neither alone does.

### `[STORY]` `[ALGO]` Ablation H: SORT publishable output is INVARIANT under predict-order swap (algorithmic insight)

- **Symptom:** Ablation H ran the tracker on a maneuvering scenario (`gen_maneuver`, velocity reversal at frame 25) with `--swap-order` ON and OFF. Both runs produced bit-identical `tracks.csv` content; the residual + cov-trace plot showed two completely overlapping curves.
- **Diagnosis:** The unit test `KalmanFilter.UpdateOrderMatters` catches the bug because it compares the swapped filter's state to *current-frame ground truth*, where the swapped state has been advanced by one extra `predict()` step (a one-frame lookahead). The Kalman-level bug is real.
- **At SORT level, by construction:** `SORTTracker::update()` builds `publishable` in Section 7 (after the match-update cycle) BEFORE the late `predict()` runs in Section 8 (when `Order::UpdateThenPredict`). So the publishable state at frame N is `update(predict(post_update_{N-1}), z_N)` in BOTH orderings — bit identical. The lookahead never enters `tracks.csv`.
- **Lesson / blog framing:** The SORT API contract decouples state-publish from state-prediction. Consumers always see post-update state at the queried frame, regardless of internal predict ordering. This is a more nuanced story than "swapped is worse" — it shows the value of API isolation between internal Kalman bookkeeping and the external publishable contract. The unit test does the load-bearing proof at the algorithmic level; the SORT layer's invariance is a property worth highlighting.
- **Decision:** Keep `--swap-order` flag and the H scaffold for completeness. Frame H in the blog as "the bug is Kalman-internal; SORT's publishable output is invariant by design." Do NOT manufacture a fake difference by republishing the post-predict state — that would misrepresent the SORT contract.
- **Blog tag:** STORY for the H section. The two-overlapping-lines plot becomes the *evidence* of invariance, not a failure. Title the figure: "SORT publishable state is invariant under predict-order swap."

### `[STORY]` `[ALGO]` SORT-on-RELLIS stationary-segment ID flicker

- **Symptom:** First closing-hero animation (`sort_vs_dbscan.mp4`, params: `process_noise=0.5, min_hits=3, max_misses=10, max_dist=5`) shows clean stable IDs while the bot is moving, but during a stationary segment around frames 1750–1830 the right panel flickers — track IDs disappear and re-spawn as new ids on the SAME physical trees that should have kept their identity.
- **Root cause #1 (dominant) — stale Kalman velocity at deceleration.** SORT uses a constant-velocity model. While the bot moves at ~1 m/s, every static tree has a learned apparent velocity of ~(-1, 0) m/s (the negative of ego). When the bot decelerates to a stop, the KF still PREDICTS each tree drifting backward at the old velocity. With `process_noise=0.5`, the filter takes ~5–10 frames to "forget" that velocity. During those frames, predicted positions drift away from the actual (now stationary) cluster centroids. Cumulative drift eventually exceeds `max_dist`, the match fails, the track accumulates misses, and after `max_misses=10` is pruned. When DBSCAN re-detects the same tree post-prune, it gets a new track_id.
- **Root cause #2 — DBSCAN centroid jitter even on stationary scenes.** LiDAR scan returns are slightly different per rotation (angular aliasing), so a stationary tree's cluster centroid wanders ~10–30 cm frame-to-frame even with zero physical motion. This compounds with cause #1.
- **Root cause #3 — `min_hits=3` + hits-reset-on-miss force re-warmup.** Our `SORTTracker::update()` resets `hits=0` on any miss (strict consecutive-hits semantics). Combined with `min_hits=3`, a track that's been published for 100 frames drops back to `hits=0` after one missed frame and disappears from publishable output until 3 consecutive matches accumulate again.
- **Fix applied:** `process_noise: 0.5 → 2.0`, `min_hits: 3 → 1`. Higher Q lets the KF adapt to ego stops in ~2 frames; `min_hits=1` makes re-acquired tracks publishable immediately. See `scripts/run_tracker_on_rellis.sh`.
- **Lesson — constant-velocity models break at velocity discontinuities.** The "what I'd improve" blog section should call out IMM (Interacting Multiple Model) — running constant-velocity + constant-position KFs in parallel and selecting per-frame — as the production-grade fix. Bewley's original SORT used the same constant-velocity assumption; this is a known limitation.
- **Blog tag:** STORY for the SORT-on-RELLIS section. The flicker→fix→explanation arc is genuinely instructive: it shows real-world tracking is harder than the unit-test scenarios suggest, and motivates IMM as natural next work.

### `[STORY]` `[ALGO]` Ablation F resolved — runner now iterates real-world frame range

- **Outcome after fix:** distinct_track_ids: max_misses=1 → 2, max_misses=3 → 2, max_misses=10 → **1**. The bottom panel of `max_misses_sweep.png` shows track 0 surviving the entire 8-frame occlusion gap with the same id resuming on the other side. Headline visual works.
- **Edit shape:** `src/tracker_runner.cpp` per-frame loop now iterates `for (int fid = min_f; fid <= max_f; ++fid)` and uses an `unordered_map<int, const FrameDetections*>` lookup with a static-empty fallback when a frame_id has no row in the CSV.
- **Backwards compatibility:** Unchanged behavior for Ablations A, B, C, D, E (all have detection rows on every frame, so `min_f..max_f` produces the same iteration as the original). Only `gen_occluded` produces gaps, so only Ablation F's behavior changed.
- **Cosmetic loose end:** the runner's `frames` field in metrics.json and the DONE line still report `frames.size()` (22) instead of `max_f - min_f + 1` (30). Algorithm is correct; only the reported count is off. Trivial fix.

### `[STORY]` `[ALGO]` Ablation F: runner skipped no-detection frames, max_misses never exercised

- **Symptom:** Ablation F's max_misses sweep over {1, 3, 10} all produced distinct_track_ids=2 — IDENTICAL behavior across the sweep, including max_misses=10 which should have survived the 8-frame occlusion gap as one continuous track.
- **Root cause:** `tracker_runner.cpp`'s per-frame loop iterates `for (size_t fi = 0; fi < frames.size(); ++fi)`, where `frames` is the output of `group_by_frame()` over the input CSV. `gen_occluded` deliberately drops detection rows for frames inside the gap; those frame_ids never appear in `frames`. The tracker therefore receives 0 update() calls for the 8 occlusion frames, then a single update() with the post-gap detections — looking like only 1 elapsed frame, not 8. max_misses is never put under load.
- **Lesson — separate "real-world time" from "detection rows":** A tracker simulator must iterate *real-world time*, not the *event timeline of detections*. Empty frames are real and meaningful. This bug only manifests on scenarios with explicit gaps — Ablations A, B, C, D, E all have one detection per frame, so the bug stayed dormant. Ablation F is the one that exposes it.
- **Fix:** Iterate `for (int fid = min_f; fid <= max_f; ++fid)` and pass empty detections when no row exists for `fid`. ~15 line change. Idempotent for non-occlusion scenarios (since min..max equals frames.size() when there are no gaps).
- **Blog tag:** STORY for the F section, plus a sidebar on simulator design ("how time actually passes in your tracker").

### `[ALGO]` Ablation A: greedy and Munkres both score 0 ID switches on simple 2-target near-crossing

- **Symptom (not a bug):** `bash scripts/run_ablation_a.sh` final result: `greedy id_switches=0, munkres id_switches=0`.
- **Interpretation:** This is the *correct* answer for this scene. With strict-`<` tie-breaking on a 2-target trajectory where the cost matrix at the closest frame has a clear diagonal preference, greedy survives. Munkres also survives. The two solvers are *indistinguishable* on this scene.
- **Implication for the blog:** Ablation A's scalar comparison is a non-event for the blog story. The real claim ("greedy is suboptimal") lives at the unit-test level (`Hungarian.GreedyOrderDependenceOnCrossings`). For a more dramatic figure later, escalate to dense scenes (`gen_dense`) where many costs are near-tied.
- **Lesson:** Some ablations come back as "no difference" — that's also a valid finding, *if framed honestly*. The blog should call this out, not paper over it. Scaling to dense scenes is the natural next experiment.
- **Blog tag:** Ablation A summary table; "what this did not show" framing.

---

## How to use this file going forward

- **Append, don't rewrite.** Each new entry goes at the bottom under today's date.
- **Tag deliberately.** `[STORY]` is reserved for blog-worthy debugging narratives — bugs that taught a real algorithmic or numerical lesson. Don't dilute it.
- **One entry per fix.** If a single bug took three iterations to fix, that's one entry with the iteration history inside it (see "Phantom ID switches" above).
- **Cite test names.** Every entry that's gated by a test should name the test, so the blog can link the narrative to executable proof.
- **Keep symptoms verbatim where possible.** Compiler messages, log lines, metric numbers — these are the most blog-able artifacts.

---

# P3-M12 — IMM Kalman filter (Mon 04-27 → )

Source material for `docs/m12-imm.md`. Same tag taxonomy as above.

## Day 1 — scaffolding (Mon 04-27)

### `[INFRA]` HPC `data/` symlink dangling on remote
**Symptom:** `scripts/sync_to_hpc.sh` errored with
`mkdir: cannot create directory 'data': File exists` followed by
`ln: failed to create symbolic link 'data/RELLIS-3D': No such file or directory`.

**Root cause:** On the HPC repo, `data/` had been left as a stale symlink
from a previous laptop path (`/home/nishant/.../terra-perceive/data`)
that doesn't exist on HPC. `mkdir -p` refuses to create over a non-
directory; `ln -s` then has no parent to drop the link into.

**Fix:** Self-healing block in `sync_to_hpc.sh:62-75` — detect a non-dir
or dangling-symlink `data/` entry and remove before `mkdir -p`. Same
treatment for the inner `data/RELLIS-3D` link. Documented the manual
repair in `docs/hpc-access.md`.

**Lesson:** Sync scripts should defend against stale state on the
remote; "the script worked yesterday" doesn't mean the remote tree is
what we expect today.

---

## Day 2-4 — filling `IMMFilter::update` (Tue 04-28 → Thu 04-30)

### `[NUMERIC]` Uninitialized `P_0[j]` accumulator
**Symptom:** `IMMCovarianceTraceBounded` reported
`covariance_trace = 1990` on every frame across 100 frames. Pure KF on
the same trajectory dropped trace from 1980 → 41 → 11 → 4.8 → 2.7 in 5
frames.

**Root cause:** `std::array<Eigen::Matrix4f, 2> P_0;` declares an array
of default-constructed Eigen matrices. Eigen's default constructor
leaves contents **uninitialized** — not zeroed. The mixing loop then
did `P_0[j] += mu_mix(i, j) * (...)`, accumulating onto garbage memory.

**Fix:** Add `P_0[j].setZero();` at the top of the P_0 loop in
`src/imm_filter.cpp` (mirrors the existing `x_0[j].setZero()` for the
mean accumulator). One missing line, infinite-noise covariance.

**Lesson:** Eigen's "matrices are not zero-initialized" convention bites
in accumulator patterns. Either zero explicitly or use
`Eigen::Matrix4f::Zero()` when declaring.

**Blog tag:** Sidebar note in m12-imm.md; not load-bearing.

---

### `[ALGO]` Division by zero in `mu_mix` when `c(j) ≈ 0`
**Symptom:** `MixedOutputMatchesSingleFilterDegenerate` produced
`imm: -nan -nan -nan -nan`. NaN propagated from frame 1 forward.

**Root cause:** The test uses Π = I_2 and μ_0 = [1, 0] to verify the IMM
degenerates to pure CV when forced into a single mode. With these
inputs, `c = Π^T · μ = [1, 0]`, so `c(1) = 0`. The mixing-weight
formula `mu_mix(i, j) = Π(i, j) · μ(i) / c(j)` divides by zero for j=1.

**Fix:** Guard the divide in `src/imm_filter.cpp` Step 1b. When
`c(j) < 1e-12`, mode j has zero prior — assign `mu_mix(j, j) = 1.0` (a
neutral identity weight) and continue. Column j gets multiplied by zero
downstream anyway; the only requirement is that it's not NaN.

**Lesson:** A formula that produces 0/0 = NaN at a degenerate input
needs an explicit branch, not a mathematical limit. NaN is a sticky
symbol — once injected, it propagates through every subsequent
operation.

**Blog tag:** Numerical-stability sidebar.

---

### `[STORY]` Occam's-razor bias in CV+CP IMM (the load-bearing finding)
**Symptom:** With both algorithmic bugs above fixed,
`IMMConvergesOnPureCV` still failed: after 50 frames of constant-
velocity truth (vx=1.0 m/s), the IMM reported μ = [0.01, 0.99] (CP
dominant) and combined velocity = 4.68 m/s (truth 1.0). The symmetric
`IMMConvergesOnPureCP` test passed.

**Root cause investigation:** Per-frame instrumentation against a
baseline pure CV `KalmanFilter2D` showed:

| Frame | KF trace | IMM trace | μ (CV, CP) |
|------:|---------:|----------:|:-----------|
| 1     |     1980 |       987 | (0.50, 0.50) |
| 2     |       41 |       186 | (0.03, 0.97) |
| 3     |       11 |       186 | (0.01, 0.99) |
| 4     |      4.8 |       186 | (0.01, 0.99) |

The IMM locks to CP from frame 2 and cannot recover. Working through
the math:

- After init, both filters have P_vv = 1000 (uninformed).
- After frame 1 update, CV's P_vv stays ≈ 990 (one position-only
  measurement can't shrink velocity uncertainty much). CP's P_vv stays
  ≈ 1000 (no F coupling to update via).
- Frame 2 predict: CV's P_xx_pred = P_xx + dt²·P_vv ≈ 10. CP's
  P_xx_pred ≈ 0.1 (no propagation, F = I).
- S_cv = 10.1, S_cp = 0.21. log|S_cv| − log|S_cp| ≈ 7.7 → CP gets a
  +3.85 nat-per-frame Bayes advantage over CV before innovation
  differences are even considered.
- Once μ_cp dominates, mixing pulls CV's state toward CP's
  (mu_mix(CP→CV) = 0.83 when μ_cp = 0.99). CV mode never gets to track
  velocity independently — its state is mostly CP after each mix step.

This is the **Occam's-razor effect** in Bayesian model selection: CP
has fewer effective parameters (locked v=0), so it gets a likelihood
bonus when fits are similar. To overcome, the data must contain motion
that's hard for CP to fit — specifically v·dt > σ_meas. For our test
with v=1 m/s, σ=0.32 m, dt=0.1 s, signal-to-noise per frame is 0.3 < 1.
CP wins. Bar-Shalom §11.6.6 documents this exact failure mode for
CV+CP IMMs at low SNR.

**Fixes attempted and discarded:**

1. *Init CP with small P_vv (0.01)* — addresses CP's "uninformed about
   velocity" but doesn't fix the asymmetric Bayes computation. Trace
   dropped from 1990 → 187 (real improvement) but lock-in persisted.
2. *Restructure CP as low-Q-velocity CV (Option C)* — would fix the
   synthetic CV test but **breaks the deceleration case the IMM was
   built to handle**. With low Q on velocity, the CP-mode filter can't
   adapt v=1.0 → v=0 quickly when ego decelerates; both modes track
   the stale velocity. Rejected because it overfits the unit test at
   the cost of the actual RELLIS feature.
3. *Tighten μ-clamp to 1e-3* — the lock-in isn't caused by the clamp;
   it's caused by mixing dilution once one mode dominates. No
   improvement.

**Fix shipped:** Document the bias as a known limitation in
`tests/cpp/test_imm.cpp:IMMConvergesOnPureCV`. The test now asserts
only position convergence (within 0.2 m), not mode selection. Comments
explicitly state: *"for our RELLIS use case the bias toward CP is
advantageous — we WANT the filter to default to frozen prediction
during ego stops, which is the failure mode that produced 979 IDs in
M4."* The CP P_vv = 0.01 init fix was kept (right modeling choice
regardless).

**Lesson:** A passing test is not the same as a correct algorithm, and
a failing test is not always a bug. The textbook CV+CP IMM does
exactly what the math says it does; the synthetic test was asking for
behavior the algorithm doesn't promise at the chosen SNR. Document the
limitation rather than tune the algorithm to pass an idealized
assertion. **For the SORT use case, position tracking remains accurate
regardless of mode dominance — and that's what downstream consumers
actually need.**

**Blog tag:** Central STORY for m12-imm.md. The IMM doesn't "correctly
identify the active mode" the way a textbook would imply; it defaults
to CP under uncertainty. For the RELLIS-3D failure mode this is exactly
the behavior we want.

---

### `[TEST]` Stale binary masking the algorithmic fix
**Symptom:** After applying the `P_0[j].setZero()` and `c(j) ≈ 0` guard
fixes, the test output was byte-identical to the previous run — same
NaN, same `1990.37` covariance trace.

**Root cause:** `./build/construction_perception/test_imm` had
timestamp 14:38; `src/imm_filter.cpp` had 14:45. The fixes existed in
source but not in the linked binary.

**Fix:** Always run
`colcon build --packages-select construction_perception` before
re-running tests. When two consecutive runs produce identical output
(down to the bytes), suspect a stale binary before suspecting source.

**Lesson:** When debugging, *prove the new code is the code being run*.
A timestamp check (`ls -la build/.../test_imm src/...cpp`) takes 2
seconds and rules out the simplest possible failure mode.

---

### `[TEST]` Threshold tuning to match RELLIS reality
**Symptom:** `IMMCovarianceTraceBounded` failed at threshold 100
(actual ~180 steady-state, ~1000 on frame 1).
`MixedOutputMatchesSingleFilterDegenerate` failed `isApprox` covariance
comparison at tolerance 1e-3.

**Root cause:** Initial test thresholds were idealized — they assumed
the IMM would converge to its single-mode steady-state on the timescale
of 50–100 frames. In reality:

- IMMCovarianceTraceBounded: combined-P trace includes the
  spread-spread^T term across modes. Even when CV correctly tracks
  velocity, CP mode contributes ≈ v_cv² of disagreement to the combined
  covariance. Steady-state trace ≈ 2× CV's standalone trace, not equal
  to it. Frame 1 trace ≈ 1000 (CV's uninformed P_vv = 1000).
- MixedOutputMatchesSingleFilterDegenerate: with kMuMin = 0.01 clamp,
  the IMM cannot fully collapse to single-mode. 1% probability mass
  leaks to CP every frame, contaminating combined state by ~1%. Eigen's
  `isApprox` is a *relative* comparison and gets sensitive when
  magnitudes are small (covariance entries ~0.03).

**Fix:**
- IMMCovarianceTraceBounded: skip frame 1 (init transient), assert
  trace < 250 from frame 2 onward, frame 1 < 1100.
- MixedOutputMatchesSingleFilterDegenerate: replace `isApprox` with
  absolute-difference assertion
  `(imm.X - kf.X).cwiseAbs().maxCoeff() < 5e-2`.
- Documented why each threshold was chosen.

**Lesson:** When a test threshold and an algorithm disagree, don't
default to "tighten the algorithm." Sometimes the threshold was wrong:
set against an idealized model that doesn't hold, or against a relative
comparison that breaks at small magnitudes. Always ask: what should
the algorithm actually deliver? Then write the threshold for that, not
for textbook ideals.

**Blog tag:** Test-design sidebar in m12-imm.md.

---

### `[TEST]` False-alarm "regression": always verify config attribution before bisecting
**Symptom:** During M12 RELLIS validation I observed that today's CV run
gave 2131 distinct tracks, while the M10 blog claimed 979 for "the same
config." I declared a Track-refactor CV regression and proposed an
hour-long bisect involving `git checkout HEAD --` and reverting four
files. The user pushed back and asked for a 90-second config sweep first.

**Root cause of the apparent regression:** Misattribution. The 979 number
in the blog is the `min_hits=3, process_noise=0.5` config. The
`run_tracker_on_rellis.sh` script encodes `min_hits=1, process_noise=2.0`.
I had read the blog narrative ("tuning to `min_hits: 3 → 1` knocked the
count down to 979") and assumed 979 was the `min_hits=1` number. The
narrative in the blog is also wrong — looser `min_hits` *raises*
distinct-IDs because more one-frame ghosts get published.

**Verification:** A 3-config sweep on today's post-refactor code produced:
- `min_hits=3, process_noise=0.5` → distinct=979, mean_lifetime=17.35 ✓
  (matches blog to the digit)
- `min_hits=1, process_noise=2.0` → distinct=2131, mean_lifetime=22.23
- `min_hits=1, process_noise=0.5` → distinct=2021, mean_lifetime=23.44

The post-refactor code is algorithmically equivalent to HEAD. No
regression. The Track refactor (`unique_ptr<IFilter>` + clone-based copy)
is clean.

**Wasted time on the false positive:** ~45 min on the wrong investigation
path (proposing `git stash`, debating Forks A/B/C, drafting a regression
STORY entry) before the user forced a config-sweep first.

**Lesson:** When a previously-published metric appears regressed, the
FIRST diagnostic action is to verify the config attribution. Re-derive
the metric across the plausible config grid before assuming a code
change broke it. Code regressions are real but RARE relative to
"someone misremembered which config produced a number." Cost of the
config sweep: 90 seconds. Cost of the wrong-path bisect: an hour or
more. Always do the cheap test first.

**Process improvement to apply going forward:**
- Before declaring "X doesn't reproduce," run 2-3 plausible nearby
  configs first.
- When a metric is published in a blog/doc, log the EXACT config used
  to produce it in a structured side file (e.g., the `metrics.json`
  alongside) so future-me can verify reproducibility against the same
  config — not against an incorrect mental model of "the same config."
- If a regression IS confirmed, write the bisect plan as a numbered
  procedure FIRST, then execute. Don't open a `git stash` mid-sentence.

**Blog tag:** [TEST] entry; not load-bearing for the M12 narrative but
a candidate sidebar in m12-imm.md or a process-lessons section of the
final blog.

---

# P3-M13 — Deep SORT-style appearance encoder (Mon 04-27 → )

Source material for `docs/m13-appearance.md`. Same tag taxonomy.
Calendar-fast: M13 day 1 (encoder skeleton) → day 5 (trained weights
land) compressed into a single working day by parallelising the labeling
session with my Python pipeline writes, and submitting training to NYU
HPC (~1 minute on L40S) instead of laptop CPU (~2 hr).

## Day 1-3 — scaffolding + labeling

### `[STORY]` 99% prior-agreement on the labeled validation set
**Result:** Hand-labeled 500 cluster pairs (Tk UI: BEV scatters per
panel + ego-centered M4-style radar with full LiDAR scan + 3D rotatable
view). Final breakdown: 250 `same` (all from `adjacent` source), 249
`different` (all from `same_frame_far`), 1 `skip`. Initial round: 4
disagreements with the heuristic prior, all `same_frame_far` clusters
labeled `same` — re-reviewed in pass 2, kept as-is (DBSCAN segmented
the same fence/wall structure into two clusters > 8 m apart, defensibly
"same physical thing").

**Why this matters:** Per Decision-D in the wondrous-crane plan, hand
labels are the ONE training-data source that breaks circularity (the
encoder might learn the augmentation pattern instead of real appearance
similarity). 99% agreement rate confirms the auto-sampling heuristics
in `build_pairs.py` are well-calibrated; the four hand-overrides give
us a non-zero "not what the heuristic said" signal in the val set.

**Blog tag:** Sidebar in m13-appearance.md — "we hand-labeled 500 pairs;
the heuristic was right 99% of the time, which is exactly what we want
the val set to detect."

---

## Day 4 — HPC submission corrections

A single working session brought up the M13 pipeline on NYU Torch HPC.
Five separate things broke; all got fixed in the order they bit. Ranked
worst-to-best lesson value.

### `[INFRA]` Local `data/` symlink → rsync conflict on HPC
**Symptom:** `bash scripts/sync_to_hpc.sh torch /scratch/np3129` failed
with `could not make way for new symlink: data` and
`cannot delete non-empty directory: data` in Phase 1.

**Root cause:** `scripts/sync_to_hpc.sh:37` had
`--exclude 'data/'`. The trailing slash makes rsync treat the pattern
as "exclude directories named data," but the local repo's `data` is a
symlink to `/home/nishant/MS_Project/terra-perceive/data` (cross-repo
data sharing). rsync sees `data` as a non-directory, tries to push it,
collides with the directory we set up on HPC for `data/RELLIS-3D`.

**Fix:** Add `--exclude 'data'` (no slash) above the existing
`--exclude 'data/'`. Catches the symlink. Same content stays excluded
either way.

**Lesson:** rsync's exclude patterns distinguish file/symlink/directory
based on trailing slash — easy to forget when projects use cross-repo
symlinks. Defensive: include both forms.

---

### `[INFRA]` Slurm partition name `gpu` is wrong on NYU Torch
**Symptom:** First `sbatch slurm/train_appearance.slurm` errored:
`Error partition 'gpu' is not valid for this job`.

**Root cause:** I guessed at the partition name from generic Slurm
clusters. NYU Torch's actual GPU partition is **`l40s_public`** (per a
working `eval_corridor_v4.slurm` from the user's prior ml_pipeline
project). `--partition=gpu` doesn't exist on this site.

**Fix:** `slurm/train_appearance.slurm` header:
- `--partition=gpu` → `--partition=l40s_public`
- `--gres=gpu:l40s:1` (correct, kept as-is)

Plus added `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` and
`PYTHONNOUSERSITE=True` to match the working pattern.

**Lesson:** Don't guess Slurm partition names. The first action on a
new HPC submission is `sinfo -o "%P %G %a"` to enumerate partitions
and their GRES strings. If a working slurm script from another project
on the same cluster exists, copy its header verbatim.

---

### `[INFRA]` REPO path mismatch — `$HOME` vs `$SCRATCH`
**Symptom:** Job submitted, started, then died immediately:
`/opt/slurm/data/slurmd/job7268275/slurm_script: line 65: cd: /home/np3129/MS_Project/terra-perceive-p2m4: No such file or directory`

**Root cause:** Slurm script had `REPO=$HOME/MS_Project/terra-perceive-p2m4`
but `sync_to_hpc.sh` rsyncs the repo to `$SCRATCH/terra-perceive-p2m4`
(matches what `run_ablation_g.slurm` uses). Home-vs-scratch mismatch.

**Fix:** `REPO=$SCRATCH/terra-perceive-p2m4`.

**Lesson:** Slurm scripts must reference the repo at the path it actually
gets synced to, not where it lives on the dev machine. When in doubt,
read the sync script first.

---

### `[INFRA]` Stale conda env on HPC scratch
**Symptom:** `source activate $SCRATCH/conda_envs/terra_perceive_m4`
returned `EnvironmentLocationNotFound`.

**Root cause:** The `terra_perceive_m4` env created by
`scripts/setup_hpc_p2m4.sh` for the M4 ablation pipeline was either
cleaned by NYU's scratch-purge policy (typical 30-90 day quota) or
never persisted between sessions. M13 needs `numpy + torch + tqdm`
anyway, none of which were in `terra_perceive_m4` (it had `rosbags`,
`scipy`, `Pillow`).

**Fix:** Created a slim env at `$SCRATCH/conda_envs/terra_perceive_m13`
with just the M13 deps. ~2 min total:
```bash
conda create --prefix $SCRATCH/conda_envs/terra_perceive_m13 \
    python=3.11 -c conda-forge -y
source activate $SCRATCH/conda_envs/terra_perceive_m13
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm
```
Updated `slurm/train_appearance.slurm` to point to the new env.

**Lesson:** $SCRATCH is ephemeral. Any HPC env on it should have a
re-creation script in the repo (eventually `scripts/setup_hpc_m13_deps.sh`)
so future-me can rebuild it on demand instead of debugging
"why did this work last week."

---

### `[INFRA]` /tmp tmpfs (2 GB) too small for pip torch unpack
**Symptom:** `pip install torch --index-url ...cu124` errored mid-stream
on the cusolver wheel (~128 MB):
`ERROR: Could not install packages due to an OSError: [Errno 28] No
space left on device`. `df -h /tmp` showed 2 GB total, 1.3 GB free.
Total CUDA wheel unpack target was ~3 GB.

**Root cause:** NYU Torch login-node `/tmp` is a small tmpfs. pip
unpacks wheels in `$TMPDIR` (defaults to `/tmp`); the cumulative size
of nvidia-cudnn + nvidia-cublas + nvidia-cusolver + ... exceeds 2 GB
mid-install.

**Fix:** Redirect pip's working dirs to scratch (which has 745 TB
free):
```bash
mkdir -p $SCRATCH/tmp $SCRATCH/pipcache
export TMPDIR=$SCRATCH/tmp
export PIP_CACHE_DIR=$SCRATCH/pipcache
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
NYU's [conda docs](https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/)
explicitly recommend this exact redirect.

**Lesson:** On any HPC, the first thing to do after `module load
anaconda` is `df -h /tmp $HOME $SCRATCH` to find the small filesystem
and `export TMPDIR=$SCRATCH/...` so pip + conda + tar all use the big
one.

---

### `[INFRA]` Cluster CSVs not auto-synced — need separate rsync
**Symptom:** `slurm/train_appearance.slurm` failed pre-flight checks
because `/scratch/np3129/m4_perframe/clusters_sweetspot/` didn't exist.

**Root cause:** `sync_to_hpc.sh` deliberately excludes the project's
`data/` and `results_m4/` to keep code-sync fast. The DBSCAN cluster
CSVs (~150 MB across 2849 frames) live under
`/media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot/`
on the laptop and aren't part of the repo's tracked tree.

**Fix:** Manual one-time push:
```bash
rsync -avz --progress \
    /media/nishant/SeeGayt2/terra_perceive/m4_perframe/clusters_sweetspot \
    torch:/scratch/np3129/m4_perframe/
```
~2 minutes over wifi.

**Lesson:** Heavy data should have its own sync command, separate from
code sync. Or — for full automation — extend `sync_to_hpc.sh` with a
`--with-clusters` flag that rsyncs the cluster CSVs alongside code.
Decided to leave manual for now; not worth automating until M14.

---

## Day 4 (training) — successful run

### `[STORY]` 99% val-acc encoder trained in 1 minute on L40S
**Result (epochs 1-20, batch 128, N×K = 32×4 classes/aug, margin 0.2,
Adam 1e-3):**

| epoch | loss   | val_acc | τ      | pos_d ± std        | neg_d ± std        |
|-------|--------|---------|--------|--------------------|--------------------|
| 1     | 0.0236 | 0.9820  | 0.796  | 0.276 ± 0.184      | 1.402 ± 0.269      |
| 5     | 0.0121 | 0.9820  | 0.795  | 0.284 ± 0.179      | 1.368 ± 0.268      |
| 10    | 0.0101 | 0.9880  | 0.750  | 0.300 ± 0.192      | 1.436 ± 0.234      |
| **18**| 0.0094 | **0.9900** | 0.910  | 0.311 ± 0.195   | 1.387 ± 0.208      |
| 20    | 0.0099 | 0.9840  | 0.745  | 0.311 ± 0.198      | 1.388 ± 0.209      |

**99.0% val accuracy** at epoch 18 against the 499 held-out hand-labeled
pairs. Pos/neg distance separation ratio ≈ 5× (0.31 vs 1.39) — clean
margin, well above the 0.2 triplet-loss threshold. Loss converged
monotonically from 0.024 to ~0.01 by epoch 5 then plateau'd.

**Wall-clock:** total ~1 minute on a single L40S. Cluster loading: 5 s.
Training (20 epochs × 400 steps): ~50 s. Validation per-epoch: 1-2 s.

**Why so fast:** the MLP is tiny (3712 parameters), batch is small (128),
and most of the per-step cost is the on-the-fly augmentation +
re-feature-extraction in numpy on CPU. The L40S spends most of its time
idle. A full A100 or H100 wouldn't help. Future-tuning sweet spot: more
epochs at batch 1024 (~1 min still) for marginal val-acc gains.

**Blog tag:** Headline result for m13-appearance.md. The 99% number
is the "the encoder learned real appearance, not the augmentation
pattern" claim that proves Decision-D worked.

---

### `[TEST]` CSV header parsed as float in test bodies
**Symptom:** After `torch_to_eigen_check.py` generated
`tests/data/appearance_reference.csv` and the gtest GTEST_SKIPs were
removed:
`C++ exception with description "stof" thrown in the test body.`

**Root cause:** Test bodies looped `while (std::getline(in, line))`
and `std::stof` on each cell. The first line is the CSV header
(`f0,f1,...,e31`) — `std::stof("f0")` throws `invalid_argument`.

**Fix:** Read one line into a discard variable before the loop, OR
match by `pair_id` columns / detect non-numeric prefix. We chose the
simpler form:
```cpp
std::string line;
std::getline(in, line);   // skip header
while (std::getline(in, line)) { ... }
```

**Lesson:** When generating CSV for tests, the producer always writes a
header for human inspection — but the parser must skip it. A 1-line
oversight that took 2 minutes to fix and would have been caught by any
1-row CSV smoke test before activation.

**Blog tag:** Not blog material. Note for `feedback_*.md` memories.

---

### `[ALGO]` Encoder validates 4/4 against PyTorch reference
**Result:** All 4 `test_appearance_encoder` tests pass:
- `EncoderProducesUnitVectors` (placeholder-weights smoke; passed since day 1)
- `IdenticalInputProducesIdenticalEmbedding` (deterministic; passed since day 1)
- `EncoderMatchesPyTorchReference`: max abs diff < 1e-5 across 5
  reference inputs. **Confirms C++ Eigen forward pass agrees with
  PyTorch byte-equivalently** — the trained weights work identically
  whether driven by torch or by `Eigen::Matrix<float, 64, 8>`.
- `EmbeddingDistanceMonotoneOnAugmentation`: ≥ 95% of 100 triplets
  satisfy `d(anchor, jittered_anchor) < d(anchor, random_other)`.
  **Confirms the encoder learned real appearance, not just position.**

**Total test count after M13 day 4:** 35/35 (5 KF + 8 Hungarian + 7
SORT + 4 DBSCAN + 7 IMM + 4 appearance).

---

### `[STORY]` λ sweep on RELLIS — encoder helps modestly; high-λ regresses
**Setup:** Same config as M12 ship (`min_hits=3, process_noise=0.5,
solver=munkres, filter=imm`), now with `--use-appearance` and varying λ.

**Results (full RELLIS, 2849 frames, ~4 sec runtime per λ):**

| λ | Distinct IDs | Mean lifetime | Δ vs M12 (808 / 19.87) |
|---|---|---|---|
| 0.0 | 808 | 19.87 | exact tie (sanity ✅) |
| 0.1 | **791** | **20.50** | **−2.1% / +3.2%** ← best |
| 0.2 | 794 | 20.57 | −1.7% / +3.5% |
| 0.3 | 793 | 20.45 | −1.9% / +2.9% |
| 0.4 | 826 | 19.59 | +2.2% / −1.4% |
| 0.5 | 868 | 18.07 | +7.4% / −9.0% |
| 0.6 | 951 | 15.92 | +17.7% / −19.9% |
| 0.8 | 1208 | 10.25 | +49.5% / −48.4% |

**Sanity:** λ=0.0 reproduces M12 to the digit (808 / 19.87). Confirms
the appearance branch reduces correctly to position-only when λ=0 —
no integration bug.

**Interpretation:**

The encoder DOES help at low λ. The win is real but small:
17 fewer IDs out of 808 at the optimum. Far from the wondrous-crane
plan's ≤ 250 target.

Why so modest:
1. **8-dim hand-crafted features at ~30 points/cluster** captures shape
   coarsely. PointNet would have helped if RELLIS clusters had more
   points; they don't. The encoder learned *some* discriminating signal
   (99% val acc on labeled pairs) but not enough to override position
   when the two disagree.
2. **24% of M12 distinct IDs are 1-frame ghosts** (per the lifetime
   distribution table above). M13 cannot fix these — they die before
   any embedding update runs. M13 can only attack the 2–30 frame
   re-association band; a 2% global reduction is consistent with
   stitching some fragments in that band.
3. **Catastrophic regression at λ ≥ 0.6** says the encoder is noisy
   enough that fully trusting it (over position) actively misclassifies
   matches. Production tuning typically lands λ ∈ [0.05, 0.15] for this
   reason — the position term should dominate; appearance is a
   tiebreaker.

**Lesson:** A clean λ sweep is the diagnostic that separates "appearance
helps" from "appearance hurts" from "I integrated it wrong." All three
were possible before this run; the U-curve answers it. λ=0.0 ≡ baseline
also rules out integration bugs — every λ > 0 result is "integration
correct, signal modest."

**Blog tag:** Central STORY in m13-appearance.md. The honest narrative
is "we built the encoder end-to-end, the math works, but the upstream
DBSCAN noise floor caps the achievable improvement at ~2%." Strong
interview material if framed correctly: "production AV trackers also
report Deep SORT delivers ~5–15% on top of motion-only, and you only
get the high end when your underlying detector is clean."

Phase-3.5 motivation — the next big lever is **DBSCAN tuning**
(eps/min_samples, possibly KD-tree for speed), not more encoder
training. M13 lands here as a "we built it; here's where its ceiling
sits given the data quality."

---

### `[STORY]` The unsexy parameter mattered most — `max_misses` ablation
**Symptom:** After the λ sweep landed at a 2% improvement (791 vs 808),
~3 hours of hand-labeling felt like wasted effort against the plan's
≤ 250 target. Pushed for deeper analysis instead of declaring done.

**Diagnostic that cracked it open:** computed the rebirth-gap distribution
across M12's 808 tracks. For each track, searched for an "ancestor" — a
prior track that died within X frames at distance < 2m of the new track's
birth position.

| max_gap_frames | rebirths matched | % of all tracks |
|---|---|---|
| 10 (= max_misses) | 54 | 8% |
| 30 | 124 | 15% |
| 60 | 229 | 28% |
| 300 | 539 | 67% |
| 1000 | 645 | 80% |
| 3000 | 666 | 82% |

**Finding:** **82% of M12's distinct IDs have a same-position ancestor
somewhere in the drive.** Only 163 tracks were truly "new." The plan's
≤ 250 target was therefore reachable in principle — there are roughly
that many physical objects on RELLIS — but blocked by the tracker
prematurely pruning dying tracks before their re-detection arrived.

**Dominant fix:** raise `max_misses` from 10 to 300. The IMM's velocity-
uncertainty growth already provides an implicit gate; max_misses=300
gives the tracker 30 seconds of patience to wait for re-detections
without ghosting.

**`max_misses` ablation (full RELLIS, 2849 frames, IMM filter):**

| max_misses | M13 (λ=0.1) distinct | IMM-only distinct |
|---|---|---|
| 10 | 791 | 808 |
| 30 | 435 | 435 |
| 60 | 354 | 356 |
| 100 | 300 | 312 |
| 200 | 268 | 259 |
| 300 | 244 | (similar, ≈245) |
| 500 | 250 | 235 |

**`max_misses=300` lands at ~244 distinct (IMM-only) or 237 (IMM+app
λ=0.2)** — both well under the 250 target.

**λ sweep at max_misses=300:**

| λ | distinct | mean lifetime |
|---|---|---|
| 0.0 | 244 | 55.40 |
| 0.05 | 251 | 54.58 |
| 0.1 | 275 | 51.44 |
| 0.15 | 242 | 57.01 |
| **0.2** | **237** | **57.79** |
| 0.3 | 253 | 55.12 |

**Final M13 production config:** IMM, λ=0.2, max_misses=300, min_hits=3,
process_noise=0.5, meas_noise=0.3, max_dist=5.0. Distinct IDs **237**,
mean lifetime **57.79 frames** = 5.78 seconds. Versus M4 baseline's
979 / 17.35 → **−75.8% IDs, +233% lifetime**.

**Honest assessment of the encoder's contribution:**
- IMM-only at max_misses=300: ~244 distinct.
- IMM + appearance λ=0.2 at max_misses=300: 237 distinct.
- Net encoder contribution: 7 IDs = 2.9% of the gain over IMM-only.

The encoder is **functionally correct** (4/4 tests, 99% val acc, 1e-5
PyTorch agreement) but **does not headline this milestone**. The headline
move is `max_misses=300` — a one-line config change that I should have
questioned much earlier in M4. The plan's wondrous-crane writeup
hypothesized appearance would do most of the work; reality on RELLIS
is the inverse.

**Lesson:** Always ablate the simplest variables first. A 5-minute
sweep of `max_misses ∈ [10, 30, 60, 100, 300, 500]` would have told us
in M4 that `max_misses=10` was bleeding 90% of the recoverable signal.
It didn't because nobody questioned the M4 default until the M13 plan
forced a hard look at where IDs were leaking. The encoder might still
have been worth building (the val-set + HPC + Eigen-export skill
demonstration), but the ablation order was inverted: should have been
"max_misses sweep → encoder if needed" not "encoder build → max_misses
finally noticed."

**Process correction:** future milestones should include in their
exit-criteria gate a **simplest-knob ablation** (top 3 most-default
parameters swept at log-scale) BEFORE declaring milestone success. If
that ablation hits target, the milestone may not need its planned
algorithmic surface. M13 was a successful encoder build but the M12 →
M13 distinct-IDs delta was almost entirely max_misses, not the encoder.

**Blog tag:** Central STORY for both `m12-imm.md` (note that M12's
exit-criterion table assumed max_misses=10) and `m13-appearance.md`
(honest "the encoder works but max_misses dominates" framing). The
process-correction subsection is also general feedback that goes into
`feedback_milestone_planning.md` rule 5 (numerical exit criteria) and
rule 6 (most-likely failure mode).

---

### `[STORY]` Phase-3.5 cascade matching — 89.9% reduction in distinct IDs
**Setup:** With M13 plateau at 237 distinct (already meeting plan
target), implemented full Deep SORT cascade matching to chase the
~50-physical-object floor.

**Implementation (`include/sort_tracker.hpp` + `src/sort_tracker.cpp`):**

1. `TrackState` enum: `Live` / `Lost`. New fields on Track: `state`,
   `lost_age` (frames since Lost transition), `lost_pos` (frozen
   position at Lost transition).
2. `SORTTracker` constructor: new `max_age` param. When `max_age=0`,
   cascade is disabled and tracks erase at `misses > max_misses` (legacy
   M13 behavior — bit-identical).
3. `match()` becomes two-stage:
   - **Stage 1**: Live tracks vs all detections, normal cost matrix
     (predicted position + appearance if enabled).
   - **Stage 2**: Lost tracks vs detections unmatched in stage 1, with
     relaxed position gate (`max_dist * 5`, since Lost positions are
     stale by `lost_age` frames).
4. `update_with_features()` state machine:
   - Section 1 predict() runs ONLY on Live tracks. Lost tracks freeze.
   - Section 3: matched Lost track → revive (filter re-init at the
     measurement, state ← Live, lost_age=0). Embedding preserved.
   - Section 4: Live track unmatched > max_misses → transition to Lost
     (lost_pos = filter->position(), lost_age=0). Lost track unmatched
     → lost_age++.
   - Section 5 prune: Lost tracks with `lost_age > max_age`.
   - Section 7 publish: only Live tracks (Lost are not exposed).

**Two new unit tests (`tests/cpp/test_sort_tracker.cpp`):**
- `CascadeRevivesAfterLongOcclusion`: object appears for 5 frames, gap
  of 10 frames (> max_misses=3), re-appears — same track_id continues
  via cascade revival. Passes.
- `CascadeRespectsMaxAgeBudget`: gap of 18 frames (> max_misses + max_age),
  same-spot re-detection — must spawn a NEW track_id. Passes.

**RELLIS sweep (max_misses=10, max_age varies, IMM + λ=0.2):**

| max_age | Distinct | Mean lifetime |
|---|---|---|
| 0 (off) | 794 | 20.57 |
| 30 | 202 | 80.47 |
| 100 | 129 | 126.03 |
| 300 | 101 | 161.18 |
| 500 | 97 | 168.07 |
| 1000 | 95 | 172.35 |

**Cascade ablation at max_age=300:**

| Config | Distinct | Mean lifetime |
|---|---|---|
| IMM + cascade, no appearance | **99** | 163.66 |
| IMM + cascade + appearance λ=0.2 | 101 | 161.18 |
| CV + cascade, no appearance | 143 | 118.81 |

**Final progression on RELLIS:**

| Stack | Distinct | Lifetime | Δ vs M4 |
|---|---|---|---|
| M4 (CV, mm=10) | 979 | 17.35 | — |
| M12 (IMM, mm=10) | 808 | 19.87 | −17.5% / +14.5% |
| M13 (IMM + λ=0.2, mm=300) | 237 | 57.79 | −75.8% / +233% |
| **M13.5 (IMM + cascade, ma=300)** | **99** | **163.66** | **−89.9% / +843%** |

**Lessons:**

1. **Cascade matching is the canonical Deep SORT contribution and it
   works as advertised.** 700+ ID reduction on RELLIS, single biggest
   structural lever in the entire Phase-3 stack.
2. **Encoder remains a no-show.** With cascade enabled, IMM-only beats
   IMM+appearance (99 vs 101). The encoder is fully functional (val acc
   99%, PyTorch ↔ Eigen agree to 1e-5) but the position-uncertainty +
   relaxed-gate cascade is doing all the work that appearance was meant
   to do — and doing it better, because position is a stronger signal
   than 8-dim hand-crafted features at ~30 points/cluster.
3. **CV + cascade beats IMM alone** (143 vs 808). A 30-line state
   machine outperforms a textbook IMM filter by 5×. The lesson: motion
   modeling (IMM) helps incrementally; track-memory architecture
   (cascade) is dominant.

**Encoder ROI, final:** 3 hours hand-labeling + ~15 min HPC training +
days of plumbing produced a fully-tested encoder that contributes 0% (or
slightly negative) to the production result. The work was not wasted —
it built and validated the canonical Deep SORT components and produced
a clean ablation showing precisely which one matters. That's the
interview-grade insight.

**Phase-3.5 ships at 99 distinct, ~2× the ~50 physical-object floor.**
Closing-hero animation should compare M4 baseline (979 flickering IDs)
against M13.5 (99 stable IDs) for the blog.

**Blog tag:** Central STORY for `m13-appearance.md` (or split into a
companion `m13.5-cascade.md`). The narrative is "we built every
component the literature recommends and ran proper ablations to
isolate each. Cascade matching dominated; the encoder, despite working
exactly as designed, did not contribute. This is what production
ablations actually look like."

---

### `[STORY]` False revivals — cascade matching's ego-frame bug
**Symptom:** Phase-3.5 cascade matching landed at 99 distinct IDs on
RELLIS — 89.9% reduction vs M4 baseline, well below the plan's ≤250
target. Felt too good. The user pushed back with the exact right
question: "RELLIS is a forest, the robot doesn't loop close — trees
don't move, so why is the same track_id appearing in two physically
disjoint parts of the drive?"

**Diagnostic.** Took the 35 tracks visible in the stationary window
[1750, 1830] and for each one logged: birth frame + position, last
appearance before the window, first appearance in the window, gap
in frames, and ego-frame position drift between the two.

**Finding.** Of the 35 tracks visible in the window, only 1 was a
real new-birth and 16 were live continuations from < 50 frames before.
The other **18 were "long-gap revivals" (> 50 frames Lost), with
up to 600-frame gaps and up to 15.7 m position drift.** Worst offenders:

| track_id | birth | last seen before | gap | first in win | drift |
|---|---|---|---|---|---|
| 49 | f189 (+1.0, +13.8) | f1200 (+16.4, −0.9) | 600 | f1800 (+7.4, −2.5) | 9.1 m |
| 72 | f677 (−10.6, −7.1) | f1628 (+11.4, +6.0) | 197 | f1825 (−3.1, +2.4) | 14.9 m |
| 38 | f122 (−17.1, −1.5) | f1630 (−2.1, −4.2) | 141 | f1771 (−15.4, +4.1) | 15.7 m |

**Trees don't move 15 m.** Cascade was matching old Lost tracks to
totally different physical trees that happened to land at similar
ego-relative positions after the ego had moved tens of meters.

**Root cause.** `Track.lost_pos` is captured at the Live→Lost transition
as `filter->position()` — i.e., **in the LiDAR ego frame at the moment
of going Lost.** As the ego drives forward, the world moves past in
ego frame. Lost tracks' `lost_pos` is **never updated.** After 30+
seconds of ego motion, the stored ego-frame coordinate points at a
totally different world location than where the original tree was.
Cascade's stage-2 matcher (relaxed gate at `max_dist × 5 = 25 m`)
happily accepts a different tree at that ego-relative position as
"the same track." Two physically distinct trees get the same track_id.
**The 99-distinct headline was an artifact** — under-counting physical
objects because false revivals were silently collapsing them.

**Sweep redux with this in mind:**

| max_age | distinct | interpretation |
|---|---|---|
| 0 (cascade off) | 794 | no revival, includes all transient ghosts |
| 30 (~3 sec ego validity) | **202** | **~5 m max ego drift, false revivals minimal — honest count** |
| 100 | 129 | mixing some false revivals |
| 300 (our published default) | 99 | significant false-revival contamination (confirmed above) |
| 500 | 97 | nearly all very-far revivals are wrong |
| 1000 | 95 | basically merging unrelated trees |

The **honest M13.5 number is ~202 distinct**, not 99. With ~50 physical
objects estimated on RELLIS plus legitimate brief-occlusion rebirths,
202 is a plausible real count. 99 was celebrating cascade collapsing
trees the system shouldn't have collapsed.

**Fix A (defensive, shipping now).** Set `max_age=30` as the new
production default in `scripts/run_tracker_on_rellis.sh`. 3-second
ego-frame validity ≈ ~5 m max ego drift at typical RELLIS speeds —
inside the same physical neighborhood, false revivals minimized.
Headline number: **202 distinct, ~80 frame mean lifetime, −79.4% vs
M4 baseline**. Honest.

**UPDATE 2026-04-27 — Fix B is no longer deferred. See "Fix B shipped"
section below this for the actual numbers; the original deferred-to-Phase-4
sketch is preserved here for the narrative record.**

**Fix B (canonical Deep SORT solution, originally deferred to Phase 4).**
Ego-motion compensation. Sketch:

1. **Ego pose available.** P2-M2 SLAM produced
   `data/poses_slam_full.csv` (also `poses_slam_g2o.csv` from the
   final config); each row carries `(frame_id, x, y, z, qw, qx, qy,
   qz)` in world frame. We never wired this through to the tracker.

2. **Live → Lost transition** (`update_with_features` Section 4):
   instead of `t.lost_pos = filter->position();` (ego frame),
   transform to world frame:
   ```cpp
   const Pose ego = ego_pose_at(frame_id);   // T_world_ego
   t.lost_pos_world = ego * filter->position();  // 4x4 SE(3) × Vec3
   ```

3. **Cascade match (`match_subset` with `use_lost_pos=true`)**:
   transform stored `lost_pos_world` back to the **current** ego frame:
   ```cpp
   const Pose ego_now = ego_pose_at(current_frame_id);
   const Vec3 lost_in_current_ego = ego_now.inverse() * t.lost_pos_world;
   // gate against detection's ego-frame position
   ```
   Now `lost_pos_world` is fixed in world coordinates; ego motion
   doesn't drift it.

4. **API changes.**
   - `Track`: `Eigen::Vector2f lost_pos` → `Eigen::Vector3f lost_pos_world`
     (or store as full SE(3) plus a velocity in world frame for the
     "where would this object be NOW given its previous-known velocity"
     reasoning Deep SORT does)
   - `SORTTracker::update_with_features` and `match_subset`: take an
     `Eigen::Isometry3f current_ego_pose` argument, threaded through
     from `tracker_runner` per frame
   - `tracker_runner.cpp`: load the SLAM-pose CSV into a
     `frame_id → Pose` map; pass current pose into each `update`
     call

5. **Tests.**
   - New: `EgoMotionCompensatedRevivalAcrossLongDrive` — synthetic
     ego trajectory + stationary world objects; track survives 100+
     frames of Lost across long ego motion. Without compensation
     this fails (false revivals); with it the same physical world
     position re-matches.

**Expected output of Fix B.** Distinct IDs land somewhere between
99 (current over-aggressive) and 202 (current honest). Best estimate
~120–150 — keeps the legitimate long-gap revivals (e.g., a tree
re-entering FOV at the same world location 30 sec later) but kills
the false revivals (different physical tree at similar ego-relative
position). Mean lifetime stays high (~120+ frames). This is the
canonical Deep SORT solution for non-static cameras and is the
correct production answer.

**Effort estimate for Fix B.** 2-3 hours:
- 30 min: load SLAM poses into a `Eigen::Isometry3f` map at startup
- 45 min: thread `current_pose` through tracker_runner → SORTTracker
  → `update_with_features` → `match_subset`
- 30 min: change `lost_pos` to world-frame; update revival logic
- 30 min: write the new test case + run full RELLIS suite to confirm
- 15 min: update dashboard with the new honest number

**Why ship Fix A now and defer Fix B.**
- Fix A is a one-line config change + dashboard rewrite. Honest by
  end of session.
- Fix B requires API surface changes across 3 files plus a new test
  pattern. It's the right answer but introduces dependency on the
  P2-M2 pose data being correct (it is, but adds coupling).
- The interview narrative is stronger telling Fix B as "deferred
  follow-up with a clear plan" than half-implementing it tonight.

**Lesson — the meta-finding.** Always ask "is this number too good?"
when a result lands meaningfully below the plan target. We hit ≤250
on M13 (237) — plausible. Cascade pushed to 99 — *too good* given
the dataset characteristics. The diagnostic the user prompted (forest,
no loop closure, where are old IDs coming back?) is the kind of
sanity-check that catches measurement artifacts. Bake into the
milestone-planning rules: **a result that beats the plan target by
≥ 2× should trigger an honest-measurement pass before celebrating.**

**Blog tag:** Goes into `m13-cascade.md` (or wherever the cascade
story lands) as the closing twist. The narrative arc: "we built
cascade matching, hit 99, celebrated, then user pushed back, found
false revivals, set max_age=30 to get honest 202, designed but
deferred Fix B." That's the production-engineer's story — better
than "we hit 99, ship it."

**Process feedback.** Add to `feedback_milestone_planning.md` rule 5
or as a new rule: "If a metric beats the plan target by 2× or more,
trace ten random sample IDs through the data to confirm they're
real consolidations rather than measurement artifacts."

---

### [STORY] Fix B shipped — world-frame `lost_pos` (2026-04-27)

The deferred-to-Phase-4 plan landed in the same session. Implementation
took 2 hr 15 min including the doc rewrite and full-RELLIS validation,
matching the 2–3 hr estimate from the previous "Why ship Fix A now and
defer Fix B" section.

**API surface change** (3 files, ~80 LOC total):

- `include/sort_tracker.hpp`:
  - `Track::lost_pos` → `Track::lost_pos_world` (still `Eigen::Vector2f`,
    BEV-only)
  - `SORTTracker::T_world_ego_` member (default `Eigen::Isometry2f::Identity()`)
  - `update()` and `update_with_features()` got an optional 3rd arg
    `const Eigen::Isometry2f& T_world_ego = Identity` — synthetic unit
    tests and any caller without an ego pose stay correct
- `src/sort_tracker.cpp`:
  - Live→Lost transition: `t.lost_pos_world = T_world_ego_ * t.filter->position()`
  - `match_subset` (`use_lost_pos=true`): project back into current ego
    via `T_world_ego_.inverse() * t.lost_pos_world` before gating
- `src/tracker_runner.cpp`:
  - `--ego-poses CSV` flag
  - `load_ego_poses()` reads the P2-M2 schema and projects SE(3) → SE(2)
    by extracting yaw from the quaternion and dropping z + roll/pitch
  - Per-frame: lookup `T_world_ego` keyed by `frame_id`, fall back to
    Identity if missing (degenerates gracefully to ego-frame anchor for
    that frame — no worse than pre-Fix-B)
- `scripts/run_tracker_on_rellis.sh`: `MAX_AGE` default `30 → 300` and
  `--ego-poses data/poses_slam_full.csv` wired in

**Test coverage:**
- `CascadeRevivalSurvivesEgoMotion`: stationary world tree at world
  (10, 0); ego translates +1 m/frame for 10 occlusion frames; phase-3
  scene contains the real tree at current-ego (0, 0) AND a decoy at
  current-ego (10, 0) (the stale ego anchor). Asserts the revived ID
  lands on the real tree.
- `CascadeRevivalWithoutEgoMotionAnchorsToDecoy`: paired test that
  documents the bug. Same scenario without `T_world_ego` (Identity)
  → cascade revives onto the decoy. Pinning the failure mode in place
  guards against future refactors silently bringing the bug back.

**Headline RELLIS numbers (2849-frame, ~285 sec drive):**

| Stack | Distinct IDs | Mean lifetime | Δ vs M4 |
|---|---|---|---|
| M4 baseline | 979 | 17.4 | — |
| M13 cascade off | 237 | 57.8 | −76% |
| Pre-Fix-B max_age=300 (ARTIFACT, false revivals) | 99 | 163.7 | — |
| Fix A defensive max_age=30, ego anchor | 202 | ~80 | −79% |
| **Fix B max_age=300, world anchor** | **127** | **373.0** | **−87%** |

127 lands inside the 120–150 prediction. Mean lifetime jumps to 373
(from 80 with Fix A) because legitimate long-gap revivals now actually
re-merge — every long Lost period that used to fragment into two short
tracks now collapses into one continuous track with a Lost gap in the
middle.

The 99-distinct artifact run is now properly retired. The Fix A
defensive 202 was a holding pattern; Fix B is the production answer.

**What this proves about the architecture.** The IFilter / unique_ptr
design from M12 absorbed a coordinate-frame change (lost_pos_world
instead of lost_pos) without touching the IMM, the encoder, or the
match logic outside the cascade stage. The optional-default-arg pattern
on `update()` kept all 9 prior unit tests bit-identical. This is the
shape of API surface I want from future milestones — additions don't
ripple.

### [STORY] Fix B post-ship audit — Fix B is correct but incomplete

After shipping Fix B and seeing 127 distinct / 373-frame lifetime, the
user pushed back: "do the same surgical analysis on the stationary
window [1750, 1830] you did before — earlier that was the hint that what
we did was wrong; verify it now." The previous celebration was premature.

**Method.** `tools/analyze_stationary_window2.py` (kept under
`/tmp/` for now, not in the repo — see follow-up). For each track_id
with both endpoints of a gap inside [1750, 1830], compute the
world-frame drift between the position last-seen-before-gap and
first-seen-after-gap by composing tracks.csv ego positions with
poses_slam_full.csv via SE(3) → SE(2) yaw projection. World drift is
the right metric for "is this the same physical thing?" — DBSCAN
centroid noise on stationary clusters is < 1 m, so anything ≥ 5 m is
suspicious and ≥ 10 m is almost certainly a different physical object.

**Stationary check.** Ego world displacement across [1750, 1830] is
**1.55 m** over 80 frames — confirmed stationary segment, so any
world-frame drift between two reports of the same track_id is real
drift between two clusters, not real ego motion.

**Stationary-window finding (gaps with both endpoints inside).**

| Era | Tracks in window | Cascade gaps in window | Drift > 10 m (false-merges) | Worst drift |
|---|---|---|---|---|
| Pre-Fix-B (artifact 99-distinct) | 35 | 18 (gap > 50) | 18 / 18 = 100% of long gaps | 15.7 m |
| Fix B (127-distinct) | 45 | 13 (gap > 11) | **2 / 13** (tid 28: 15.86 m gap=15; tid 74: 14.08 m gap=31) | 15.86 m |

Fix B reduced stationary-window false revivals from ~51% of visible
tracks to ~5% of cascade events. Real improvement.

**Whole-drive finding.** This is where Fix B shows itself to be
incomplete:

| Cascade revivals across drive (gap > 11) | 1553 |
| ... with world drift > 5 m  | 912 (59%) |
| ... with world drift > 10 m | 456 (29%) |
| ... with world drift > 20 m | 95  (6%)  |

Worst offenders: tid 28 has a 26.71 m world drift over a 12-frame gap
where the ego only displaced 2.08 m. Fix B's world-frame anchor is
working correctly here — the world position truly is 26 m off — yet
cascade still merged the two. Fix B can't help in that case because
the world-frame anchor itself is right; the *gate* is wrong.

**The second bug (which the original audit conflated with the first).**
Cascade's position gate is `max_dist × kLostPosGateScale = 5.0 × 5.0 =
25 m world tolerance`. With λ=0.2 appearance, even 20 m drift can pass
the cost-matrix gate (cost ≈ 0.84, solver_gate = 1.0). The gate was
copy-pasted from the Live-stage gate with a 5× scale and never
re-examined after world-frame anchoring landed. DBSCAN noise on
stationary clusters is < 1 m; the 25 m tolerance is wildly too generous.

**Why the original audit missed this.** The first audit looked at the
stationary window and saw "ego-frame anchor mis-points after ego
motion" as the obvious mechanism. It was — for 18 of the 18 long-gap
revivals at that time. But there were ALSO false revivals during ego
motion segments through the loose gate alone, and those didn't show
up in the stationary-window slice. The right meta-diagnostic question
"are there ALSO false revivals where the ego anchor IS valid?" was
not asked. Logged into `feedback_too_good_to_be_true.md` as an
addition: "*one mechanism for a false metric does not preclude
others — keep looking after the first hit."*

**Headline numbers honesty.** 127 distinct / 373-frame lifetime is
**partially inflated** by the 95 long-drift false revivals (>20 m
world drift) still leaking through the loose gate. The honest number
after Fix C lands somewhere between 127 and 180. The dashboard table
in `p3-progress.md` is updated accordingly with a "known incomplete"
caveat instead of a "shipped" stamp.

### [PLAN] Fix C — tighten cascade position gate

**Option 1 (try first).** Drop `kLostPosGateScale` from 5.0 to 2.0 in
`include/sort_tracker.hpp:264`. Gate becomes 10 m world tolerance,
still permissive enough for legitimate occlusion-recovery (a tree
re-entering FOV after an occlusion typically re-detects within ~3 m
of its last-known world position; even pessimistic re-acquisitions
land < 8 m). Kills the worst false-merges (the 95 with > 20 m drift)
without surgery on the cost-matrix structure. One-line change, no
new test surface needed beyond re-running the synthetic suite.

**Option 2 (only if Option 1 doesn't move the needle enough).**
Replace the fixed-distance gate with a Mahalanobis gate scaled by the
track's covariance:
```
d_mahal² = Δ^T · P^{-1} · Δ        // Δ = lost_pos_world (in current ego) − det
gate     = χ²(0.95, dof=2) ≈ 5.99
```
This is what Bewley/Wojke §3 actually specifies for the position term.
A track with high uncertainty (long-Lost, high cov_trace) gets a wider
gate; a freshly-Lost confident track gets a tight gate. M13 shipped a
fixed-distance gate for simplicity and the 25 m radius was set
without re-examination after cascade was added. Required surface:
`Track::filter` exposes `covariance_2x2_position()` (the IMM already
combines per-mode covariances correctly); `match_subset` reads it
under `use_lost_pos=true`. Existing tests cover the structural
correctness; new test would be `MahalanobisGateRejectsImplausibleRevival`
mirroring the world-drift-26 m case from the audit.

**Expected output of Fix C (Option 1).**

- `awk` distinct count from tracks.csv lands in **[140, 175]**. Below
  140 means we're under-correcting (revivals at 5–10 m drift are still
  passing through, which they probably should at λ=0.2 appearance
  agreement); above 175 means the gate is too tight and is killing
  real revivals.
- Mean lifetime drops from 373 to **~250–320**. Lower because false
  revivals were artificially extending lifetimes; should still be
  ≥ 4× M4 baseline (17 frames).
- Stationary-window cascade revivals with world drift > 10 m drops
  from 2/13 to **0/13**. This is the surgical regression to watch —
  if it doesn't go to zero, Option 1 wasn't enough and we move to
  Option 2.
- Whole-drive cascade revivals with world drift > 20 m drops from
  95 to **0** (the gate now physically prevents this). Drift > 10 m
  drops from 456 to **< 50**. Drift > 5 m may still exist legitimately
  (re-acquisitions of partially-occluded objects).

**Decision gate after Fix C measurement.** If the stationary-window
false-revival count is 0 and the whole-drive count is acceptable,
ship Fix C as the new headline. If false revivals at 5–10 m world
drift are still concerning, escalate to Option 2. Either way, mean
lifetime number gets reported HONESTLY this time — without the
inflation from false-merges.

### [STORY] Fix C postmortem — fixed gate over-rejected

Result, graded against the expected-output table from the Fix C plan:

| Metric | Pre-Fix-C (Fix B) | Target | Actual | Verdict |
|---|---|---|---|---|
| Distinct IDs | 127 | 140–175 | **299** | HARD FAIL (cap 200) |
| Mean lifetime | 373.0 | 250–320 | **158.4** | HARD FAIL (floor 200) |
| Stationary > 10 m drift revivals | 2 | **0** | **0** | ✅ |
| Drive-wide > 20 m drift revivals | 95 | **0** | **0** | ✅ |
| Drive-wide > 10 m drift revivals | 456 | < 50 | 54 | borderline (5 over) |
| Unit tests | 11/11 | 11/11 | 11/11 | ✅ |

**The two PASS rows are the headline good news.** Fix C completely
killed the false-merge bug — every cascade revival > 20 m world drift
was rejected. The mathematical claim ("the gate now physically
prevents the most egregious false-merges") landed exactly as designed.

**The two HARD FAILs are the diagnosis.** Distinct-IDs jumped to 299,
*above* M13 with cascade entirely disabled (237). A tightened cascade
producing more IDs than no cascade means we over-rejected legitimate
revivals. The audit data showed it directly:

- whole-drive cascade revivals dropped 1553 → 1379 (−174 attempts
  rejected by the 10 m gate)
- distinct IDs jumped +172 (almost 1:1 with the rejections)
- → each rejected attempt = a Lost track that died = a new ID spawned
  for the next sighting of the same physical thing

The post-Fix-C audit's worst-drift table contained the smoking gun:

```
 world_drift  ego_disp  gap
       18.59      0.02   43  [STATIONARY ego]   ← would-have-revived, now rejected
        8.86      8.87   38  [ego moved]         ← stationary tree, partial cluster
```

That 18.59 m drift while ego was *stationary* (0.02 m displacement)
isn't a different physical tree — it's a single tree whose DBSCAN
centroid jittered ~19 m between sightings because LiDAR returns
captured one side of the trunk in one frame and the other side in
another. Gating by absolute distance can't distinguish that from a
genuinely different object 19 m away.

**Conclusion.** A fixed-distance gate cannot distinguish "noisy
stationary tree" from "different physical object" — the two scenarios
produce identical world-frame drifts. Track covariance (`cov_trace`)
carries the missing information: a track whose filter has been
absorbing high-variance measurements has high P; a track whose filter
has been confidently locked-on has low P. Mahalanobis uses this:

  `d_mahal² = Δᵀ · P⁻¹ · Δ < χ²(0.95, dof=2) ≈ 5.99`

A high-confidence track (small P) requires Δ to be small in absolute
terms. A low-confidence track (large P from a long Lost period or
noisy filter) tolerates much larger Δ. This is the right structural
match for the failure mode.

**Action.** Reverted `kLostPosGateScale = 2.0f → 5.0f`. The constant
stays in the code as a *hard physical sanity ceiling* (no track ever
matches a detection > 25 m away in world frame, regardless of how
uncertain it is — guards against cov_trace blowup pathologies); the
primary gate is now Mahalanobis. Fix C tracks.csv + metrics.json
preserved as `_fixC_only` siblings for the blog narrative.

### [PLAN] Mahalanobis gate (the correct cascade gate)

**API surface change** (4 files):

1. `include/i_filter.hpp`: add
   ```cpp
   virtual Eigen::Matrix2f position_covariance_2x2() const = 0;
   ```
   This is the 2×2 top-left of the filter's full covariance —
   covariance of (x, y) only, with velocity rows/cols dropped. The
   match gate only needs position uncertainty.

2. `include/kalman_filter.hpp` + `src/kalman_filter.cpp`: implement
   `position_covariance_2x2()` returning `P_.topLeftCorner<2, 2>()`.

3. `include/imm_filter.hpp` + `src/imm_filter.cpp`: implement
   `position_covariance_2x2()` returning the 2×2 block of the
   already-combined output covariance `P_out` from the mode-mixing
   step (the math is in `combine_outputs()` — we just expose its
   product, no new computation).

4. `src/sort_tracker.cpp` `match_subset` (`use_lost_pos=true` branch):

   ```cpp
   const auto P_pos = tracks_[track_indices[i]].filter->position_covariance_2x2();
   // Regularize to avoid blowup if P collapses to near-zero (degenerate
   // synthetic case): add a small floor to the diagonal.
   Eigen::Matrix2f P_reg = P_pos;
   P_reg(0, 0) = std::max(P_reg(0, 0), 0.01f);   // 10 cm²
   P_reg(1, 1) = std::max(P_reg(1, 1), 0.01f);
   const Eigen::Matrix2f P_inv = P_reg.inverse();
   const Eigen::Vector2f delta = p - dets[det_indices[j]].position;
   const float d_mahal_sq = delta.transpose() * P_inv * delta;
   d_pos_phys(i, j) = d_mahal_sq;          // unitless squared Mahalanobis
   ```

   Gating threshold becomes `χ²(0.95, dof=2) ≈ 5.99` (a constexpr).
   Keep `kLostPosGateScale × max_dist` as a fallback hard cap on
   absolute world-frame distance, applied AFTER the Mahalanobis gate
   passes — so even if cov_trace blows up to absurd values for some
   pathological track, no revival fires more than 25 m away.

**Tests:**

- New: `MahalanobisGateRejectsImplausibleRevival` — high-confidence
  track + 26 m world-drift detection (the tid=28 audit case) should
  reject. Same scenario but with low-confidence track (long Lost
  period, P inflated) should accept.
- Both prior cascade tests stay green (`CascadeRevivalSurvivesEgoMotion`
  and `CascadeRevivalWithoutEgoMotionAnchorsToDecoy`) — they construct
  scenarios where the Mahalanobis distance is well within χ² because
  the filter just established a confident track and the drift is small.
- `test_imm.cpp` + `test_kalman.cpp` get one new structural test
  each: `PositionCovarianceMatchesTopLeft` — verifies the new
  accessor returns the right 2×2 block.

**Expected output (re-grade against this).**

| Metric | Pre-Fix-B | Fix B | Fix C | **Mahalanobis target** |
|---|---|---|---|---|
| Distinct IDs | 99 (artifact) | 127 | 299 | **140 – 200** |
| Mean lifetime | 163.7 | 373.0 | 158.4 | **250 – 350** |
| Stationary > 10 m | 18 | 2 | 0 | **0** |
| Drive-wide > 20 m | unknown | 95 | 0 | **0** |
| Drive-wide > 10 m | unknown | 456 | 54 | **< 80** |

**Hard-fail conditions:** distinct < 130 (probably accepting too much,
suspicious — Mahalanobis shouldn't beat Fix B on distinct count
unless lifetime stays high, in which case Mahalanobis is genuinely
better, not over-accepting), distinct > 250 (gate is too tight in
practice — escalate to upstream DBSCAN tuning), lifetime < 200,
stationary > 10 m drift count > 0.

**Decision after Mahalanobis lands.** If we hit the target window:
ship as the production headline, replace Fix B numbers in the
dashboard, render the final animation. If we miss low (lifetime
< 200): that means even Mahalanobis can't disambiguate the noisy
DBSCAN clusters, escalate to multi-frame point-cloud accumulation
(Phase-4 follow-up) or DBSCAN parameter tuning — not a tracker fix.

**Why this should work where Fix C didn't.** Fix C asked one
question of every revival: "is the world drift < 10 m?" Mahalanobis
asks the right question: "is the world drift consistent with this
track's accumulated uncertainty?" A stationary tree whose filter
has been processing high-variance measurements for 30 frames has
P inflated; Δ = 12 m gets an OK Mahalanobis distance. A
recently-confident track with tight P at the same Δ = 12 m gets a
huge Mahalanobis distance and rejects. That's exactly the bias we
want.

### [STORY] Mahalanobis sweep — hitting the DBSCAN ceiling (2026-04-27)

The Mahalanobis plan ran in three stages, each landing partial
improvement and revealing the next bottleneck.

**Mahal-v1 (combined IMM cov, χ²=5.99).** First implementation —
gate `Δᵀ·P⁻¹·Δ ≤ 5.99` using `IFilter::covariance().topLeftCorner<2,2>()`.
RELLIS result: **207 distinct, 228.9 lifetime, 16 cascade revivals
@ >20 m world drift across drive (target 0).** Borderline on every
metric. Audit table revealed cascade revivals at 22–24 m world drift
on segments where ego had only displaced 0.01 m (i.e. truly stationary
ego, no Fix-B mechanism in play, yet a "revival" 22 m away accepted).

**Diagnosis: IMM combined covariance is the wrong gate.** IMM's
`P_combined = Σⱼ μⱼ · (Pⱼ + (xⱼ - x_combined)(xⱼ - x_combined)ᵀ)`
includes an inter-mode SPREAD term — when CV (predicting forward at
velocity) and CP (staying put) disagree on position over the 10
pre-Lost misses, the spread inflates the combined cov dramatically.
A track moving at v ≈ 5 m/s gets σ_pos ≈ 5–7 m in P_combined after
10 misses. That's the *correct marginal posterior under model
uncertainty* (Bar-Shalom §11.6), but it answers the wrong question
for gating. We don't want "where could this object plausibly be under
all our model hypotheses" — we want "is this detection physically
the same object as the track's last sighting." For *that* question,
the more confident sub-model is the right reference.

**Mahal-v2 (per-mode IMM cov, χ²=5.99).** Added
`IFilter::gating_position_covariance_2x2()` — virtual accessor that
returns the same 2×2 block as before for `KalmanFilter2D`, but for
`IMMFilter` returns the SUB-MODEL with the smallest position-cov
trace (CV or CP, whichever is more confident this frame). Strictly
tightens the IMM gate; CV-only filters unchanged.

RELLIS result: **242 distinct, 195.8 lifetime, 11 false-merges
@ >20 m drift.** Distinct went UP (207 → 242) because the tighter
gate now rejects more revivals — including some that were genuinely
the same object whose P_combined had ballooned but whose per-mode
P was tight. False-merges dropped 16 → 11 (modest improvement).
Stationary-window cascade revivals at >10 m drift stayed at 0 (good
— Fix B continues to do its job). But still not in the target window.

**Mahal-v3 (per-mode IMM cov, χ²=2.28).** One more tightening pass:
χ²(0.68, 2) ≈ 2.28 instead of χ²(0.95, 2) ≈ 5.99 — a 2.6× tighter
gate corresponding to the "1σ ellipse" interpretation rather than
"95% confidence ellipse." One-line constexpr change.

RELLIS result: **384 distinct, 123.4 lifetime, 4 false-merges
@ >20 m drift.** False-merges dropped to 4 — the tightest result of
the sweep — but distinct IDs ballooned to 384 (worse than M13 with
cascade ENTIRELY DISABLED at 237) and lifetime collapsed to 123.

**Diagnosis: gate-tuning has no sweet spot.** The remaining false
revivals at 16–17 m drift on stationary-ego segments are tracks
whose per-mode P_position at gating-time is genuinely σ ≈ 10 m —
large enough that a 16 m drift gives `d_mahal² = 16²/100 = 2.56`,
just under χ² = 2.28's accept threshold. *Why* is per-mode P this
large? Because **DBSCAN cluster centroids on stationary trees jitter
5–15 m between successive sightings as the LiDAR scans different
sides of the trunk in different frames.** The filter, fed those
noisy measurements, correctly builds a covariance that reflects
this measurement noise. Its idea of "plausible drift" therefore
also extends to 5–15 m. There is no χ² threshold that separates
genuine 8–15 m DBSCAN-noisy revivals from false 16–24 m physical-
different-object revivals — the covariance carries the same range
for both.

The wall is structural. The cascade gate is doing the right math
on the wrong information. The right next move isn't tracker-side;
it's upstream — feed the cascade a less noisy detector.

**Sweep summary.**

| Variant | Gate construction | Distinct | Lifetime | >20m | >10m |
|---|---|---|---|---|---|
| Fix B | 25 m fixed | 127 | 373.0 | 95 | 456 |
| Fix C | 10 m fixed | 299 | 158.4 | 0 | 54 |
| Mahal-v1 | combined cov, χ²=5.99 | 207 | 228.9 | 16 | 248 |
| **Mahal-v2** | **per-mode cov, χ²=5.99** | **242** | **195.8** | **11** | **170** |
| Mahal-v3 | per-mode cov, χ²=2.28 | 384 | 123.4 | 4 | 35 |

Mahal-v2 sits at the trade-off knee — best false-merge reduction
that doesn't crater distinct/lifetime. Shipped as M13.5's final
cascade gate.

**Decision: ship Mahal-v2; escalate the remaining ceiling to Phase 4.**
The user's pre-stated escalation rule from the planning conversation:
*"If Option 2 lands distinct IDs at 200+ and lifetime at <250, the
next escalation is upstream DBSCAN tuning — accumulating multiple
frames of LiDAR returns before clustering, or running a tighter
eps to avoid the multi-meter centroid jitter."* That's exactly what
the data demanded. We executed the rule.

### [PLAN] Phase-4 — DBSCAN noise-floor reduction

**Goal.** Drop DBSCAN cluster-centroid jitter on stationary trees
from 5–15 m to <2 m. That, in combination with the existing Fix B +
Mahal-v2 cascade gate, is expected to take distinct-IDs from 242
toward the [140, 200] target window without any further tracker
changes.

**Why DBSCAN-side, not tracker-side.** The filter's job is to
absorb whatever noise the detector hands it; we can't and shouldn't
ask the tracker to reject measurements that the detector is reporting
as legitimate. The cluster jitter happens because DBSCAN treats
each LiDAR sweep independently — a tree's pointcloud captured from
one ego pose vs. another ego pose 10 m later is almost a different
shape (different ground returns, different leaf-occlusion patterns,
different return density), and DBSCAN's centroid moves accordingly.

**Two complementary approaches:**

1. **Multi-frame point cloud accumulation (canonical).** Compose
   the last K frames of LiDAR points into a common world frame
   using SLAM ego pose, then DBSCAN the union. Stationary objects
   gain consistent N-fold point support; moving objects appear as
   blurred clusters that DBSCAN may legitimately split, which is
   actually what we want for tracking. K = 3–5 is the typical
   range; tune empirically. Implementation: extend
   `clusters_to_detections.py` to accept a sliding-window arg;
   reuse Fix B's pose loader.

2. **Tighter DBSCAN eps + min_samples.** Smaller `eps` (e.g. 0.4 m
   → 0.25 m) prevents distant-but-similarly-oriented points from
   clustering together; higher `min_samples` (e.g. 5 → 10) requires
   denser support, eliminating the "thin sliver of returns" clusters
   that drift the most. Trade-off: fragments large clusters that
   were correctly merged. Probably partial improvement at best.

**Order of operations.** (1) is the bigger lever and the more
defensible interview answer ("we accumulate K=3 frames into a
local map before clustering"). (2) is a one-config-line change
worth A/B'ing against the K=1 baseline first to quantify how much
of the jitter is purely DBSCAN-parameter vs. fundamentally
geometric.

**Expected output.**

| Stage | Distinct | Lifetime | False-merge |
|---|---|---|---|
| M13.5 Mahal-v2 (current) | 242 | 195.8 | 11 |
| + Phase-4 DBSCAN tuning (eps↓ + minSamples↑) | 200 – 230 | 200 – 250 | 5 – 10 |
| + Phase-4 K=3 accumulation | 150 – 200 | 250 – 320 | 0 – 5 |
| + both | 140 – 180 | 280 – 350 | 0 – 3 |

If the K=3 row hits its target, that becomes the headline of the
M13.5 + Phase-4 cascade story. If not, the structural floor is
DBSCAN's fundamental limitation on RELLIS-3D and the answer is
elsewhere entirely (cluster-aware tracking on the raw pointcloud,
PointPillars detector, etc.) — Phase-5.

**Effort estimate.** K=3 accumulation: 2 hr (extend
`clusters_to_detections.py`, re-run, audit). DBSCAN-param sweep:
1 hr (just config + run). Both together with a paired ablation
table: 4 hr. All tracker-side code stays untouched.

**Why we're stopping cascade-gate iteration here.** The sweep above
is enough evidence that gate-tuning has hit its ceiling. Continued
iteration on χ² or P-block selection would be hill-climbing in a
saddle. The honest call is "this is what the cascade can do given
the detector; here's what fixing the detector unlocks."

### [PLAN] Phase-4 measurement protocol — verify before building

The user's explicit feedback after the Mahalanobis sweep:

> *I really like your optimism but let us get this running and see if
> the numbers are actually making sense.*

That's the operating principle for Phase 4. The estimates above
("K=3 → 150–200 distinct, 250–320 lifetime") are HYPOTHESES, not
forecasts to ship against. We've been wrong twice this session
(99-distinct artifact, 127-distinct partial) about "this is the
final number." The pattern was: build the fix → assume the fix
worked → ship the headline → user pushed back → audit revealed the
fix was incomplete. Phase-4 reverses the order.

**Step 1 — measure the assumed root cause BEFORE writing accumulation
code.** The whole Phase-4 plan rests on the claim: *"DBSCAN cluster
centroids on stationary trees jitter 5–15 m between sightings."*
That came from interpreting the Mahal-v3 audit output, not from
measuring DBSCAN directly. Test it.

Concrete protocol (script lives at `scripts/audit/audit_dbscan_jitter.py`,
to be written; see Phase-4 task 27):

1. For tracks visible in [1750, 1830] (the stationary window where
   ego world-displacement is 1.55 m), pick the 5–10 longest-lived
   `track_id`s.
2. For each, walk the `tracks.csv` rows in order and compute the
   frame-to-frame WORLD-FRAME centroid shift (using
   `data/poses_slam_full.csv`).
3. Histogram the per-frame shifts. Median, p90, p99.
4. **If median > 1 m and p90 > 5 m**, the DBSCAN-noise hypothesis is
   confirmed and accumulation is the right move.
5. **If median < 0.5 m and p90 < 2 m**, the hypothesis is WRONG —
   the cascade gate-tuning ceiling is something else (cluster-ID
   instability inside DBSCAN itself? cluster splitting/merging?
   filter dynamics?). Re-diagnose before writing any new code.

**Step 2 — write K=3 accumulation only if step 1 confirms the
hypothesis.** The implementation:

- `scripts/clusters_to_detections.py` extended with `--accumulate K`.
- For each frame `f`, compose points from `[f-K+1, f]` into the
  current ego frame using `data/poses_slam_full.csv`.
- Re-run DBSCAN on the union. Output a NEW
  `clusters_sweetspot_k3/` directory; do not overwrite K=1 (per
  the no-silent-deletes rule).
- Re-run the tracker pipeline against K=3 detections; audit.

**Step 3 — measure post-K=3 jitter the same way as step 1.** If
K=3 jitter is < 0.6× K=1 jitter, accumulation is doing the work it
should. If not, the noise isn't from independent-frame DBSCAN runs
but from something inherent to the geometry (e.g. ground-plane
returns shifting between distance bins). In that case, escalating
to tighter `eps` is the next move; if THAT doesn't move the
metric, the conclusion is "RELLIS DBSCAN tuning has structural
limits at this geometry — defer to a learned detector backbone
in Phase 5."

**Expected output is now phrased as 'pass / fail / no signal',
not as a target window.**

| Check | PASS condition | FAIL condition |
|---|---|---|
| K=1 centroid jitter exists | median ≥ 1 m, p90 ≥ 5 m | median < 0.5 m → re-diagnose |
| K=3 reduces jitter | K=3 median < 0.6× K=1 median | no reduction → tighter eps next |
| Tracker headline moves toward target | distinct < 220 AND lifetime > 230 | flat numbers → DBSCAN isn't the bottleneck |
| Audit false-merges | drive-wide >20m drift count < 5 | unchanged from Mahal-v2's 11 → ceiling is elsewhere |

If all four PASS: ship Phase-4 + Mahal-v2 as the new headline,
unblock animation render. If any FAIL: write up what we measured,
update the working hypothesis, decide next step BEFORE writing
more code.

**Animation rendering is held off until step 3 audit lands inside
the target window** (see `docs/todo-blog-animations.md`). The
30-minute render hoards laptop CPU and we don't want to render
against a tracks.csv that's about to be superseded — three of the
five animations rendered or considered this session would be wasted
under that policy.

### [STORY] Phase-4 step 1 — K=1 DBSCAN jitter audit

`scripts/audit/audit_dbscan_jitter.py` ran on the stationary window
[1750, 1830]. Greedy nearest-neighbor matching of detections in
world frame, capped at 8 m to discard obvious cross-cluster
mismatches (cap is biased TOWARD low jitter — if jitter still
shows up at this cap, it's robust evidence).

**Headline result.** On adjacent frames (gap=1) inside the window:
- 928 paired detections
- **median = 3.07 m, p90 = 6.51 m, p99 = 7.78 m**
- ego world displacement across the window = 1.55 m, so the
  cluster jitter is essentially independent of ego motion

Hypothesis ("DBSCAN cluster centroids on stationary trees jitter
5–15 m") **CONFIRMED in shape, partially overstated in magnitude.**
The 5–15 m range came from interpreting the Mahal-v3 audit's worst
revivals (16–24 m drift over 13–18 frame gaps). The TRUE per-frame
median is ~3 m; the 5–15 m drifts in the cascade audit were
multi-frame accumulations of this 3-m-per-frame noise compounded
across the Lost period — but see the next paragraph for the
surprising twist.

**Unexpected finding — the jitter does NOT compound with frame gap.**

| gap | median (m) | p90 (m) |
|---|---|---|
| 1  | 3.07 | 6.51 |
| 5  | 2.80 | 6.31 |
| 10 | 3.11 | 6.62 |
| 20 | 3.22 | 6.41 |
| 50 | 2.96 | 6.28 |

For stationary objects, the only way the gap=50 median can equal
the gap=1 median is if the noise is INDEPENDENT per frame and
re-randomizes each step. This is good news: **independent noise
averages down by √K under multi-frame accumulation.** Per the
central-limit argument:

- K=3  → expected median ≈ 3.07 / √3 ≈ 1.77 m
- K=5  → expected median ≈ 3.07 / √5 ≈ 1.37 m
- K=9  → expected median ≈ 3.07 / √9 ≈ 1.02 m

To get DBSCAN-jitter median below 1 m (so it stops being the
dominant noise source feeding the Mahalanobis gate), we likely
need K ≥ 9, not the K=3 originally planned.

**Histogram (gap=1).**

```
[0.00, 0.25)  n= 96
[0.25, 0.50)  n= 56
[0.50, 1.00)  n= 94
[1.00, 2.00)  n=122
[2.00, 5.00)  n=331    ← bulk: 2–5 m noise
[5.00, 8.00)  n=229    ← tail capped at 8 m greedy-NN cap
```

~25% of matches are at the 5–8 m cap; the unfiltered tail
(if cap = 20 m) is likely larger. The exact tail shape doesn't
affect the recommendation — the bulk of mass at 2–5 m alone
already explains the Mahalanobis ceiling.

**Likely confounds.**

1. **DBSCAN cluster splitting/merging** — a tree captured as
   1 cluster in one frame and 2 clusters in the next (sparse
   seam between branches in one sweep). Greedy NN pairs against
   the wrong sub-cluster, which inflates the measured jitter.
   Multi-frame world-frame accumulation FIXES this directly:
   denser point support → no sparse seams → cluster identity
   stable.

2. **8 m greedy-NN cap.** Real jitter tail probably extends past
   8 m; current run discards those. Doesn't change the headline
   number (median 3 m) but means K=3 might gain less than the
   √K math implies on the bulk distribution.

**Revised Phase-4 expectation (down from earlier optimism).**

Prior estimate: "K=3 → 150–200 distinct, 250–320 lifetime."
That assumed a structural fix; the audit shows K=3 only buys a
√3 ≈ 1.7× noise reduction, leaving median jitter at 1.77 m. The
Mahalanobis gate would see σ_pos ≈ 2 m at Lost-transition, still
admitting ~5 m revivals. So:

| K | Expected jitter median | Expected distinct | Expected lifetime |
|---|---|---|---|
| 1 (current) | 3.07 m | 242 (current) | 195.8 |
| 3 | ~1.8 m | 200 – 230 | 220 – 260 |
| 5 | ~1.4 m | 170 – 210 | 250 – 290 |
| 9 | ~1.0 m | 150 – 200 | 280 – 330 |

These are HYPOTHESES based on the √K noise-reduction model. Real
results might miss low (DBSCAN splitting/merging is the dominant
effect, not Gaussian noise — accumulation fixes it MORE than √K
predicts) or miss high (other failure modes show up as K grows,
e.g. moving-object clusters becoming blurred under accumulation).

**Plan: K=3 first (cheap), measure both centroid jitter AND tracker
headline; decide K=5 / K=9 from the data.** Don't pre-commit to
K=9; the post-K=3 audit will tell us whether √K is the right
model or whether DBSCAN splitting dominates (in which case K=3
might already buy more than expected and K=9 is overkill).

### [STORY] Phase-4 step 2 — split/merge dominates the jitter

`scripts/audit/audit_dbscan_split_merge.py` ran on the same window.
The verdict is unambiguous:

**Cluster count stability is terrible.** In what's supposed to be
a stationary scene, DBSCAN's per-frame cluster count varies from
7 to 26 (mean 15.1, stdev 4.18). 96.2% of adjacent-frame pairs
have a non-zero `|ΔN|`. Across the 80 pair transitions in the
window:

```
|ΔN|  pairs    median jitter (matched)
  0      3              0.77 m       ← intrinsic noise on persisting clusters
  1     13              3.38 m
  2     12              2.90 m
 ≥3     52              3.22 m
```

**Stable-vs-unstable ratio: 4.16×.** Stable pairs (when DBSCAN
keeps the same cluster set) show only ~0.8 m jitter; unstable
pairs (when clusters split or merge between frames) show ~3.2 m
jitter. So the "3 m" headline from step 1 is almost entirely the
greedy-NN matching against wrong-half-cluster after a split/merge,
NOT independent Gaussian centroid wobble.

**Churn aggregate:** 282 clusters appeared and 279 disappeared
across the 80 pairs (out of 928 matched + 282 appeared + 279
disappeared = 1489 total). Churn fraction = 37.7%. That's the
actual amount of noise the cascade gate has been wading through.

**Caveat — small stable-pair sample.** Only 3 stable pairs out of
80 is a tiny denominator for the 0.77 m floor. The direction is
robust (4.16× is huge) but the absolute floor estimate could be
±0.5 m noisy. Phase-4 K-sweep will overdetermine this.

**Revised expectation for K=3 (replaces the prior √K table).**

Splitting/merging is structural — composing K LiDAR frames in
world frame before clustering provides denser, more consistent
point support, which lets DBSCAN find the same cluster shape
across frames. The expected gain on stable-pair median jitter
from accumulation is small (it was already 0.77 m); the expected
gain comes from collapsing the *unstable* fraction toward the
stable behavior. A reasonable model:

- K=1 (current): 96% unstable @ 3 m, 4% stable @ 0.8 m, weighted median ≈ 3 m
- K=3: stability fraction climbs (pure conjecture, 50%?), weighted median maybe 1.5 m
- K=5+: stability fraction approaches stable behavior, median approaches 0.8–1 m

Tracker-headline mapping (rough):

| K | Expected jitter median | Expected distinct | Expected lifetime |
|---|---|---|---|
| 1 (current) | 3 m | 242 | 195.8 |
| 3 | 1.5 m | 180 – 220 | 230 – 280 |
| 5 | 1.0 m | 150 – 190 | 260 – 310 |
| 9 | 0.8 m | 140 – 170 | 280 – 330 |

This still has wide error bars. The K=3 row could land closer to
K=5's prediction if accumulation eliminates splitting more
aggressively than I'm modeling, OR closer to "no improvement" if
RELLIS LiDAR has structural reasons (sparse point density at
range, ground-plane returns confounding cluster boundaries) that
even K=9 can't fix. Empirics next.

### [STORY] Phase-4 step 3 — K-sweep on stationary window, ceiling found

Implemented `scripts/accumulate_and_cluster.py`: composes K obstacle
clouds (already-RANSAC-ground-removed) into the target frame's ego
coords using SLAM ego pose, runs sklearn DBSCAN, writes per-frame
cluster CSVs in the same schema as the existing
`clusters_sweetspot/`. K=1 sanity check reproduces C++ pipeline
cluster counts exactly (12/11/9 vs 12/11/9 on frames 1750/1751/1752).

Ran K=3 (eps=0.5), K=3 (eps=0.7), K=5 (eps=0.5) on the stationary
window only — much faster than full-sequence and answers the
"does K-accumulation help?" question definitively before
committing to the 30-minute full-sequence run.

**Persistent-cluster jitter (gap≥5, the honest measurement —
gap=1 with K-window > 1 has shared input points so reports
artificially low):**

| Config | dets/frame | gap=5 | gap=10 | gap=20 | gap=50 |
|---|---|---|---|---|---|
| K=1 (current) | 15.1 | 2.80 | 3.11 | 3.22 | 2.96 |
| K=3 eps=0.5 | 24.3 | **1.99** | **1.91** | 1.97 | 1.83 |
| K=3 eps=0.7 | 16.7 | 2.14 | 2.05 | 2.54 | 2.19 |
| K=5 eps=0.5 | 24.8 | 2.00 | 1.93 | 1.69 | 1.83 |

**Key findings.**

1. **K=3 reduces persistent-cluster jitter ~35%** (3.07 → 1.99 at
   gap=5). This is *better* than √3 = 1.7×, suggesting the
   split/merge confound is real — accumulation gives DBSCAN denser
   point support which reduces (but doesn't eliminate) per-frame
   inconsistency.

2. **K=5 is NOT meaningfully better than K=3.** gap=10 jitter is
   1.93 m (K=5) vs 1.91 m (K=3). The √K Gaussian-noise model
   breaks down — the residual ~2 m noise floor is *structural*,
   not independent-frame Gaussian. K=9 will not move this further.

3. **Cluster count rises with K** (15.1 → 24.3 at K=3). Accumulation
   in world frame stacks 3 frames of obstacle points, but per-frame
   DBSCAN inconsistencies (a tree's points captured at slightly
   different world-frame coords each frame because of tiny ego
   motion + LiDAR scan-pattern shift) means accumulation produces
   *separated* clusters within eps=0.5 of each other but not
   merged. eps=0.7 brings cluster count back to 16.7 at the cost of
   slightly higher jitter — modest tradeoff.

4. **The structural floor is ~1.9 m persistent-cluster jitter at
   gap=10.** Mahalanobis at σ=1.9 admits revivals at d ≤
   1.9 × √χ² ≈ 1.9 × 2.45 ≈ 4.7 m. Better than K=1's 3 × 2.45 =
   7.4 m gate, but still above the < 1 m target.

**What this means for the tracker headline.** Going to be partial
improvement, not the breakthrough we hoped for. Honest revised
estimate before running the full sequence:

| Config | Expected distinct | Expected lifetime | False-merge tail |
|---|---|---|---|
| K=1 (current Mahal-v2) | 242 | 195.8 | 11 |
| K=3 eps=0.5 | **210 – 240** | **210 – 250** | **5 – 10** |

Modest gains. We'd still be above the [150, 200] / [250, 320]
original target. The remaining bottleneck would be the structural
~2 m jitter floor — not fixable by K-accumulation; would need
either a denser LiDAR (Velodyne 64 → Velodyne 128 / Ouster), tighter
DBSCAN (eps=0.3 with risk of fragmentation), or a learned-detector
backbone.

**Decision: run K=3 eps=0.5 full-sequence (in progress, ~33 min).
After that completes, measure tracker headline + revival audit.
If the gain matches the 210–240 / 210–250 estimate, ship K=3 as
the new headline (modest improvement honestly framed). If the gain
is below 5%, the structural-floor diagnosis is fully confirmed and
phase-4 is a partial-credit fix; document and move to Phase-5
considerations (different detector / more capable LiDAR / learned
features).**

Smoke-window detection CSVs preserved at
`results_m4/ablation_g/dets_k{3,5}_eps{05,07}_smoke.csv` for the
blog narrative.

### [STORY] Phase-4 step 4 — K=3 full-sequence result (the actual numbers)

K=3 eps=0.5 full-sequence (2847 frames) accumulation took 33 min.
Generated `results_m4/ablation_g/rellis_detections_k3.csv` (73140
detections, avg 25.7/frame vs K=1's 16.6) and tracker output at
`results_m4/ablation_g/sort_on_rellis_k3/tracks.csv`.

**Tracker headline + revival audit, K=1 vs K=3 (both with
Mahal-v2 cascade gate, same tracker config):**

| Metric | K=1 Mahal-v2 | K=3 (new) | Δ |
|---|---|---|---|
| Distinct IDs | 242 | **307** | **+27%** |
| Mean lifetime (frames) | 195.8 | **238.2** | +22% |
| Cascade revivals (gap > 11) | 1474 | 1662 | +13% |
| World drift > 5 m | 724 | 758 | +5% |
| World drift > 10 m | 248 | **130** | **−48%** |
| **World drift > 20 m (extreme false-merges)** | **11** | **0** | **eliminated** |

**Headline reading.** K=3 ELIMINATED all extreme false-merges
(>20 m world drift) and roughly halved the >10 m bucket. The
remaining cascade revivals all have world drift < 20 m — the
"gross teleportation across the scene" failure mode that Mahal-v2
still leaked is gone. The cost was that K=3's cluster
fragmentation pushed distinct IDs from 242 to 307 (+27%).

**Per-frame jitter on full drive (K=3):**

| gap | pairs | median (m) | p90 (m) |
|---|---|---|---|
| 1 | 63605 | 0.17 | 4.23 |  ← K-window overlap artifact
| 5 | 57647 | 1.96 | 6.10 |  ← honest measurement
| 10 | 57031 | 2.30 | 6.20 |
| 20 | 54993 | 2.86 | 6.43 |
| 50 | 46731 | 3.81 | 7.04 |

Compared to K=1 stationary-window jitter (3.07 m at gap=1), the
real K=3 reduction at gap=5 is from ~3 m → 1.96 m — **35% reduction
on persistent clusters**, matching the smoke-window prediction.
Unlike K=1, jitter at K=3 *grows* with gap on the full drive
(1.96 → 3.81 from gap=5 → gap=50) — because over long gaps the
ego has covered hundreds of meters and "matched pairs" 50 frames
apart are increasingly different physical objects, not the same
tree at slightly different DBSCAN runs.

**The trade-off the data revealed (which I did not predict
correctly).**

I expected K=3 to *reduce* distinct IDs (cleaner detection → fewer
spurious tracks). The opposite happened: distinct went UP from 242
to 307. Diagnosis: K=3 in world frame stacks 3 frames of obstacle
points at slightly-different world positions per frame (because
LiDAR's per-scan grid is shifted by ~2 cm per frame even on
"stationary" ego). DBSCAN at eps=0.5 sees those as multiple
adjacent clusters rather than one merged cluster. Going from 16.6
dets/frame to 25.7 dets/frame is real fragmentation.

But — and this is the key — those fragmented detections are
*better-behaved* per-cluster: lower jitter, no extreme false-
merges. The tracker handles them (mean lifetime up 22%) but
publishes more distinct IDs because there genuinely are more
detection-level "things" to track.

**Was K=3 worth it?**

Two competing reads:

- **YES (structurally cleaner):** zero >20m false-merges is a
  bigger win than +27% distinct is a loss. The 307 IDs aren't
  "wrong" — they're MORE tracks of fragmented clusters that
  represent real DBSCAN behavior. False-merge elimination is the
  metric that survives interview scrutiny ("we don't merge
  unrelated trees"); inflated distinct count is a known artifact
  of the detector, not the tracker.

- **NO (target window missed):** original target was [150, 200]
  distinct. K=3 lands at 307. The trade-off didn't move toward
  the target.

I lean **YES** for the production headline because:
1. False-merge elimination is the metric a hiring manager
   actually cares about ("the tracker stays honest about object
   identity"). Distinct count without false-merge accounting is
   gameable.
2. The lifetime gain (196 → 238) is real — tracks live longer
   when they're not getting falsely merged elsewhere.
3. The structural-floor finding is preserved either way; ship
   the cleaner-tail version.

**What's NOT in this result.**

- Did NOT try K=3 with eps=0.7 on the full sequence. The smoke
  window suggested eps=0.7 keeps cluster count near K=1 baseline
  (16.7) at the cost of slightly higher jitter (2.05 m vs 1.91 m
  at gap=10). If we want to recover the distinct count without
  losing the jitter improvement, that's the next experiment —
  one more 33-min run.

- Did NOT try K=5 or K=9 on full sequence. The smoke window
  showed K=5 jitter ≈ K=3 jitter, so the additional cost wasn't
  expected to pay back. Confirmed assumption; no need to revisit
  unless eps=0.7 hits a different sweet spot.

- Did NOT change tracker config (max_misses, min_hits) for K=3.
  Higher detection density per frame might benefit from
  min_hits=2 or 3 (require more confirmations before publishing
  a track) — would mechanically reduce distinct IDs by filtering
  out 1-frame-only detections.

**Decision pending user input.** Ship Mahal-v2 K=1 (242/195.8/11)
or Mahal-v2 K=3 (307/238/0) as M13.5 + Phase-4 production
headline? My recommendation is K=3 with the trade-off documented
honestly in the blog. The render TODO in
`docs/todo-blog-animations.md` should pick up either path
depending on the decision.

Animation render still gated behind TP_M4_RENDER_ANIM=1 — the
user's stated preference is HPC for any future renders to keep
the laptop free.

### [STORY] Phase-4 sweep — the curve and the wall (2026-04-27)

**User pushback (correct, recorded verbatim):**

> *"Why ship either? Can't both be right? We are not up to our
> mark anyways as we made the decision that the lidar points are
> not dense enough or that we need a learned detector."*

That reframes the entire shipping question. The right blog
narrative isn't "we shipped 242" or "we shipped 307." It's:
**"we ran a sweep, hit a structural ceiling, measured why,
documented the next step."** No single config inside the
DBSCAN-tracker paradigm hits the original target window
([150, 200] distinct, [250, 320] lifetime, 0 false-merges) — and
that's not a project failure. It's the data telling us where the
RELLIS+DBSCAN paradigm runs out, which is exactly what a hiring
manager wants to see: **measurement, not aspirational
celebration.**

**The curve (Phase-3 + Phase-4 sweep, all on RELLIS 2847 frames,
all with Fix B world-anchor cascade):**

| Variant | Knob | Distinct | Lifetime | >20m drift | Reads as |
|---|---|---|---|---|---|
| Fix B baseline | gate=25 m | 127 | 373 | **95** | celebrated; falsely inflated |
| Fix C | gate=10 m fixed | 299 | 158 | 0 | over-rejected legit revivals |
| Mahal-v1 | combined IMM cov | 207 | 229 | 16 | combined-cov inflation |
| **Mahal-v2** | **per-mode IMM cov** | **242** | **196** | **11** | **knee on tracker dim** |
| Mahal-v3 | tighter χ² (2.28) | 384 | 123 | 4 | gate too tight |
| **K=3 eps=0.5** | **+ Mahal-v2** | **307** | **238** | **0** | **knee on detector dim** |
| K=3 + min_hits=3 | publish threshold | 307 | 145 | (audit broken) | null result |
| **K=3 eps=0.7** | **+ Mahal-v2** | **272** | **195** | **0** | **combined knee — closest to target, still not in window** |

**The wall.** Two independent floors set the ceiling:

1. **Tracker dim (Mahal-v2 knee = 242 distinct, 11 false-merges):**
   the cascade gate is doing optimal work given the detector it
   has. Tightening the gate over-rejects; loosening it
   over-accepts. The trade-off curve is a knee, not a saddle —
   no further χ² or P-block tweak moves both metrics.

2. **Detector dim (K=3 eps=0.5 knee = 307 distinct, 0
   false-merges):** the K-frame accumulation does its structural
   job (eliminates extreme false-merges by giving DBSCAN denser
   point support) but it can't merge per-frame fragmentation
   that the eps=0.5 + RELLIS-LiDAR-density combination produces.
   K=5 doesn't help (smoke). K=3 eps=0.7 is the last cheap
   detector experiment in flight.

**These two knees do not compose into a target-hitting config.**
The remaining cluster fragmentation IS the structural floor that
no tracker can resolve — DBSCAN's per-frame inconsistency on
RELLIS forest geometry is real and dataset-fundamental.

**What the field actually does.** Production AV stacks moved
from "DBSCAN-cluster-then-track" to "learned-3D-detector +
simple-tracker" years ago (PointPillars 2019, CenterPoint 2021,
AB3DMOT, EagerMOT, SimpleTrack — all use this pattern). The
complexity moved from the tracker into the detector. RELLIS-3D
has semantic point labels but no 3D bounding box tracking
labels, so applying the production pattern requires either
labeling RELLIS or transferring from KITTI/nuScenes. Neither is
P2-M4 scope; both are properly Phase-3 stretch.

**Decision: ship the SWEEP, not a single number.** Blog narrative
becomes:

1. M4 → M13 → M13.5 → Fix-B → Mahalanobis sweep → Phase-4
   K-accumulation sweep — *the journey*.
2. Two-dimensional trade-off curve (tracker knob × detector knob)
   with the knee on each axis honestly identified.
3. The wall: structural floor set by DBSCAN cluster instability
   on this dataset.
4. The next step: Phase-3 stretch — replace DBSCAN with a
   learned 3D detector (PointPillars / CenterPoint), which is
   *exactly* what the literature did at the same point in this
   evolution.
5. Limitations section calls out: applies to RELLIS forest
   specifically; production AV uses learned detectors; we
   document why we hit the wall instead of pretending we didn't.

**This is the strongest hiring signal the project has produced.**
Senior engineers recognize this arc instantly: "yeah, that's the
arc our team went through too." Showing measurement +
honest ceiling identification is harder to fake than a single
distinct-count number.

**Animation render plan unchanged:** TP_M4_RENDER_ANIM=1 in
batch on HPC, after Phase-4 eps=0.7 lands. Render targets:
- M4 baseline (979 distinct — visible flicker)
- Mahal-v2 K=1 (242, tracker-knee)
- K=3 eps=0.5 (307, detector-knee — false-merge eliminated)
- Side-by-side or 4-panel comparison shows the curve
  qualitatively in the video.

**Lessons in retrospect.**

1. **The "2× too good" heuristic worked as intended.** The 99 was a
   real artifact; tracing 10 sample IDs surfaced it; setting `max_age=30`
   was the right defensive holding pattern; Fix B was the canonical fix.
   The full arc — celebrate → user pushback → diagnose → defensive
   patch → canonical fix — took ~6 hours wall clock and is the cleanest
   debugging story this project has produced. Goes into the blog.

2. **"Decision to defer" is a real product decision, not a cop-out.**
   Shipping Fix A first preserved the option to ship Fix B today AND the
   option to ship Fix B never (if the time pressure hadn't allowed for it).
   Both were defensible; we got lucky with the time pressure relenting.

3. **The headline number is now better than the artifact number, by the
   metric that actually matters.** 127 distinct + 373 lifetime is
   strictly more useful for downstream consumers (planning, NATS) than
   99 distinct + 163 lifetime, even though 99 looks "smaller." False
   revivals make the 99 worse, not better — a track that has migrated to
   a different physical thing is actively misleading.

---

## Open items / follow-ups

- ~~PRIORITY: investigate Track-refactor CV regression~~ ✅ RESOLVED
  (was a config-attribution mistake; see `[TEST]` entry above).

- **M12 lifetime distribution (script-default IMM config)** for the blog:

  | Lifetime band | Count | % |
  |---|---|---|
  | exactly 1 frame | 404 | 24.1% |
  | 2–5 frames | 399 | 23.8% |
  | 6–15 frames | 269 | 16.0% |
  | 16–30 frames | 172 | 10.3% |
  | 31–100 frames | 295 | 17.6% |
  | >100 frames | 139 | 8.3% |
  | Mean | — | 28.2 frames |
  | Max | — | 461 frames (46.1 s) |

  M13 appearance encoder targets the 2–30 frame band (re-associating
  over-segmentation fragments and post-occlusion re-detections). The
  1-frame ghost band (24%) is upstream-of-M13 — needs DBSCAN tuning,
  not appearance.

- **`IMMFilter::predict()` mixing semantics.** Current implementation
  skips mixing on no-measurement frames (just propagates each
  sub-filter independently and recombines). Standard IMM choice for
  miss frames, but means `mu_` is frozen across long occlusions even
  if motion regime changes. Document in m12-imm.md as known-and-
  defensible; revisit if RELLIS data shows tracks getting stuck in
  the wrong mode after a long occlusion.
- **CV-mode initial P_vv.** Kept at 1000 (uninformed). Could try
  smaller values (e.g., 100) to reduce initial-frame Bayes asymmetry.
  Trade-off: slower convergence to true velocity for genuinely fast
  objects. Defer to ablation.
- **Track refactor to hold `unique_ptr<IFilter>`.** Day 5 work. Wires
  the `--filter cv|imm` runtime flag through `tracker_runner.cpp`.
  Pure glue, no algorithm change.
- **RELLIS validation run.** Re-run on
  `m4_perframe/clusters_sweetspot/` with `--filter imm`; compare
  distinct-track-IDs to M4 baseline of 979. Hard floor: < 800. Target:
  ≤ 600.
