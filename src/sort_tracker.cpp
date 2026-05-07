// sort_tracker.cpp
// SORT multi-object tracker — implementation.
//
// References:
//   Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016).
//   Wojke et al.,  "Simple Online and Realtime Tracking with a Deep
//                   Association Metric" (ICIP 2017) — appearance term in
//                   the cost matrix; this file's M13 integration mirrors
//                   §3 of that paper.
//
// Implementation notes:
//   - This file implements the per-frame state machine described in
//     sort_tracker.hpp's "Pipeline per call to update()".
//   - Track holds its filter via std::unique_ptr<IFilter> (not an inline
//     value), so SORTTracker can swap in a CV KF or an IMM at runtime.
//     New tracks are default-constructed, then `filter` is assigned a
//     freshly built filter from the make_filter() factory and immediately
//     init()-seeded at the detection position.
//   - Order of pruning vs new-track creation matters: prune AFTER matching
//     is complete and AFTER unmatched-track misses are incremented. New
//     tracks must be appended only AFTER pruning, otherwise their indices
//     in tracks_ would shift mid-iteration.
//   - P3-M13 adds an appearance term to the cost matrix and a per-track
//     embedding running-mean. Both are gated by `use_appearance_`; when
//     false, the file's behavior is bit-identical to the M12 path.

#include "sort_tracker.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>

#include "hungarian.hpp"
#include "imm_filter.hpp"

namespace tracker {

namespace {
// Factory: build the concrete filter for a new Track based on the
// SORTTracker's configured FilterKind. Centralizes the CV-vs-IMM dispatch
// so the new-track branch in update() doesn't need an if/else.
std::unique_ptr<IFilter> make_filter(FilterKind kind,
                                     float dt,
                                     float process_noise,
                                     float meas_noise) {
    switch (kind) {
        case FilterKind::CV:
            return std::make_unique<KalmanFilter2D>(dt, process_noise, meas_noise);
        case FilterKind::IMM:
            return std::make_unique<IMMFilter>(dt, process_noise, meas_noise);
    }
    return std::make_unique<KalmanFilter2D>(dt, process_noise, meas_noise);
}
}  // namespace

// =============================================================================
// Constructor — store configuration. No tracks exist yet.
// =============================================================================
SORTTracker::SORTTracker(float max_dist,
                         int max_misses,
                         int min_hits,
                         Solver solver,
                         Order order,
                         float dt,
                         float process_noise,
                         float meas_noise,
                         FilterKind filter_kind,
                         bool  use_appearance,
                         float appearance_weight,
                         float embedding_alpha,
                         int   max_age)
    : max_dist_(max_dist),
      max_misses_(max_misses),
      min_hits_(min_hits),
      solver_(solver),
      dt_(dt),
      process_noise_(process_noise),
      meas_noise_(meas_noise),
      order_(order),
      filter_kind_(filter_kind),
      use_appearance_(use_appearance),
      appearance_weight_(appearance_weight),
      embedding_alpha_(embedding_alpha),
      encoder_(),
      max_age_(max_age) {}

// =============================================================================
// match — build the NxM cost matrix and dispatch to the configured solver.
//
// Cost matrix layout:
//   rows = current live tracks (in tracks_ order)
//   cols = detections (in dets order)
//
// Pure-position case (M4/M12, use_appearance_ = false):
//   cost(i, j) = ‖predicted_pos(track_i) − det_pos(j)‖   (meters)
//   Solver gates by max_dist_ in meters. Same as before M13.
//
// Position+appearance case (P3-M13, use_appearance_ = true):
//   d_pos(i, j)  = ‖p_i − z_j‖ / max_dist_              ∈ [0, ~1]
//   d_emb(i, j)  = 1 − e_track_i · e_det_j              ∈ [0, 2]
//   cost(i, j)   = (1 − λ) · d_pos + λ · d_emb           ∈ [0, ~1.5]
//   Solver gating threshold is 1.0 (one normalized "track-radius") in
//   this case rather than max_dist_ — the cost is unitless after
//   normalization. We still post-filter on the underlying physical
//   distance d_pos · max_dist_ ≤ max_dist_ to retain Bewley's gate.
//
// Note: this is called AFTER predict() has been run on every track, so the
// "position" we read from each track's filter is the predicted (a-priori) one.
// =============================================================================
// =============================================================================
// match_subset — build cost matrix + solve for one stage of cascade matching.
// =============================================================================
std::vector<std::pair<int, int>> SORTTracker::match_subset(
    const std::vector<int>& track_indices,
    const std::vector<int>& det_indices,
    const std::vector<Detection>& dets,
    const std::vector<Embedding>& det_embeddings,
    bool use_lost_pos) {
    const int N = static_cast<int>(track_indices.size());
    const int M = static_cast<int>(det_indices.size());
    if (N == 0 || M == 0) return {};

    // Build per-pair physical position distance.
    //
    // Fix B: when use_lost_pos=true, the track's lost_pos_world is in the
    // WORLD frame, but detections live in the CURRENT ego frame. We project
    // the world-frame anchor back into current-ego with T_world_ego_.inverse()
    // before computing distance. T_ego_world is the same matrix per update()
    // call, so we compute it once.
    const Eigen::Isometry2f T_ego_world = T_world_ego_.inverse();

    Eigen::MatrixXf d_pos_phys(N, M);
    for (int i = 0; i < N; ++i) {
        const Track& t = tracks_[track_indices[i]];
        // Lost stage: project world-frame frozen position back into the
        //   current ego frame, then gate against detections.
        // Live stage: use filter->position() — already in current ego
        //   frame (the predicted a-priori position after Section 1).
        const Eigen::Vector2f p = use_lost_pos
                                      ? Eigen::Vector2f(T_ego_world * t.lost_pos_world)
                                      : t.filter->position();
        for (int j = 0; j < M; ++j) {
            d_pos_phys(i, j) = (p - dets[det_indices[j]].position).norm();
        }
    }

    // Mahalanobis gate (cascade / Lost stage only).
    //
    // After Fix C demonstrated that a fixed-distance gate cannot
    // disambiguate "DBSCAN-noisy stationary tree" from "different physical
    // object" — both produce identical world-frame drifts of 5–15 m on
    // RELLIS — we gate Lost-stage matches by the Mahalanobis distance
    // through each track's position covariance:
    //
    //   d_mahal² = Δᵀ · (P_pos)⁻¹ · Δ < χ²(0.95, dof=2) ≈ 5.99
    //
    // P_pos is the 2×2 top-left block of the filter's full covariance
    // (covariance of (x, y); velocity rows/cols dropped — the gate only
    // cares about position uncertainty). For IMM, this is the mode-mixed
    // P_combined the filter already computes.
    //
    // Confident track + small drift  → small d_mahal²  → accept
    // Confident track + 12 m drift   → huge d_mahal²   → reject
    // Long-Lost (P inflated) + 12 m  → moderate d_mahal² → accept
    //
    // The fixed-distance `pos_gate` below is retained as a HARD physical
    // ceiling above Mahalanobis — even if cov_trace blows up, no Lost
    // track ever revives more than `kLostPosGateScale × max_dist` away in
    // world frame. This guards against numerical pathology, not normal
    // operation.
    //
    // Live stage keeps the simple Euclidean gate — match_subset for Live
    // tracks runs every frame on small drift, the Mahalanobis machinery
    // is overkill there and would change M4/M12 baseline behavior.
    // χ² threshold for the Mahalanobis cascade gate (dof = 2).
    //
    // Sweep history (RELLIS, max_age=300, ego-poses on, IMM filter):
    //   χ²=5.99 + combined cov (Mahal v1) → 207/229, 16 @ >20m drift
    //   χ²=5.99 + per-mode cov (Mahal v2) → 242/196, 11 @ >20m drift  ← shipped
    //   χ²=2.28 + per-mode cov (Mahal v3) → 384/123,  4 @ >20m drift
    //
    // Mahal-v3 (tighter χ²) drove false-merges down further but distinct
    // IDs ballooned to 384 and lifetime collapsed — the gate was rejecting
    // legitimate DBSCAN-noisy revivals (centroid jitter of 5–15 m on
    // stationary trees as LiDAR scans different sides of the trunk).
    // Mahal-v2 sits at the trade-off knee; further tightening hits the
    // structural ceiling set by DBSCAN noise, not the tracker. The right
    // next move is upstream — multi-frame point cloud accumulation or
    // tighter DBSCAN eps — not gate-tuning. Phase-4 follow-up.
    //
    // χ²(0.95, 2) ≈ 5.99 — the standard "95% confidence ellipse" gate.
    constexpr float kChiSq2DoF95 = 5.99f;
    Eigen::MatrixXf d_mahal_sq;
    if (use_lost_pos) {
        d_mahal_sq.resize(N, M);
        for (int i = 0; i < N; ++i) {
            const Track& t = tracks_[track_indices[i]];
            // Use the gating-specific 2x2 position cov, not the combined
            // covariance(). For IMMFilter this returns the more confident
            // sub-model — see IFilter::gating_position_covariance_2x2() docs.
            const Eigen::Matrix2f P_pos = t.filter->gating_position_covariance_2x2();
            // Regularize: floor diagonal at (0.1 m)² so a freshly-init'd
            // filter with near-zero P doesn't yield infinite Mahalanobis
            // for any non-zero drift. 0.01 m² ≈ DBSCAN centroid noise on
            // a confident cluster.
            Eigen::Matrix2f P_reg = P_pos;
            P_reg(0, 0) = std::max(P_reg(0, 0), 0.01f);
            P_reg(1, 1) = std::max(P_reg(1, 1), 0.01f);
            const Eigen::Matrix2f P_inv = P_reg.inverse();
            const Eigen::Vector2f p =
                Eigen::Vector2f(T_ego_world * t.lost_pos_world);
            for (int j = 0; j < M; ++j) {
                const Eigen::Vector2f delta = p - dets[det_indices[j]].position;
                d_mahal_sq(i, j) = delta.transpose() * P_inv * delta;
            }
        }
    }

    // Position gate. Lost stage uses kLostPosGateScale × max_dist as the
    // HARD physical ceiling above Mahalanobis (see comment block above);
    // Live stage uses the standard max_dist.
    const float pos_gate = use_lost_pos
                               ? max_dist_ * kLostPosGateScale
                               : max_dist_;

    // Combined cost. When appearance is off and we're in the Lost stage,
    // we still allow the relaxed position match — that lets the
    // implementation degrade gracefully when no encoder is available
    // (max_misses=300 + max_age=100 still catches some rebirths).
    Eigen::MatrixXf cost(N, M);
    float solver_gate;
    if (!use_appearance_) {
        cost = d_pos_phys;
        solver_gate = pos_gate;
    } else {
        for (int i = 0; i < N; ++i) {
            const Embedding& e_t = tracks_[track_indices[i]].embedding;
            for (int j = 0; j < M; ++j) {
                const float dp = d_pos_phys(i, j) / pos_gate;
                const float de = 1.0f - e_t.dot(det_embeddings[det_indices[j]]);
                cost(i, j) = (1.0f - appearance_weight_) * dp
                           + appearance_weight_         * de;
            }
        }
        solver_gate = 1.0f;
    }

    auto raw_pairs = hungarian_solve(cost, solver_, solver_gate);

    // Three-step gate (Lost stage adds the Mahalanobis test):
    //   1. cost-matrix gate (already enforced by solver)
    //   2. Mahalanobis: d_mahal² ≤ χ²(0.95, dof=2) ≈ 5.99 — only when
    //      use_lost_pos. This is the primary admission gate for cascade
    //      revivals; it's the answer Fix C couldn't give with a fixed
    //      distance because it can't distinguish DBSCAN noise from
    //      different physical objects without using track confidence.
    //   3. physical-distance HARD CEILING: d_pos ≤ kLostPosGateScale ×
    //      max_dist. Guards against numerical pathology where cov_trace
    //      blows up to absurd values and Mahalanobis would let a track
    //      teleport across the scene.
    //
    // Live stage skips step 2 — it runs every frame, drift is small,
    // appearance does the disambiguation, and the M4/M12 unit tests
    // depend on the simple Euclidean cost matrix being unchanged.
    std::vector<std::pair<int, int>> gated;
    gated.reserve(raw_pairs.size());
    for (auto [li, lj] : raw_pairs) {
        if (cost(li, lj) > solver_gate) continue;
        if (d_pos_phys(li, lj) > pos_gate) continue;
        if (use_lost_pos && d_mahal_sq(li, lj) > kChiSq2DoF95) continue;
        gated.emplace_back(track_indices[li], det_indices[lj]);
    }
    return gated;
}

// =============================================================================
// match — two-stage cascade: Live tracks first, then Lost tracks on the
// remaining detections (P3.5).
// =============================================================================
std::vector<std::pair<int, int>> SORTTracker::match(
    const std::vector<Detection>& dets,
    const std::vector<Embedding>& det_embeddings) {
    const int N = static_cast<int>(tracks_.size());
    const int M = static_cast<int>(dets.size());
    if (N == 0 || M == 0) return {};

    // Partition tracks by state.
    std::vector<int> live_idx, lost_idx;
    live_idx.reserve(N);
    lost_idx.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (tracks_[i].state == TrackState::Live) live_idx.push_back(i);
        else                                       lost_idx.push_back(i);
    }

    // All det indices to start.
    std::vector<int> all_dets(M);
    for (int j = 0; j < M; ++j) all_dets[j] = j;

    // Stage 1: Live tracks against all detections. Standard cost matrix.
    auto live_pairs = match_subset(live_idx, all_dets, dets, det_embeddings,
                                   /*use_lost_pos=*/false);

    // Stage 2: Lost tracks against UNMATCHED detections only, with
    // relaxed position gating + appearance-weighted cost.
    std::vector<std::pair<int, int>> all_pairs = live_pairs;
    if (!lost_idx.empty()) {
        std::set<int> matched_dets;
        for (auto [_, dj] : live_pairs) matched_dets.insert(dj);
        std::vector<int> unmatched_dets;
        unmatched_dets.reserve(M - matched_dets.size());
        for (int j = 0; j < M; ++j) {
            if (matched_dets.count(j) == 0) unmatched_dets.push_back(j);
        }
        auto lost_pairs = match_subset(lost_idx, unmatched_dets,
                                       dets, det_embeddings,
                                       /*use_lost_pos=*/true);
        all_pairs.insert(all_pairs.end(), lost_pairs.begin(), lost_pairs.end());
    }
    return all_pairs;
}

// =============================================================================
// update_with_features (M13 primary — full Detection input).
//
// The Bewley pipeline, with the M13 appearance hooks marked YOUR CODE.
// Distinct name from update() to avoid brace-init overload ambiguity
// that broke compilation of `tracker.update({}, {})` style test calls.
// =============================================================================
std::vector<Track> SORTTracker::update_with_features(
    const std::vector<Detection>& detections,
    const std::vector<int>& class_ids,
    const Eigen::Isometry2f& T_world_ego) {
    // Fix B: stash the per-frame world←ego transform so match_subset() can
    // reach it without threading it through every helper signature. Reset
    // every frame; defaults to Identity for callers that don't supply a pose.
    T_world_ego_ = T_world_ego;

    // Compute per-detection embeddings once for this frame. Skipped when
    // appearance is off, so the M4/M12 path pays no cost.
    std::vector<Embedding> det_embeddings;
    if (use_appearance_) {
        det_embeddings.reserve(detections.size());
        for (const auto& d : detections) {
            det_embeddings.push_back(encoder_.encode(d.features));
        }
    }

    // ---- Section 1: PREDICT every LIVE track ----------------------------
    // Lost tracks have frozen position (lost_pos); we deliberately skip
    // their predict() so their position doesn't drift while waiting for
    // re-association. They re-enter the predict-update loop on revival.
    if (order_ == Order::PredictThenUpdate) {
        for (auto& t : tracks_) {
            if (t.state == TrackState::Live) t.filter->predict();
        }
    }

    // ---- Section 2: MATCH (two-stage cascade: Live first, then Lost) ---
    const auto pairs = match(detections, det_embeddings);

    // ---- Section 3: UPDATE each matched track. --------------------------
    // If the match was on a Lost track (cascade stage 2 hit), revive it:
    // re-init the filter at the matched position (the frozen filter has
    // stale state) and transition state Lost → Live. Track ID and
    // appearance embedding are preserved; this is exactly the cascade-
    // matching contribution — same physical thing, same ID, after a
    // potentially long occlusion gap.
    std::set<int> matched_tracks, matched_dets;
    for (auto [ti, dj] : pairs) {
        const auto& z = detections[dj].position;
        Track& t = tracks_[ti];
        if (t.state == TrackState::Lost) {
            // Revive: re-init the filter at the new measurement.
            t.filter->init(z.x(), z.y());
            t.state    = TrackState::Live;
            t.lost_age = 0;
        } else {
            t.filter->update(z.x(), z.y());
        }
        t.hits  += 1;
        t.misses = 0;

        // ─────────────────────────────────────────────────────────────
        // YOUR CODE — running-mean embedding update for this track.
        //
        // When appearance is on, blend the detection's embedding into
        // the track's stored embedding using exponential moving average:
        //    e_track ← (1−α) · e_track + α · e_det
        // then re-normalize to unit length so subsequent cosine-similarity
        // comparisons stay valid.
        //
        //   if (use_appearance_) {
        //       Embedding e = (1.0f - embedding_alpha_) * tracks_[ti].embedding
        //                   + embedding_alpha_         * det_embeddings[dj];
        //       const float n = e.norm();
        //       if (n > 1e-12f) e /= n;
        //       tracks_[ti].embedding = e;
        //   }
        //
        // α (Wojke 2017): default 0.1. Higher α = track follows the most
        // recent detection's appearance closely; lower α = track holds
        // its long-term mean (more robust to occlusion-time appearance
        // shift, slower to adapt to legitimate drift).
        // ─────────────────────────────────────────────────────────────
        if (use_appearance_) {
            Embedding e = (1.0f - embedding_alpha_) * tracks_[ti].embedding
                        + embedding_alpha_         * det_embeddings[dj];
            const float n = e.norm();
            if (n > 1e-12f) e /= n;
            tracks_[ti].embedding = e;
        }
        matched_tracks.insert(ti);
        matched_dets  .insert(dj);
    }

    // ---- Section 4: increment MISSES on unmatched tracks. ---------------
    // Live unmatched: bump misses. If misses exceeds max_misses, transition
    //   Live → Lost and freeze position at the current filter state.
    // Lost unmatched: bump lost_age (Lost tracks don't accumulate misses;
    //   they were already over the threshold).
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        if (matched_tracks.count(i) > 0) continue;
        Track& t = tracks_[i];
        if (t.state == TrackState::Live) {
            t.misses += 1;
            t.hits    = 0;
            if (t.misses > max_misses_) {
                t.state          = TrackState::Lost;
                // Fix B: store the freeze position in WORLD frame so it
                // remains physically meaningful after ego translates. The
                // filter's position() is in current ego frame; T_world_ego_
                // takes it to world.
                t.lost_pos_world = T_world_ego_ * t.filter->position();
                t.lost_age       = 0;
            }
        } else {
            // Already Lost — age it.
            t.lost_age += 1;
        }
    }

    // ---- Section 5: PRUNE final-erased tracks ---------------------------
    // Two erase conditions:
    //   (a) max_age_ == 0  → cascade matching DISABLED. Erase any track
    //       whose misses crossed max_misses (legacy P3-M13 behavior).
    //   (b) max_age_ > 0   → cascade matching ENABLED. Lost tracks stay
    //       in tracks_ until lost_age > max_age_, then erase. Live tracks
    //       never erase here (they erase via state transition + age).
    if (max_age_ > 0) {
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this](const Track& t) {
                    return t.state == TrackState::Lost && t.lost_age > max_age_;
                }),
            tracks_.end());
    } else {
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this](const Track& t) {
                    return t.state == TrackState::Lost;   // any Lost = prune
                }),
            tracks_.end());
    }

    // ---- Section 6: CREATE new tracks for unmatched detections. ---------
    for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
        if (matched_dets.count(j) > 0) continue;
        const int cls = (j < static_cast<int>(class_ids.size()))
                        ? class_ids[j]
                        : 0;
        Track t;
        t.id        = next_id_++;
        t.filter    = make_filter(filter_kind_, dt_, process_noise_, meas_noise_);
        t.hits      = 1;
        t.misses    = 0;
        t.class_id  = cls;
        t.z_3d      = 0.0f;
        t.filter->init(detections[j].position.x(), detections[j].position.y());

        // ─────────────────────────────────────────────────────────────
        // YOUR CODE — embedding init for a new track.
        //
        // Seed the new track's embedding with this detection's embedding.
        // The running-mean update in Section 3 won't run on this track
        // until next frame, so without this init the embedding stays at
        // its default (zero), and matching against this track via cosine
        // will return 1.0 (since dot of zero with anything is zero,
        // 1 - 0 = 1 → maximally dissimilar). That kills appearance-aided
        // matching for the entire first second of a new track's life.
        //
        //   if (use_appearance_) {
        //       t.embedding = det_embeddings[j];
        //   }
        // ─────────────────────────────────────────────────────────────
        if (use_appearance_) {
            t.embedding = det_embeddings[j];
        }

        tracks_.push_back(std::move(t));
    }

    // ---- Section 7: RETURN only PUBLISHABLE tracks (hits >= min_hits). --
    // Lost tracks are not published (they have stale state until revived);
    // only Live tracks reach downstream consumers.
    std::vector<Track> publishable;
    publishable.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.state == TrackState::Live && t.hits >= min_hits_) {
            publishable.push_back(t);
        }
    }

    // section 8 — UpdateThenPredict variant runs predict AFTER update.
    // Same Live-only restriction.
    if (order_ == Order::UpdateThenPredict) {
        for (auto& t : tracks_) {
            if (t.state == TrackState::Live) t.filter->predict();
        }
    }
    return publishable;
}

// =============================================================================
// update (M4-shim — Vector2f overload).
//
// Existing call sites (M4 unit tests, ros2_nodes/tracker_node.cpp, the
// non-appearance branch of tracker_runner) pass plain (x, y) detections.
// Build empty-feature Detections and forward to update_with_features().
// When use_appearance_ is false (the default), the empty features are
// never read.
// =============================================================================
std::vector<Track> SORTTracker::update(
    const std::vector<Eigen::Vector2f>& detections,
    const std::vector<int>& class_ids,
    const Eigen::Isometry2f& T_world_ego) {
    std::vector<Detection> dets;
    dets.reserve(detections.size());
    for (const auto& p : detections) {
        Detection d;
        d.position = p;
        d.features = FeatureVector::Zero();
        dets.push_back(std::move(d));
    }
    return update_with_features(dets, class_ids, T_world_ego);
}

}  // namespace tracker
