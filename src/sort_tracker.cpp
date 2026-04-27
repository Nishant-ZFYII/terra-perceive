// sort_tracker.cpp
// SORT multi-object tracker — implementation.
//
// References:
//   Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016).
//
// Implementation notes:
//   - This file implements the per-frame state machine described in
//     sort_tracker.hpp's "Pipeline per call to update()".
//   - The Track struct embeds a KalmanFilter2D with no default constructor.
//     New tracks are brace-initialized with a freshly built KF, then
//     immediately .init()-seeded at the detection position.
//   - Order of pruning vs new-track creation matters: prune AFTER matching
//     is complete and AFTER unmatched-track misses are incremented. New
//     tracks must be appended only AFTER pruning, otherwise their indices
//     in tracks_ would shift mid-iteration.

#include "sort_tracker.hpp"

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
    // Unreachable; silences -Wreturn-type on some compilers.
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
                         FilterKind filter_kind)
    : max_dist_(max_dist),
      max_misses_(max_misses),
      min_hits_(min_hits),
      solver_(solver),
      dt_(dt),
      process_noise_(process_noise),
      meas_noise_(meas_noise),
      order_(order),
      filter_kind_(filter_kind) {}

// =============================================================================
// match — build the NxM cost matrix and dispatch to the configured solver.
//
// rows of the cost matrix = current live tracks (in tracks_ order)
// cols = detections (in dets order)
// cost(i, j) = euclidean distance from predicted position of track_i to det_j
//
// Note: this is called AFTER predict() has been run on every track, so the
// "position" we read from each track's KF is the predicted (a-priori) one.
// =============================================================================
std::vector<std::pair<int, int>> SORTTracker::match(
    const std::vector<Eigen::Vector2f>& dets) {
    // YOUR CODE:
    //   1. const int N = static_cast<int>(tracks_.size());
    //      const int M = static_cast<int>(dets.size());
    //      if (N == 0 || M == 0) return {};
    //   2. Build cost matrix:
    //        Eigen::MatrixXf cost(N, M);
    //        for (int i = 0; i < N; ++i) {
    //            const Eigen::Vector2f p = tracks_[i].kf.position();
    //            for (int j = 0; j < M; ++j) {
    //                cost(i, j) = (p - dets[j]).norm();
    //            }
    //        }
    //   3. return hungarian_solve(cost, solver_, max_dist_);
    //
    // The dispatcher handles greedy vs Munkres internally — match() does not
    // care which solver is in use. That isolation is the entire reason the
    // Solver enum exists.
    const int N = static_cast<int>(tracks_.size());
    const int M = static_cast<int>(dets.size());
    if (N == 0 || M == 0) return {};
    Eigen::MatrixXf cost(N, M);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector2f p = tracks_[i].filter->position();
        for (int j = 0; j < M; ++j) {
            cost(i, j) = (p - dets[j]).norm();
        }
    }
    auto pairs = hungarian_solve(cost, solver_, max_dist_);
    std::vector<std::pair<int, int>> gated;
    gated.reserve(pairs.size());
    for (auto [ti, dj] : pairs) {
        if (cost(ti, dj) <= max_dist_){
            gated.push_back({ti, dj});
        }
    }
    return gated;
}

// =============================================================================
// update — one frame of the tracker.
//
// Bewley's pipeline, broken into named sections you can step through with a
// debugger. Mind the iteration-order traps marked with ⚠.
// =============================================================================
std::vector<Track> SORTTracker::update(
    const std::vector<Eigen::Vector2f>& detections,
    const std::vector<int>& class_ids) {
    // ---- Section 1: PREDICT every existing track. -----------------------
    //
    // YOUR CODE:
    //   for (auto& t : tracks_) t.kf.predict();
    //
    // After this, t.kf.position() is the a-priori (predicted) position used
    // by match() to build the cost matrix.

    // ---- Section 2: MATCH (build cost matrix, call solver). --------------
    //
    // YOUR CODE:
    //   const auto pairs = match(detections);
    //
    // pairs is a list of (track_idx, det_idx). Both indices reference the
    // CURRENT tracks_ vector and the detections argument respectively.

    // ---- Section 3: UPDATE each matched track. --------------------------
    //
    // We need to know which tracks and which detections were matched, to
    // figure out who's "unmatched" in sections 4 and 5.
    //
    // YOUR CODE:
    //   std::set<int> matched_tracks, matched_dets;
    //   for (auto [ti, dj] : pairs) {
    //       const auto& z = detections[dj];
    //       tracks_[ti].kf.update(z.x(), z.y());
    //       tracks_[ti].hits  += 1;
    //       tracks_[ti].misses = 0;
    //       matched_tracks.insert(ti);
    //       matched_dets  .insert(dj);
    //   }

    // ---- Section 4: increment MISSES on unmatched tracks. ---------------
    //
    // YOUR CODE:
    //   for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
    //       if (matched_tracks.count(i) == 0) {
    //           tracks_[i].misses += 1;
    //           tracks_[i].hits    = 0;   // optional: reset hits on miss.
    //                                     // SORT spec is ambiguous; the
    //                                     // reference impl resets. Pick one
    //                                     // and document in the blog.
    //       }
    //   }

    // ---- Section 5: PRUNE tracks with too many misses. ------------------
    //
    // ⚠ Erase-remove idiom — std::erase_if (C++20) is cleanest. If you're on
    // C++17, use the std::remove_if + tracks_.erase combo.
    //
    // YOUR CODE (C++20 form):
    //   std::erase_if(tracks_, [this](const Track& t) {
    //       return t.misses > max_misses_;
    //   });
    //
    // YOUR CODE (C++17 form):
    //   tracks_.erase(
    //       std::remove_if(tracks_.begin(), tracks_.end(),
    //                      [this](const Track& t) { return t.misses > max_misses_; }),
    //       tracks_.end());

    // ---- Section 6: CREATE new tracks for unmatched detections. ---------
    //
    // ⚠ Append AFTER pruning (section 5). If you append before, the indices
    // in matched_tracks become invalid because section-5 erase shifts later
    // entries down.
    //
    // YOUR CODE:
    //   for (int j = 0; j < static_cast<int>(detections.size()); ++j) {
    //       if (matched_dets.count(j) > 0) continue;
    //       const int cls = (j < static_cast<int>(class_ids.size()))
    //                       ? class_ids[j]
    //                       : 0;
    //       Track t{
    //           /*id=*/      next_id_++,
    //           /*kf=*/      KalmanFilter2D(dt_, process_noise_, meas_noise_),
    //           /*hits=*/    1,
    //           /*misses=*/  0,
    //           /*class_id=*/cls,
    //           /*z_3d=*/    0.0f,
    //       };
    //       t.kf.init(detections[j].x(), detections[j].y());
    //       tracks_.push_back(std::move(t));
    //   }

    // ---- Section 7: RETURN only PUBLISHABLE tracks (hits >= min_hits). --
    //
    // The full tracks_ vector is the tracker's internal state and is kept
    // alive. The return value is the filtered "for downstream consumption"
    // view. tracker_runner will log both — internal state for debugging,
    // returned vector for metrics.
    //
    // YOUR CODE:
    //   std::vector<Track> publishable;
    //   publishable.reserve(tracks_.size());
    //   for (const auto& t : tracks_) {
    //       if (t.hits >= min_hits_) publishable.push_back(t);
    //   }
    //   return publishable;
    
    // ---- Section 1: PREDICT every existing track. -----------------------
    if (order_ == Order::PredictThenUpdate) {
        for (auto& t : tracks_) t.filter->predict();
    }

    // ---- Section 2: MATCH (build cost matrix, call solver). --------------
    const auto pairs = match(detections);
    // ---- Section 3: UPDATE each matched track. --------------------------
    std::set<int> matched_tracks, matched_dets;
    for (auto [ti, dj] : pairs) {
        const auto& z = detections[dj];
        tracks_[ti].filter->update(z.x(), z.y());
        tracks_[ti].hits  += 1;
        tracks_[ti].misses = 0;
        matched_tracks.insert(ti);
        matched_dets  .insert(dj);
    }
    // ---- Section 4: increment MISSES on unmatched tracks. ---------------
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        if (matched_tracks.count(i) == 0) {
            tracks_[i].misses += 1;
            tracks_[i].hits    = 0;   // optional: reset hits on miss.
                                      // SORT spec is ambiguous; the reference impl resets. Pick one
                                      // and document in the blog.  
        }
    }
    // ---- Section 5: PRUNE tracks with too many misses. ------------------
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) { return t.misses > max_misses_; }),
        tracks_.end());
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
        t.filter->init(detections[j].x(), detections[j].y());
        tracks_.push_back(std::move(t));
    }
    // ---- Section 7: RETURN only PUBLISHABLE tracks (hits >= min_hits). --
    std::vector<Track> publishable;
    publishable.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.hits >= min_hits_) publishable.push_back(t);    
    }

    //section 8 only if order::UpdateThenPredict
    if (order_ == Order::UpdateThenPredict) {
        for (auto& t : tracks_) t.filter->predict();
    }
    return publishable; 
}

}  // namespace tracker
