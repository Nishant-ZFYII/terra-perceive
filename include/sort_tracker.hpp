// sort_tracker.hpp
// SORT (Simple Online and Realtime Tracking) multi-object tracker.
//
// References:
//   Bewley, Ge, Ott, Ramos, Upcroft, "Simple Online and Realtime Tracking"
//     (ICIP 2016) — 4 pages. THE algorithm we implement here.
//   github.com/abewley/sort — Bewley's reference Python implementation. Read
//     AFTER yours works; do not crib structure from it before then.
//
// Pipeline per call to update():
//   1. predict() every existing track (Kalman propagates position + covariance).
//   2. Build NxM cost matrix of Euclidean distances:
//        cost(track_i, det_j) = ||predicted_pos(track_i) - det_pos(j)||
//   3. hungarian_solve(cost, solver_, max_dist_) → list of (track, det) pairs.
//      Pairs above max_dist_ are gated out by greedy; Munkres returns them
//      anyway and we filter post-hoc.
//   4. Matched pair → KF.update(det); ++hits; misses = 0.
//   5. Unmatched track  → ++misses; prune if misses > max_misses_.
//   6. Unmatched det    → create new Track, init KF at det, hits=1, misses=0.
//   7. Return only tracks with hits >= min_hits_ (false-positive suppression).
//
// Key parameters (set via constructor):
//   max_dist     — gating threshold in measurement units (meters for LiDAR,
//                  pixels for image-plane). Pairs above this get dropped.
//   max_misses   — how many consecutive misses before a track is deleted.
//                  Higher → robust to occlusion; too high → ghost tracks.
//   min_hits     — confirmation threshold. New tracks only become "published"
//                  after this many consecutive hits. Suppresses single-frame
//                  false positives.
//   solver       — Greedy or Munkres. Greedy is O(N*M) but order-dependent on
//                  ambiguous costs; Munkres is O(N^3) and globally optimal.
//   dt, process_noise, meas_noise — Kalman parameters baked into every NEW
//                  track this tracker creates. Existing tracks keep the values
//                  they were created with.

#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

#include "appearance_encoder.hpp"   // P3-M13: Embedding type alias (32-d)
#include "hungarian.hpp"            // for tracker::Solver
#include "i_filter.hpp"
#include "kalman_filter.hpp"

namespace tracker {
enum class Order { PredictThenUpdate, UpdateThenPredict };

// Which concrete filter to instantiate for new tracks. P3-M12 added IMM as
// an alternative to the M4 baseline CV Kalman filter. Switchable at runtime
// via SORTTracker's constructor (and tracker_runner's --filter flag).
enum class FilterKind { CV, IMM };

// Phase-3.5 cascade matching states.
//
//   Live: track has matched within the last `max_misses` frames. Filter
//         is predicted/updated normally. Eligible in the primary
//         match stage and can be revived on subsequent matches.
//   Lost: track exceeded `max_misses` consecutive misses. Filter is FROZEN
//         (no predict()) so the position doesn't drift; the track waits
//         in a retired pool for up to `max_age` more frames trying to be
//         re-associated by appearance in the cascade stage. Revives to
//         Live if matched. Final-erased after `max_age` frames in Lost.
//
// When `use_appearance_` is false, cascade matching is a no-op — Lost
// tracks are eligible in the primary stage with their frozen position
// only. This keeps the M4/M12 path bit-identical when appearance is off.
enum class TrackState { Live, Lost };

// Per-frame detection input. P3-M13 extends the M4 (x, y) point with
// optional appearance features. When appearance is disabled (M4/M12 path),
// `features` is ignored — the cost matrix uses position alone.
struct Detection {
    Eigen::Vector2f position;
    FeatureVector   features;        // 8-dim hand-crafted geometric features
                                     // (see include/appearance_encoder.hpp)
                                     // unused when use_appearance_ = false
};

// A single live track. id is unique per SORTTracker instance; persists across
// frames as long as the track is alive. class_id and z_3d are passthroughs
// that the tracker preserves but does not interpret (M5 will use them when
// fusing YOLO class labels and 3D-lifted bbox depths).
//
// Track holds its filter via unique_ptr<IFilter> so SORTTracker can swap in
// a CV KF or an IMM at runtime without templating. Track is copyable: the
// copy ctor/assignment use IFilter::clone() to deep-copy the polymorphic
// filter. This preserves the M4 API where update() returns std::vector<Track>
// by value.
struct Track {
    int id;
    std::unique_ptr<IFilter> filter;
    int hits;       // consecutive successful matches since creation
    int misses;     // consecutive frames with no matching detection
    int class_id;
    float z_3d;

    // P3-M13 appearance: 32-dim unit-norm embedding maintained as the
    // running mean of all detection embeddings ever matched to this track.
    // On creation: copy the first detection's embedding. On match: update
    // via e_track ← (1−α)·e_track + α·e_det, then re-normalize. When
    // use_appearance_ is false, this stays at its default (zero) — no
    // updates run, no matching cost contribution.
    Embedding embedding = Embedding::Zero();

    // Phase-3.5 cascade matching state.
    //
    // lost_pos_world is the track's position captured at the Live→Lost
    // transition, expressed in the WORLD frame (Fix B). On a cascade
    // re-association attempt the tracker transforms it back into the
    // CURRENT ego frame using the runtime T_world_ego, so a re-detection
    // is gated against where the physical object ACTUALLY is now — not
    // where it was relative to a previous ego pose. Pre-Fix-B this field
    // was stored in ego frame and the same coordinate pointed at totally
    // different world locations once ego had translated, silently merging
    // unrelated stationary trees. See docs/m10-debug-log.md
    // "False revivals — cascade matching's ego-frame bug".
    TrackState      state          = TrackState::Live;
    int             lost_age       = 0;                       // frames since entering Lost
    Eigen::Vector2f lost_pos_world = Eigen::Vector2f::Zero(); // frozen WORLD-frame position at Lost transition

    Track() = default;
    Track(Track&&) noexcept = default;
    Track& operator=(Track&&) noexcept = default;

    Track(const Track& o)
        : id(o.id),
          filter(o.filter ? o.filter->clone() : nullptr),
          hits(o.hits),
          misses(o.misses),
          class_id(o.class_id),
          z_3d(o.z_3d),
          embedding(o.embedding),
          state(o.state),
          lost_age(o.lost_age),
          lost_pos_world(o.lost_pos_world) {}

    Track& operator=(const Track& o) {
        if (this != &o) {
            id = o.id;
            filter = o.filter ? o.filter->clone() : nullptr;
            hits = o.hits;
            misses = o.misses;
            class_id = o.class_id;
            z_3d = o.z_3d;
            embedding = o.embedding;
            state = o.state;
            lost_age = o.lost_age;
            lost_pos_world = o.lost_pos_world;
        }
        return *this;
    }

    // Convenience accessors — delegate to the filter so call sites in
    // tracker_runner don't have to dereference filter and don't need to
    // know the concrete filter type.
    Eigen::Vector2f position() const { return filter->position(); }
    Eigen::Vector2f velocity() const { return filter->velocity(); }
    float covariance_trace() const { return filter->covariance_trace(); }
};

class SORTTracker {
   public:
    SORTTracker(float max_dist,
                int max_misses,
                int min_hits,
                Solver solver = Solver::Greedy,
                Order order = Order::PredictThenUpdate,
                float dt = 0.1f,
                float process_noise = 0.01f,
                float meas_noise = 0.1f,
                FilterKind filter_kind = FilterKind::CV,
                // P3-M13 appearance knobs.  When use_appearance=false (the
                // default, M4/M12 path), the encoder is never called and
                // the cost matrix is pure-position. When true, the cost
                // matrix becomes a convex combination of position and
                // appearance distance, weighted by appearance_weight.
                bool  use_appearance     = false,
                float appearance_weight  = 0.4f,    // λ in (1-λ)·d_pos + λ·d_emb
                float embedding_alpha    = 0.1f,    // running-mean rate per match
                // Phase-3.5 cascade matching:
                //   max_age = additional frames a Lost track stays in the
                //             retired pool waiting to be revived. 0 disables
                //             cascade (M13 behavior: tracks erase at
                //             misses > max_misses). Recommended ≥ 100 when
                //             use_appearance=true; no-op when false.
                int   max_age            = 0);

    // Run one frame. M4/M12 entry point — accepts (x, y) detections and
    // does NOT compute appearance. Internally this constructs a
    // Detection per (x, y) with zero features and forwards.
    //
    // T_world_ego (Fix B): rigid 2D transform from the current ego frame
    // (where detections live) to the world frame. Used by cascade matching
    // to anchor Lost-track positions in world coordinates so they survive
    // ego motion. Defaults to Identity, which makes update() bit-identical
    // to the pre-Fix-B implementation — synthetic unit tests and any
    // caller that doesn't have an ego pose stays correct.
    std::vector<Track> update(
        const std::vector<Eigen::Vector2f>& detections,
        const std::vector<int>& class_ids,
        const Eigen::Isometry2f& T_world_ego = Eigen::Isometry2f::Identity());

    // P3-M13 entry point — accepts full Detections (position + features).
    // Distinct name (not just an overload of update()) to avoid the
    // brace-init ambiguity that breaks `update({}, {})` test calls and
    // the M4-shim above. When use_appearance_ is true, the embedded
    // features are passed to encoder_.encode() to produce per-detection
    // embeddings, and the cost matrix uses both position and appearance.
    //
    // T_world_ego (Fix B): see update() above.
    std::vector<Track> update_with_features(
        const std::vector<Detection>& detections,
        const std::vector<int>& class_ids,
        const Eigen::Isometry2f& T_world_ego = Eigen::Isometry2f::Identity());

    // Read-only access to the full live-track set (including unconfirmed).
    // Useful for tests and for the runner's debug log; production callers
    // should use the return value of update() instead.
    const std::vector<Track>& tracks() const { return tracks_; }

   private:
    std::vector<Track> tracks_;
    int next_id_ = 0;

    // Configuration (baked in at construction time).
    float max_dist_;
    int max_misses_;
    int min_hits_;
    Solver solver_;
    float dt_;
    float process_noise_;
    float meas_noise_;
    Order order_;
    FilterKind filter_kind_;

    // P3-M13 appearance.
    bool  use_appearance_;
    float appearance_weight_;          // λ
    float embedding_alpha_;            // α
    AppearanceEncoder encoder_;        // weights compiled in at build time

    // Phase-3.5 cascade matching.
    int   max_age_;                    // Lost track lifetime budget after going Lost

    // Fix B: rigid transform from the CURRENT frame's ego coordinates to
    // the world frame, captured at the start of every update_with_features
    // call. match_subset() reads this when use_lost_pos=true to project
    // each Lost track's world-frame lost_pos back into ego coordinates
    // before gating against detections (which live in current-ego frame).
    // Defaults to Identity → ego-frame-only behavior (legacy / unit tests).
    Eigen::Isometry2f T_world_ego_ = Eigen::Isometry2f::Identity();

    // Build cost matrix (rows = current tracks, cols = detections) and run
    // hungarian_solve. Returns (track_index, det_index) pairs.
    //
    // Phase-3.5 two-stage cascade:
    //   Stage 1: Live tracks against ALL detections, normal cost matrix.
    //   Stage 2: Lost tracks against UNMATCHED detections (those not paired
    //            in stage 1), with relaxed position gating because the
    //            Lost track's position was frozen at the Lost transition
    //            and is stale. Appearance dominates the matching here.
    //
    // det_embeddings: per-detection unit-norm embeddings (size = dets.size()
    // when use_appearance_ is true; empty otherwise). Lifetime-bound to the
    // call — match() does not store them.
    std::vector<std::pair<int, int>> match(
        const std::vector<Detection>& dets,
        const std::vector<Embedding>& det_embeddings);

    // Helper: build the cost matrix and run hungarian_solve for a SUBSET
    // of tracks against a subset of detections. Used twice in match() for
    // the Live and Lost stages of cascade matching.
    //
    // track_indices / det_indices : indices into tracks_ / dets respectively
    // use_lost_pos                : if true, read each track's `lost_pos`
    //                               instead of `filter->position()` (Lost
    //                               tracks have frozen positions). Also
    //                               relaxes the position-distance gate by
    //                               kLostPosGateScale.
    // Returns: list of (track_global_index, det_global_index) pairs that
    //          passed both the solver's gate and the physical-distance gate.
    std::vector<std::pair<int, int>> match_subset(
        const std::vector<int>& track_indices,
        const std::vector<int>& det_indices,
        const std::vector<Detection>& dets,
        const std::vector<Embedding>& det_embeddings,
        bool use_lost_pos);

    // Position-gate HARD CAP for Lost-stage matching. After Mahalanobis
    // gating (see match_subset), this acts as a physical sanity ceiling
    // on world-frame drift — Mahalanobis can match a high-uncertainty
    // track to anything within its covariance ellipsoid, but no track
    // should ever match a detection more than `kLostPosGateScale ×
    // max_dist` away in world frame regardless of how uncertain it is.
    // Set to 5.0 (= 25 m at max_dist=5), the same value Fix B shipped
    // with — Mahalanobis does the per-track-confidence cutting; this is
    // just the floor against pathological cov_trace blowup.
    //
    // Fix C history (kept for narrative). Fix C tried tightening this
    // constant 5.0 → 2.0 (= 10 m hard gate). On RELLIS that produced
    // 299 distinct / 158-frame mean lifetime — worse than M13 with
    // cascade entirely disabled. Diagnosis: DBSCAN's cluster-centroid
    // jitter on stationary trees regularly drifts 5–15 m between
    // sightings (partial-tree returns vs whole-tree returns), and a
    // fixed 10 m gate rejects most of those legitimate re-acquisitions
    // while keeping the actually-false ones at 12–18 m drift. Conclusion:
    // a fixed-distance gate cannot distinguish "noisy stationary tree"
    // from "different physical object." Track covariance carries that
    // information; Mahalanobis uses it. See docs/m10-debug-log.md
    // "Fix C postmortem" + "Mahalanobis gate" entries.
    static constexpr float kLostPosGateScale = 5.0f;
};

}  // namespace tracker
