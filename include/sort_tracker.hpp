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
#include <vector>

#include "hungarian.hpp"   // for tracker::Solver
#include "kalman_filter.hpp"

namespace tracker {

// A single live track. id is unique per SORTTracker instance; persists across
// frames as long as the track is alive. class_id and z_3d are passthroughs
// that the tracker preserves but does not interpret (M5 will use them when
// fusing YOLO class labels and 3D-lifted bbox depths).
struct Track {
    int id;
    KalmanFilter2D kf;
    int hits;       // consecutive successful matches since creation
    int misses;     // consecutive frames with no matching detection
    int class_id;
    float z_3d;
};

class SORTTracker {
   public:
    SORTTracker(float max_dist,
                int max_misses,
                int min_hits,
                Solver solver = Solver::Greedy,
                float dt = 0.1f,
                float process_noise = 0.01f,
                float meas_noise = 0.1f);

    // Run one frame. detections[i] is the (x, y) of the i-th detection in
    // measurement frame; class_ids[i] is its class label (passthrough).
    // class_ids may be empty — in which case all tracks get class_id = 0.
    // Returns ONLY the tracks publishable this frame (hits >= min_hits).
    std::vector<Track> update(const std::vector<Eigen::Vector2f>& detections,
                              const std::vector<int>& class_ids);

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

    // Build cost matrix (rows = current tracks, cols = detections) and run
    // hungarian_solve. Returns (track_index, det_index) pairs.
    std::vector<std::pair<int, int>> match(const std::vector<Eigen::Vector2f>& dets);
};

}  // namespace tracker
