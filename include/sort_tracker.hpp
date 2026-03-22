// sort_tracker.hpp
// SORT multi-object tracker: Kalman + Hungarian — implement from scratch.
//
// PDF reference: Part 5.3 (page 16)
// Read: Bewley et al., "Simple Online and Realtime Tracking" (ICIP 2016) — 4 pages
// Reference impl (read AFTER understanding math): github.com/abewley/sort
//
// YOUR TASK:
//   1. Track struct with KF, hit/miss counters, class_id
//   2. Predict all tracks
//   3. Build cost matrix (Euclidean distance predicted pos -> detections)
//   4. Solve assignment via hungarian_solve()
//   5. Update matched tracks, create new tracks for unmatched detections
//   6. Remove tracks with too many consecutive misses
//   7. Only publish tracks with enough consecutive hits (min_hits)

#pragma once
#include <Eigen/Dense>
#include <vector>

#include "kalman_filter.hpp"

struct Track {
    int id;
    KalmanFilter2D kf;
    int hits;
    int misses;
    int class_id;
    float z_3d;
};

class SORTTracker {
   public:
    SORTTracker(float max_dist, int max_misses, int min_hits);
    std::vector<Track> update(const std::vector<Eigen::Vector2f>& detections,
                              const std::vector<int>& class_ids);

   private:
    std::vector<Track> tracks_;
    int next_id_ = 0;
    float max_dist_;
    int max_misses_;
    int min_hits_;

    std::vector<std::pair<int, int>> match(const std::vector<Eigen::Vector2f>& dets);
};
