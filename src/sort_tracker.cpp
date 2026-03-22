// sort_tracker.cpp
// TODO: Implement SORT multi-object tracker from scratch.
// See include/sort_tracker.hpp for interface and instructions.

#include "sort_tracker.hpp"

#include "hungarian.hpp"

SORTTracker::SORTTracker(float max_dist, int max_misses, int min_hits)
    : max_dist_(max_dist), max_misses_(max_misses), min_hits_(min_hits) {}

std::vector<Track> SORTTracker::update(const std::vector<Eigen::Vector2f>& detections,
                                       const std::vector<int>& class_ids) {
    // TODO: implement
    // 1. Predict all existing tracks
    // 2. Build cost matrix
    // 3. Solve assignment
    // 4. Update matched, create new, prune stale
    return {};
}

std::vector<std::pair<int, int>> SORTTracker::match(
    const std::vector<Eigen::Vector2f>& dets) {
    // TODO: build cost matrix from track predictions vs detections
    return {};
}
