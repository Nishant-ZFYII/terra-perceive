// scan_context.hpp
// Scan Context loop closure detection — Kim & Kim (IROS 2018).
// Encodes a 3D LiDAR scan as a 2D polar descriptor (rings × sectors)
// where each bin stores the max height. Loop closure candidates are found
// by column-shifting one descriptor against another and computing cosine
// distance. Rotation-invariant because column shift simulates yaw change.
//
// Depends on: Eigen
// Used by:    pose_graph_slam.cpp (as loop closure edges)
//
// YOUR TASK (P2-M2.1):
//   P2-M2.1.1: buildDescriptor — polar grid binning, max-z per bin
//   P2-M2.1.2: columnShiftDistance — cosine distance between shifted descriptors
//   P2-M2.1.3: matchDescriptors — try all N_SECTORS shifts, return best
//   P2-M2.1.4: detectLoopClosures — batch over history, min-gap + threshold filter
//
// Key insight (Kim & Kim, Section 3):
//   A 3D scan is projected into a polar grid centered on the sensor.
//   Each point at (x, y, z) maps to:
//     ring index  = floor(sqrt(x²+y²) / max_range * N_RINGS)     ← radial distance
//     sector index = floor((atan2(y,x) + π) / (2π) * N_SECTORS)  ← azimuth angle
//   The max height z in each bin is stored. This produces a compact
//   N_RINGS × N_SECTORS descriptor that is:
//     - viewpoint invariant (egocentric)
//     - rotation invariant (via column-shift alignment)
//     - translation invariant (up to revisit proximity)
//
// Cosine distance (Kim & Kim, Section 4):
//   For each column c of descriptor A, compare to column (c + shift) of
//   descriptor B. Cosine distance = 1 - (a·b)/(‖a‖·‖b‖) per column pair,
//   averaged over all columns. Try all N_SECTORS shifts, keep the minimum.
//
// References:
//   Kim & Kim, "Scan Context: Egocentric Spatial Descriptor for Place
//     Recognition Within 3D Point Cloud Map" (IROS 2018), Sections 3–4
//   gisbi-kim/scancontext (GitHub) — reference C++ implementation
//
// Numerical gotchas:
//   - Points with range > max_range or range ≈ 0 should be skipped
//   - Empty bins (no points fell in) stay at 0.0; this affects cosine
//     distance — a column of all zeros has zero norm → division by zero.
//     Handle by skipping zero-norm columns in the distance computation.
//   - atan2 returns [-π, π]; shift to [0, 2π) for sector indexing

#pragma once
#include <Eigen/Dense>
#include <vector>

namespace scan_context {

// Grid resolution — Kim & Kim use 20 rings × 60 sectors
// Increase for finer resolution (more discriminative but slower matching)
static constexpr int N_RINGS = 20;
static constexpr int N_SECTORS = 60;

// Descriptor: N_RINGS × N_SECTORS matrix of max-height values.
// Row = ring (radial distance from sensor), Column = sector (azimuth angle).
// Stored as Eigen matrix for convenient column operations during matching.
using Descriptor = Eigen::Matrix<float, N_RINGS, N_SECTORS>;

// Loop closure candidate returned by detectLoopClosures.
struct LoopCandidate {
    int query_id;       // index of the query scan (current frame)
    int match_id;       // index of the matched scan (from history)
    float distance;     // cosine distance [0, 2] — lower = better match
    int best_shift;     // column shift that produced the minimum distance
};

// -----------------------------------------------------------------------------
// P2-M2.1.1: Descriptor construction — Kim & Kim Section 3
// -----------------------------------------------------------------------------

// Build a Scan Context descriptor from a raw 3D point cloud.
//
// For each point (x, y, z):
//   range = sqrt(x² + y²)
//   ring  = floor(range / max_range * N_RINGS),  clamp to [0, N_RINGS-1]
//   sector = floor((atan2(y, x) + π) / (2π) * N_SECTORS), clamp to [0, N_SECTORS-1]
//   desc(ring, sector) = max(desc(ring, sector), z)
//
// Points with range < 1.0m (too close, sensor noise) or range > max_range
// (too far, unreliable) should be skipped.
//
// max_range: maximum radial distance in meters. Kim & Kim use 80m for
// outdoor LiDAR; Ouster OS1-64 effective range is ~120m but 80m is fine
// for RELLIS-3D (mostly <50m useful range under canopy).
Descriptor buildDescriptor(const std::vector<Eigen::Vector3f>& points,
                           float max_range = 80.0f);

// -----------------------------------------------------------------------------
// P2-M2.1.2: Column-shift distance — Kim & Kim Section 4
// -----------------------------------------------------------------------------

// Compute the cosine distance between two descriptors with a given column shift.
// Column c of descriptor `a` is compared to column (c + shift) % N_SECTORS
// of descriptor `b`.
//
// Per-column cosine distance: d_c = 1 - (a_c · b_{c+shift}) / (‖a_c‖ · ‖b_{c+shift}‖)
// Overall distance = mean of d_c over all valid columns (skip if either norm ≈ 0).
//
// Returns a value in [0, 2]. 0 = identical, 2 = maximally different.
float columnShiftDistance(const Descriptor& a, const Descriptor& b, int shift);

// -----------------------------------------------------------------------------
// P2-M2.1.3: Best-match search — Kim & Kim Section 4
// -----------------------------------------------------------------------------

// Try all N_SECTORS column shifts and return the (min_distance, best_shift).
// This makes the descriptor rotationally invariant: if the vehicle revisits
// a location from a different heading, the column shift compensates for yaw.
std::pair<float, int> matchDescriptors(const Descriptor& a, const Descriptor& b);

// -----------------------------------------------------------------------------
// P2-M2.1.4: Loop closure detection over a history of descriptors
// -----------------------------------------------------------------------------

// Given a history of descriptors (one per LiDAR frame), find loop closure
// candidates for a query descriptor at index `query_id`.
//
// Filtering:
//   - min_gap: skip candidates within min_gap frames of query_id.
//     Nearby frames naturally look similar (the vehicle hasn't moved far);
//     these are NOT loop closures, just temporal neighbors. Typical: 50–100.
//   - dist_threshold: reject matches with cosine distance > threshold.
//     Kim & Kim use 0.1–0.3 depending on environment. Start with 0.3 for
//     RELLIS-3D (off-road has more variability than urban).
//
// Returns: all candidates passing both filters, sorted by distance (best first).
std::vector<LoopCandidate> detectLoopClosures(
    const std::vector<Descriptor>& history,
    int query_id,
    float dist_threshold = 0.3f,
    int min_gap = 50);

}  // namespace scan_context
