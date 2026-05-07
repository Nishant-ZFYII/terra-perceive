// dbscan.hpp
// Density-Based Spatial Clustering of Applications with Noise — from scratch.
//
// References:
//   Ester, Kriegel, Sander, Xu, "A Density-Based Algorithm for Discovering
//     Clusters in Large Spatial Databases with Noise" (KDD 1996).
//   Schubert, Sander, Ester, Kriegel, Xu, "DBSCAN Revisited, Revisited:
//     Why and How You Should (Still) Use DBSCAN" (TODS 2017) — modern
//     critique and clarifications.
//
// Algorithm (Ester 1996, condensed):
//   For every unvisited point p:
//     N = neighbors(p, eps)
//     if |N| < min_points:
//       mark p as NOISE  (may later be reclassified as a border point if
//                         pulled into a cluster's expansion)
//     else:
//       start a new cluster, add p
//       process every point in N:
//         if a queued point q has |neighbors(q, eps)| >= min_points,
//         it's a CORE point — add ITS neighbors to the queue too.
//         Border points are added to the cluster but don't expand it.
//
// Output:
//   - vector<vector<int>>: one inner vector per cluster, holding indices
//     into the input `pts`. Noise indices are NOT returned.
//   - cluster ordering is implementation-defined (determined by the order
//     in which seed points are encountered in the outer loop).
//
// Complexity:
//   - O(N^2) using a brute-force neighbor search (this implementation).
//   - O(N log N) with a KD-tree; future work (mention in blog).
//   - For RELLIS obstacle clouds (≤ ~10K points after RANSAC), the brute
//     force is fast enough — well under 1 second.
//
// Edge cases the test suite WILL hit:
//   * pts.empty()                     → return {}
//   * min_points <= 0                 → behavior undefined; assert in DEBUG
//   * eps <= 0                        → every point is its own neighbor
//                                       only; everything is noise unless
//                                       min_points == 1
//   * all points identical            → one giant cluster of size N
//                                       (everyone is a core point of
//                                       everyone else)
//   * min_points > N                  → no point has enough neighbors;
//                                       everything is noise; return {}

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace tracker {

// Run DBSCAN on a 3D point set.
//
// Parameters:
//   pts        — input points. Indices in the returned clusters reference
//                this vector. Coordinates can be in any units; eps must
//                match.
//   eps        — neighborhood radius. Two points p, q are neighbors iff
//                ||p - q|| <= eps (Euclidean).
//   min_points — density threshold. A point with at least min_points
//                neighbors (INCLUDING ITSELF) within eps is a core point.
//                The conventional choice in the original paper.
//
// Returns:
//   List of clusters. Each cluster is a list of point indices into `pts`.
//   Noise points (not core, not in any core's neighborhood) are NOT
//   present in any cluster. Indices within a cluster are in encounter
//   order (the order DBSCAN's BFS-style expansion visits them), NOT
//   sorted.
std::vector<std::vector<int>> dbscan(const std::vector<Eigen::Vector3f>& pts,
                                     float eps,
                                     int min_points);

}  // namespace tracker
