// dbscan.cpp
// Implementation of DBSCAN with brute-force neighbor search.
//
// Reference: Ester, Kriegel, Sander, Xu (KDD 1996), Algorithm boxes 1 and 2.
//
// Implementation notes:
//   - Per-point state tracked via two parallel vectors: `state` (UNVISITED,
//     NOISE, ASSIGNED) and `cluster_id` (-1 if unassigned, else cluster
//     index). NOISE is a tentative classification — a NOISE point can be
//     promoted to ASSIGNED later if it falls into a core point's eps
//     neighborhood (these are "border points" in the paper's terminology).
//   - Cluster expansion uses a BFS-style queue (std::deque) seeded with the
//     core point's neighbors. The textbook formulation uses a recursive
//     "ExpandCluster" routine; the iterative form below is equivalent and
//     safer (no deep stack on large clusters).
//   - Convention: the radius check `dist <= eps` includes the point itself
//     (distance 0). So a point's neighbor count INCLUDES itself.
//     min_points = 1 therefore makes every point a core point of itself.
//   - Brute-force: every neighbor query scans all N points. O(N^2) total.
//     KD-tree replacement is straight-forward future work; mention in blog.

#include "dbscan.hpp"

#include <cassert>
#include <deque>
#include <vector>

namespace tracker {

namespace {  // file-local helpers

enum class PointState : uint8_t {
    UNVISITED = 0,
    NOISE     = 1,
    ASSIGNED  = 2,
};

// Brute-force neighbor query.
// Returns indices i where ||pts[query_idx] - pts[i]|| <= eps.
// Includes query_idx itself (distance 0 to itself).
std::vector<int> region_query(const std::vector<Eigen::Vector3f>& pts,
                              int query_idx,
                              float eps_sq) {
    // YOUR CODE:
    //   std::vector<int> out;
    //   const Eigen::Vector3f& p = pts[query_idx];
    //   for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
    //       if ((pts[i] - p).squaredNorm() <= eps_sq) {
    //           out.push_back(i);
    //       }
    //   }
    //   return out;
    //
    // Notes:
    //   - Compare squared distance to eps_sq to avoid the sqrt() per pair.
    //     Caller passes eps_sq (= eps*eps) so we don't repeat the square
    //     across N^2 calls.
    //   - The query point IS included in its own neighborhood (squaredNorm
    //     of zero vector). That's deliberate: the min_points threshold
    //     counts the query point itself per Ester 1996.
    std::vector<int> out;
    const Eigen::Vector3f& p = pts[query_idx];
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if ((pts[i] - p).squaredNorm() <= eps_sq) {
            out.push_back(i);
        }
    }
    return out;
}

}  // anonymous namespace

// =============================================================================
// dbscan — main entry point.
// =============================================================================
std::vector<std::vector<int>> dbscan(const std::vector<Eigen::Vector3f>& pts,
                                     float eps,
                                     int min_points) {
    // ---- Edge cases ---------------------------------------------------------
    //
    // YOUR CODE:
    //   if (pts.empty()) return {};
    //
    // Optional defensive check (uncomment to be strict; production callers
    // should generally pre-validate):
    //   assert(min_points > 0 && "DBSCAN min_points must be positive");
    //   assert(eps >= 0.0f   && "DBSCAN eps must be non-negative");

    // ---- Setup --------------------------------------------------------------
    //
    // YOUR CODE:
    //   const int N = static_cast<int>(pts.size());
    //   const float eps_sq = eps * eps;   // compare squared distances; saves N^2 sqrts
    //
    //   std::vector<PointState> state(N, PointState::UNVISITED);
    //   std::vector<int>        cluster_id(N, -1);
    //
    //   std::vector<std::vector<int>> clusters;

    // ---- Main loop: scan every point, seed a cluster from each core --------
    //
    // YOUR CODE:
    //   for (int i = 0; i < N; ++i) {
    //       if (state[i] != PointState::UNVISITED) continue;
    //
    //       // Find i's neighborhood.
    //       std::vector<int> nbrs = region_query(pts, i, eps_sq);
    //
    //       // Density check — if too few neighbors, mark NOISE for now.
    //       // (Could be reclassified to a border point later if some core's
    //       //  neighborhood pulls it in.)
    //       if (static_cast<int>(nbrs.size()) < min_points) {
    //           state[i] = PointState::NOISE;
    //           continue;
    //       }
    //
    //       // i is a core point — start a new cluster, BFS-expand it.
    //       const int cid = static_cast<int>(clusters.size());
    //       clusters.emplace_back();
    //       std::vector<int>& this_cluster = clusters.back();
    //
    //       state[i]      = PointState::ASSIGNED;
    //       cluster_id[i] = cid;
    //       this_cluster.push_back(i);
    //
    //       // BFS queue seeded with i's neighbors (excluding i; or include
    //       //   it — the loop short-circuits since state[i] is ASSIGNED).
    //       std::deque<int> queue(nbrs.begin(), nbrs.end());
    //
    //       while (!queue.empty()) {
    //           const int j = queue.front();
    //           queue.pop_front();
    //
    //           // ⚠ Three cases for j's current state:
    //           //   - UNVISITED: process it normally.
    //           //   - NOISE:     reclassify as border (assigned to this cluster);
    //           //                do NOT expand from it (border points don't expand).
    //           //   - ASSIGNED:  skip; already in some cluster.
    //
    //           if (state[j] == PointState::NOISE) {
    //               state[j]      = PointState::ASSIGNED;   // promote NOISE to border
    //               cluster_id[j] = cid;
    //               this_cluster.push_back(j);
    //               continue;     // border points do NOT expand the cluster
    //           }
    //
    //           if (state[j] != PointState::UNVISITED) continue;
    //
    //           // j is UNVISITED. Add to cluster.
    //           state[j]      = PointState::ASSIGNED;
    //           cluster_id[j] = cid;
    //           this_cluster.push_back(j);
    //
    //           // If j is also a core point, queue its neighbors so the
    //           // cluster keeps growing.
    //           std::vector<int> nbrs_j = region_query(pts, j, eps_sq);
    //           if (static_cast<int>(nbrs_j.size()) >= min_points) {
    //               for (int k : nbrs_j) {
    //                   // We could filter by state here to shrink the queue,
    //                   // but the inner-loop short-circuits handle re-visits
    //                   // correctly, and the resulting algorithm is the
    //                   // textbook one. Keep it simple.
    //                   queue.push_back(k);
    //               }
    //           }
    //       }
    //   }

    // ---- Return -------------------------------------------------------------
    //
    // YOUR CODE:
    //   return clusters;
    //
    // Notes:
    //   - `state` and `cluster_id` go out of scope here. They were only
    //     scratch state during expansion. The caller doesn't need them.
    //   - Noise points are NOT in any cluster. To recover the noise set,
    //     the caller would need to compute it as the complement: indices
    //     not present in any returned cluster. We don't expose this
    //     separately; the convention is "if you wanted the noise list, you
    //     can derive it." Future work: a second overload that returns
    //     (clusters, noise_indices) if callers need it.
    if (pts.empty()) return {};
    const int N = static_cast<int>(pts.size());
    const float eps_sq = eps * eps;
    std::vector<PointState> state(N, PointState::UNVISITED);
    std::vector<int>        cluster_id(N, -1);
    std::vector<std::vector<int>> clusters;
    for (int i = 0; i < N; ++i) {
        if (state[i] != PointState::UNVISITED) continue;
        std::vector<int> nbrs = region_query(pts, i, eps_sq);
        if (static_cast<int>(nbrs.size()) < min_points) {
            state[i] = PointState::NOISE;
            continue;
        }
        const int cid = static_cast<int>(clusters.size());
        clusters.emplace_back();
        std::vector<int>& this_cluster = clusters.back();
        state[i]      = PointState::ASSIGNED;
        cluster_id[i] = cid;
        this_cluster.push_back(i);
        std::deque<int> queue(nbrs.begin(), nbrs.end());
        while (!queue.empty()) {
            const int j = queue.front();
            queue.pop_front();
            if (state[j] == PointState::NOISE) {
                state[j]      = PointState::ASSIGNED;
                cluster_id[j] = cid;
                this_cluster.push_back(j);
                continue;
            }
            if (state[j] != PointState::UNVISITED) continue;
            state[j]      = PointState::ASSIGNED;
            cluster_id[j] = cid;
            this_cluster.push_back(j);
            std::vector<int> nbrs_j = region_query(pts, j, eps_sq);
            if (static_cast<int>(nbrs_j.size()) >= min_points) {
                for (int k : nbrs_j) {
                    queue.push_back(k);
                }
            }
        }
    }
    return clusters;
}

}  // namespace tracker
