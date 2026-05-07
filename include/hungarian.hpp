// hungarian.hpp
// Assignment algorithms for multi-object tracking data association.
//
// Two solvers behind one dispatcher:
//   greedy_solve()  — O(N*M) nearest-unmatched, with optional gating threshold.
//                     Fast, simple, *order-dependent on ambiguous costs* — see
//                     debugging story #3 in the M4 blog (greedy's order-
//                     dependence on track crossings).
//   munkres_solve() — O(N^3) optimal assignment. Munkres (1957) variant of the
//                     Kuhn (1955) Hungarian algorithm. Handles rectangular
//                     matrices via internal dummy-row/column padding.
//
// References:
//   Kuhn, "The Hungarian Method for the Assignment Problem" (Naval Res.
//     Logistics Q., 1955) — foundational; read the algorithm, skim the proof.
//   Munkres, "Algorithms for the Assignment and Transportation Problems"
//     (J. SIAM 1957) — the practical O(N^3) version.
//
// Input:  NxM cost matrix. Rows = tracks (or "left" set), cols = detections
//         (or "right" set). Entry (r, c) = cost of pairing row r with col c.
// Output: vector of (row, col) assignment pairs, total cost minimized subject
//         to the gating threshold. Pairs above max_cost (greedy) are dropped.

#pragma once
#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace tracker {

// Which solver to dispatch to. Wired through SORTTracker so the CLI flag
// --solver {greedy|munkres} can swap implementations without recompilation.
enum class Solver { Greedy, Munkres };

// Greedy nearest-unmatched matching with a gating threshold.
// For each detection (column), assign the lowest-cost unmatched track (row)
// with cost <= max_cost. Pairs with cost > max_cost are dropped.
//
// NOT globally optimal. The order in which detections are visited can change
// the result on ambiguous cost matrices. This is by design — the
// `GreedyOrderDependenceOnCrossings` test exercises it as debugging story #3.
//
// max_cost: gating threshold. Use std::numeric_limits<float>::infinity() to
//           disable gating.
std::vector<std::pair<int, int>> greedy_solve(const Eigen::MatrixXf& cost_matrix,
                                              float max_cost);

// Munkres O(N^3) optimal assignment. Globally minimizes total cost.
// Handles rectangular matrices internally via dummy padding to square.
// Pairs involving dummy rows/cols are filtered out before returning.
std::vector<std::pair<int, int>> munkres_solve(const Eigen::MatrixXf& cost_matrix);

// Dispatcher: pick one based on the Solver enum. SORT calls this so the swap
// is one constructor argument, not a recompile. max_cost is forwarded to
// greedy_solve and ignored by munkres_solve (Munkres doesn't gate; if you
// want gating, post-filter the result).
std::vector<std::pair<int, int>> hungarian_solve(const Eigen::MatrixXf& cost_matrix,
                                                 Solver solver,
                                                 float max_cost);

}  // namespace tracker
