// hungarian.cpp
// Assignment-algorithm implementations for SORT data association.
//
// Two solvers in this file:
//   greedy_solve()  — O(N*M) nearest-unmatched with gating.
//   munkres_solve() — O(N^3) optimal assignment (classical Munkres 1957).
//
// References:
//   Kuhn, "The Hungarian Method for the Assignment Problem" (Naval Res.
//     Logistics Q., 1955) — foundational.
//   Munkres, "Algorithms for the Assignment and Transportation Problems"
//     (J. SIAM 1957) — the practical O(N^3) variant.
//   Pilgrim, "Munkres' Assignment Algorithm: Modified for Rectangular
//     Matrices" (Murray State Univ., 2000) — the canonical state-machine
//     formulation that this file follows.
//   Guo, github.com/xg590/munkres — pure-C port of Pilgrim's reference.
//
// Implementation pattern (Munkres):
//   - Maintain a `mask` matrix in parallel with the cost matrix:
//       mask(i, j) == 0  → unmarked
//       mask(i, j) == 1  → STARRED (tentative assignment)
//       mask(i, j) == 2  → PRIMED  (alternate-path candidate)
//   - Step functions (step_three .. step_six) return the next step number.
//   - Outer loop is a state machine: while (step != Done) dispatch.
//   - Steps 1 and 2 run once at entry; the 3 ↔ 4 ↔ 5 ↔ 6 loop runs until
//     all columns are covered (Step 7 / Done).
//
// Edge cases the test suite WILL hit:
//   * 0x0 matrix       → return {}
//   * 1x1 matrix       → return {{0, 0}}
//   * all-equal costs  → any valid permutation is optimal
//   * rectangular N!=M → pad to K x K with zeros, filter dummies on output
//
// Cross-validation: the test file should compare against
// scipy.optimize.linear_sum_assignment on at least one random-matrix case.
// Sum of selected costs must match exactly.

#include "hungarian.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tracker {

// =============================================================================
// greedy_solve
// =============================================================================
//
// Algorithm sketch:
//   matched_rows = {}; out = {}
//   for each col c (in column order):
//       find row r* not in matched_rows minimizing cost(r, c)
//       if cost(r*, c) < max_cost:                    (strict <, to make
//           out.push_back({r*, c}); matched_rows.add(r*)   ties order-dependent)
//   return out
//
// Notes:
//   - Iterating columns-then-rows is one valid order; rows-then-cols yields
//     the symmetric variant. Pick ONE order and stick to it. The blog story
//     "greedy is order-dependent on crossings" relies on the iteration order
//     being deterministic. Recommended: outer loop = columns (detections).
//   - max_cost = std::numeric_limits<float>::infinity() disables gating.
//   - Complexity: O(N*M). For N, M <= 50 this is trivial.
// =============================================================================
std::vector<std::pair<int, int>> greedy_solve(const Eigen::MatrixXf& cost_matrix,
                                              float max_cost) {
    // YOUR CODE:
    //   1. const int N = cost_matrix.rows();
    //      const int M = cost_matrix.cols();
    //      if (N == 0 || M == 0) return {};
    //   2. std::vector<bool> row_used(N, false);
    //      std::vector<std::pair<int, int>> out;
    //      out.reserve(std::min(N, M));
    //   3. For each column c in [0, M):
    //        a. int   best_row  = -1;
    //           float best_cost = max_cost;
    //        b. For each row r in [0, N):
    //             if (!row_used[r] && cost_matrix(r, c) < best_cost) {
    //                 best_cost = cost_matrix(r, c);
    //                 best_row  = r;
    //             }
    //        c. if (best_row != -1) {
    //               row_used[best_row] = true;
    //               out.push_back({best_row, c});
    //           }
    //   4. return out;
    //
    // Gotcha: use strictly < (not <=) so ties are broken in favor of the
    // first-seen row. That deterministic-but-arbitrary choice is what makes
    // the order-dependence story possible.
    const int N = cost_matrix.rows();
    const int M = cost_matrix.cols();
    if (N == 0 || M == 0) return {};
    std::vector<bool> row_used(N, false);
    std::vector<std::pair<int, int>> out;
    out.reserve(std::min(N, M));
    for (int c = 0; c < M; ++c) {
        int best_row = -1;
        float best_cost = max_cost;
        for (int r = 0; r < N; ++r) {
            if (!row_used[r] && cost_matrix(r, c) < best_cost) {
                best_cost = cost_matrix(r, c);
                best_row = r;
            }
        }
        if (best_row != -1) {
            row_used[best_row] = true;
            out.push_back({best_row, c});
        }    
    }
    return out;
}

// =============================================================================
// munkres_solve — classical Munkres state machine.
// =============================================================================

namespace {  // file-local helpers; not exported

// Mask values.
constexpr int UNMARKED = 0;
constexpr int STARRED  = 1;
constexpr int PRIMED   = 2;

// Step return codes — readable names beat magic numbers in the dispatch loop.
enum class Step { Three = 3, Four = 4, Five = 5, Six = 6, Done = 7 };

// -----------------------------------------------------------------------------
// step_three — Cover every column that contains a starred zero. If all K
//   columns are covered, the starred zeros ARE the optimal assignment → Done.
//   Otherwise → Step 4.
//
// Pilgrim's pseudocode:
//   for (i, j) in K x K: if mask(i, j) == STARRED: col_covered[j] = true
//   if count(col_covered) >= K: return Done
//   else: return Four
// -----------------------------------------------------------------------------
Step step_three(const Eigen::MatrixXi& mask,
                std::vector<bool>& /*row_covered*/,
                std::vector<bool>& col_covered,
                int K) {
    // YOUR CODE:
    //   1. for (i = 0; i < K; ++i)
    //        for (j = 0; j < K; ++j)
    //            if (mask(i, j) == STARRED) col_covered[j] = true;
    //   2. count = number of true entries in col_covered.
    //   3. return (count >= K) ? Step::Done : Step::Four;
    //
    // Note: row_covered is intentionally NOT touched here. Step 3 only covers
    //       COLUMNS in classical Munkres. row_covered is reset to all-false
    //       before the dispatch loop starts.
    int count = 0;

    for(int i = 0; i<K; ++i){
        for(int j = 0; j<K; ++j){
            if(mask(i, j) == STARRED){
                col_covered[j] = true;
            }
        }
    }
    for (int j = 0; j < K; ++j) {
        if (col_covered[j]) count++;
    }
    if (count >= K) return Step::Done;
    else return Step::Four;

}

// -----------------------------------------------------------------------------
// find_uncovered_zero — scan for any zero with both row and col uncovered.
//   Returns {-1, -1} if none found. Pilgrim's "find_a_noncovered_zero".
// -----------------------------------------------------------------------------
std::pair<int, int> find_uncovered_zero(const Eigen::MatrixXf& padded,
                                        const std::vector<bool>& row_covered,
                                        const std::vector<bool>& col_covered,
                                        int K) {
    // YOUR CODE:
    //   for (i = 0; i < K; ++i)
    //     for (j = 0; j < K; ++j)
    //       if (padded(i, j) == 0.0f && !row_covered[i] && !col_covered[j])
    //         return {i, j};
    //   return {-1, -1};
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (padded(i, j) == 0.0f && !row_covered[i] && !col_covered[j]) {
                return {i, j};
            }
        }
    }
    return {-1, -1};
}

// -----------------------------------------------------------------------------
// find_star_in_row — column index of the starred zero in `row`, or -1.
//   Used by step 4 when an uncovered zero's row already has a star.
// -----------------------------------------------------------------------------
int find_star_in_row(const Eigen::MatrixXi& mask, int row, int K) {
    // YOUR CODE:
    //   for (j = 0; j < K; ++j) if (mask(row, j) == STARRED) return j;
    //   return -1;
    for (int j = 0; j < K; ++j) {
        if (mask(row, j) == STARRED) return j;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// find_star_in_col / find_prime_in_row — symmetric scanners used by step 5
//   when building the alternating zig-zag path.
// -----------------------------------------------------------------------------
int find_star_in_col(const Eigen::MatrixXi& mask, int col, int K) {
    // YOUR CODE: scan column `col`, return the row of any STARRED entry, else -1.
    for (int i = 0; i < K; ++i) {
        if (mask(i, col) == STARRED) return i;
    }

    return -1;
}

int find_prime_in_row(const Eigen::MatrixXi& mask, int row, int K) {
    // YOUR CODE: scan row `row`, return the column of any PRIMED entry, else -1.
    for (int j = 0; j < K; ++j) {
        if (mask(row, j) == PRIMED) return j;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// step_four — Find an uncovered zero and PRIME it.
//   - If its row contains no STARRED zero, set (path_row_0, path_col_0) and go
//     to Step 5 (build augmenting path).
//   - Otherwise: cover this row, uncover the column of the starred zero in
//     this row, and continue searching for another uncovered zero.
//   - If no uncovered zero exists, go to Step 6 (adjust costs).
// -----------------------------------------------------------------------------
Step step_four(const Eigen::MatrixXf& padded,
               Eigen::MatrixXi& mask,
               std::vector<bool>& row_covered,
               std::vector<bool>& col_covered,
               int& path_row_0,
               int& path_col_0,
               int K) {
    // YOUR CODE: loop until either we return Step::Five or Step::Six.
    //   while (true) {
    //       auto [r, c] = find_uncovered_zero(padded, row_covered, col_covered, K);
    //       if (r == -1) return Step::Six;          // no uncovered zero left
    //       mask(r, c) = PRIMED;
    //       int starred_col = find_star_in_row(mask, r, K);
    //       if (starred_col == -1) {                // no star in this row
    //           path_row_0 = r;
    //           path_col_0 = c;
    //           return Step::Five;
    //       }
    //       row_covered[r] = true;
    //       col_covered[starred_col] = false;
    //   }
    while (true) {
        auto [r, c] = find_uncovered_zero(padded, row_covered, col_covered, K);
        if (r == -1) return Step::Six;
        mask(r, c) = PRIMED;
        int starred_col = find_star_in_row(mask, r, K);
        if (starred_col == -1) {
            path_row_0 = r;
            path_col_0 = c;
            return Step::Five;
        }
        row_covered[r] = true;
        col_covered[starred_col] = false;
    }
    return Step::Six;
}

// -----------------------------------------------------------------------------
// step_five — Build an alternating path of PRIMED and STARRED zeros, then:
//   - Unstar each starred zero on the path.
//   - Star each primed zero on the path.
//   - Erase all primes.
//   - Uncover every row and column.
//   - Return to Step 3.
//
// Path construction:
//   path[0] = (path_row_0, path_col_0)   ← the primed zero from Step 4
//   loop:
//       look for a starred zero in the same column as path[end]
//       if none → done building
//       append (its_row, path[end].col)  to path
//       look for a primed zero in the same row as the new entry
//       (Pilgrim guarantees exactly one exists)
//       append (its_row, its_col)  to path
// -----------------------------------------------------------------------------
Step step_five(Eigen::MatrixXi& mask,
               std::vector<bool>& row_covered,
               std::vector<bool>& col_covered,
               int path_row_0,
               int path_col_0,
               int K) {
    // YOUR CODE:
    //   1. std::vector<std::pair<int, int>> path;
    //      path.reserve(2 * K);
    //      path.emplace_back(path_row_0, path_col_0);
    //   2. while (true) {
    //          int r = find_star_in_col(mask, path.back().second, K);
    //          if (r == -1) break;
    //          path.emplace_back(r, path.back().second);
    //          int c = find_prime_in_row(mask, path.back().first, K);
    //          path.emplace_back(path.back().first, c);
    //      }
    //   3. Augment: for each (r, c) in path,
    //        mask(r, c) = (mask(r, c) == STARRED) ? UNMARKED : STARRED;
    //   4. Erase primes: for any (i, j) with mask == PRIMED, set to UNMARKED.
    //   5. Reset covers: fill row_covered and col_covered with false.
    //   6. return Step::Three;
    std::vector<std::pair<int, int>> path;
    path.reserve(2 * K);
    path.emplace_back(path_row_0, path_col_0);
    while (true) {
        int r = find_star_in_col(mask, path.back().second, K);
        if (r == -1) break;
        path.emplace_back(r, path.back().second);
        int c = find_prime_in_row(mask, path.back().first, K);
        path.emplace_back(path.back().first, c);
    }
    for (const auto& [r, c] : path) {
        mask(r, c) = (mask(r, c) == STARRED) ? UNMARKED : STARRED;
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (mask(i, j) == PRIMED) {
                mask(i, j) = UNMARKED;
            }
        }
    }
    std::fill(row_covered.begin(), row_covered.end(), false);
    std::fill(col_covered.begin(), col_covered.end(), false);
    return Step::Three;
}

// -----------------------------------------------------------------------------
// step_six — Find the smallest UNCOVERED entry e in `padded`.
//   Subtract e from every UNCOVERED row, add e to every COVERED column.
//   This creates new zeros without invalidating existing stars/primes.
//   Return to Step 4.
// -----------------------------------------------------------------------------
Step step_six(Eigen::MatrixXf& padded,
              const std::vector<bool>& row_covered,
              const std::vector<bool>& col_covered,
              int K) {
    // YOUR CODE:
    //   1. float e = +inf;
    //      for (i = 0; i < K; ++i)
    //        if (!row_covered[i])
    //          for (j = 0; j < K; ++j)
    //            if (!col_covered[j])
    //              e = std::min(e, padded(i, j));
    //   2. for (i = 0; i < K; ++i)
    //        for (j = 0; j < K; ++j) {
    //          if (!row_covered[i]) padded(i, j) -= e;
    //          if ( col_covered[j]) padded(i, j) += e;
    //        }
    //   3. return Step::Four;
    float e = std::numeric_limits<float>::infinity();
    for (int i = 0; i < K; ++i) {
        if (!row_covered[i]) {
            for (int j = 0; j < K; ++j) {
                if (!col_covered[j]) {
                    e = std::min(e, padded(i, j));
                }
            }
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (!row_covered[i]) padded(i, j) -= e;
            if ( col_covered[j]) padded(i, j) += e;
        }
    }
    return Step::Four;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// munkres_solve — public entry point. Runs the state machine and extracts
//   the assignment from the STARRED zeros in the final mask.
// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>> munkres_solve(const Eigen::MatrixXf& cost_matrix) {
    // YOUR CODE:
    //
    // ---- Setup --------------------------------------------------------------
    //   const int N = cost_matrix.rows();
    //   const int M = cost_matrix.cols();
    //   if (N == 0 || M == 0) return {};
    //   const int K = std::max(N, M);
    //
    //   Eigen::MatrixXf padded = Eigen::MatrixXf::Zero(K, K);
    //   padded.topLeftCorner(N, M) = cost_matrix;          // dummies = 0
    //
    //   Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(K, K);
    //   std::vector<bool> row_covered(K, false);
    //   std::vector<bool> col_covered(K, false);
    //   int path_row_0 = 0, path_col_0 = 0;
    //
    // ---- Step 1 — row min subtract -----------------------------------------
    //   for (int i = 0; i < K; ++i)
    //       padded.row(i).array() -= padded.row(i).minCoeff();
    //
    // ---- Step 2 — initial starring (uses covers as a temp scratchpad) -----
    //   for (int i = 0; i < K; ++i)
    //       for (int j = 0; j < K; ++j)
    //           if (padded(i, j) == 0.0f
    //               && !row_covered[i] && !col_covered[j]) {
    //               mask(i, j) = STARRED;
    //               row_covered[i] = true;
    //               col_covered[j] = true;
    //           }
    //   std::fill(row_covered.begin(), row_covered.end(), false);
    //   std::fill(col_covered.begin(), col_covered.end(), false);
    //
    // ---- Dispatch loop -----------------------------------------------------
    //   Step step = Step::Three;
    //   const int kIterCap = 10 * K * K;     // generous safety guard
    //   int iter = 0;
    //   while (step != Step::Done) {
    //       if (++iter > kIterCap)
    //           throw std::runtime_error("munkres_solve: iteration cap hit");
    //       switch (step) {
    //           case Step::Three:
    //               step = step_three(mask, row_covered, col_covered, K); break;
    //           case Step::Four:
    //               step = step_four(padded, mask, row_covered, col_covered,
    //                                path_row_0, path_col_0, K); break;
    //           case Step::Five:
    //               step = step_five(mask, row_covered, col_covered,
    //                                path_row_0, path_col_0, K); break;
    //           case Step::Six:
    //               step = step_six(padded, row_covered, col_covered, K); break;
    //           case Step::Done:
    //               break;
    //       }
    //   }
    //
    // ---- Step 7 — extract & filter dummies --------------------------------
    //   std::vector<std::pair<int, int>> out;
    //   out.reserve(std::min(N, M));
    //   for (int i = 0; i < N; ++i)              // bounds = N, M (NOT K) so
    //       for (int j = 0; j < M; ++j)          // padded dummies are filtered
    //           if (mask(i, j) == STARRED)
    //               out.emplace_back(i, j);
    //   return out;
    const int N = cost_matrix.rows();
    const int M = cost_matrix.cols();
    if (N == 0 || M == 0) return {};
    const int K = std::max(N, M);

    Eigen::MatrixXf padded = Eigen::MatrixXf::Zero(K, K);
    padded.topLeftCorner(N, M) = cost_matrix;          // dummies = 0
    Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(K, K);
    std::vector<bool> row_covered(K, false);
    std::vector<bool> col_covered(K, false);
    int path_row_0 = 0, path_col_0 = 0;

    // ---- Step 1 — row min subtract ------------------------------------------------
    for (int i = 0; i < K; ++i) {
        padded.row(i).array() -= padded.row(i).minCoeff();
    }   
    // ---- Step 2 — initial starring (uses covers as a temp scratchpad) ----------------
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            if (padded(i, j) == 0.0f && !row_covered[i] && !col_covered[j]) {
                mask(i, j) = STARRED;
                row_covered[i] = true;
                col_covered[j] = true;
            }
        }
    }
    std::fill(row_covered.begin(), row_covered.end(), false);
    std::fill(col_covered.begin(), col_covered.end(), false);

    // ---- Dispatch loop -----------------------------------------------------
    Step step = Step::Three;
    const int kIterCap = 10 * K * K;     // generous safety guard
    int iter = 0;
    while (step != Step::Done) {
        if (++iter > kIterCap)            throw std::runtime_error("munkres_solve: iteration cap hit");
        switch (step) {
            case Step::Three:
                step = step_three(mask, row_covered, col_covered, K); break;
            case Step::Four:
                step = step_four(padded, mask, row_covered, col_covered,
                                 path_row_0, path_col_0, K); break;
            case Step::Five:
                step = step_five(mask, row_covered, col_covered,
                                 path_row_0, path_col_0, K); break;
            case Step::Six:
                step = step_six(padded, row_covered, col_covered, K); break;
            case Step::Done:
                break;
        }
    }
    // ---- Step 7 — extract & filter dummies --------------------------------
    std::vector<std::pair<int, int>> out;
    out.reserve(std::min(N, M));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (mask(i, j) == STARRED) {
                out.emplace_back(i, j);
            }
        }
    }
    return out;
}

// =============================================================================
// hungarian_solve — dispatcher.
// =============================================================================
std::vector<std::pair<int, int>> hungarian_solve(const Eigen::MatrixXf& cost_matrix,
                                                 Solver solver,
                                                 float max_cost) {
    // YOUR CODE:
    //   switch (solver) {
    //       case Solver::Greedy:  return greedy_solve(cost_matrix, max_cost);
    //       case Solver::Munkres: return munkres_solve(cost_matrix);
    //   }
    //   throw std::logic_error("hungarian_solve: unknown Solver value");
    switch (solver) {
        case Solver::Greedy:
            return greedy_solve(cost_matrix, max_cost);
        case Solver::Munkres:
            return munkres_solve(cost_matrix);
    }
    throw std::logic_error("hungarian_solve: unknown Solver value");
    return {};
}

}  // namespace tracker
