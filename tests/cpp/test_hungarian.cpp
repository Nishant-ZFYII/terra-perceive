// test_hungarian.cpp — Tests for assignment-algorithm primitives.
//
// P2-M4.2 checkpoints:
//   M4.2.1: GreedySmallKnown                  — 3x3 with one obvious minimum
//   M4.2.2: GreedyRectangular                 — N!=M cases, leftover row/col ok
//   M4.2.3: MunkresOptimalOn3x3              — Munkres matches greedy when
//                                                costs are well-separated
//   M4.2.4: GreedyOrderDependenceOnCrossings  — debugging story #3:
//                                                greedy ID-swaps on ambiguous
//                                                2x2; Munkres preserves
//   M4.2.5: MunkresHandlesRectangular         — pad-and-filter contract
//   M4.2.6: SolverAgreesOnSeparatedCosts      — random matrices with unique
//                                                minima: greedy == Munkres
//   M4.2.7: HungarianDispatcherForwardsArgs   — dispatcher routes to the
//                                                right solver
//   M4.2.8: EmptyAndSingleton                 — 0x0 and 1x1 edge cases
//
// References:
//   Kuhn (1955), Munkres (1957) — see hungarian.hpp / hungarian.cpp.
//   Pilgrim 2000 — state-machine recipe used by munkres_solve.
//
// Author note: follow the test style of tests/cpp/test_kalman.cpp and
// tests/cpp/test_so3.cpp — fixed seeds, tight per-test tolerances, comments
// describing the invariant. Do NOT put TEST() blocks inside namespace tracker;
// gtest macros expand to global registrations. The `using` below is enough.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "hungarian.hpp"

using tracker::greedy_solve;
using tracker::hungarian_solve;
using tracker::munkres_solve;
using tracker::Solver;

// -----------------------------------------------------------------------------
// Helper: total cost of an assignment vector against a cost matrix.
// -----------------------------------------------------------------------------
static float total_cost(const Eigen::MatrixXf& C,
                        const std::vector<std::pair<int, int>>& assign) {
    float s = 0.0f;
    for (auto [r, c] : assign) s += C(r, c);
    return s;
}

// Helper: assert no row or column appears twice in an assignment.
static void expect_no_duplicate_rows_or_cols(
    const std::vector<std::pair<int, int>>& assign) {
    std::set<int> rows, cols;
    for (auto [r, c] : assign) {
        EXPECT_TRUE(rows.insert(r).second) << "row " << r << " repeated";
        EXPECT_TRUE(cols.insert(c).second) << "col " << c << " repeated";
    }
}

constexpr float kInf = std::numeric_limits<float>::infinity();

// =============================================================================
// M4.2.1 — GreedySmallKnown
// 3x3 with a clearly separated minimum on each row. Greedy should find it.
// =============================================================================
TEST(Hungarian, GreedySmallKnown) {
    // YOUR CODE:
    //   1. Build a 3x3 Eigen::MatrixXf where the optimal assignment is
    //      obvious — e.g.
    //          [ 1   8   8 ]      → row 0 ↔ col 0
    //          [ 8   2   8 ]      → row 1 ↔ col 1
    //          [ 8   8   3 ]      → row 2 ↔ col 2
    //      Total optimal cost = 6.
    //   2. auto result = greedy_solve(C, /*max_cost=*/kInf);
    //   3. EXPECT_EQ(result.size(), 3u);
    //   4. expect_no_duplicate_rows_or_cols(result);
    //   5. EXPECT_NEAR(total_cost(C, result), 6.0f, 1e-5f);
    Eigen::MatrixXf C(3, 3);
    C << 1, 8, 8,
         8, 2, 8,
         8, 8, 3;
    auto result = greedy_solve(C, kInf);
    EXPECT_EQ(result.size(), 3u);
    expect_no_duplicate_rows_or_cols(result);
    EXPECT_NEAR(total_cost(C, result), 6.0f, 1e-5f);
}

// =============================================================================
// M4.2.2 — GreedyRectangular
// More cols than rows AND more rows than cols. Each detection (col) gets at
// most one track (row); leftover detections/tracks remain unassigned.
// =============================================================================
TEST(Hungarian, GreedyRectangular) {
    // YOUR CODE:
    //   Case A: 2x4 cost matrix. result.size() should be 2 (one per row).
    //   Case B: 4x2 cost matrix. result.size() should be 2 (one per col).
    //   In both cases: expect_no_duplicate_rows_or_cols(result);
    //   Suggested matrices:
    //     A: [ 1 5 5 5 ]   B: A.transpose()
    //        [ 5 5 5 1 ]
    //   In A, optimal pairs: {0,0} and {1,3}, total cost 2.
    //   In B, the same pairs flipped: {0,0} and {3,1}, total cost 2.
    Eigen::MatrixXf C(2, 4);
    C << 1, 5, 5, 5,
         5, 1, 5, 5;
    auto result_A = greedy_solve(C, kInf);
    EXPECT_EQ(result_A.size(), 2u);
    expect_no_duplicate_rows_or_cols(result_A);
    EXPECT_NEAR(total_cost(C, result_A), 2.0f, 1e-5f);
}

// =============================================================================
// M4.2.3 — MunkresOptimalOn3x3
// On a well-separated 3x3, Munkres returns the same minimum as greedy.
// =============================================================================
TEST(Hungarian, MunkresOptimalOn3x3) {
    // YOUR CODE:
    //   Reuse the 3x3 from M4.2.1.
    //   auto greedy_result  = greedy_solve(C, kInf);
    //   auto munkres_result = munkres_solve(C);
    //   EXPECT_EQ(munkres_result.size(), 3u);
    //   expect_no_duplicate_rows_or_cols(munkres_result);
    //   EXPECT_NEAR(total_cost(C, greedy_result),
    //               total_cost(C, munkres_result),
    //               1e-5f);
    Eigen::MatrixXf C(3, 3);
    C << 1, 8, 8,
         8, 2, 8,
         8, 8, 3;
    auto greedy_result = greedy_solve(C, kInf);
    auto munkres_result = munkres_solve(C);
    EXPECT_EQ(munkres_result.size(), 3u);
    expect_no_duplicate_rows_or_cols(munkres_result);
    EXPECT_NEAR(total_cost(C, greedy_result), total_cost(C, munkres_result), 1e-5f);
}

// =============================================================================
// M4.2.4 — GreedyOrderDependenceOnCrossings  (debugging story #3)
//
// Construct a 2x2 cost matrix where BOTH permutations are feasible (no
// gating cuts either) but they have DIFFERENT total costs. Greedy locks in
// the first detection's nearest track even when the swap would have been
// globally cheaper. Munkres always picks the min-total permutation.
//
// Concrete construction:
//     C = [ 1.0  1.5 ]      perm A: (0,0)+(1,1) = 1.0 + 1.0 = 2.0   ← optimal
//         [ 1.5  1.0 ]      perm B: (0,1)+(1,0) = 1.5 + 1.5 = 3.0
// Greedy iterating columns: col 0 picks row 0 (cost 1.0); col 1 picks
//     row 1 (cost 1.0). That is actually permutation A — total 2.0. Match.
//
// To trigger the order-dependent loss, asymmetrize:
//     C = [ 1.0   1.0 ]    perm A (0,0)+(1,1) = 1.0 + 2.0 = 3.0
//         [ 0.5   2.0 ]    perm B (0,1)+(1,0) = 1.0 + 0.5 = 1.5  ← optimal
// Greedy iterating columns: col 0 picks row 1 (cost 0.5); col 1 picks
//     row 0 (cost 1.0). Total 1.5 — this is permutation B, optimal.
// Greedy iterating ROWS instead: row 0 picks col 0 (cost 1.0); row 1 picks
//     col 1 (cost 2.0). Total 3.0 — suboptimal.
// Pick the construction that exposes YOUR greedy's iteration order.
// =============================================================================
TEST(Hungarian, GreedyOrderDependenceOnCrossings) {
    // YOUR CODE:
    //   1. Hand-craft C such that greedy_solve(C, kInf) returns a sub-optimal
    //      total cost. Confirm by inspection at compile time (write down
    //      both permutations and pick C accordingly).
    //   2. auto greedy_result  = greedy_solve(C, kInf);
    //      auto munkres_result = munkres_solve(C);
    //   3. EXPECT_GT(total_cost(C, greedy_result),
    //               total_cost(C, munkres_result));
    //      // strictly greater — greedy lost
    //   4. Optional: also assert the *pairs* differ (greedy and Munkres
    //      return different permutations), which is the visible bug in the
    //      blog GIF. Use sorting + EXPECT_NE on the sorted vectors.
    //
    // If this test is hard to construct, that's a hint: greedy's iteration
    // order may not be making the choice you think it is. Print the cost
    // matrix and step through greedy by hand on a 2x2.
    Eigen::MatrixXf C(2, 2);
    C << 1.0, 1.5,
         1.0, 2.0;
    auto greedy_result = greedy_solve(C, kInf);
    auto munkres_result = munkres_solve(C);
    EXPECT_GT(total_cost(C, greedy_result), total_cost(C, munkres_result));
}

// =============================================================================
// M4.2.5 — MunkresHandlesRectangular
// Munkres on a rectangular matrix must pad to K x K, run, and filter dummies.
// The returned pairs should all satisfy r < N && c < M, and the size must
// equal min(N, M).
// =============================================================================
TEST(Hungarian, MunkresHandlesRectangular) {
    // YOUR CODE:
    //   2x4 matrix: result.size() == 2, all r in {0,1}, all c in {0..3}.
    //   4x2 matrix: result.size() == 2, all r in {0..3}, all c in {0,1}.
    //   Use the same matrix shape as GreedyRectangular if you want easy
    //   cross-comparison.
    Eigen::MatrixXf C(2, 4);
    C << 1, 5, 5, 5,
         5, 5, 5, 1;
    auto result = munkres_solve(C);
    EXPECT_EQ(result.size(), 2u);
    for (auto [r, c] : result) {
        EXPECT_TRUE(r == 0 || r == 1) << "row " << r << " out of bounds";
        EXPECT_TRUE(c >= 0 && c < 4) << "col " << c << " out of bounds";
    }   
}

// =============================================================================
// M4.2.6 — SolverAgreesOnSeparatedCosts
// Generate a handful of random matrices where minima are unambiguous (no
// near-ties). Greedy and Munkres must return the same TOTAL COST (not
// necessarily the same pair list — both can be valid optima).
// =============================================================================
TEST(Hungarian, SolverAgreesOnSeparatedCosts) {
    // YOUR CODE:
    //   1. std::mt19937 rng(123);
    //      std::uniform_real_distribution<float> u(0.0f, 100.0f);
    //   2. for (int trial = 0; trial < 10; ++trial) {
    //        int N = 5, M = 5;
    //        Eigen::MatrixXf C(N, M);
    //        for (int i = 0; i < N; ++i)
    //          for (int j = 0; j < M; ++j) C(i, j) = u(rng);
    //        // Boost separation: penalize off-diagonal so a unique minimum exists.
    //        for (int i = 0; i < N; ++i) C(i, i) -= 50.0f;
    //
    //        auto g = greedy_solve(C, kInf);
    //        auto m = munkres_solve(C);
    //        EXPECT_NEAR(total_cost(C, g), total_cost(C, m), 1e-3f)
    //            << "trial " << trial;
    //      }
    //
    // Note: this is a probabilistic-style test, but the seed is fixed →
    // deterministic. If a trial flakes, lower the off-diagonal penalty, or
    // raise N — more dimensions = lower probability of greedy hitting a
    // pathological case.
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(0.0f, 100.0f);
    for (int trial = 0; trial < 10; ++trial) {
        int N = 5, M = 5;
        Eigen::MatrixXf C(N, M);
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < M; ++j) C(i, j) = u(rng);
        // Boost separation: penalize off-diagonal so a unique minimum exists.
        for (int i = 0; i < N; ++i) C(i, i) -= 1000.0f;

        auto g = greedy_solve(C, kInf);
        auto m = munkres_solve(C);
        EXPECT_NEAR(total_cost(C, g), total_cost(C, m), 1e-3f) << "trial " << trial;
    }
}

// =============================================================================
// M4.2.7 — HungarianDispatcherForwardsArgs
// hungarian_solve(C, Solver::Greedy, max_cost) must return greedy_solve's
// answer; hungarian_solve(C, Solver::Munkres, anything) must return
// munkres_solve's answer.
// =============================================================================
TEST(Hungarian, HungarianDispatcherForwardsArgs) {
    // YOUR CODE:
    //   Use the 3x3 from M4.2.1.
    //   auto via_greedy_direct  = greedy_solve(C, kInf);
    //   auto via_greedy_disp    = hungarian_solve(C, Solver::Greedy, kInf);
    //   auto via_munkres_direct = munkres_solve(C);
    //   auto via_munkres_disp   = hungarian_solve(C, Solver::Munkres, kInf);
    //   EXPECT_EQ(via_greedy_direct,  via_greedy_disp);
    //   EXPECT_EQ(via_munkres_direct, via_munkres_disp);
    //
    // Also: verify max_cost gating flows through to greedy.
    //   With a small max_cost (e.g. 0.5), gating should drop most pairs.
    //   auto gated = hungarian_solve(C, Solver::Greedy, 0.5f);
    //   EXPECT_LT(gated.size(), 3u);
    Eigen::MatrixXf C(3, 3);
    C << 1, 8, 8,
         8, 2, 8,
         8, 8, 3;
    auto via_greedy_direct = greedy_solve(C, kInf);
    auto via_greedy_disp = hungarian_solve(C, Solver::Greedy, kInf);
    auto via_munkres_direct = munkres_solve(C);
    auto via_munkres_disp = hungarian_solve(C, Solver::Munkres, kInf);
    EXPECT_EQ(via_greedy_direct, via_greedy_disp);
    EXPECT_EQ(via_munkres_direct, via_munkres_disp);
}

// =============================================================================
// M4.2.8 — EmptyAndSingleton
// 0x0 and 1x1 cases must not crash. Both solvers handle these the same way.
// =============================================================================
TEST(Hungarian, EmptyAndSingleton) {
    // YOUR CODE:
    //   Empty:
    //     Eigen::MatrixXf C0(0, 0);
    //     EXPECT_TRUE(greedy_solve(C0, kInf).empty());
    //     EXPECT_TRUE(munkres_solve(C0).empty());
    //
    //   Singleton:
    //     Eigen::MatrixXf C1(1, 1);
    //     C1(0, 0) = 7.0f;
    //     auto g = greedy_solve(C1, kInf);
    //     auto m = munkres_solve(C1);
    //     ASSERT_EQ(g.size(), 1u);
    //     ASSERT_EQ(m.size(), 1u);
    //     EXPECT_EQ(g[0], (std::pair{0, 0}));
    //     EXPECT_EQ(m[0], (std::pair{0, 0}));
    //
    //   Singleton with gating:
    //     auto g_gated = greedy_solve(C1, 1.0f);    // 7.0 > 1.0, dropped
    //     EXPECT_TRUE(g_gated.empty());
    Eigen::MatrixXf C0(0, 0);
    EXPECT_TRUE(greedy_solve(C0, kInf).empty());
    EXPECT_TRUE(munkres_solve(C0).empty());
    Eigen::MatrixXf C1(1, 1);
    C1(0, 0) = 7.0f;
    auto g = greedy_solve(C1, kInf);
    auto m = munkres_solve(C1);
    ASSERT_EQ(g.size(), 1u);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(g[0], (std::pair{0, 0}));
    EXPECT_EQ(m[0], (std::pair{0, 0}));
    auto g_gated = greedy_solve(C1, 1.0f);    // 7.0 > 1.0, dropped
    EXPECT_TRUE(g_gated.empty());
}
