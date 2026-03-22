// hungarian.hpp
// Hungarian (Munkres) algorithm for optimal assignment — implement from scratch.
//
// PDF reference: Part 5.2 (page 16)
// Read: Kuhn, "The Hungarian Method for the Assignment Problem" (1955)
// Alt: brilliant.org/wiki/hungarian-matching/ or TopCoder tutorial
//
// YOUR TASK:
//   Option A (recommended for time): Greedy distance matching — O(NM)
//   Option B (impressive for interviews): Full Munkres — O(N^3), ~200 lines
//
// Input:  NxM cost matrix (e.g., Euclidean distance between detections and tracks)
// Output: vector of (row, col) assignment pairs minimizing total cost

#pragma once
#include <Eigen/Dense>
#include <utility>
#include <vector>

std::vector<std::pair<int, int>> hungarian_solve(const Eigen::MatrixXf& cost_matrix);
