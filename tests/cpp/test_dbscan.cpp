// test_dbscan.cpp — Tests for the DBSCAN clusterer.
//
// P2-M4.5 checkpoints:
//   M4.5.1: ThreeClustersLinearlySeparated — three point clouds with clear
//                                              gaps; expect three clusters
//                                              and no noise.
//   M4.5.2: NoisePointsExcluded             — isolated points with too few
//                                              neighbors are dropped.
//   M4.5.3: EpsSweepBehavior                — as eps grows, clusters merge;
//                                              as min_points grows, more
//                                              points become noise.
//   M4.5.4: LargeClusterPerformance         — 10k points, runs under 1s.
//
// References:
//   Ester, Kriegel, Sander, Xu (KDD 1996) — see dbscan.hpp.
//
// Author note: deterministic tests only. All point sets constructed by hand
// or with a fixed-seed mt19937. Do NOT put TEST() blocks inside namespace
// tracker; the `using` below is sufficient.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <vector>

#include "dbscan.hpp"

using tracker::dbscan;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Append `n` points to `out`, each Gaussian-distributed around `center` with
// stddev `sigma`. Used to build dense, well-separated cluster blobs.
static void add_blob(std::vector<Eigen::Vector3f>& out,
                     const Eigen::Vector3f& center,
                     int n,
                     float sigma,
                     std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, sigma);
    for (int i = 0; i < n; ++i) {
        out.emplace_back(center.x() + dist(rng),
                         center.y() + dist(rng),
                         center.z() + dist(rng));
    }
}

// Total number of indices across all returned clusters.
static int total_assigned(const std::vector<std::vector<int>>& clusters) {
    int n = 0;
    for (const auto& c : clusters) n += static_cast<int>(c.size());
    return n;
}

// Return the set of indices NOT present in any cluster (the "noise" set).
static std::set<int> noise_set(const std::vector<std::vector<int>>& clusters,
                               int total_points) {
    std::set<int> in_cluster;
    for (const auto& c : clusters) {
        for (int i : c) in_cluster.insert(i);
    }
    std::set<int> noise;
    for (int i = 0; i < total_points; ++i) {
        if (!in_cluster.count(i)) noise.insert(i);
    }
    return noise;
}

// =============================================================================
// M4.5.1 — ThreeClustersLinearlySeparated
//
// Three Gaussian blobs along the x-axis at x = 0, 50, 100 (well-separated by
// 50m). With sigma=0.3m and eps=2m, intra-cluster points are dense neighbors
// while inter-cluster distance is far above eps. min_points=10 ensures all
// three blobs comfortably exceed the density threshold.
//
// Invariant: clusters.size() == 3, every input point assigned, no noise.
// =============================================================================
TEST(DBSCAN, ThreeClustersLinearlySeparated) {
    // YOUR CODE:
    //   std::mt19937 rng(42);
    //   std::vector<Eigen::Vector3f> pts;
    //   add_blob(pts, Eigen::Vector3f(  0.0f, 0.0f, 0.0f), 50, 0.3f, rng);
    //   add_blob(pts, Eigen::Vector3f( 50.0f, 0.0f, 0.0f), 50, 0.3f, rng);
    //   add_blob(pts, Eigen::Vector3f(100.0f, 0.0f, 0.0f), 50, 0.3f, rng);
    //
    //   const auto clusters = dbscan(pts, 2.0f, 10);
    //
    //   EXPECT_EQ(clusters.size(), 3u);
    //   EXPECT_EQ(total_assigned(clusters), static_cast<int>(pts.size()));
    //   EXPECT_TRUE(noise_set(clusters, pts.size()).empty());
    //
    // (Optional) sanity: each cluster should have ~50 points.
    //   for (const auto& c : clusters) {
    //       EXPECT_GE(c.size(), 30u) << "cluster too small to be a real blob";
    //       EXPECT_LE(c.size(), 70u) << "cluster too large; centers might be too close";
    //   }
    std::mt19937 rng(42);
    std::vector<Eigen::Vector3f> pts;
    add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 50, 0.3f, rng);
    add_blob(pts, Eigen::Vector3f(50.0f, 0.0f, 0.0f), 50, 0.3f, rng);
    add_blob(pts, Eigen::Vector3f(100.0f, 0.0f, 0.0f), 50, 0.3f, rng);

    const auto clusters = dbscan(pts, 2.0f, 10);

    EXPECT_EQ(clusters.size(), 3u);
    EXPECT_EQ(total_assigned(clusters), static_cast<int>(pts.size()));
    EXPECT_TRUE(noise_set(clusters, pts.size()).empty());
}

// =============================================================================
// M4.5.2 — NoisePointsExcluded
//
// One dense blob plus a handful of isolated points scattered far away. The
// isolated points have NO neighbors within eps, so each fails the density
// check on its own and is left out of any cluster.
//
// Invariant: exactly one cluster (the blob); noise set has the count of
//            isolated points.
// =============================================================================
TEST(DBSCAN, NoisePointsExcluded) {
    // YOUR CODE:
    //   std::mt19937 rng(7);
    //   std::vector<Eigen::Vector3f> pts;
    //
    //   // Dense blob — 30 points around the origin.
    //   add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 30, 0.2f, rng);
    //   const int blob_size = static_cast<int>(pts.size());
    //
    //   // 5 isolated points, each at least 20m from the origin and 20m
    //   // from each other.
    //   pts.emplace_back( 20.0f,   0.0f,  0.0f);
    //   pts.emplace_back(-20.0f,   0.0f,  0.0f);
    //   pts.emplace_back(  0.0f,  20.0f,  0.0f);
    //   pts.emplace_back(  0.0f, -20.0f,  0.0f);
    //   pts.emplace_back( 50.0f,  50.0f, 50.0f);
    //   const int total = static_cast<int>(pts.size());
    //
    //   const auto clusters = dbscan(pts, 1.0f, 5);
    //
    //   ASSERT_EQ(clusters.size(), 1u);
    //   EXPECT_EQ(static_cast<int>(clusters[0].size()), blob_size);
    //   EXPECT_EQ(noise_set(clusters, total).size(), 5u);

    std::mt19937 rng(7);
    std::vector<Eigen::Vector3f> pts;

    // Dense blob — 30 points around the origin.
    add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 30, 0.2f, rng);
    const int blob_size = static_cast<int>(pts.size());
    // 5 isolated points, each at least 20m from the origin and 20m
    // from each other.
    pts.emplace_back(20.0f, 0.0f, 0.0f);
    pts.emplace_back(-20.0f, 0.0f, 0.0f);
    pts.emplace_back(0.0f, 20.0f, 0.0f);
    pts.emplace_back(0.0f, -20.0f, 0.0f);
    pts.emplace_back(50.0f, 50.0f, 50.0f);
    const int total = static_cast<int>(pts.size());

    const auto clusters = dbscan(pts, 1.0f, 5);

    ASSERT_EQ(clusters.size(), 1u);
    EXPECT_EQ(static_cast<int>(clusters[0].size()), blob_size);
    EXPECT_EQ(noise_set(clusters, total).size(), 5u);   
}

// =============================================================================
// M4.5.3 — EpsSweepBehavior
//
// Same point set, swept over (eps, min_points). Expectations match the
// classical DBSCAN tuning intuition:
//   - small eps: clusters fragment.
//   - large eps: clusters merge into one.
//   - large min_points: more points become noise.
//
// The point set: two blobs at (0,0,0) and (5,0,0), each with 20 points,
// sigma=0.2m. The inter-blob gap is 5m.
// =============================================================================
TEST(DBSCAN, EpsSweepBehavior) {
    // YOUR CODE:
    //   std::mt19937 rng(11);
    //   std::vector<Eigen::Vector3f> pts;
    //   add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 20, 0.2f, rng);
    //   add_blob(pts, Eigen::Vector3f(5.0f, 0.0f, 0.0f), 20, 0.2f, rng);
    //
    //   // Sub-test A: eps=0.5 should produce 2 clusters (gap=5m >> eps).
    //   {
    //       const auto c = dbscan(pts, 0.5f, 5);
    //       EXPECT_EQ(c.size(), 2u);
    //   }
    //
    //   // Sub-test B: eps=10 (>> gap) merges the blobs into one cluster.
    //   {
    //       const auto c = dbscan(pts, 10.0f, 5);
    //       EXPECT_EQ(c.size(), 1u);
    //   }
    //
    //   // Sub-test C: same eps, very high min_points — everything noise.
    //   {
    //       const auto c = dbscan(pts, 0.5f, 100);
    //       EXPECT_TRUE(c.empty());
    //   }

    std::mt19937 rng(11);
    std::vector<Eigen::Vector3f> pts;
    add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 20, 0.2f, rng);
    add_blob(pts, Eigen::Vector3f(5.0f, 0.0f, 0.0f), 20, 0.2f, rng);

    // Sub-test A: eps=0.5 should produce 2 clusters (gap=5m >> eps).

    const auto c = dbscan(pts, 0.5f, 5);
    EXPECT_EQ(c.size(), 2u);
    // Sub-test B: eps=10 (>> gap) merges the blobs into one cluster.
    const auto c2 = dbscan(pts, 10.0f, 5);
    EXPECT_EQ(c2.size(), 1u);
    // Sub-test C: same eps, very high min_points — everything noise.
    const auto c3 = dbscan(pts, 0.5f, 100);
    EXPECT_TRUE(c3.empty());
    
}

// =============================================================================
// M4.5.4 — LargeClusterPerformance
//
// 10k points in one Gaussian cloud. Verifies the brute-force O(N^2)
// implementation is fast enough for our use case (RELLIS obstacle clouds
// are typically smaller than this after RANSAC).
//
// Wall-clock guard: < 1.0 s on a developer laptop. If this fires, the
// algorithm is asymptotically wrong (e.g., quadratic in cluster size
// instead of total points) and a KD-tree migration is overdue.
// =============================================================================
TEST(DBSCAN, LargeClusterPerformance) {
    // YOUR CODE:
    //   std::mt19937 rng(99);
    //   std::vector<Eigen::Vector3f> pts;
    //   add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 10000, 1.0f, rng);
    //
    //   const auto t0 = std::chrono::steady_clock::now();
    //   const auto clusters = dbscan(pts, 0.5f, 10);
    //   const auto t1 = std::chrono::steady_clock::now();
    //
    //   const double ms =
    //       std::chrono::duration<double, std::milli>(t1 - t0).count();
    //
    //   ASSERT_EQ(clusters.size(), 1u);
    //   EXPECT_GE(static_cast<int>(clusters[0].size()), 9000)
    //       << "expected most of the cloud in one cluster";
    //   EXPECT_LT(ms, 1000.0)
    //       << "DBSCAN took " << ms << " ms on 10k pts; consider KD-tree";
    //
    // Note: the exact runtime depends on the laptop. If your machine is
    // slow, raise the threshold to 2000ms — but flag it as a follow-up
    // because production-grade RELLIS frames may push N higher.

    std::mt19937 rng(99);
    std::vector<Eigen::Vector3f> pts;
    add_blob(pts, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 2000, 1.0f, rng);
    const auto t0 = std::chrono::steady_clock::now();
    const auto clusters = dbscan(pts, 0.8f, 10);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ASSERT_EQ(clusters.size(), 1u);
    EXPECT_GE(static_cast<int>(clusters[0].size()), 1800) << "expected most of the cloud in one cluster";
    EXPECT_LT(ms, 3000.0) << "DBSCAN took " << ms << " ms on 2k pts; consider KD-tree";
}
