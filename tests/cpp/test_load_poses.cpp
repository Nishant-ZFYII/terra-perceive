// =============================================================================
// test_load_poses.cpp — quaternion-order + header-driven CSV parser (P2-M3 step 0)
// =============================================================================
//
// Test plan:
//
//   Positive cases
//     - LoadsWFirstFile                  (poses_wfirst.csv)
//     - LoadsWLastFile                   (poses_wlast.csv)
//     - BothFilesProduceEqualRotations   (gold-standard equivalence test)
//
//   Negative cases (return empty, no crash)
//     - MissingFileReturnsEmpty
//     - MissingQwColumnReturnsEmpty
//     - MalformedHeaderReturnsEmpty
//
// Test files live at: tests/data/poses_wfirst.csv, poses_wlast.csv,
//                     poses_missing_qw.csv
//
// Both positive fixtures encode the same three rotations:
//   frame 0: identity
//   frame 1: 90° yaw around z
//   frame 2: 180° yaw around z
// The files differ ONLY in column order (W-first vs W-last) and time-column
// naming (timestamp vs frame_id). The rotation matrices MUST be bit-identical
// within float tolerance — if they aren't, the quaternion loader is
// misinterpreting column positions.
//
// DO NOT fill in:
//   - Expected rotation matrices (compute from axis-angle by hand, paste in)
//
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "pose_graph_slam.hpp"

using pose_graph::Pose;

// Test WORKING_DIRECTORY is set to ${CMAKE_SOURCE_DIR} in CMakeLists.txt so
// relative paths resolve to the repo root.
static const std::string kWFirstPath     = "tests/data/poses_wfirst.csv";
static const std::string kWLastPath      = "tests/data/poses_wlast.csv";
static const std::string kMissingQwPath  = "tests/data/poses_missing_qw.csv";

// -----------------------------------------------------------------------------
// Positive cases
// -----------------------------------------------------------------------------

TEST(LoadPoses, LoadsWFirstFile) {
    // YOUR CODE:
    //   auto [poses, timestamps] = pose_graph::loadPosesFromCSV(kWFirstPath);
    //   ASSERT_EQ(poses.size(), 3u);
    //   ASSERT_EQ(timestamps.size(), 3u);
    //
    //   Frame 0 should be identity:
    //     EXPECT_TRUE(poses[0].R.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
    //     EXPECT_TRUE(poses[0].t.isApprox(Eigen::Vector3d::Zero()));
    //
    //   Frame 1 (90° yaw):
    //     Eigen::Matrix3d R_yaw90;
    //     R_yaw90 << 0, -1, 0,
    //                1,  0, 0,
    //                0,  0, 1;
    //     EXPECT_TRUE(poses[1].R.isApprox(R_yaw90, 1e-5));
    //
    //   Frame 2 (180° yaw):
    //     Eigen::Matrix3d R_yaw180;
    //     R_yaw180 << -1,  0, 0,
    //                  0, -1, 0,
    //                  0,  0, 1;
    //     EXPECT_TRUE(poses[2].R.isApprox(R_yaw180, 1e-5));

    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(kWFirstPath);
    ASSERT_EQ(poses.size(), 3u);
    ASSERT_EQ(timestamps.size(), 3u);
    EXPECT_TRUE(poses[0].R.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
    EXPECT_TRUE(poses[0].t.isApprox(Eigen::Vector3d::Zero()));
    Eigen::Matrix3d R_yaw90;
    R_yaw90 << 0, -1, 0,
               1,  0, 0,
               0,  0, 1;
    EXPECT_TRUE(poses[1].R.isApprox(R_yaw90, 1e-5));
    Eigen::Matrix3d R_yaw180;
    R_yaw180 << -1,  0, 0,
                 0, -1, 0,
                 0,  0, 1;
    EXPECT_TRUE(poses[2].R.isApprox(R_yaw180, 1e-5));
}

TEST(LoadPoses, LoadsWLastFile) {
    // YOUR CODE:
    //   Same assertions as LoadsWFirstFile, loading kWLastPath.
    //   If the loader is broken (e.g. constructs Quaterniond from qx,qy,qz,qw
    //   instead of qw,qx,qy,qz), the rotations will be garbage for W-last
    //   files and this test will fail.

    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(kWLastPath);
    ASSERT_EQ(poses.size(), 3u);
    ASSERT_EQ(timestamps.size(), 3u);
    EXPECT_TRUE(poses[0].R.isApprox(Eigen::Matrix3d::Identity(), 1e-5));
    EXPECT_TRUE(poses[0].t.isApprox(Eigen::Vector3d::Zero()));
    Eigen::Matrix3d R_yaw90;
    R_yaw90 << 0, -1, 0,
               1,  0, 0,
               0,  0, 1;
    EXPECT_TRUE(poses[1].R.isApprox(R_yaw90, 1e-5));
    Eigen::Matrix3d R_yaw180;
    R_yaw180 << -1,  0, 0,
                 0, -1, 0,
                 0,  0, 1;
    EXPECT_TRUE(poses[2].R.isApprox(R_yaw180, 1e-5));
}

TEST(LoadPoses, BothFilesProduceEqualRotations) {
    // YOUR CODE — the gold-standard equivalence test:
    //   auto [p_wfirst, _t1] = pose_graph::loadPosesFromCSV(kWFirstPath);
    //   auto [p_wlast,  _t2] = pose_graph::loadPosesFromCSV(kWLastPath);
    //   ASSERT_EQ(p_wfirst.size(), p_wlast.size());
    //   for (size_t i = 0; i < p_wfirst.size(); ++i) {
    //       EXPECT_TRUE(p_wfirst[i].R.isApprox(p_wlast[i].R, 1e-5))
    //           << "frame " << i << ": W-first/W-last disagreement";
    //       EXPECT_TRUE(p_wfirst[i].t.isApprox(p_wlast[i].t, 1e-5));
    //   }
    //
    //   This is the single test that proves the header-order fix works.
    //   If it passes, ablation A (four-source map comparison) can proceed.

    auto [p_wfirst, _t1] = pose_graph::loadPosesFromCSV(kWFirstPath);
    auto [p_wlast,  _t2] = pose_graph::loadPosesFromCSV(kWLastPath);
    ASSERT_EQ(p_wfirst.size(), p_wlast.size());
    for (size_t i = 0; i < p_wfirst.size(); ++i) {
        EXPECT_TRUE(p_wfirst[i].R.isApprox(p_wlast[i].R, 1e-5))
            << "frame " << i << ": W-first/W-last disagreement";
        EXPECT_TRUE(p_wfirst[i].t.isApprox(p_wlast[i].t, 1e-5));
    }
}

// -----------------------------------------------------------------------------
// Negative cases
// -----------------------------------------------------------------------------

TEST(LoadPoses, MissingFileReturnsEmpty) {
    auto [poses, timestamps] = pose_graph::loadPosesFromCSV("/nonexistent/path.csv");
    EXPECT_TRUE(poses.empty());
    EXPECT_TRUE(timestamps.empty());
}

TEST(LoadPoses, MissingQwColumnReturnsEmpty) {
    // YOUR CODE:
    //   Load kMissingQwPath (has timestamp,x,y,z,qx,qy,qz — no qw).
    //   ASSERT_TRUE(poses.empty());   // loader must bail on missing required column
    //   The file's stderr will log "missing column: qw" — stderr noise is OK.
    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(kMissingQwPath);
    EXPECT_TRUE(poses.empty());
    EXPECT_TRUE(timestamps.empty());
}

TEST(LoadPoses, MalformedHeaderReturnsEmpty) {
    // YOUR CODE (optional — exercises the empty-column-name guard):
    //   Write a temp file with header "timestamp,,x,y,z,qw,qx,qy,qz" (double
    //   comma → empty column name).
    //   Assert loadPosesFromCSV returns empty.
    //   Cleanup the temp file.
    //   Skip if you'd rather spend the time on other tests — the negative
    //   path is already well-covered by MissingQw.

    std::string tmp_path = ::testing::TempDir() + "/malformed_header.csv";
    std::ofstream tmp_file(tmp_path);
    tmp_file << "timestamp,,x,y,z,qw,qx,qy,qz\n";  // double comma → empty column name
    tmp_file.close();
    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(tmp_path);
    EXPECT_TRUE(poses.empty());
    EXPECT_TRUE(timestamps.empty());
    std::remove(tmp_path.c_str());
}
