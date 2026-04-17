// test_scan_context.cpp — Tests for Scan Context loop closure detection.
// Verifies descriptor construction, cosine distance, and loop detection.                                                                                                 
//                                          
// P2-M2.1 checkpoints:                                                                                                                                                   
//   M2.1.1: Descriptor has correct dimensions and bins points correctly
//   M2.1.2: Cosine distance properties (identity, symmetry, bounds)                                                                                                      
//   M2.1.3: Column shift recovers known yaw rotation
//   M2.1.4: Loop detection filters by min-gap and threshold                                                                                                              
//                                                                                                                                                                        
// References:                                                                                                                                                            
//   Kim & Kim, "Scan Context" (IROS 2018), Sections 3–4                                                                                                                  
                                                                                                                                                                        
#include <gtest/gtest.h>                                  
#include <Eigen/Dense>                                                                                                                                                    
#include <cmath>                                                                                                                                                          
#include <vector>                           
#include "scan_context.hpp"                                                                                                                                               
                                                        
#ifndef M_PI                                                                                                                                                              
#define M_PI 3.14159265358979323846
#endif                                                                                                                                                                    
                                                        
static constexpr float kTolStrict = 1e-6f;  
static constexpr float kTolNumeric = 1e-4f;
                                            
// Helper: generate a ring of points at a given range and height                                                                                                          
static std::vector<Eigen::Vector3f> makeRingCloud(float range, float z, int n_points) {
    // YOUR HELPER:                                                                                                                                                       
    //   Generate n_points equally spaced around a circle at (range, z)                                                                                                   
    //   x = range * cos(theta), y = range * sin(theta), z = z                                                                                                            
    //   theta = i * 2π / n_points for i = 0..n_points-1                                                                                                                  
    std::vector<Eigen::Vector3f> pts;                                                                                                                                     
    for (int i = 0; i < n_points; ++i) {
        float theta = i * 2.0f * M_PI / n_points;
        float x = range * std::cos(theta);
        float y = range * std::sin(theta);
        pts.emplace_back(x, y, z);
    }
    return pts;                                                                                                                                                           
}                                                                                                                                                                         
                                                                                                                                                                        
// Helper: generate a single point at known (range, azimuth, z)
static std::vector<Eigen::Vector3f> makeSinglePoint(float range, float azimuth_rad, float z) {                                                                            
    // YOUR HELPER:                                       
    //   x = range * cos(azimuth), y = range * sin(azimuth)                                                                                                               
    //   return vector with one point       
    std::vector<Eigen::Vector3f> pts;                                                                                                                                     
    float x = range * std::cos(azimuth_rad);
    float y = range * std::sin(azimuth_rad);
    pts.emplace_back(x, y, z);
    return pts;
}                                                                                                                                                                         
                                                                                                                                                                        
// Helper: rotate a point cloud by yaw angle (z-axis rotation)                                                                                                            
static std::vector<Eigen::Vector3f> rotateCloudYaw(                                                                                                                       
    const std::vector<Eigen::Vector3f>& pts, float yaw_rad) {
    // YOUR HELPER:                                                                                                                                                       
    //   For each point, apply 2D rotation:
    //     x' = x*cos(yaw) - y*sin(yaw)                                                                                                                                   
    //     y' = x*sin(yaw) + y*cos(yaw)                                                                                                                                   
    //     z' = z
    std::vector<Eigen::Vector3f> rotated;       
    float cos_yaw = std::cos(yaw_rad);
    float sin_yaw = std::sin(yaw_rad);
    for (const auto& pt : pts) {
        float x = pt.x();
        float y = pt.y();
        float z = pt.z();
        float x_rot = x * cos_yaw - y * sin_yaw;
        float y_rot = x * sin_yaw + y * cos_yaw;
        rotated.emplace_back(x_rot, y_rot, z);
    }                                                                                                                          
    return rotated;                                       
}                                                                                                                                                                         
                                                                                                                                                                        
// -----------------------------------------------------------------------------
// M2.1.1 — Descriptor construction                                                                                                                                       
// -----------------------------------------------------------------------------
                                            
TEST(ScanContext, EmptyCloudGivesZeroDescriptor) {
    // YOUR TEST:                       
    //   - empty point cloud → all-zero descriptor                                                                                                                        
    //   - assert desc.norm() == 0 
    scan_context::Descriptor desc = scan_context::buildDescriptor({});
    ASSERT_NEAR(desc.norm(), 0.0f, kTolStrict);                                                                                                                                       
}                                                                                                                                                                         
                                                                                                                                                                        
TEST(ScanContext, DescriptorDimensions) {                                                                                                                                 
    // YOUR TEST:                       
    //   - build descriptor from any non-empty cloud                                                                                                                      
    //   - assert desc.rows() == N_RINGS && desc.cols() == N_SECTORS     
    auto cloud = makeRingCloud(5.0f, 1.0f, 100);
    scan_context::Descriptor desc = scan_context::buildDescriptor(cloud);
    ASSERT_EQ(desc.rows(), scan_context::N_RINGS);
    ASSERT_EQ(desc.cols(), scan_context::N_SECTORS);                                                                                                     
}                                           
                                                                                                                                                                        
TEST(ScanContext, SinglePointInCorrectBin) {              
    // YOUR TEST:                                                                                                                                                         
    //   - place one point at known (range, azimuth, z)
    //   - compute expected ring and sector indices manually                                                                                                              
    //   - assert desc(ring, sector) == z                 
    //   - assert all other bins are 0
    float range = 10.0f;
    float azimuth = M_PI / 4.0f; // 45 degrees
    float z = 2.5f;
    auto cloud = makeSinglePoint(range, azimuth, z);
    scan_context::Descriptor desc = scan_context::buildDescriptor(cloud);
    int expected_ring = std::min(static_cast<int>(std::floor(range / 80.0f * scan_context::N_RINGS)), scan_context::N_RINGS - 1);
    int expected_sector = std::min(static_cast<int>(std::floor((azimuth + M_PI) / (2 * M_PI) * scan_context::N_SECTORS)), scan_context::N_SECTORS - 1);
    ASSERT_NEAR(desc(expected_ring, expected_sector), z, kTolStrict);
    for (int r = 0; r < desc.rows(); ++r) {
        for (int c = 0; c < desc.cols(); ++c) {
            if (r == expected_ring && c == expected_sector) continue;
            ASSERT_NEAR(desc(r, c), 0.0f, kTolStrict);  
        }
    }
}                                                         
                                                                                                                                                                        
TEST(ScanContext, MaxHeightPerBin) {
    // YOUR TEST:                                                                                                                                                         
    //   - place two points in the same (ring, sector) at z=1.0 and z=3.0
    //   - assert desc(ring, sector) == 3.0 (max, not mean or last)           
    float range = 10.0f;
    float azimuth = M_PI / 4.0f; // 45 degrees
    auto cloud = makeSinglePoint(range, azimuth, 1.0f);
    auto cloud2 = makeSinglePoint(range, azimuth, 3.0f);
    cloud.insert(cloud.end(), cloud2.begin(), cloud2.end());
    scan_context::Descriptor desc = scan_context::buildDescriptor(cloud);
    int expected_ring = std::min(static_cast<int>(std::floor(range / 80.0f * scan_context::N_RINGS)), scan_context::N_RINGS - 1);
    int expected_sector = std::min(static_cast<int>(std::floor((azimuth + M_PI) / (2 * M_PI) * scan_context::N_SECTORS)), scan_context::N_SECTORS - 1);
    ASSERT_NEAR(desc(expected_ring, expected_sector), 3.0f, kTolStrict);                                                                                            
}                                                         
                                        
TEST(ScanContext, PointsOutOfRangeSkipped) {                                                                                                                              
    // YOUR TEST:                                                                                                                                                         
    //   - place one point at range > max_range, one at range < 1.0                                                                                                       
    //   - assert descriptor is all zeros (both skipped)        
    auto cloud = makeSinglePoint(100.0f, 0.0f, 1.0f); // out of range
    auto cloud2 = makeSinglePoint(0.5f, 0.0f, 1.0f); // too close
    cloud.insert(cloud.end(), cloud2.begin(), cloud2.end());
    scan_context::Descriptor desc = scan_context::buildDescriptor(cloud);
    ASSERT_NEAR(desc.norm(), 0.0f, kTolStrict);                                                                                                           
}                                                         
                                                                                                                                                                        
// -----------------------------------------------------------------------------                                                                                          
// M2.1.2 — Cosine distance                                                                                                                                               
// -----------------------------------------------------------------------------                                                                                          
                                                                                                                                                                        
TEST(ScanContext, IdenticalDescriptorsZeroDistance) {     
    // YOUR TEST:                       
    //   - build descriptor from a ring cloud
    //   - distance to itself at shift=0 should be 0.0                                                                                                                    
    //   - assert columnShiftDistance(desc, desc, 0) < kTolStrict
    auto cloud = makeRingCloud(5.0f, 1.0f, 100);
    scan_context::Descriptor desc = scan_context::buildDescriptor(cloud);
    float dist = scan_context::columnShiftDistance(desc, desc, 0);
    ASSERT_NEAR(dist, 0.0f, kTolStrict);
}                                                                                                                                                                         
                                                                                                                                                                        
TEST(ScanContext, DistanceIsSymmetric) {
    // YOUR TEST:                                                                                                                                                         
    //   - build two different descriptors (different ring radii or heights)
    //   - assert columnShiftDistance(a, b, 0) == columnShiftDistance(b, a, 0)       
    auto cloud1 = makeRingCloud(5.0f, 1.0f, 100);
    auto cloud2 = makeRingCloud(10.0f, 2.0f, 100);
    scan_context::Descriptor desc1 = scan_context::buildDescriptor(cloud1);
    scan_context::Descriptor desc2 = scan_context::buildDescriptor(cloud2);
    float dist1 = scan_context::columnShiftDistance(desc1, desc2, 0);
    float dist2 = scan_context::columnShiftDistance(desc2, desc1, 0);
    ASSERT_NEAR(dist1, dist2, kTolNumeric);                                                                                      
}                                                         
                                                                                                                                                                        
TEST(ScanContext, DistanceBounded) {                                                                                                                                      
    // YOUR TEST:                                                                                                                                                         
    //   - for any two descriptors, distance should be in [0, 2]                                                                                                          
    //   - build a few pairs, check bounds              
    auto cloud1 = makeRingCloud(5.0f, 1.0f, 100);
    auto cloud2 = makeRingCloud(10.0f, 2.0f, 100);
    auto cloud3 = makeRingCloud(20.0f, 3.0f, 100);
    scan_context::Descriptor desc1 = scan_context::buildDescriptor(cloud1);
    scan_context::Descriptor desc2 = scan_context::buildDescriptor(cloud2);
    scan_context::Descriptor desc3 = scan_context::buildDescriptor(cloud3); 
    float dist12 = scan_context::columnShiftDistance(desc1, desc2, 0);
    float dist23 = scan_context::columnShiftDistance(desc2, desc3, 0);
    float dist13 = scan_context::columnShiftDistance(desc1, desc3, 0);
    ASSERT_GE(dist12, 0.0f);
    ASSERT_LE(dist12, 2.0f);
    ASSERT_GE(dist23, 0.0f);
    ASSERT_LE(dist23, 2.0f);
    ASSERT_GE(dist13, 0.0f);
    ASSERT_LE(dist13, 2.0f);                                                                                                                  
}                                           
                                                                                                                                                                        
// -----------------------------------------------------------------------------
// M2.1.3 — Column shift alignment                                                                                                                                        
// -----------------------------------------------------------------------------
                                                                                                                                                                        
TEST(ScanContext, ShiftRecoverKnownYaw) {                 
    // YOUR TEST — the core rotation-invariance test:                                                                                                                     
    //   - build a descriptor from a non-symmetric cloud (e.g., points only
    //     in one sector quadrant)                                                                                                                                        
    //   - rotate the cloud by exactly (2π / N_SECTORS) * K radians (K sectors)
    //   - build descriptor from the rotated cloud                                                                                                                        
    //   - matchDescriptors should return best_shift ≈ K (or N_SECTORS - K)                                                                                               
    //   - distance should be near 0 
    
    std::vector<Eigen::Vector3f> cloud;                                                                                                                                       
    // Put points only in the first quadrant, at varying ranges depending on angle                                                                                            
    for (int i = 0; i < 25; i++) {                                                                                                                                            
        float angle = (M_PI / 2.0f) * i / 25.0f;   // angle: 0 to π/2                                                                                                         
        float range = 5.0f + angle * 10.0f;         // range: 5m to ~20m as angle grows
        float x = range * std::cos(angle);                                                                                                                                    
        float y = range * std::sin(angle);                    
        cloud.push_back(Eigen::Vector3f(x, y, 1.0f));                                                                                                                         
    }                                                                                                                                                                         
                                                                                                                                                                            
    auto cloud_rot = rotateCloudYaw(cloud, 2.0f * M_PI / scan_context::N_SECTORS * 10);                                                                                       
    auto desc1 = scan_context::buildDescriptor(cloud);                                                                                                                        
    auto desc2 = scan_context::buildDescriptor(cloud_rot);
    auto [dist, shift] = scan_context::matchDescriptors(desc1, desc2);                                                                                                        
                                                            
    ASSERT_EQ(shift, 10);                     // EQ not NEAR — shift is an int                                                                                                
    ASSERT_NEAR(dist, 0.0f, 1e-1f);           // loosen tolerance — discrete binning introduces small error                                                                                                         
}                                                                                                                                                                         
                                                        
// -----------------------------------------------------------------------------                                                                                          
// M2.1.4 — Loop closure detection                                                                                                                                        
// -----------------------------------------------------------------------------                                                                                          
                                                                                                                                                                        
TEST(ScanContext, MinGapRejectsNearbyFrames) {            
    // YOUR TEST:                       
    //   - create 100 identical descriptors (every frame looks the same)
    //   - detectLoopClosures(history, query_id=99, thresh=1.0, min_gap=50)                                                                                               
    //   - all returned candidates should have match_id <= 49
    //   - no candidate should have match_id in [50, 98]      
    std::vector<scan_context::Descriptor> history(100, scan_context::Descriptor::Zero());
    auto candidates = scan_context::detectLoopClosures(history, 99, 1.0f, 50);
    for (const auto& candidate : candidates) {
        ASSERT_LE(candidate.match_id, 49);
        ASSERT_TRUE(candidate.match_id < 50 || candidate.match_id >= 99);
    }                                                                                                            
}                                                                                                                                                                         
                                                                                                                                                                        
TEST(ScanContext, ThresholdRejectsWeakMatches) {                                                                                                                          
    // YOUR TEST:                                                                                                                                                         
    //   - create history with diverse descriptors (different heights/ranges)                                                                                             
    //   - set threshold very low (e.g., 0.01)                                                                                                                            
    //   - assert no candidates returned (nothing matches that tightly)
    std::vector<scan_context::Descriptor> history;
    for (int i = 0; i < 100; ++i) {
        auto cloud = makeRingCloud(5.0f + i, 1.0f + i * 0.1f, 100);
        history.push_back(scan_context::buildDescriptor(cloud));
    }
    auto candidates = scan_context::detectLoopClosures(history, 99, 0.01f, 50);
    ASSERT_TRUE(candidates.empty());
}                                                                                                                                                                         
                                                                                                                                                                        
TEST(ScanContext, DetectsExactRevisit) {    
    // YOUR TEST:                                                                                                                                                         
    //   - create history: frames 0–49 at location A, frames 50–99 at location B,
    //     frame 100 = same descriptor as frame 0 (exact revisit)                                                                                                         
    //   - detectLoopClosures(history, 100, thresh=0.3, min_gap=50)
    //   - assert at least one candidate with match_id == 0 (or near 0)                                                                                                   
    //   - assert candidate distance ≈ 0               
    std::vector<scan_context::Descriptor> history;
    auto cloudA = makeRingCloud(5.0f, 1.0f, 100);
    auto cloudB = makeRingCloud(10.0f, 2.0f, 100);
    for (int i = 0; i < 50; ++i) {
        history.push_back(scan_context::buildDescriptor(cloudA));
        history.push_back(scan_context::buildDescriptor(cloudB));
    }
    history.push_back(scan_context::buildDescriptor(cloudA));
    auto candidates = scan_context::detectLoopClosures(history, 100, 0.3f, 50);
    bool found_revisit = false;
    for (const auto& candidate : candidates) {
        if (candidate.match_id == 0) {
            found_revisit = true;
            ASSERT_NEAR(candidate.distance, 0.0f, kTolStrict);  
        }
    }
    ASSERT_TRUE(found_revisit);                                                                                                                                       
                                                                                                               
}