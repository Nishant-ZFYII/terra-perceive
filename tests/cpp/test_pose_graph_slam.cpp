// test_pose_graph_slam.cpp — Tests for pose graph SLAM.
// Build tests incrementally with the implementation layers:
//   Layer 1 tests: graph assembly (addEdge, setFixed, setInitialPoses)
//   Layer 2 tests: residuals + Jacobians (one edge type at a time)
//   Layer 3 tests: optimizer convergence on toy problems
//   Layer 4 tests: manifold vs Euclidean ATE on the same graph
//
// P2-M2.3 + P2-M2.4 checkpoints:
//   M2.3.1: Graph assembly
//   M2.4.1: Per-edge residuals correct (zero at ground truth)
//   M2.4.2: Analytical Jacobian matches numerical Jacobian
//   M2.4.3: Gauss-Newton converges on a simple 2-node graph
//   M2.4.4: Fixed anchor stays put after optimization
//   M2.4.5: Loop closure reduces ATE vs odometry-only baseline
//   M2.4.6: Manifold retraction keeps R ∈ SO(3); Euclidean drifts
//
// References:
//   Grisetti et al. 2010, Forster et al. 2015, Solà micro Lie theory

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "pose_graph_slam.hpp"
#include "so3.hpp"
#include "imu_preintegration.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr double kTolStrict = 1e-10;
static constexpr double kTolNumeric = 1e-6;
static constexpr double kTolOptimized = 1e-3;   // after optimization, looser

// -----------------------------------------------------------------------------
// Test fixtures / helpers
// -----------------------------------------------------------------------------

// Helper: build a simple 3-node chain with known ground-truth relative transforms.
// Frame 0: identity
// Frame 1: translated +1m in x
// Frame 2: translated another +1m in x, yawed 10°
static std::vector<pose_graph::Pose> makeGroundTruthChain() {
    // YOUR HELPER:
    //   Return 3 poses:
    //     p0: R = I, t = (0, 0, 0)
    //     p1: R = I, t = (1, 0, 0)
    //     p2: R = Exp((0, 0, π/18)) yaw, t = (2, 0, 0)
    std::vector<pose_graph::Pose> gt;
    pose_graph::Pose p0;
    p0.R = Eigen::Matrix3d::Identity();
    p0.t = Eigen::Vector3d::Zero();
    gt.push_back(p0);

    pose_graph::Pose p1;
    p1.R = Eigen::Matrix3d::Identity();
    p1.t = Eigen::Vector3d(1, 0, 0);
    gt.push_back(p1);

    pose_graph::Pose p2;
    double yaw = M_PI / 18;  // 10 degrees in radians
    p2.R = so3::Exp(Eigen::Vector3d(0, 0, yaw));
    p2.t = Eigen::Vector3d(2, 0, 0);
    gt.push_back(p2);

    return gt;
}

// Helper: produce a noisy initial estimate from ground-truth poses
static std::vector<pose_graph::Pose> perturbPoses(
    const std::vector<pose_graph::Pose>& gt, double noise_trans, double noise_rot) {
    // YOUR HELPER:
    //   For each pose, add Gaussian noise (fixed seed for determinism):
    //     t += noise_trans * random_vec
    //     R = R * so3::Exp(noise_rot * random_vec)
    std::vector<pose_graph::Pose> noisy;
    for (int i = 0; i < gt.size(); i++) {   
        pose_graph::Pose p = gt[i];
        // Add noise to translation
        Eigen::Vector3d noise_t = noise_trans * Eigen::Vector3d::Random();
        p.t += noise_t;
        // Add noise to rotation
        Eigen::Vector3d noise_r = noise_rot * Eigen::Vector3d::Random();
        p.R = so3::Exp(noise_r) * p.R;
        noisy.push_back(p);
    }
    return noisy;
}

// Helper: build an ICP edge from two ground-truth poses
static pose_graph::ICPEdge makeICPEdgeFromGT(
    const pose_graph::Pose& p_from, const pose_graph::Pose& p_to,
    int i, int j, double information_scale = 100.0) {
    // YOUR HELPER:
    //   dR = p_from.R.transpose() * p_to.R
    //   dp = p_from.R.transpose() * (p_to.t - p_from.t)
    //   information = Matrix<6,6>::Identity() * information_scale
    pose_graph::ICPEdge e;
    e.from = i; e.to = j;
    e.dR_meas = p_from.R.transpose() * p_to.R;
    e.dp_meas = p_from.R.transpose() * (p_to.t - p_from.t);
    e.information = Eigen::Matrix<double, 6, 6>::Identity() * information_scale;
    return e;
}
//Helper: numerical Jacobian via central differences (for testing/validation)
template<typename ResidualFunc>  
Eigen::MatrixXd numericalJacobian(std::function<ResidualFunc()> residual_fn,pose_graph::Pose& pose, double eps = 1e-6){
    auto r0 = residual_fn();
    int residual_dim = r0.size();
    Eigen::MatrixXd J(residual_dim, 6);
    pose_graph::Pose original = pose;

    for (int i = 0; i<6; ++i){
        Eigen::Matrix<double,6,1> delta = Eigen::Matrix<double,6,1>::Zero();
        delta[i] = eps;

        //Perturn +eps
        if (i < 3) {
            pose.R = pose.R * so3::Exp(delta.head<3>());
        } else {
            pose.t += delta.tail<3>();
        }
        auto r_plus = residual_fn();

        //Reset pose
        pose = original;
        delta[i] = -eps;
        if(i < 3) {
            pose.R = pose.R * so3::Exp(delta.head<3>());
        } else {
            pose.t += delta.tail<3>();
        }
        auto r_minus = residual_fn();
        //Central difference
        J.col(i) = (r_plus - r_minus) / (2 * eps);
        pose = original;
    }
    return J;
}
// -----------------------------------------------------------------------------
// Layer 1 — Graph assembly
// -----------------------------------------------------------------------------

TEST(PoseGraph, InitialPosesSet) {
    // YOUR TEST:
    //   config = default
    //   graph = PoseGraphSLAM(config)
    //   graph.setInitialPoses( makeGroundTruthChain() )
    //   — nothing to assert beyond "doesn't crash"; this mostly exercises
    //     the interface. Optimizing without edges should return same poses.
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    std::vector<pose_graph::Pose> gt = makeGroundTruthChain();
    graph.setInitialPoses(gt);
    graph.setFixed(0, true);  // fix the first node as the anchor

    //NO assertions, verifying it does not crash

}

TEST(PoseGraph, OptimizeWithNoEdgesReturnsInitial) {
    // YOUR TEST:
    //   graph with 3 poses, no edges
    //   result = graph.optimize()
    //   for each i: assert result[i].t == initial[i].t, result[i].R == initial[i].R
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    std::vector<pose_graph::Pose> gt = makeGroundTruthChain();
    graph.setInitialPoses(gt);
    std::vector<pose_graph::Pose> result = graph.optimize();
    ASSERT_EQ(result.size(), gt.size());
    for (size_t i = 0; i < gt.size(); ++i) {
        ASSERT_TRUE(result[i].t.isApprox(gt[i].t));
        ASSERT_TRUE(result[i].R.isApprox(gt[i].R));
    }
}

// -----------------------------------------------------------------------------
// Layer 2 — Residuals (zero at ground truth)
// -----------------------------------------------------------------------------

TEST(PoseGraph, ICPResidualZeroAtGroundTruth) {
    // YOUR TEST:
    //   gt = makeGroundTruthChain()
    //   edge = makeICPEdgeFromGT(gt[0], gt[1], 0, 1)
    //   graph.setInitialPoses(gt) — exact truth
    //   graph.addICPEdge(edge)
    //   — reach into graph to compute residual (may need friend access or
    //     an accessor, e.g., evaluateCost()). Alternative: just check that
    //     optimize() doesn't move the poses.
    //   For a proper residual test, expose a public `costAtCurrent()` method
    //   on PoseGraphSLAM that returns Σ r^T Ω r.
    auto gt = makeGroundTruthChain();
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(gt);
    graph.addICPEdge(makeICPEdgeFromGT(gt[0], gt[1], 0, 1));
    graph.addICPEdge(makeICPEdgeFromGT(gt[1], gt[2], 1, 2));
    EXPECT_NEAR(graph.evaluateCost(), 0.0, kTolStrict);
}

TEST(PoseGraph, GPSResidualZeroAtGroundTruth) {
    // YOUR TEST:
    //   Analogous to above — GPS measurement = pose's true position.

    auto gt = makeGroundTruthChain();
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(gt);
    // Add GPS edge for node 1 with measurement equal to gt[1].t
    pose_graph::GPSEdge gps_edge;
    gps_edge.frame_id = 1;
    gps_edge.position_meas = gt[1].t;
    gps_edge.information = Eigen::Matrix<double, 3, 3>::Identity();
    graph.addGPSEdge(gps_edge);
    EXPECT_NEAR(graph.evaluateCost(), 0.0, kTolStrict);
}

TEST(PoseGraph, IMUResidualReasonableAtGroundTruth) {     
    auto gt = makeGroundTruthChain();                                                                                                                                     
    pose_graph::PoseGraphConfig cfg;                      
    pose_graph::PoseGraphSLAM graph(cfg);                                                                                                                                 
    graph.setInitialPoses(gt);                            
                                                                                                                                                                        
    double dt = 1.0;
    pose_graph::IMUEdge imu_edge;                                                                                                                                         
    imu_edge.from = 0;                  // ← MUST initialize
    imu_edge.to = 1;                                                                                                                                                      
    imu_edge.factor.dt = dt;                              
    imu_edge.factor.dR = gt[0].R.transpose() * gt[1].R;                                                                                                                   
    imu_edge.factor.dp = gt[0].R.transpose() * (gt[1].t - gt[0].t - 0.5 * pose_graph::kGravityWorld * dt * dt);
    imu_edge.factor.covariance = Eigen::Matrix<double,9,9>::Identity() * 0.01;                                                                                            
    graph.addIMUEdge(imu_edge);                                                                                                                                           
                                        
    EXPECT_NEAR(graph.evaluateCost(), 0.0, kTolNumeric);                                                                                                                  
}    

// -----------------------------------------------------------------------------
// Layer 2b — Jacobian sanity (analytical vs numerical)
// -----------------------------------------------------------------------------

TEST(PoseGraph, ICPAnalyticalJacobianMatchesNumerical) {
    // YOUR TEST — a "secret weapon" for validating Jacobian derivations.
    //   1. Build a random ICPEdge with a generic (non-GT) measurement
    //   2. Compute analytical J_from, J_to via graph.jacobianICP()
    //      (will need to expose this method or make the test a friend)
    //   3. Compute numerical Jacobian via central differences:
    //        for each of 6 perturbation directions:
    //            poses_ += epsilon perturbation
    //            r_plus = residual()
    //            poses_ -= 2*epsilon
    //            r_minus = residual()
    //            J_numerical.col(i) = (r_plus - r_minus) / (2 * epsilon)
    //   4. assert (J_analytical - J_numerical).norm() < 1e-4
    //
    // If this test fails, the Jacobian derivation is wrong — far better to
    // catch here than to debug optimizer divergence.
    auto gt = makeGroundTruthChain();
    auto perturbed = perturbPoses(gt, 0.05, 0.02);

    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(perturbed);

    auto edge = makeICPEdgeFromGT(gt[0], gt[1], 0, 1);
    graph.addICPEdge(edge);

    // Compute analytical Jacobians
    Eigen::Matrix<double, 6, 6> J_from_analytical, J_to_analytical;
    graph.jacobianICP(edge, J_from_analytical, J_to_analytical);

    // Compute numerical Jacobians
    pose_graph::Pose& pose_from = graph.mutablePose(0);
    pose_graph::Pose& pose_to = graph.mutablePose(1);
    
    auto residula_fn = [&]() -> Eigen::Matrix<double, 6, 1> {
        return graph.residualICP(edge);
    };

    auto J_from_numerical = numericalJacobian<Eigen::Matrix<double, 6, 1>>(residula_fn, pose_from);
    EXPECT_NEAR((J_from_analytical - J_from_numerical).norm(), 0.0, 1e-2);

    auto J_to_numerical = numericalJacobian<Eigen::Matrix<double, 6, 1>>(residula_fn, pose_to);
    EXPECT_NEAR((J_to_analytical - J_to_numerical).norm(), 0.0, 1e-2);
}

TEST(PoseGraph, IMUAnalyticalJacobianMatchesNumerical) {                                                                                                                  
    auto gt = makeGroundTruthChain();       
    auto perturbed = perturbPoses(gt, 0.05, 0.02);                                                                                                                        
                                                                                                                                                                        
    pose_graph::PoseGraphConfig cfg;                                                                                                                                      
    pose_graph::PoseGraphSLAM graph(cfg);                                                                                                                               
    graph.setInitialPoses(perturbed);                                                                                                                                     
                                                                                                                                                                        
    double dt = 1.0;                                                                                                                                                      
    pose_graph::IMUEdge edge;                                                                                                                                             
    edge.from = 0;                                                                                                                                                      
    edge.to = 1;                                                                                                                                                          
    edge.factor.dt = dt;                                                                                                                                                
    edge.factor.dR = gt[0].R.transpose() * gt[1].R;                                                                                                                       
    edge.factor.dp = gt[0].R.transpose() * (gt[1].t - gt[0].t - 0.5 * pose_graph::kGravityWorld * dt * dt);
    edge.factor.covariance = Eigen::Matrix<double,9,9>::Identity() * 0.01;                                                                                                
    graph.addIMUEdge(edge);                                                                                                                                               
                                        
    Eigen::Matrix<double,6,6> J_from_a, J_to_a;                                                                                                                           
    graph.jacobianIMU(edge, J_from_a, J_to_a);                                                                                                                            
                                                                                                                                                                        
    auto& pose_from = graph.mutablePose(0);                                                                                                                               
    auto& pose_to   = graph.mutablePose(1);                                                                                                                             
                                                                                                                                                                        
    auto residual_fn = [&]() -> Eigen::Matrix<double,6,1> {                                                                                                               
        return graph.residualIMU(edge); 
    };                                                                                                                                                                    
                                                                                                                                                                        
    auto J_from_n = numericalJacobian<Eigen::Matrix<double,6,1>>(residual_fn, pose_from);                                                                                 
    auto J_to_n   = numericalJacobian<Eigen::Matrix<double,6,1>>(residual_fn, pose_to);                                                                                   
                                                                                                                                                                        
    EXPECT_NEAR((J_from_a - J_from_n).norm(), 0.0, 1e-2);                                                                                                                 
    EXPECT_NEAR((J_to_a - J_to_n).norm(), 0.0, 1e-2);                                                                                                                   
}  

// -----------------------------------------------------------------------------
// Layer 3 — Optimizer convergence
// -----------------------------------------------------------------------------

                                                                                                                                                                            
TEST(PoseGraph, TwoNodeChainConverges) {                                                                                                                                  
    std::vector<pose_graph::Pose> gt(2);                                                                                                                                  
    gt[0].t = Eigen::Vector3d::Zero();                                                                                                                                    
    gt[1].t = Eigen::Vector3d(1, 0, 0);                                                                                                                                   
                                                                                                                                                                        
    auto perturbed = perturbPoses(gt, 0.05, 0.02);
    perturbed[0] = gt[0];   // anchor at GT                                                                                                                               
                                                                                                                                                                        
    pose_graph::PoseGraphConfig cfg;                                                                                                                                      
    pose_graph::PoseGraphSLAM graph(cfg);                                                                                                                                 
    graph.setInitialPoses(perturbed);                                                                                                                                     
    graph.setFixed(0, true);                                                                                                                                              
    graph.addICPEdge(makeICPEdgeFromGT(gt[0], gt[1], 0, 1));
                                                                                                                                                                        
    auto result = graph.optimize();                       
    EXPECT_NEAR((result[1].t - gt[1].t).norm(), 0.0, kTolOptimized);                                                                                                      
    EXPECT_NEAR((result[1].R - gt[1].R).norm(), 0.0, kTolOptimized);                                                                                                      
}                                           
            

TEST(PoseGraph, FixedAnchorStaysPut) {
    // YOUR TEST:
    //   Any graph with node 0 fixed
    //   After optimize(): result[0] must equal initial[0] EXACTLY
    //   (not "close to" — fixed nodes should not move at all)
    auto gt = makeGroundTruthChain();
    auto perturbed = perturbPoses(gt, 0.05, 0.02);
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(perturbed);
    graph.setFixed(0, true);  // fix the first node as the anchor
    graph.addICPEdge(makeICPEdgeFromGT(gt[0], gt[1], 0, 1));
    graph.addICPEdge(makeICPEdgeFromGT(gt[1], gt[2], 1, 2));
    auto result = graph.optimize();
    EXPECT_TRUE(result[0].t.isApprox(perturbed[0].t));
    EXPECT_TRUE(result[0].R.isApprox(perturbed[0].R));
}

TEST(PoseGraph, GPSEdgesPullTowardMeasurements) {
    // YOUR TEST:
    //   Graph with a single node (nothing else)
    //   Initial guess: (10, 10, 10)
    //   GPS edge: measurement = (0, 0, 0), high information
    //   optimize
    //   assert result.t is close to (0, 0, 0)
    
    //build graph with singe node 
    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);

    //Inital guess = (10, 10, 10)
    std::vector<pose_graph::Pose> poses(2);
    poses[0].t = Eigen::Vector3d::Zero(); 
    poses[1].t = Eigen::Vector3d(10, 10, 10);  
    graph.setInitialPoses(poses);
    graph.setFixed(0, true);  // fix the first node as the anchor


    //Add GPS edge with measurement = (0, 0, 0) and high information
    pose_graph::ICPEdge edge;
    edge.dR_meas = Eigen::Matrix3d::Identity();                                                                                                                             
    edge.dp_meas = Eigen::Vector3d::Zero();                                                                                                                                   
    edge.information = Eigen::Matrix<double,6,6>::Identity() * 1.0;
    edge.from = 0;
    edge.to = 1; // Self-loop to apply GPS constraint on node 1
    graph.addICPEdge(edge);

    pose_graph::GPSEdge gps_edge;
    gps_edge.frame_id = 1;
    gps_edge.position_meas = Eigen::Vector3d::Zero();;
    gps_edge.information = Eigen::Matrix<double, 3, 3>::Identity() * 100.0; // High information
    graph.addGPSEdge(gps_edge);

    //Optimize
    auto result = graph.optimize();

    //Assert result.t is close to (0, 0, 0)
    EXPECT_NEAR(result[1].t.norm(), 0.0, kTolOptimized);

}

TEST(PoseGraph, LoopClosureReducesATE) {
    // YOUR TEST — the flagship convergence test:
    //   1. Build a 10-node chain with accumulated translational drift
    //      (each ICP edge has dp_meas slightly off from ground truth — simulates
    //       ICP odometry drift)
    //   2. Optimize WITHOUT loop closure → record ATE vs GT
    //   3. Add a loop closure edge between node 0 and node 9 with correct
    //      relative transform
    //   4. Optimize WITH loop closure → record ATE vs GT
    //   5. assert ATE_with_loop < ATE_without_loop
    //   Tolerance can be generous; just verify the qualitative effect.

    //build a 10 node chain with drift
    //each ICP edge has a dp_meas slightly off from ground truth to simultae ICP odom drift
    int num_nodes = 10;
    std::vector<pose_graph::Pose> gt_chain;
    for (int i = 0; i < num_nodes; ++i) {
        pose_graph::Pose p;
        p.R = Eigen::Matrix3d::Identity();
        p.t = Eigen::Vector3d(i * 1.0, 0, 0); // Ground truth: 1m apart in x
        gt_chain.push_back(p);
    }

    pose_graph::PoseGraphConfig cfg;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(gt_chain);
    graph.setFixed(0, true); // Fix the first node as the anchor

    // Add ICP edges with drift
    for (int i = 0; i < num_nodes - 1; ++i) {
        pose_graph::ICPEdge edge;
        edge.from = i;
        edge.to = i + 1;
        edge.dR_meas = Eigen::Matrix3d::Identity();
        edge.dp_meas = Eigen::Vector3d(1.0 + 0.05, 0, 0); // Simulate drift: 5cm too long
        edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0; // High information
        graph.addICPEdge(edge);
    }

    // Optimize without loop closure
    auto result_without_loop = graph.optimize();
    double ate_without_loop = (result_without_loop[num_nodes - 1].t - gt_chain[num_nodes - 1].t).norm();

    // Add a loop closure edge
    pose_graph::ICPEdge loop_closure_edge;
    loop_closure_edge.from = num_nodes - 1;
    loop_closure_edge.to = 0;
    loop_closure_edge.dR_meas = Eigen::Matrix3d::Identity();
    loop_closure_edge.dp_meas = Eigen::Vector3d(-9.0, 0, 0); // Correct relative transform from node 9 back to node 0
    loop_closure_edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0; // High information
    graph.addICPEdge(loop_closure_edge);

    // Optimize with loop closure
    auto result_with_loop = graph.optimize();
    double ate_with_loop = (result_with_loop[num_nodes - 1].t - gt_chain[num_nodes - 1].t).norm();

    // Assert ATE_with_loop < ATE_without_loop
    EXPECT_LT(ate_with_loop, ate_without_loop);

}

// -----------------------------------------------------------------------------
// Layer 4 — Manifold vs Euclidean ablation
// -----------------------------------------------------------------------------

TEST(PoseGraph, ManifoldRetractionKeepsRotationValid) {
    // YOUR TEST:
    //   Build a 5-node graph, perturb initial poses heavily, run optimize()
    //   with config.optimizer = Manifold.
    //   For each resulting pose:
    //     assert (R * R^T - I).norm() < kTolNumeric
    //     assert |det(R) - 1| < kTolNumeric

    //test retract() function
    
    //build graph with 5 nodes
    int num_nodes = 5;
    std::vector<pose_graph::Pose> gt_chain;
    for (int i = 0; i < num_nodes; ++i) {
        pose_graph::Pose p;
        p.R = Eigen::Matrix3d::Identity();
        p.t = Eigen::Vector3d(i * 1.0, 0, 0); // Ground truth: 1m apart in x
        gt_chain.push_back(p);
    }
    auto perturbed = perturbPoses(gt_chain, 0.05, 0.02);

    pose_graph::PoseGraphConfig cfg;
    cfg.optimizer = pose_graph::PoseGraphConfig::OptimizerType::Manifold;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(perturbed);
    graph.setFixed(0, true); // Fix the first node as the anchor
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[0], gt_chain[1], 0, 1));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[1], gt_chain[2], 1, 2));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[2], gt_chain[3], 2, 3));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[3], gt_chain[4], 3, 4));

    auto poses = graph.optimize();
    for (const auto& pose : poses) {
        double rot_validity = (pose.R * pose.R.transpose() - Eigen::Matrix3d::Identity()).norm();
        double det_validity = std::abs(pose.R.determinant() - 1.0);
        EXPECT_NEAR(rot_validity, 0.0, kTolNumeric);
        EXPECT_NEAR(det_validity, 0.0, kTolNumeric);
    }
}

TEST(PoseGraph, EuclideanRetractionNeedsSVDReprojection) {
    // YOUR TEST — verify the Euclidean variant actually runs and stays valid
    // (thanks to the SVD reprojection step):
    //   Same graph as above but config.optimizer = Euclidean.
    //   R * R^T - I should still be within tolerance (because SVD reprojects).
    //   The INTERESTING comparison (ATE) belongs in the ablation script,
    //   not a unit test — this test just confirms the code runs end-to-end.
    int num_nodes = 5;
    std::vector<pose_graph::Pose> gt_chain;
    for (int i = 0; i < num_nodes; ++i) {
        pose_graph::Pose p;
        p.R = Eigen::Matrix3d::Identity();
        p.t = Eigen::Vector3d(i * 1.0, 0, 0); // Ground truth: 1m apart in x
        gt_chain.push_back(p);
    }
    auto perturbed = perturbPoses(gt_chain, 0.05, 0.02);
    pose_graph::PoseGraphConfig cfg;
    cfg.optimizer = pose_graph::PoseGraphConfig::OptimizerType::Euclidean;
    pose_graph::PoseGraphSLAM graph(cfg);
    graph.setInitialPoses(perturbed);
    graph.setFixed(0, true); // Fix the first node as the anchor
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[0], gt_chain[1], 0, 1));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[1], gt_chain[2], 1, 2));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[2], gt_chain[3], 2, 3));
    graph.addICPEdge(makeICPEdgeFromGT(gt_chain[3], gt_chain[4], 3, 4));

    auto poses = graph.optimize();
    for (const auto& pose : poses) {
        double rot_validity = (pose.R * pose.R.transpose() - Eigen::Matrix3d::Identity()).norm();
        double det_validity = std::abs(pose.R.determinant() - 1.0);
        EXPECT_NEAR(rot_validity, 0.0, kTolNumeric);
        EXPECT_NEAR(det_validity, 0.0, kTolNumeric);
    }

}

TEST(PoseGraph, RetractAppliesUpdate) {                                                                                                                                   
    pose_graph::PoseGraphConfig cfg;  // default = Manifold                                                                                                               
    pose_graph::PoseGraphSLAM graph(cfg);                                                                                                                                 
    std::vector<pose_graph::Pose> poses(2);  // 2 identity poses                                                                                                          
    graph.setInitialPoses(poses);                         
    graph.setFixed(0, true); // Fix the first node as the anchor
                                                                                                                                                                        
    // dx: zero for pose 0, known update for pose 1                                                                                                                       
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(12);                                                                                                                       
    dx.segment<3>(6) = Eigen::Vector3d(0, 0, M_PI / 4);  // δφ for pose 1                                                                                                 
    dx.segment<3>(9) = Eigen::Vector3d(1, 2, 3);         // δt for pose 1                                                                                                 
                                                                                                                                                                        
    graph.retract(dx);   // will need to expose retract as public                                                                                                         
                                                                                                                                                                        
    // Pose 0 unchanged                                                                                                                                                   
    EXPECT_NEAR((graph.pose(0).R - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolStrict);                                                                                 
    EXPECT_NEAR(graph.pose(0).t.norm(), 0.0, kTolStrict);                                                                                                                 
                                                                                                                                                                        
    // Pose 1: R rotated 45° around z, t = (1,2,3)        
    Eigen::Matrix3d expected_R = so3::Exp(Eigen::Vector3d(0, 0, M_PI / 4));                                                                                               
    EXPECT_NEAR((graph.pose(1).R - expected_R).norm(), 0.0, kTolNumeric);
    EXPECT_NEAR((graph.pose(1).t - Eigen::Vector3d(1, 2, 3)).norm(), 0.0, kTolStrict);                                                                                    
} 

// -----------------------------------------------------------------------------
// CSV I/O (simple sanity checks)
// -----------------------------------------------------------------------------

TEST(PoseGraph, DISABLED_CSVRoundTrip) {
    // (Enable when savePosesToCSV + loadPosesFromCSV are both implemented.)
    // YOUR TEST:
    //   Save a vector of poses, load it back, compare.
}
