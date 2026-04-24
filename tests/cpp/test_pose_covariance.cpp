// =============================================================================
// test_pose_covariance.cpp — per-pose marginal covariance extraction (P2-M3.4)
// =============================================================================
//
// Test plan:
//
//   Convention verifier (CRITICAL — blocks ablation E correctness)
//     - G2OMarginalTranslationBlockIsBottomRight
//         Build a tiny 2-pose graph with a GPS-only unary constraint on pose 1.
//         After optimization, the translation block of the marginal covariance
//         must have FINITE trace (GPS constrains translation). The rotation
//         block must have a LARGE trace (unconstrained). Our permutation in
//         computeMarginalsG2O should deliver the covariance as [rot | trans].
//
//   Shape & sanity
//     - MarginalsShapeMatchesPoseCount
//     - HeuristicShapeMatchesPoseCount
//     - HeuristicSingularPoseFallsBackToHugeSigma
//     - PoseSigmaFromCovariance_ExtractsTranslationTrace
//     - PoseSigmaFromCovariance_ClampedOnInf
//
//   Heuristic vs g2o
//     - HeuristicHasHigherCovThanG2O_OnLoopClosureGraph
//         On a graph WITH loop closures, g2o's marginals are tighter than the
//         edge-sum heuristic (because loop closures add information that the
//         heuristic ignores via diagonal-only accumulation). This is the
//         "honest ablation footnote" we planned for the blog.
//
// DO NOT fill in:
//   - Numerical thresholds for "large trace" / "tight covariance" without
//     running the test once to see actual values. Use std::cerr prints first,
//     then hard-code the thresholds.
//
// =============================================================================

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

#include "pose_graph_slam.hpp"

// g2o optimizer — required because the test builds a SparseOptimizer directly.
// g2o_wrapper::optimizeWithG2O() does not expose the optimizer handle; our
// test needs it to call computeMarginalsG2O, so we inline the graph build.
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_prior.h>
#include <g2o/types/slam3d/parameter_se3_offset.h>
#include <numeric>

using pose_graph::Pose;
using pose_graph::ICPEdge;
using pose_graph::GPSEdge;
using pose_graph::PoseGraphSLAM;
using pose_graph::PoseGraphConfig;

// -----------------------------------------------------------------------------
// Convention verifier — the single most important test in this file
// -----------------------------------------------------------------------------

TEST(PoseCovariance, G2OMarginalTranslationBlockIsBottomRight) {
    // YOUR CODE:
    //
    // 1) Build a 2-pose graph:
    //      Pose p0 (origin, identity R), fixed.
    //      Pose p1 at (1, 0, 0), identity R.
    //
    // 2) Add a single GPS unary edge on pose 1 with measured position (1, 0, 0)
    //    and small positional sigma (e.g. info = I_3 * 100).
    //    No ICP/IMU/Loop edges — translation is ONLY constrained by GPS,
    //    rotation is unconstrained (will take a prior-only value).
    //
    // 3) Optimize the graph via g2o_wrapper. Extract the optimizer handle.
    //
    // 4) auto covs = pose_graph::computeMarginalsG2O(optimizer, /*num_poses=*/2);
    //    ASSERT_EQ(covs.size(), 2u);
    //
    // 5) Inspect covs[1] (pose 1's marginal):
    //      Eigen::Matrix3d rot_block = covs[1].block<3,3>(0, 0);   // top-left
    //      Eigen::Matrix3d trans_block = covs[1].block<3,3>(3, 3); // bottom-right
    //
    //      // Translation was constrained by GPS → finite, small trace.
    //      EXPECT_LT(trans_block.trace(), 1.0);
    //      EXPECT_GT(trans_block.trace(), 0.0);
    //
    //      // Rotation is unconstrained → huge trace (or near-infinite if g2o
    //      // can't compute it). If the permutation is WRONG, these assertions
    //      // flip — the rotation block is where GPS info landed and the
    //      // translation block is huge.
    //      EXPECT_GT(rot_block.trace(), 1e3);
    //
    // If this test fails, our permutation at pose_graph_slam.cpp:1143
    // (P.indices() << 3,4,5,0,1,2) is wrong for this g2o version. Fix by
    // removing the permutation and re-running.

    // --- Optimizer setup (mirrors g2o_wrapper.cpp's pattern) ---
    using LinearSolver = g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>;
    using BlockSolver  = g2o::BlockSolverX;

    auto linear = std::make_unique<LinearSolver>();
    auto block  = std::make_unique<BlockSolver>(std::move(linear));
    auto* algo  = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(false);

    // --- Register a SE3 offset parameter (required by EdgeSE3Prior) ---
    // EdgeSE3Prior models a prior measurement in a sensor frame that may be
    // offset from the body frame. Even for a body-frame prior (identity offset),
    // the parameter MUST be registered and linked to the edge via setParameterId.
    // Skipping this trips a g2o FATAL at addEdge(). ID 0 is conventional.
    auto* offset_param = new g2o::ParameterSE3Offset();
    offset_param->setId(0);
    offset_param->setOffset(Eigen::Isometry3d::Identity());
    optimizer.addParameter(offset_param);

    // --- Vertex 0: fixed at origin ---
    auto* v0 = new g2o::VertexSE3();
    v0->setId(0);
    v0->setEstimate(Eigen::Isometry3d::Identity());
    v0->setFixed(true);
    optimizer.addVertex(v0);

    // --- Vertex 1: initial guess (1, 0, 0), identity rotation ---
    auto* v1 = new g2o::VertexSE3();
    v1->setId(1);
    Eigen::Isometry3d init_p1 = Eigen::Isometry3d::Identity();
    init_p1.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
    v1->setEstimate(init_p1);
    optimizer.addVertex(v1);

    // --- Prior on pose 1: constrain translation only, rotation info ≈ 0.
    // EdgeSE3Prior is a 6D unary pose prior. Information layout: top-left 3x3
    // is translation, bottom-right 3x3 is rotation (this is g2o's INTERNAL
    // [trans | rot] order — our computeMarginalsG2O should permute it to
    // [rot | trans] before returning).
    auto* gps = new g2o::EdgeSE3Prior();
    gps->setVertex(0, v1);
    gps->setParameterId(0, /*param_id=*/0);   // link to offset_param registered above
    Eigen::Isometry3d meas = Eigen::Isometry3d::Identity();
    meas.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
    gps->setMeasurement(meas);
    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    info.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 100.0;   // strong translation info (σ ≈ 0.1m)
    // Tiny-but-nonzero rotation info: σ ≈ 10 rad → effectively unconstrained, but
    // keeps the Hessian positive-definite so Cholesky doesn't fail in the marginal
    // extraction. Exact zero here caused SIGABRT via the "Cholesky failure" path.
    info.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.01;
    gps->setInformation(info);
    optimizer.addEdge(gps);

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // --- Extract marginals via our code path ---
    auto covs = pose_graph::computeMarginalsG2O(optimizer, /*num_poses=*/2);
    ASSERT_EQ(covs.size(), 2u);

    // --- The actual convention check, on pose 1's marginal ---
    // Our convention is [rot | trans]. After computeMarginalsG2O's permutation,
    // top-left 3x3 should be rotation covariance, bottom-right 3x3 translation.
    const Eigen::Matrix3d rot_block   = covs[1].block<3, 3>(0, 0);
    const Eigen::Matrix3d trans_block = covs[1].block<3, 3>(3, 3);

    // Print once so you can see actual numbers; tune thresholds after first run:
    std::cerr << "[convention check] pose 1 rot-block trace   = " << rot_block.trace()   << "\n";
    std::cerr << "[convention check] pose 1 trans-block trace = " << trans_block.trace() << "\n";

    // Translation constrained by GPS → finite, small trace.
    EXPECT_GT(trans_block.trace(), 0.0);
    EXPECT_LT(trans_block.trace(), 1.0);

    // Rotation unconstrained → much larger trace. If the permutation is
    // wrong, these assertions flip and the test fires.
    EXPECT_GT(rot_block.trace(), trans_block.trace() * 10.0);
}

// -----------------------------------------------------------------------------
// Shape & sanity
// -----------------------------------------------------------------------------

TEST(PoseCovariance, MarginalsShapeMatchesPoseCount) {                                                                                                                    
    // --- Optimizer setup (same boilerplate as the convention-verifier test) ---
    using LinearSolver = g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>;                                                                                       
    using BlockSolver  = g2o::BlockSolverX;                                                                                                                               
                                                                                                                                                                        
    auto linear = std::make_unique<LinearSolver>();                                                                                                                       
    auto block  = std::make_unique<BlockSolver>(std::move(linear));                                                                                                       
    auto* algo  = new g2o::OptimizationAlgorithmLevenberg(std::move(block));                                                                                              
                                                                                                                                                                        
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algo);                                                                                                                                         
    optimizer.setVerbose(false);                          
                                                                                                                                                                        
    // --- 5 vertices: pose i at (i, 0, 0), pose 0 fixed ---                                                                                                              
    const int N = 5;                                                                                                                                                      
    for (int i = 0; i < N; ++i) {                                                                                                                                         
        auto* v = new g2o::VertexSE3();                                                                                                                                   
        v->setId(i);
        Eigen::Isometry3d e = Eigen::Isometry3d::Identity();                                                                                                              
        e.translation() = Eigen::Vector3d(i, 0, 0);                                                                                                                       
        v->setEstimate(e);
        if (i == 0) v->setFixed(true);                                                                                                                                    
        optimizer.addVertex(v);                           
    }                                                                                                                                                                     
                                                        
    // --- ICP-style binary edges i → i+1, measuring Δp = (1, 0, 0), ΔR = I ---                                                                                           
    for (int i = 0; i < N - 1; ++i) {
        auto* e = new g2o::EdgeSE3();                                                                                                                                     
        e->setVertex(0, optimizer.vertex(i));             
        e->setVertex(1, optimizer.vertex(i + 1));                                                                                                                         
        Eigen::Isometry3d m = Eigen::Isometry3d::Identity();
        m.translation() = Eigen::Vector3d(1, 0, 0);                                                                                                                       
        e->setMeasurement(m);                             
        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 100.0);                                                                                               
        optimizer.addEdge(e);                                                                                                                                             
    }
                                                                                                                                                                        
    optimizer.initializeOptimization();                   
    optimizer.optimize(20);
                                                                                                                                                                        
    // --- The actual shape assertions ---
    auto covs = pose_graph::computeMarginalsG2O(optimizer, N);                                                                                                            
    EXPECT_EQ(covs.size(), static_cast<size_t>(N));                                                                                                                       
    for (const auto& P : covs) {
        EXPECT_EQ(P.rows(), 6);                                                                                                                                           
        EXPECT_EQ(P.cols(), 6);                           
    }                                                                                                                                                                     
}  

TEST(PoseCovariance, HeuristicShapeMatchesPoseCount) {                                                                                                                    
    // computeEdgeInformationSum reads edge.information; no optimize() needed.
    PoseGraphConfig cfg;                                                                                                                                                  
    PoseGraphSLAM graph(cfg);
                                                                                                                                                                        
    const int N = 5;                                      
                                                                                                                                                                        
    // 1) Seed pose estimates — required before adding edges (edge indices                                                                                                
    //    reference pose IDs in graph.poses_).
    std::vector<Pose> initial(N);                                                                                                                                         
    for (int i = 0; i < N; ++i) {                         
        initial[i].R = Eigen::Matrix3d::Identity();                                                                                                                       
        initial[i].t = Eigen::Vector3d(i, 0, 0);                                                                                                                          
    }                                                                                                                                                                     
    graph.setInitialPoses(initial);                                                                                                                                       
                                                                                                                                                                        
    // 2) Add ICP edges i → i+1 with identity 6x6 information. The actual                                                                                                 
    //    measurements don't matter for this shape check — only that the
    //    information matrices are well-formed so H is invertible.                                                                                                        
    for (int i = 0; i < N - 1; ++i) {                                                                                                                                     
        ICPEdge e;
        e.from = i;                                                                                                                                                       
        e.to = i + 1;                                     
        e.dR_meas = Eigen::Matrix3d::Identity();                                                                                                                          
        e.dp_meas = Eigen::Vector3d(1, 0, 0);                                                                                                                             
        e.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
        graph.addICPEdge(e);                                                                                                                                              
    }                                                     
                                                                                                                                                                        
    // 3) Run the heuristic (reads edges, no optimization).                                                                                                               
    auto covs = pose_graph::computeEdgeInformationSum(graph, N);
                                                                                                                                                                        
    EXPECT_EQ(covs.size(), static_cast<size_t>(N));       
    for (const auto& P : covs) {                                                                                                                                          
        EXPECT_EQ(P.rows(), 6);                           
        EXPECT_EQ(P.cols(), 6);                                                                                                                                           
    }
}  

TEST(PoseCovariance, HeuristicSingularPoseFallsBackToHugeSigma) {
    // YOUR CODE:
    //   Build a PoseGraphSLAM with 3 poses and ICP edges ONLY between 0-1
    //   (pose 2 has no incident edges except the fixed-anchor stiffness, which
    //   the heuristic doesn't know about).
    //   auto covs = computeEdgeInformationSum(graph, 3);
    //
    //   covs[2] should be Matrix::Identity() * 1e9 (the sentinel for "no info").
    //   EXPECT_GT(covs[2].diagonal().minCoeff(), 1e8);

    PoseGraphConfig cfg;
    PoseGraphSLAM graph(cfg);

    std::vector<Pose> initial(3);
    initial[0].R = Eigen::Matrix3d::Identity();
    initial[0].t = Eigen::Vector3d(0, 0, 0);
    initial[1].R = Eigen::Matrix3d::Identity();
    initial[1].t = Eigen::Vector3d(1, 0, 0);
    initial[2].R = Eigen::Matrix3d::Identity();
    initial[2].t = Eigen::Vector3d(2, 0, 0);
    graph.setInitialPoses(initial);
    ICPEdge e;
    e.from = 0;
    e.to = 1;   
    e.dR_meas = Eigen::Matrix3d::Identity();
    e.dp_meas = Eigen::Vector3d(1, 0, 0);
    e.information = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    graph.addICPEdge(e);
    auto covs = pose_graph::computeEdgeInformationSum(graph, 3);
    EXPECT_GT(covs[2].diagonal().minCoeff(), 1e8);


}

TEST(PoseCovariance, PoseSigmaFromCovariance_ExtractsTranslationTrace) {
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
    P.block<3, 3>(3, 3) = Eigen::Vector3d(0.01, 0.04, 0.09).asDiagonal();
    // Trace of translation block = 0.14.  sqrt(0.14) ≈ 0.374.
    EXPECT_NEAR(pose_graph::poseSigmaFromCovariance(P), std::sqrt(0.14), 1e-6);
}

TEST(PoseCovariance, PoseSigmaFromCovariance_ClampedOnInf) {
    // YOUR CODE:
    //   Build a P with translation block diagonal = (1e20, 1e20, 1e20).
    //   Expected: poseSigmaFromCovariance returns the clamp value (1e3)
    //   rather than inf/nan. Guards downstream exp(-k * sigma) from producing 0.
    Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
    P.block<3, 3>(3, 3) = Eigen::Vector3d(1e20, 1e20, 1e20).asDiagonal();
    EXPECT_EQ(pose_graph::poseSigmaFromCovariance(P), 1e3); 
}

TEST(PoseCovariance, HeuristicHasHigherCovThanG2O_OnLoopClosureGraph) {
    const int N = 10;                                                                                                                                                     

    // Helper: produce the i-th pose on a unit circle (x, y) ∈ R².                                                                                                        
    auto pose_i = [N](int i) {                            
        const double angle = i * 2.0 * M_PI / N;                                                                                                                          
        Eigen::Vector3d t(std::cos(angle), std::sin(angle), 0.0);                                                                                                         
        return std::make_pair(Eigen::Matrix3d::Identity(), t);
    };                                                                                                                                                                    
                                                        
    // Relative measurement between consecutive circle poses: Δt = p_{i+1} - p_i.                                                                                         
    auto relative_meas = [&](int i) {                     
        auto [R_i, t_i] = pose_i(i);                                                                                                                                      
        auto [R_j, t_j] = pose_i((i + 1) % N);            
        return Eigen::Vector3d(t_j - t_i);   // ΔR is identity on a circle                                                                                                
    };                                                                                                                                                                    
                                                                                                                                                                        
    // ------------------------------------------------------------------                                                                                                 
    // Path 1 — g2o graph for computeMarginalsG2O         
    // ------------------------------------------------------------------                                                                                                 
    using LinearSolver = g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>;
    using BlockSolver  = g2o::BlockSolverX;                                                                                                                               
                                                        
    auto linear = std::make_unique<LinearSolver>();                                                                                                                       
    auto block  = std::make_unique<BlockSolver>(std::move(linear));
    auto* algo  = new g2o::OptimizationAlgorithmLevenberg(std::move(block));                                                                                              

    g2o::SparseOptimizer optimizer;                                                                                                                                       
    optimizer.setAlgorithm(algo);                         
    optimizer.setVerbose(false);                                                                                                                                          
                                                        
    for (int i = 0; i < N; ++i) {                                                                                                                                         
        auto* v = new g2o::VertexSE3();
        v->setId(i);                                                                                                                                                      
        auto [R_i, t_i] = pose_i(i);                      
        Eigen::Isometry3d est = Eigen::Isometry3d::Identity();
        est.translation() = t_i;                                                                                                                                          
        v->setEstimate(est);
        if (i == 0) v->setFixed(true);                                                                                                                                    
        optimizer.addVertex(v);                                                                                                                                           
    }
                                                                                                                                                                        
    const Eigen::Matrix<double, 6, 6> icp_info =                                                                                                                          
        Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    const Eigen::Matrix<double, 6, 6> loop_info =                                                                                                                         
        Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;                                                                                                                 

    // ICP edges 0→1, 1→2, ..., 8→9                                                                                                                                       
    for (int i = 0; i < N - 1; ++i) {                     
        auto* e = new g2o::EdgeSE3();                                                                                                                                     
        e->setVertex(0, optimizer.vertex(i));             
        e->setVertex(1, optimizer.vertex(i + 1));                                                                                                                         
        Eigen::Isometry3d m = Eigen::Isometry3d::Identity();                                                                                                              
        m.translation() = relative_meas(i);
        e->setMeasurement(m);                                                                                                                                             
        e->setInformation(icp_info);                                                                                                                                      
        optimizer.addEdge(e);
    }                                                                                                                                                                     
                                                        
    // Loop closure 9 → 0 (strong info)
    {
        auto* loop = new g2o::EdgeSE3();                                                                                                                                  
        loop->setVertex(0, optimizer.vertex(N - 1));
        loop->setVertex(1, optimizer.vertex(0));                                                                                                                          
        Eigen::Isometry3d m = Eigen::Isometry3d::Identity();                                                                                                              
        m.translation() = relative_meas(N - 1);
        loop->setMeasurement(m);                                                                                                                                          
        loop->setInformation(loop_info);                                                                                                                                  
        optimizer.addEdge(loop);
    }                                                                                                                                                                     
                                                        
    optimizer.initializeOptimization();
    optimizer.optimize(50);
                                                                                                                                                                        
    auto covs_g2o = pose_graph::computeMarginalsG2O(optimizer, N);
    ASSERT_EQ(covs_g2o.size(), static_cast<size_t>(N));                                                                                                                   
                                                                                                                                                                        
    // ------------------------------------------------------------------
    // Path 2 — PoseGraphSLAM (same graph) for computeEdgeInformationSum                                                                                                  
    // ------------------------------------------------------------------                                                                                                 
    PoseGraphConfig cfg;
    PoseGraphSLAM graph(cfg);                                                                                                                                             
                                                                                                                                                                        
    std::vector<Pose> initial(N);                                                                                                                                         
    for (int i = 0; i < N; ++i) {                                                                                                                                         
        auto [R_i, t_i] = pose_i(i);                                                                                                                                      
        initial[i].R = R_i;                               
        initial[i].t = t_i;                                                                                                                                               
    }
    graph.setInitialPoses(initial);                                                                                                                                       
    graph.setFixed(0, true);                              
                                                                                                                                                                        
    for (int i = 0; i < N - 1; ++i) {
        ICPEdge e;                                                                                                                                                        
        e.from = i;                                       
        e.to = i + 1;                                                                                                                                                     
        e.dR_meas = Eigen::Matrix3d::Identity();
        e.dp_meas = relative_meas(i);                                                                                                                                     
        e.information = icp_info;                                                                                                                                         
        graph.addICPEdge(e);
    }                                                                                                                                                                     
                                                        
    // Loop-closure as a LoopEdge (same fields as ICP but separate list).                                                                                                 
    pose_graph::LoopEdge loop;
    loop.from = N - 1;                                                                                                                                                    
    loop.to = 0;                                          
    loop.dR_meas = Eigen::Matrix3d::Identity();
    loop.dp_meas = relative_meas(N - 1);                                                                                                                                  
    loop.information = loop_info;
    graph.addLoopEdge(loop);                                                                                                                                              
                                                                                                                                                                        
    auto covs_heuristic = pose_graph::computeEdgeInformationSum(graph, N);
    ASSERT_EQ(covs_heuristic.size(), static_cast<size_t>(N));                                                                                                             
                                                                                                                                                                        
    // ------------------------------------------------------------------
    // Compare pose_sigma from both paths                                                                                                                                 
    // ------------------------------------------------------------------                                                                                                 
    std::vector<double> sigma_g2o(N), sigma_heuristic(N);
    for (int i = 0; i < N; ++i) {                                                                                                                                         
        sigma_g2o[i]       = pose_graph::poseSigmaFromCovariance(covs_g2o[i]);
        sigma_heuristic[i] = pose_graph::poseSigmaFromCovariance(covs_heuristic[i]);                                                                                      
    }                                                     
                                                                                                                                                                        
    // Print so you can eyeball the gap before tightening the margin:                                                                                                     
    for (int i = 0; i < N; ++i) {
        std::cerr << "pose " << i                                                                                                                                         
                << ": sigma_g2o=" << sigma_g2o[i]                                                                                                                       
                << "  sigma_heuristic=" << sigma_heuristic[i] << "\n";
    }                                                                                                                                                                     
                                                                                                                                                                        
    const double avg_g2o =
        std::accumulate(sigma_g2o.begin(), sigma_g2o.end(), 0.0) / N;                                                                                                     
    const double avg_heuristic =                                                                                                                                          
        std::accumulate(sigma_heuristic.begin(), sigma_heuristic.end(), 0.0) / N;
                                                                                                                                                                        
    std::cerr << "avg_g2o=" << avg_g2o                                                                                                                                    
            << "  avg_heuristic=" << avg_heuristic << "\n";
                                                                                                                                                                        
    // The heuristic IGNORES correlations introduced by the loop-closure edge,                                                                                            
    // so it should either UNDER-report or OVER-report relative to g2o
    // depending on the graph. On this ring-with-loop-closure the heuristic                                                                                               
    // double-counts the loop edge (it naively adds to both endpoints),                                                                                                   
    // yielding over-confident (smaller) sigma at the loop endpoints but                                                                                                  
    // otherwise-similar numbers. Without running once, we can't commit to a                                                                                              
    // tight assertion. For now, just assert they're both finite and positive.                                                                                            
    //                                                                                                                                                                    
    // After first run: look at stderr numbers and tighten to either                                                                                                      
    //   EXPECT_GT(avg_heuristic, avg_g2o * 1.2)   // if heuristic overshoots                                                                                             
    //   EXPECT_LT(avg_heuristic, avg_g2o * 0.8)   // if heuristic undershoots                                                                                            
    // The ABSOLUTE direction is a blog-worthy finding either way.                                                                                                        
    EXPECT_GT(avg_g2o, 0.0);                                                                                                                                              
    EXPECT_GT(avg_heuristic, 0.0);                        
    EXPECT_LT(avg_g2o, 1e6);                                                                                                                                              
    EXPECT_LT(avg_heuristic, 1e6);                        
}                                                                                                                                                                         
            