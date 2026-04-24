#include "g2o_wrapper.hpp"
                                                                                                                                                                        
#include <g2o/core/sparse_optimizer.h>                                                                                                                                    
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>                                                                                                                    
#include <g2o/solvers/eigen/linear_solver_eigen.h>        
#include <g2o/types/slam3d/vertex_se3.h>                                                                                                                                  
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/base_unary_edge.h>  
                                                                                                                                                                        
#include <iostream>                                       
                                                                                                                                                                        
namespace g2o_wrapper {                                   

// Custom unary edge for GPS position constraint                                                                                                                          
// g2o doesn't have a built-in unary position-only edge
class EdgeGPS : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3> {                                                                                           
    public:                                                                                                                                                                
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                                                                                                                                                                        
    void computeError() override {                        
        // YOUR CODE:
        //   const auto* v = static_cast<const g2o::VertexSE3*>(_vertices[0]);                                                                                            
        //   Eigen::Vector3d pos = v->estimate().translation();                                                                                                           
        //   _error = pos - _measurement;         
        const auto* v = static_cast<const g2o::VertexSE3*>(_vertices[0]);
        Eigen::Vector3d pos = v->estimate().translation();
        _error = pos - _measurement;                                                                                                                        
    }                                                                                                                                                                     
                                                        
    bool read(std::istream&) override { return true; }                                                                                                                    
    bool write(std::ostream&) const override { return true; }
};                                                                                                                                                                        

// Helper: convert our Pose to g2o's Isometry3d (SE3Quat internally) 
/*                                                                                                     
static g2o::SE3Quat poseToSE3Quat(const pose_graph::Pose& p) {
    // YOUR CODE:                                                                                                                                                         
    //   Eigen::Quaterniond q(p.R);                       
    //   return g2o::SE3Quat(q, p.t);     
    Eigen::Quaterniond q(p.R);
    return g2o::SE3Quat(q, p.t);                              
} */

static Eigen::Isometry3d poseToIsometry(const pose_graph::Pose& p) {
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = p.R;
    iso.translation() = p.t;
    return iso;
}
                                                        
// Helper: convert relative transform to g2o::SE3Quat measurement  
/*                                                                                                       
static g2o::SE3Quat relativeToSE3Quat(const Eigen::Matrix3d& dR,
                                        const Eigen::Vector3d& dp) {                                                                                                      
    // YOUR CODE:                                                                                                                                                         
    //   Eigen::Quaterniond q(dR);                                                                                                                                        
    //   return g2o::SE3Quat(q, dp);                                                                                                                                      
    Eigen::Quaterniond q(dR);
    return g2o::SE3Quat(q, dp);                                
}                                                                                                                                                                         
*/

static Eigen::Isometry3d relativeToIsometry(const Eigen::Matrix3d& dR,
                                            const Eigen::Vector3d& dp) {
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = dR;
    iso.translation() = dp;
    return iso;
}

std::vector<pose_graph::Pose> optimizeWithG2O(                                                                                                                            
    const std::vector<pose_graph::Pose>& initial_poses,   
    const std::vector<pose_graph::ICPEdge>& icp_edges,
    const std::vector<pose_graph::IMUEdge>& imu_edges,                                                                                                                    
    const std::vector<pose_graph::GPSEdge>& gps_edges,
    const std::vector<pose_graph::LoopEdge>& loop_edges,                                                                                                                  
    int fixed_node,                                                                                                                                                       
    int max_iterations,
    bool verbose) {                                                                                                                                                       
                                                        
    // -------------------------------------------------------------------------
    // 1. Create optimizer
    // -------------------------------------------------------------------------                                                                                          
    
    auto linearSolver = std::make_unique<                                                                                                                            
        g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();                                                                                                
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));                                                                                
                                                                                                                                                                     
    g2o::SparseOptimizer optimizer;                                                                                                                                  
    optimizer.setAlgorithm(algorithm);                                                                                                                               
    optimizer.setVerbose(verbose);  
    
    

    // -------------------------------------------------------------------------
    // 2. Add vertices (one per pose)
    // -------------------------------------------------------------------------                                                                                          
    
    for (int i = 0; i < initial_poses.size(); ++i) {                                                                                                                 
        auto* v = new g2o::VertexSE3();                                                                                                                              
        v->setId(i);
        v->setEstimate(poseToIsometry(initial_poses[i]));                                                                                                             
        if (i == fixed_node) v->setFixed(true);                                                                                                                      
        optimizer.addVertex(v);
    }                                                                                                                                                                
                                                        
    // -------------------------------------------------------------------------                                                                                          
    // 3. Add ICP edges (binary SE3)                      
    // -------------------------------------------------------------------------
    
    for (const auto& e : icp_edges) {                                                                                                                                
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.dR_meas, e.dp_meas));
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                     
    }                                                                                                                                                                
                                                        
    // -------------------------------------------------------------------------                                                                                          
    // 4. Add IMU edges (same structure as ICP, different measurement)
    // -------------------------------------------------------------------------                                                                                          
    for (const auto& e : imu_edges) {            
        if (e.factor.covariance.block<3,3>(0,0).determinant() < 1e-20 ||                                                                                                      
            e.factor.covariance.block<3,3>(6,6).determinant() < 1e-20) {                                                                                                      
            continue;                                                                                                                                                         
        }                  
        Eigen::Matrix<double, 6, 6> info;
        info.setZero();     
        info.block<3,3>(0,0) = e.factor.covariance.block<3,3>(0,0).inverse();
        info.block<3,3>(3,3) = e.factor.covariance.block<3,3>(6,6).inverse();                                                                                                  
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.factor.dR, e.factor.dp));
        edge->setInformation(info);                                                                                                                         
        optimizer.addEdge(edge);                     
    }                                                                                                                                                                
                                                                                                                                                                        
    // -------------------------------------------------------------------------
    // 5. Add GPS edges (custom unary)                                                                                                                                    
    // -------------------------------------------------------------------------
    for (const auto& e : gps_edges) {
        auto* edge = new EdgeGPS();
        edge->setVertex(0, optimizer.vertex(e.frame_id));                                                                                                            
        edge->setMeasurement(e.position_meas);
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                                                                                                                                     
    }
                                                                                                                                                                        
    // -------------------------------------------------------------------------
    // 6. Add Loop edges (same as ICP)
    // -------------------------------------------------------------------------                                                                                          
    // YOUR CODE: same pattern as ICP
    for (const auto& e : loop_edges) {                                                                                                                                
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.dR_meas, e.dp_meas));
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                     
    }

    // -------------------------------------------------------------------------
    // 7. Optimize
    // -------------------------------------------------------------------------
    // YOUR CODE:
    //   optimizer.initializeOptimization();
    //   optimizer.optimize(max_iterations);    
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);                                                                                                                          
    std::cout << "g2o chi2: " << optimizer.chi2()
            << " edges: " << optimizer.edges().size()
            << " vertices: " << optimizer.vertices().size() << std::endl;
    // -------------------------------------------------------------------------                                                                                          
    // 8. Extract results                                 
    // -------------------------------------------------------------------------
    
    std::vector<pose_graph::Pose> result(initial_poses.size());                                                                                                      
    for (int i = 0; i < initial_poses.size(); ++i) {
        auto* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));                                                                                                 
        auto est = v->estimate();                    
        result[i].R = est.rotation();                                                                                                             
        result[i].t = est.translation();             
    }                                                                                                                                                                
    return result;                                                                                                                                                   

}                                                         


std::pair<std::vector<pose_graph::Pose>,std::vector<Eigen::Matrix<double, 6, 6>>>                                                                                                                       
optimizeWithG2OAndMarginals(
    const std::vector<pose_graph::Pose>& initial_poses,                                                                                                                   
    const std::vector<pose_graph::ICPEdge>& icp_edges,                                                                                                                    
    const std::vector<pose_graph::IMUEdge>& imu_edges,
    const std::vector<pose_graph::GPSEdge>& gps_edges,                                                                                                                    
    const std::vector<pose_graph::LoopEdge>& loop_edges,  
    int fixed_node,                                                                                                                                                       
    int max_iterations,                                                                                                                                                   
    bool verbose) {

      // YOUR CODE:                                         
      //   1. Reuse the exact same body as optimizeWithG2O(...) — build optimizer,
      //      add vertices/edges, ::initializeOptimization(), ::optimize(...).                                                                                              
      //                                                                                                                                                                    
      //   2. AFTER optimization completes but BEFORE destroying the optimizer,                                                                                             
      //      call the existing computeMarginalsG2O(optimizer, N):                                                                                                          
      //                                                                                                                                                                    
      //        auto marginals = pose_graph::computeMarginalsG2O(                                                                                                           
      //            optimizer, static_cast<int>(initial_poses.size()));                                                                                                     
      //                                                    
      //   3. Extract optimized poses as usual (identical to optimizeWithG2O).                                                                                              
      //                                                                                                                                                                    
      //   4. Return { poses, marginals }.
      //                                                                                                                                                                    
      //   Gotcha reminder: g2o returns covariance in [trans|rot] block layout,
      //   but our Matrix6 convention is [rot|trans]. computeMarginalsG2O                                                                                                   
      //   already applies the permutation — no additional work here.   
      
      //build optimizer
        auto linearSolver = std::make_unique<                                                                                                                            
            g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
                                                                                                                                                                            
    g2o::SparseOptimizer optimizer;                                                                                                                                  
    optimizer.setAlgorithm(algorithm);                                                                                                                               
    optimizer.setVerbose(verbose);  

    //add vertices
    for (int i = 0; i < initial_poses.size(); ++i) {                                                                                                                 
        auto* v = new g2o::VertexSE3();                                                                                                                              
        v->setId(i);
        v->setEstimate(poseToIsometry(initial_poses[i]));                                                                                                             
        if (i == fixed_node) v->setFixed(true);                                                                                                                      
        optimizer.addVertex(v); 


    }

    //add ICP edges
    for (const auto& e : icp_edges) {                                                                                                                                
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.dR_meas, e.dp_meas));
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                     
    }

    //add IMU edges
    for (const auto& e : imu_edges) {            
        if (e.factor.covariance.block<3,3>(0,0).determinant() < 1e-20 ||                                                                                                      
            e.factor.covariance.block<3,3>(6,6).determinant() < 1e-20) {                                                                                                      
            continue;                                                                                                                                                         
        }                  
        Eigen::Matrix<double, 6, 6> info;
        info.setZero();     
        info.block<3,3>(0,0) = e.factor.covariance.block<3,3>(0,0).inverse();
        info.block<3,3>(3,3) = e.factor.covariance.block<3,3>(6,6).inverse();                                                                                                  
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.factor.dR, e.factor.dp));
        edge->setInformation(info);                                                                                                                         
        optimizer.addEdge(edge);                     
    }

    //add GPS edges
    for (const auto& e : gps_edges) {
        auto* edge = new EdgeGPS();
        edge->setVertex(0, optimizer.vertex(e.frame_id));                                                                                                            
        edge->setMeasurement(e.position_meas);
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                                                                                                                                     
    }

    //add Loop edges
    for (const auto& e : loop_edges) {                                                                                                                                
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, optimizer.vertex(e.from));                                                                                                                
        edge->setVertex(1, optimizer.vertex(e.to));                                                                                                                  
        edge->setMeasurement(relativeToIsometry(e.dR_meas, e.dp_meas));
        edge->setInformation(e.information);                                                                                                                         
        optimizer.addEdge(edge);                     
    }

    //optimize
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);                                                                                                                                      
    std::cout << "g2o chi2: " << optimizer.chi2()
            << " edges: " << optimizer.edges().size()
            << " vertices: " << optimizer.vertices().size() << std::endl;

    //extract marginals
    auto marginals = pose_graph::computeMarginalsG2O(                                                                                                           
            optimizer, static_cast<int>(initial_poses.size()));
    
    //extract optimized poses
    std::vector<pose_graph::Pose> result(initial_poses.size());                                                                                                      
    for (int i = 0; i < initial_poses.size(); ++i) {
        auto* v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));                                                                                                 
        auto est = v->estimate();                    
        result[i].R = est.rotation();                                                                                                             
        result[i].t = est.translation();             
    }
    return { result, marginals };
    }  // namespace g2o_wrapper
}