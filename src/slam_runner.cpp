// slam_runner.cpp                          
// Main executable: loads all data, builds pose graph, runs optimizer,
// saves optimized trajectory. Supports ablation via command-line flags.                                                                                                  
//                                                                                                                                                                        
// Usage:                                                                                                                                                                 
//   ./slam_runner --icp data/poses_icp.csv \                                                                                                                             
//                 --imu data/imu_raw.csv \                                                                                                                               
//                 --gps data/poses_gps.csv \                                                                                                                             
//                 --lidar data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin/ \                                                  
//                 --out data/poses_slam.csv \            
//                 --use-icp --use-imu --use-gps --use-loop \                                                                                                             
//                 --optimizer manifold \   
//                 --verbose                                                                                                                                              
                                                        
#include <iostream>                                                                                                                                                       
#include <string>                                         
#include <vector>                                                                                                                                                         
#include <filesystem>                                     

#include "pose_graph_slam.hpp"                                                                                                                                            
#include "imu_preintegration.hpp"
#include "scan_context.hpp"                                                                                                                                               
#include "point_cloud_loader.hpp"      
#include <g2o/core/sparse_optimizer.h>                                                                                                                                    
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>                                                                                                                    
#include <g2o/solvers/eigen/linear_solver_eigen.h>        
#include <g2o/types/slam3d/vertex_se3.h>                                                                                                                                  
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/base_unary_edge.h>  
#include "g2o_wrapper.hpp"                   
                                        
// Simple arg parser — returns value after flag, or default if not found
static std::string getArg(int argc, char** argv, const std::string& flag, const std::string& def = "") {                                                                  
    for (int i = 1; i < argc - 1; ++i)      
        if (std::string(argv[i]) == flag) return argv[i + 1];                                                                                                             
    return def;                                           
}                                                                                                                                                                         
                                                                                                                                                                        
static bool hasFlag(int argc, char** argv, const std::string& flag) {                                                                                                     
    for (int i = 1; i < argc; ++i)                                                                                                                                        
        if (std::string(argv[i]) == flag) return true;    
    return false;                                                                                                                                                         
}      

static std::string formatFrameId(int id) {                
    char buf[16];                                                                                                                                                         
    snprintf(buf, sizeof(buf), "%06d", id);               
    return std::string(buf) + ".bin";                                                                                                                                     
}    
                                                                                                                                                                        
int main(int argc, char** argv) {                         
    // -------------------------------------------------------------------------                                                                                          
    // 1. Parse command-line arguments
    // -------------------------------------------------------------------------                                                                                          
    std::string icp_path   = getArg(argc, argv, "--icp", "data/poses_icp.csv");
    std::string imu_path   = getArg(argc, argv, "--imu", "data/imu_raw.csv");
    std::string gps_path   = getArg(argc, argv, "--gps", "data/poses_gps.csv");
    std::string lidar_dir  = getArg(argc, argv, "--lidar", "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin/");                
    std::string out_path   = getArg(argc, argv, "--out", "data/poses_slam.csv");
    std::string opt_type   = getArg(argc, argv, "--optimizer", "manifold"); 
    std::string cov_out_base = getArg(argc, argv, "--cov-out-base", "");                                                                                                      
    std::string cov_method   = getArg(argc, argv, "--cov-method",   "none");                                                                                                  
    // valid: "none", "heuristic", "g2o", "both"                                                                                              
                                                        
    pose_graph::PoseGraphConfig config;                                                                                                                                   
    config.use_icp_edges  = hasFlag(argc, argv, "--use-icp");
    config.use_imu_edges  = hasFlag(argc, argv, "--use-imu");                                                                                                             
    config.use_gps_edges  = hasFlag(argc, argv, "--use-gps");
    config.use_loop_edges = hasFlag(argc, argv, "--use-loop");                                                                                                            
    config.verbose        = hasFlag(argc, argv, "--verbose");                                                                                                             
                                        
    if (opt_type == "euclidean")                                                                                                                                          
        config.optimizer = pose_graph::PoseGraphConfig::OptimizerType::Euclidean;                                                                                         
    else
        config.optimizer = pose_graph::PoseGraphConfig::OptimizerType::Manifold;                                                                                          
                                                        
    // If no edge flags specified, enable all (default full system)                                                                                                       
    if (!hasFlag(argc, argv, "--use-icp") && !hasFlag(argc, argv, "--use-imu") &&
        !hasFlag(argc, argv, "--use-gps") && !hasFlag(argc, argv, "--use-loop")) {                                                                                        
        config.use_icp_edges = config.use_imu_edges = config.use_gps_edges = config.use_loop_edges = true;
    }                                                                                                                                                                     
                                                                                                                                                                        
    std::cout << "Config: ICP=" << config.use_icp_edges                                                                                                                   
            << " IMU=" << config.use_imu_edges                                                                                                                          
            << " GPS=" << config.use_gps_edges                                                                                                                          
            << " Loop=" << config.use_loop_edges                                                                                                                        
            << " Optimizer=" << opt_type << "\n";
                                                                                                                                                                        
    // -------------------------------------------------------------------------                                                                                          
    // 2. Load initial poses from KISS-ICP
    // -------------------------------------------------------------------------                                                                                          
    // YOUR CODE:                                                                                                                                                         
    //   auto poses = pose_graph::loadPosesFromCSV(icp_path);
    //   std::cout << "Loaded " << poses.size() << " poses from " << icp_path << "\n";  
    
    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(icp_path);
    std::cout << "Loaded " << poses.size() << " poses from " << icp_path << "\n";
                                                        
    // -------------------------------------------------------------------------                                                                                          
    // 3. Build pose graph
    // -------------------------------------------------------------------------                                                                                          
    // YOUR CODE:                                         
    //   pose_graph::PoseGraphSLAM graph(config);                                                                                                                         
    //   graph.setInitialPoses(poses);                    
    //   graph.setFixed(0, true);  // anchor frame 0

    pose_graph::PoseGraphSLAM graph(config);
    graph.setInitialPoses(poses);
    graph.setFixed(0, true); // Fix the first node as the anchor
                                            
    // -------------------------------------------------------------------------                                                                                          
    // 3a. Add ICP edges (consecutive relative transforms from poses_icp)                                                                                                 
    // -------------------------------------------------------------------------                                                                                                                                                                                                                                                  
    for (int i = 0; i < poses.size() - 1; ++i) {  
        if(i%100 == 0 || i == (int) poses.size() - 1)
            std::cerr<< "\rAdding ICP edges: " << i+1 << "/" << poses.size() << std::flush;                                                                                                                   
        pose_graph::ICPEdge edge;                                                                                                                                    
        edge.from = i; edge.to = i + 1;
        edge.dR_meas = poses[i].R.transpose() * poses[i+1].R;                                                                                                        
        edge.dp_meas = poses[i].R.transpose() * (poses[i+1].t - poses[i].t);
        edge.information = Eigen::Matrix<double,6,6>::Identity() * 100.0;                                                                                            
        graph.addICPEdge(edge);                                                                                                                                      
    }                                                                                                                                                                
    std::cout << "Added " << poses.size()-1 << " ICP edges\n";
    
    // -------------------------------------------------------------------------
    // 3c. Add GPS edges (unary position constraints)                                                                                                                     
    // -------------------------------------------------------------------------
                                                                                                                                                            
    auto [gps_poses, gps_timestamps] = pose_graph::loadPosesFromCSV(gps_path);
    // GPS CSV might have different format — just need (t, x, y, z)                                                                                                  
    // For each GPS reading matched to a LiDAR frame:
    for (int i = 0; i < gps_poses.size() && i < (int)poses.size(); ++i) {
        if(i%100 == 0 || i == (int) gps_poses.size() - 1)
            std::cerr<< "\rAdding GPS edges: " << i+1 << "/" << gps_poses.size() << std::flush;
        pose_graph::GPSEdge edge;                                                                                                                                    
        edge.frame_id = i;                                                                                                                                           
        edge.position_meas = gps_poses[i].t;                                                                                                                         
        // HDOP-weighted: sigma = 0.5 + 2.0 * hdop (if available)                                                                                                    
        double sigma ;   // default; refine with HDOP later       
        if ( i >= 1800 && i<=2200) sigma = 50.0;
        else if (i >= 2400) sigma = 15.0;
        else sigma = 8.0;                                                                                              
        edge.information = Eigen::Matrix3d::Identity() / (sigma * sigma);
        graph.addGPSEdge(edge);                                                                                                                                      
    }                                                                                                                                                                
    std::cout << "\nAdded " << std::min(gps_poses.size(), poses.size()) << " GPS edges\n";       
    
        // -------------------------------------------------------------------------                                                                                          
    // 3b. Add IMU edges (preintegrated factors)                                                                                                                          
    // -------------------------------------------------------------------------                                                                                          
                              
    auto imu_data = imu::loadIMUFromCSV(imu_path);
    imu::Bias bias;
    if (hasFlag(argc, argv, "--zero-bias")) {
        std::cout << "IMU bias: ZERO (forced)\n";
    } else {
        bias = imu::estimateBiasFromStatic(imu_data, 100);
        std::cout << "IMU bias: gyro=" << bias.gyro.transpose()
                  << " accel=" << bias.accel.transpose() << "\n";
    }
                                                                                                                                                                     
    // Extract LiDAR timestamps from poses (column 0 of poses_icp.csv)                                                                                               
    // You'll need to store timestamps when loading poses, or load separately
    double t0_epoch = gps_timestamps[0] * 1e-9;  // first GPS timestamp as epoch seconds                                                                                      
    std::vector<double> lidar_times;                                                                                                                                          
    lidar_times.reserve(timestamps.size()); 
    for (size_t i = 0; i < timestamps.size(); ++i) {                                                                                                                          
        lidar_times.push_back(t0_epoch + timestamps[i] * 1e-9);                                                                                                               
    }                                                                                                                  
                                                     
    auto imu_factors = imu::preintegrateTrajectory(imu_data, lidar_times, bias);                                                                                     
    for (int i = 0; i < imu_factors.size(); ++i) {   
        if(i%100 == 0 || i == (int) imu_factors.size() - 1)
            std::cerr<< "\rAdding IMU edges: " << i+1 << "/" << imu_factors.size() << std::flush;                                                                                                               
        pose_graph::IMUEdge edge;                                                                                                                                    
        edge.from = imu_factors[i].frame_from;                                                                                                                       
        edge.to = imu_factors[i].frame_to;                                                                                                                           
        edge.factor = imu_factors[i];                
        graph.addIMUEdge(edge);                                                                                                                                      
    }                                                
    std::cout << "Added " << imu_factors.size() << " IMU edges\n";      
                                                                                                                                                                        
    // -------------------------------------------------------------------------                                                                                          
    // 3d. Add loop closure edges (Scan Context + ICP alignment)
    // -------------------------------------------------------------------------                                                                                          
                           
    // Build Scan Context descriptors for all frames                                                                                                                 
    std::vector<scan_context::Descriptor> descriptors;                                                                                                               
    for (int i = 0; i < poses.size(); ++i) {
        if(i%100 == 0 || i == (int) poses.size() - 1)
            std::cerr<< "\rBuilding Scan Context descriptors: " << i+1 << "/" << poses.size() << std::flush;
        std::string bin_path = lidar_dir + "/" + formatFrameId(i);                                                                                        
        auto points = load_bin(bin_path);  // from point_cloud_loader                                                                                          
        // Convert to Vector3f if needed
        descriptors.push_back(scan_context::buildDescriptor(points));                                                                                              
    }                                                                                                                                                                
    std::cout<<"\nStarting Loop Closure Detection\n";                                                              
    // Detect loop closures                                                                                                                                          
    for (int i = 50; i < descriptors.size(); i += 500) {         
        if(i%100 == 0 || i == (int) descriptors.size() - 1)
            std::cerr<< "\rDetecting loop closures: " << i+1 << "/" << descriptors.size() << std::flush;
        auto candidates = scan_context::detectLoopClosures(descriptors, i);                                                                                          
        for (const auto& c : candidates) {                                                                                                                           
            // ICP align scan[c.match_id] to scan[i] for relative transform
            // (or use the existing pose estimates as approximation)                                                                                                 
            pose_graph::LoopEdge edge; 
            edge.from = c.match_id;                                                                                                                                  
            edge.to = c.query_id;                                                                                                                                    
            edge.dR_meas = poses[c.match_id].R.transpose() * poses[c.query_id].R;                                                                                    
            edge.dp_meas = poses[c.match_id].R.transpose() * (poses[c.query_id].t - poses[c.match_id].t);                                                            
            edge.information = Eigen::Matrix<double,6,6>::Identity() * 200.0;                                                                                        
            graph.addLoopEdge(edge);                                                                                                                                 
        }                                            
    }                                                                                                                                                                
    std::cout << "Added loop closure edges\n";       
    std::vector<pose_graph::Pose> result_g2o;                                                                                                                                 
    if (cov_method != "g2o" && cov_method != "both") {
        result_g2o = g2o_wrapper::optimizeWithG2O(                                                                                                                            
            poses, graph.getICPEdges(), graph.getIMUEdges(),                                                                                                                  
            graph.getGPSEdges(), graph.getLoopEdges(),                                                                                                                        
            0, 100, config.verbose);                                                                                                                                          
        pose_graph::savePosesToCSV(result_g2o, "data/poses_slam_g2o.csv");                                                                                                    
        std::cout << "Saved g2o result to data/poses_slam_g2o.csv\n";
    }                                                                                                                                                                
    // -------------------------------------------------------------------------
    // 4. Optimize                                                                                                                                                        
    // -------------------------------------------------------------------------
    
    auto result = graph.optimize();
    std::cout << "Optimization done: " << graph.lastIterations() << " iterations, "
              << "final cost = " << graph.lastCost() << "\n";
                                            
    // -------------------------------------------------------------------------                                                                                          
    // 5. Save result                                                                                                                                                     
    // -------------------------------------------------------------------------  
    
 // -------------------------------------------------------------------------                                                                                              
  // 5a. P2-M3 Ablation E — compute per-pose covariances for confidence scaling
  // -------------------------------------------------------------------------  
    if (cov_method != "none" && cov_out_base.empty()) {
    std::cerr << "WARNING: --cov-method " << cov_method                                                                                                                   
            << " requires --cov-out-base; skipping covariance save.\n";                                                                                                 
    }                                                                                            
    if (cov_method != "none" && !cov_out_base.empty()) {                                                                                                                      
        const int N = static_cast<int>(result.size());                                                                                                                        
                                                                                                                                                                                
        // --- Heuristic path (cheap — only needs the graph) --------------------                                                                                             
        if (cov_method == "heuristic" || cov_method == "both") {                                                                                                              
            auto covs_h = pose_graph::computeEdgeInformationSum(graph, N);                                                                                                    
            bool ok = pose_graph::saveCovariancesToCSV(                                                                                                                     
                covs_h, cov_out_base + "_cov_heuristic.csv");                                                                                                                 
            if (!ok) {                                                                                                                                                        
                std::cerr << "ERROR: saveCovariancesToCSV failed for heuristic at "                                                                                           
                            << cov_out_base << "_cov_heuristic.csv\n";                                                                                                          
                return 1;                                                                                                                                                   
            }                                                                                                                                                                 
            std::cout << "Saved heuristic covariances (" << covs_h.size()                                                                                                   
                        << ") to " << cov_out_base << "_cov_heuristic.csv\n";                                                                                                   
            if (!covs_h.empty()) {                                                                                                                                            
                std::cout << "  heuristic pose_sigma[0]="                                                                                                                     
                            << pose_graph::poseSigmaFromCovariance(covs_h.front())                                                                                              
                            << "  pose_sigma[N-1]="                                                                                                                             
                            << pose_graph::poseSigmaFromCovariance(covs_h.back())
                            << "\n";                                                                                                                                            
            }                                                                                                                                                               
        }                                                                                                                                                                     
                                                                                                                                                                            
        // --- True-marginal path (needs a live g2o::SparseOptimizer) ----------                                                                                              
        if (cov_method == "g2o" || cov_method == "both") {
            auto [poses_g2o, covs_g] = g2o_wrapper::optimizeWithG2OAndMarginals(                                                                                              
                poses, graph.getICPEdges(), graph.getIMUEdges(),                                                                                                              
                graph.getGPSEdges(), graph.getLoopEdges(),                                                                                                                    
                0, 100, config.verbose);                                                                                                                                      
            bool ok = pose_graph::saveCovariancesToCSV(                                                                                                                       
                covs_g, cov_out_base + "_cov_g2o.csv");                                                                                                                       
            if (!ok) {                                                                                                                                                        
                std::cerr << "ERROR: saveCovariancesToCSV failed for g2o at "                                                                                               
                            << cov_out_base << "_cov_g2o.csv\n";                                                                                                                
                return 1;                                                                                                                                                     
            }
            std::cout << "Saved g2o marginals (" << covs_g.size()                                                                                                             
                        << ") to " << cov_out_base << "_cov_g2o.csv\n";                                                                                                       
            if (!covs_g.empty()) {                                                                                                                                            
                std::cout << "  g2o pose_sigma[0]="
                            << pose_graph::poseSigmaFromCovariance(covs_g.front())                                                                                              
                            << "  pose_sigma[N-1]="                                                                                                                             
                            << pose_graph::poseSigmaFromCovariance(covs_g.back())
                            << "\n";                                                                                                                                            
            }    
            pose_graph::savePosesToCSV(poses_g2o, "data/poses_slam_g2o.csv");       
            std::cout << "Saved g2o poses to data/poses_slam_g2o.csv\n";                                                                                                                                       
        }
    }
    pose_graph::savePosesToCSV(result, out_path);                                                                                                                    
    std::cout << "Saved " << result.size() << " optimized poses to " << out_path << "\n";
                                            
    return 0;                           
}
