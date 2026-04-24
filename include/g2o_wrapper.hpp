// g2o_wrapper.hpp                                                                                                                                                        
// Thin wrapper around g2o for pose graph optimization comparison.                                                                                                      
// Takes the same edges as PoseGraphSLAM, runs g2o's LM solver,                                                                                                           
// returns optimized poses for ATE comparison.                                                                                                                            
                                                                                                                                                                        
#pragma once                                                                                                                                                              
#include <vector>                                         
#include "pose_graph_slam.hpp"                                                                                                                                            

namespace g2o_wrapper {                                                                                                                                                   
                                                        
// Run g2o optimization on the same graph data.                                                                                                                           
// Returns optimized poses for comparison against custom optimizer.
std::vector<pose_graph::Pose> optimizeWithG2O(                                                                                                                            
    const std::vector<pose_graph::Pose>& initial_poses,                                                                                                                   
    const std::vector<pose_graph::ICPEdge>& icp_edges,
    const std::vector<pose_graph::IMUEdge>& imu_edges,                                                                                                                    
    const std::vector<pose_graph::GPSEdge>& gps_edges,    
    const std::vector<pose_graph::LoopEdge>& loop_edges,                                                                                                                  
    int fixed_node = 0,                                                                                                                                                   
    int max_iterations = 100,                                                                                                                                             
    bool verbose = false);   

  // Optimize AND compute per-vertex marginals in one pass.
  // Returns (optimized_poses, marginals_6x6_per_pose).                                                                                                                     
  // Marginals order matches the poses vector.                                                                                                                              
std::pair<std::vector<pose_graph::Pose>,                                                                                                                                  
        std::vector<Eigen::Matrix<double, 6, 6>>>                                                                                                                       
optimizeWithG2OAndMarginals(                                                                                                                                              
    const std::vector<pose_graph::Pose>& initial_poses,   
    const std::vector<pose_graph::ICPEdge>& icp_edges,                                                                                                                    
    const std::vector<pose_graph::IMUEdge>& imu_edges,    
    const std::vector<pose_graph::GPSEdge>& gps_edges,                                                                                                                    
    const std::vector<pose_graph::LoopEdge>& loop_edges,                                                                                                                  
    int fixed_node = 0,
    int max_iterations = 100,                                                                                                                                             
    bool verbose = false); 
                                                                                                                                                                    
}  // namespace g2o_wrapper