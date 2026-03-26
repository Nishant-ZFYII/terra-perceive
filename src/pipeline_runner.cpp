#include "traversability.hpp"
#include "ransac_ground_seg.hpp"
#include "point_cloud_loader.hpp"
#include <iostream>
#include <cmath>
#include <algorithm> 
#include <limits>
#include <fstream> 

int main(int argc, char* argv[]) {                                                                                              
    // 1. Check argc — need argv[1] (bin path) and argv[2] (output path) 
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bin_path> <output_grid_path>" << std::endl;
        return 1;
    }                                                       
                                                                                                                                
    // 2. load_bin(argv[1]) → vector<Eigen::Vector3f> cloud 
    auto cloud = load_bin(argv[1]);                                                                    
                                                                                                                                
    // 3. sector_segment_ground(cloud, ransac_params, sector_params)                                                            
    //    → SectorSegmentationResult — use result.ground_points
    RANSACParams ransac_params;
    ransac_params.distance_threshold = 0.15f;
    ransac_params.max_iterations = 1000;
    ransac_params.confidence = 0.99f;
    ransac_params.min_inliers = 100;

    SectorParams sector_params;
    sector_params.sector_size_x = 5.0f;
    sector_params.sector_size_y = 5.0f;
    sector_params.continuity_angle_thresh_deg = 30.0f;

    SectorSegmentationResult result = sector_segment_ground(cloud, ransac_params, sector_params);



                                                                                                                                
    // 4. TraversabilityGrid grid(params, vehicle)                                                                              
    //    grid.compute(result.ground_points)              
    
    GridParams grid_params;
    grid_params.x_min = -5.0f;
    grid_params.x_max = 30.0f;
    grid_params.y_min = -15.0f;
    grid_params.y_max = 15.0f;
    grid_params.resolution = 0.5f;
    grid_params.min_points_per_cell = 3;

    VehicleKinematics vehicle;  
    vehicle.max_climbable_grade_deg = 30.0f;
    vehicle.max_roughness_tolerance = 0.275f;
    vehicle.max_step_height = 0.2f;
    vehicle.vehicle_mass_kg = 590.0f;
    vehicle.max_speed_mps = 5.0f;

    TraversabilityGrid grid(grid_params, vehicle);
    grid.compute(result.ground_points);
                                                                                                                                
    // 5. Write grid to binary file (argv[2])                                                                                   
    //    — loop over rows/cols, write each CellFeatures field

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::cerr << "Error opening output file: " << argv[2] << std::endl;
        return 1;
    }
    for (int ix = 0; ix < grid.rows(); ++ix) {
        for (int iy = 0; iy < grid.cols(); ++iy) {
            const CellFeatures& c = grid.at(ix, iy);
            out.write(reinterpret_cast<const char*>(&c.risk), sizeof(float));
            out.write(reinterpret_cast<const char*>(&c.confidence), sizeof(float));
            out.write(reinterpret_cast<const char*>(&c.slope_deg), sizeof(float));
            out.write(reinterpret_cast<const char*>(&c.roughness), sizeof(float));
            out.write(reinterpret_cast<const char*>(&c.step_height), sizeof(float));
            out.write(reinterpret_cast<const char*>(&c.mean_z), sizeof(float));
        }
    }

    out.close();
    std::cout << "Grid written to: " << argv[2] << std::endl;                                                                       
    std::cout << "Ground points: " << result.ground_points.size() << std::endl;                                                     
    return 0;    
  }             