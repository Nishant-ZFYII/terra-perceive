// ransac_ground_seg.cpp

#include "ransac_ground_seg.hpp"

//define model for RANSAC plane fitting.
float PlaneModel::distance(const Eigen::Vector3f& pt) const {
    // TODO (P1-M2.1): |normal.dot(pt) + d|

    return std::abs(normal.dot(pt) + d);
}

bool is_valid_index(int ix, int iy, size_t num_sectors_x, size_t num_sectors_y){
    return (ix >=0 && ix < num_sectors_x && iy >= 0 && iy < num_sectors_y);
}

SegmentationResult segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                  const RANSACParams& params) {
    SegmentationResult result;
    if(cloud.size() < 3){
        return result;
    }
    // TODO (P1-M2.1): Global RANSAC — implement first, then observe it fail on slopes
    //
    // Step 1: Random 3-point sampling with std::mt19937
    // Step 2: Plane fitting: n = (p2-p1).cross(p3-p1), normalize, d = -n.dot(p1)
    // Step 3: Degenerate check: if cross product norm < 1e-6, skip
    // Step 4: Inlier counting: |n.dot(pt) + d| < distance_threshold
    // Step 5: Track best model (most inliers)
    // Step 6: Early termination if inlier_ratio > 0.8
    // Step 7: Iteration count: N = log(1-p) / log(1-(1-e)^s)
    //
    
    //sample 3 points with random sample std::m19937 
    std::random_device rd;
    std::mt19937 gen(rd());
    //find no.of samples in the cloud.
    int num_points = cloud.size();

    //find 3 points, within the uniform standard distribution . 
    std::uniform_int_distribution<> dis(0, num_points - 1);

    size_t best_inlier_count = 0;
    PlaneModel best_plane;
    size_t iteration_count = 0;
    while(iteration_count < params.max_iterations){
        //repeat the above steps for max_iterations or until early termination
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        int idx3 = dis(gen);
        iteration_count++;
        if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3){
            continue;
        }

        //get the points corresponding to the random indices
        Eigen::Vector3f p1 = cloud[idx1];
        Eigen::Vector3f p2 = cloud[idx2];
        Eigen::Vector3f p3 = cloud[idx3];

        //plane fitting; n = (p2-p1).cross(p3-p1),
        Eigen::Vector3f normal = (p2 - p1).cross(p3 - p1);
        if (normal.z() < 0) normal = -normal;  
        
        
        //degenerate check: if cross product norm < 1e-6, skip
        if (normal.norm() < 1e-6) {
            continue;
        }

        //normalize, d = -n.dot(p1)
        normal.normalize();
        //find the number of inliers for this plane
        int inlier_count = 0;
        float d = -normal.dot(p1);
        PlaneModel candidate_plane{normal, d};
        for(const auto& pt : cloud){
            if (candidate_plane.distance(pt) < params.distance_threshold){
                inlier_count++;
            }
        }

        //update best model if this one has more inliers
        if (inlier_count > best_inlier_count){
            best_inlier_count = inlier_count;
            best_plane.normal = normal;
            best_plane.d = d;

        }
        // terminate if inlier ration >0.8
        
        float inlier_ratio = static_cast<float>(inlier_count) / num_points;
        if (inlier_ratio > 0.8f){
            break;
        }

    }

    if (best_inlier_count < static_cast<size_t>(params.min_inliers)) {                                                             
      return result;  // no valid plane found                                                                                    
    }
         
    result.ground_plane = best_plane;
    result.iterations_used = iteration_count;
    result.inlier_ratio = static_cast<float>(best_inlier_count) / num_points;

    //classify best_pane to split coud -> ground_points and obstacle_points

    for(const auto&pt : cloud){

        if (best_plane.distance(pt) < params.distance_threshold){
            result.ground_points.push_back(pt);
        } else {
            result.obstacle_points.push_back(pt);
        }

    }
    
    //SVD refinement : improve accuracy of the estimation
    if(result.ground_points.size() < 3)
        return result;
    //Compute centroid of ground points
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& pt : result.ground_points){
        centroid += pt;
    }

    centroid /= result.ground_points.size();

    //3xN matrix M of centered inliers. 
    Eigen::MatrixXf M(3, result.ground_points.size());
    for (size_t i = 0; i < result.ground_points.size(); ++i){
        M.col(i) = result.ground_points[i] - centroid;
    }

    //Covariance C = M * M.T / N
    Eigen::Matrix3f C = (M * M.transpose()) / result.ground_points.size();

    //SeflAdjointEigenSolver to find eigenvector with smallest eigenvalue
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(C);

    //smalles eigen Col 0
    Eigen::Vector3f refined_normal = eigen_solver.eigenvectors().col(0);
    if (refined_normal.dot(best_plane.normal) < 0) {
      refined_normal = -refined_normal;                                                                                          
    }
      

    //update d and the result.ground_pane with refined normal

    float refined_d = -refined_normal.dot(centroid);
    result.ground_plane.normal = refined_normal;
    result.ground_plane.d = refined_d;

    // re-classify using refined plane                                                                                             
    result.ground_points.clear();
    result.obstacle_points.clear();
    for (const auto& pt : cloud) {
        if (result.ground_plane.distance(pt) < params.distance_threshold) {                                                        
            result.ground_points.push_back(pt);                                                                                    
        } else {                                                                                                                   
            result.obstacle_points.push_back(pt);                                                                                  
        }           
    }



    return result;

    
}

//Check if the index is a valid and in bounds



SectorSegmentationResult sector_segment_ground(const std::vector<Eigen::Vector3f>& cloud,
                                               const RANSACParams& ransac_params,
                                               const SectorParams& sector_params) {
    SectorSegmentationResult result;
    // TODO (P1-M2.4): Sector-based RANSAC — the upgrade that handles graded terrain
    //
    // Step 1: Compute x-y bounds of the cloud
    // Step 2: Divide into grid of sectors based on sector_params.sector_size
    // Step 3: Assign points to sectors
    // Step 4: For each sector, run RANSAC (call segment_ground internally)
    // Step 5: (P1-M2.5) Continuity check: verify normal alignment with neighbors
    // Step 6: Consolidate ground/obstacle points from all reliable sectors

    //function to hande graded terrains. 

    //compute x-y bound of the cloud

    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();

    for(const auto& pt : cloud){
        if (pt.x() < x_min) x_min = pt.x();
        if (pt.x() > x_max) x_max = pt.x();
        if (pt.y() < y_min) y_min = pt.y();
        if (pt.y() > y_max) y_max = pt.y();
    }

    //find the number of sectors that fint in that range. 

    size_t num_sectors_x = static_cast<size_t>(std::ceil((x_max - x_min) / sector_params.sector_size_x));
    size_t num_sectors_y = static_cast<size_t>(std::ceil((y_max - y_min) / sector_params.sector_size_y));

    //for each point assign to the sector it belons to.

    std::vector<std::vector<std::vector<Eigen::Vector3f>>> sectors(num_sectors_x, std::vector<std::vector<Eigen::Vector3f>>(num_sectors_y));
    for (const auto& pt : cloud){
        int ix = static_cast<int>((pt.x() - x_min) / sector_params.sector_size_x);
        int iy = static_cast<int>((pt.y() - y_min) / sector_params.sector_size_y);
        ix = std::min(ix, static_cast<int>(num_sectors_x) - 1);
        iy = std::min(iy, static_cast<int>(num_sectors_y) - 1);
        sectors[ix][iy].push_back(pt);
    }

    //running RANSAC for each secotr

    for (int ix = 0; ix < num_sectors_x; ++ix){
        for (int iy = 0; iy < num_sectors_y; ++iy){
            if (sectors[ix][iy].size() < ransac_params.min_inliers){
                //add to result.secotr_planes as unreliable
                SectorPlane sector_plane;
                sector_plane.sector_ix = ix;
                sector_plane.sector_iy = iy;
                sector_plane.num_inliers = sectors[ix][iy].size();
                sector_plane.inlier_ratio = 0.0f;
                sector_plane.reliable = false;
                result.sector_planes.push_back(sector_plane);
                continue;
            }
            SegmentationResult seg_result = segment_ground(sectors[ix][iy], ransac_params);
            SectorPlane sector_plane;
            sector_plane.plane = seg_result.ground_plane;
            sector_plane.sector_ix = ix;
            sector_plane.sector_iy = iy;
            sector_plane.num_inliers = seg_result.ground_points.size();
            sector_plane.inlier_ratio = static_cast<float>(seg_result.ground_points.size()) / sectors[ix][iy].size();
            //reliability check based on inlier ratio and continuity with neighbors
            sector_plane.reliable = (sector_plane.inlier_ratio > 0.5f); // simple threshold, can be improved with continuity check
            result.sector_planes.push_back(sector_plane);
        }
    }

    //check if the neighboring sectors have similar normals, if not mark the sector as unreliable.
    //check ix +-1 and iy +-1 for each sector 
    //check the angle between the normals ,if the angle is greater than the threshold, mark as unreliable.

    //auto it = std::find_if(result.sector_planes.begin(), result.sector_planes.end(),[&](const SectorPlane& sp) { return sp.sector_ix == neighbor_ix && sp.sector_iy == neighbor_iy; });
    //for each reliable sector, append ground/obstacle points to the result.
    //cound num_sectors and num_reliable_sectors
    //populating result.ground_points, result.obstacle_points, result.num_sectors, result.num_reliable_sectors
    // But where do you get the per-sector ground/obstacle points? You don't store them from segment_ground. You need to either re-run or store seg_result.ground_points somewhere in step 4.
    //is_valid_index is undefined — you call it on line 244 but never define it. It will fail to compile. Either define it as a lambda above the loop, or inline the bounds check directly.
    //Neighbor offsets include diagonals — x_dirs = {-1, 0, 1} × y_dirs = {-1, 0, 1} gives 8 neighbors (skipping 0,0). The spec says 4 neighbors (cardinal directions only). Change to {-1, +1} for x with dy=0, and {-1, +1} for y with dx=0. Or use a fixed array of 4 pairs: {-1,0}, {+1,0}, {0,-1}, {0,+1}.

    int dirs_x[4] = {-1, 1, 0, 0};
    int dirs_y[4] = {0, 0, -1, 1};

    for (auto& sector_plane : result.sector_planes){
        if (!sector_plane.reliable){
            continue;
        }
        //check neighbors
        for (int d = 0; d < 4; ++d){
            int neighbor_ix = sector_plane.sector_ix + dirs_x[d];
            int neighbor_iy = sector_plane.sector_iy + dirs_y[d];
            if (!is_valid_index(neighbor_ix, neighbor_iy, num_sectors_x, num_sectors_y)){
                continue;
            }
            auto it = std::find_if(result.sector_planes.begin(), result.sector_planes.end(),[&](const SectorPlane& sp) { return sp.sector_ix == neighbor_ix && sp.sector_iy == neighbor_iy; });
            if (it != result.sector_planes.end() && it->reliable){
                float dot = std::max(-1.0f, std::min(1.0f, sector_plane.plane.normal.dot(it->plane.normal)));
                 float angle = std::acos(dot);
                if (angle > sector_params.continuity_angle_thresh_deg * M_PI / 180.0f){
                    //mark as unreliable
                    sector_plane.reliable = false;
                    break;
                }
            }
        }
        if (sector_plane.reliable){
            //append ground/obstacle points to the result
            int ix = sector_plane.sector_ix;
            int iy = sector_plane.sector_iy;
            for (const auto& pt : sectors[ix][iy]){
                if (sector_plane.plane.distance(pt) < ransac_params.distance_threshold){
                    result.ground_points.push_back(pt);
                } else {
                    result.obstacle_points.push_back(pt);
                }
            }
        }
    }
    result.num_sectors = num_sectors_x * num_sectors_y;
    result.num_reliable_sectors = std::count_if(result.sector_planes.begin(), result.sector_planes.end(), [](const SectorPlane& sp) { return sp.reliable; });









    








    return result;
}
