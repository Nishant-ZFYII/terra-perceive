// Scan Context loop closure detection — Kim & Kim (IROS 2018), Sections 3–4.                                                                                             
                                                                                                                                                                        
#include "scan_context.hpp"                                                                                                                                               
#include <cmath>                                                                                                                                                          
#include <algorithm>                                                                                                                                                      
                                                        
namespace scan_context {                                                                                                                                                  
                                                        
Descriptor buildDescriptor(const std::vector<Eigen::Vector3f>& points,
                            float max_range) {                                                                                                                                                                                                                                         
    Descriptor desc = Descriptor::Zero();         
    for(const auto& point : points){
        float x = point.x();
        float y = point.y();
        float z = point.z();
        float range = std::sqrt(x*x + y*y);
        if(range < 1.0f || range > max_range) continue;
        int ring = std::min(static_cast<int>(std::floor(range / max_range * N_RINGS)), N_RINGS - 1);
        int sector = std::min(static_cast<int>(std::floor((std::atan2(y, x) + M_PI) / (2 * M_PI) * N_SECTORS)), N_SECTORS - 1);
        desc(ring, sector) = std::max(desc(ring, sector), z);
    }

    return desc;                                                                                                                                                          
}                                                                                                                                                                         
                                                                                                                                                                        
float columnShiftDistance(const Descriptor& a, const Descriptor& b, int shift) {                                                                                          
    // YOUR CODE — Kim & Kim Section 4:                                                                                                                                   
    //   - for each column c = 0..N_SECTORS-1:            
    //       col_a = a.col(c)                          ← N_RINGS × 1 vector
    //       col_b = b.col((c + shift) % N_SECTORS)    ← shifted column
    //       norm_a = col_a.norm(), norm_b = col_b.norm()                                                                                                                 
    //       if norm_a < 1e-6 or norm_b < 1e-6: skip (empty column)
    //       cosine_dist = 1.0 - col_a.dot(col_b) / (norm_a * norm_b)                                                                                                     
    //       accumulate cosine_dist, count valid columns                                                                                                                  
    //   - return average cosine distance over valid columns                                                                                                              
    //   - if no valid columns: return 2.0 (maximum distance)     
    float total_dist = 0.0f;                                                                                                                                               
    int valid_cols = 0;                                                                                                                                                    
    for(int c = 0; c < N_SECTORS; ++c) {
        Eigen::VectorXf col_a = a.col(c);
        Eigen::VectorXf col_b = b.col((c + shift) % N_SECTORS);
        float norm_a = col_a.norm();
        float norm_b = col_b.norm();
        if(norm_a < 1e-6f || norm_b < 1e-6f) continue; // skip empty columns
        float cosine_dist = 1.0f - col_a.dot(col_b) / (norm_a * norm_b);
        total_dist += cosine_dist;
        valid_cols++;
    }
    if(valid_cols == 0) return 2.0f;
    return total_dist / valid_cols;                                                                                                                                            
}                                                                                                                                                                         
                                                                                                                                                                        
std::pair<float, int> matchDescriptors(const Descriptor& a, const Descriptor& b) {                                                                                        
    // YOUR CODE:                                                                                                                                                         
    //   - try all N_SECTORS shifts (0 to N_SECTORS-1)    
    //   - call columnShiftDistance(a, b, shift) for each                                                                                                                 
    //   - track minimum distance and corresponding shift
    //   - return {min_distance, best_shift}                                                                                                                              
    float best_dist = 2.0f;                                                                                                                                               
    int best_shift = 0;                     
    for (int shift = 0; shift < N_SECTORS; ++shift) {
        float dist = columnShiftDistance(a, b, shift);
        if (dist < best_dist) {
            best_dist = dist;
            best_shift = shift;
        }
    }
    return {best_dist, best_shift};                                                                                                                                       
}                                                         
                                                                                                                                                                        
std::vector<LoopCandidate> detectLoopClosures(            
    const std::vector<Descriptor>& history,                                                                                                                               
    int query_id,                                         
    float dist_threshold,               
    int min_gap) {
    // YOUR CODE:                                                                                                                                                         
    //   - for each candidate i in history:
    //       skip if abs(query_id - i) < min_gap (temporal neighbor, not a loop)                                                                                          
    //       auto [dist, shift] = matchDescriptors(history[query_id], history[i])
    //       if dist < dist_threshold: add LoopCandidate{query_id, i, dist, shift}
    //   - sort candidates by distance (best first)
    //   - return candidates                                                                                                                                              
    std::vector<LoopCandidate> candidates;  
    for (int i = 0; i < history.size(); ++i) {
        if (std::abs(query_id - i) < min_gap) continue;
        auto [dist, shift] = matchDescriptors(history[query_id], history[i]);
        if (dist < dist_threshold) {
            candidates.push_back({query_id, i, dist, shift});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const LoopCandidate& a, const LoopCandidate& b) {
        return a.distance < b.distance;
    });
    return candidates;                                                                                                                                                    
}                                                         
                                                                                                                                                                        
}  // namespace scan_context    