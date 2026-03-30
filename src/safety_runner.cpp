#include "safety_supervisor.hpp"
#include <iostream>                                                                                                            
#include <fstream>
#include <vector>
#include <string>
#include <cmath>                                               
                                                                                                                                
int main(int argc, char* argv[]) {
    // 1. Check argc, parse: grid.bin, vehicle_speed, output.csv    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <grid.bin> <vehicle_speed_mps> [output.csv]" << std::endl;
        return 1;
    }
    std::string grid_path = argv[1];
    float vehicle_speed_mps = std::stof(argv[2]);
    std::string output_csv = (argc >= 4) ? argv[3] : "output.csv";
    // Load grid.bin (float32) into std::vector<float>  
    std::ifstream infile(grid_path, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file: " << grid_path << std::endl;
        return 1;
    }
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    size_t num_floats = file_size / sizeof(float);
    std::vector<float> grid_data(num_floats);
    infile.read(reinterpret_cast<char*>(grid_data.data()), file_size);
    infile.close();                                                        
                                                                                                                                
    //Create SafetyConfig, StoppingDistanceModel, SafetySupervisor  

    SafetyConfig config;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    float trav_sum = 0.0f;
    int trav_count = 0;
    for (size_t i = 0; i < num_floats / 6; ++i) {
        float confidence = grid_data[i * 6 + 1];
        if (confidence > 0.5f) {
            float risk = grid_data[i * 6 + 0];
            trav_sum += (1.0f - risk);
            trav_count++;
        }
    }
    float mean_trav = (trav_count > 0) ? (trav_sum / trav_count) : 0.0f;
    std::cout << "Mean Traversability: " << mean_trav << " (from " << trav_count << " cells)" << std::endl;
                                                                                                                                
    // 4. Simulate 30 timesteps at dt=0.1s                                                                                     
    //    worker_d starts at 15.0f, decreases by 1.2f * dt each step
    //    call supervisor.update_lidar_timestamp(t) before first evaluate()                                                    
    //    call supervisor.evaluate(vehicle_speed, worker_d, 1.2f, mean_trav, t) 
    float worker_d = 15.0f;
    supervisor.update_lidar_timestamp(0.0); // initialize LiDAR timestamp
    for (int step = 0; step < 30; ++step) {
        double t = step * 0.1;
        supervisor.update_lidar_timestamp(t);
        SafetyIntervention intervention = supervisor.evaluate(vehicle_speed_mps, worker_d, -1.2f, mean_trav, t);
        std::cout << "Time: " << t << "s, Worker Distance: " << worker_d << "m, Intervention: "
                  << (intervention.level == SafetyIntervention::EMERGENCY_STOP ? "EMERGENCY_STOP" :
                      (intervention.level == SafetyIntervention::HARD_BRAKE ? "HARD_BRAKE" :
                      (intervention.level == SafetyIntervention::PROPORTIONAL_SCALE ? "PROPORTIONAL_SCALE" : "NONE")))
                  << ", Scale Factor: " << intervention.scale_factor << ", Reason: " << intervention.reason << std::endl;
        worker_d -= 1.2f * 0.1f; // decrease distance by 1.2 m/s * dt
    }                                              
                                                                                                                                
    //Write supervisor.event_log() to CSV                                                                                  
    std::ofstream outfile(output_csv);
    if (!outfile) {
        std::cerr << "Error opening output file: " << output_csv << std::endl;
        return 1;
    }
    outfile << "timestamp,rule,d_worker,d_stop,ttc,mu,vel_before,vel_after\n";
    const auto& events = supervisor.event_log();
    for (const auto& event : events) {
        outfile << event.timestamp << ","
                << event.rule << ","
                << event.d_worker << ","
                << event.d_stop << ","
                << event.ttc << ","
                << event.friction_mu << ","
                << event.vel_before << ","
                << event.vel_after << "\n";
    }
    outfile.close();
    std::cout << "Wrote " << events.size() << " events to " << output_csv << std::endl;
    supervisor.report_latency_stats();  
    
    
                                                                                                                                
    return 0;
}  