// safety_supervisor.cpp
// TODO: Implement kinematic safety supervisor from scratch.
// See include/safety_supervisor.hpp for interface and P1-M5 sub-goals.

#include "safety_supervisor.hpp"

#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iostream>

float StoppingDistanceModel::compute(float velocity_mps, float friction_mu) const {

    if(velocity_mps <= 0.0f) {
        return 0.0f; // No stopping distance if not moving
    }
    //if friction <=0, return error.
    if(friction_mu <= 0.0f) {
        return std::numeric_limits<float>::infinity(); // Infinite stopping distance on zero friction
    }
    float braking_term = (velocity_mps * velocity_mps) / (2.0f * friction_mu * gravity);  
    float reaction_term = velocity_mps * reaction_time_s;

    float d_stop = braking_term + reaction_term;
    return d_stop;
}

TTCResult SafetySupervisor::compute_ttc(float vehicle_velocity, float d_worker,
                                        float worker_speed, float friction_mu) const {
    TTCResult r;
    // distance to collision within a safety wedge.

    //case 1: not moving or worker moving away -> safe
    r.v_relative = vehicle_velocity - worker_speed;

    r.d_stop = model_.compute(vehicle_velocity, friction_mu);
    r.d_worker = d_worker;
    if (r.v_relative <= 0.0f) {
        r.ttc_seconds = std::numeric_limits<float>::infinity(); // Diverging, safe
    } else {
        r.ttc_seconds = (d_worker - r.d_stop) / r.v_relative;
    }
    r.is_safe = (d_worker > r.d_stop) && (r.ttc_seconds > config_.ttc_proportional);
    return r;
}

SafetyIntervention SafetySupervisor::evaluate(float vehicle_velocity_mps,
                                              float d_to_nearest_worker,
                                              float worker_approach_speed,
                                              float terrain_traversability,
                                              double current_timestamp) {
    auto start = std::chrono::high_resolution_clock::now();
    SafetyIntervention intervention;
    float mu = traversability_to_friction(terrain_traversability); 
    TTCResult ttc = compute_ttc(vehicle_velocity_mps, d_to_nearest_worker,
                                worker_approach_speed, mu);
    //Decision
    if(!lidar_initialized_ || (current_timestamp - last_lidar_timestamp_) * 1000.0 > config_.lidar_timeout_ms) {
        //Emergency stop if LiDAR timeout
        intervention.level = SafetyIntervention::EMERGENCY_STOP;
        intervention.scale_factor = 0.0f;
        intervention.reason = "LiDAR timeout";

    }
    else if (ttc.ttc_seconds <= 0.0f ) {
        //Emergency stop
        intervention.level = SafetyIntervention::EMERGENCY_STOP;
        intervention.scale_factor = 0.0f;
        if(ttc.d_worker < ttc.d_stop) {
            intervention.reason = "Worker too close: d_worker < d_stop";
        } else {
            intervention.reason = "TTC <= 0";
        }
    }
    else if (ttc.ttc_seconds < config_.ttc_hard_brake) {
        //Hard brake
        intervention.level = SafetyIntervention::HARD_BRAKE;
        intervention.scale_factor = 0.1f; // Scale to 10% of current speed
        intervention.reason = "TTC < 2s";
    }
    else if (ttc.ttc_seconds < config_.ttc_proportional) {
        //Proportional scale
        intervention.level = SafetyIntervention::PROPORTIONAL_SCALE;
        intervention.scale_factor = (ttc.ttc_seconds - config_.ttc_hard_brake) / 
                                    (config_.ttc_proportional - config_.ttc_hard_brake);
        intervention.reason = "TTC < 5s";
    }
    else {
        //No intervention
        intervention.level = SafetyIntervention::NONE;
        intervention.scale_factor = 1.0f; // No change
        intervention.reason = "TTC >= 5s";
    }

    //log events
    events_.push_back(SafetyEvent{
        .timestamp = current_timestamp,
        .rule = intervention.reason,
        .d_worker = ttc.d_worker,
        .d_stop = ttc.d_stop,
        .ttc = ttc.ttc_seconds,
        .friction_mu = mu,
        .vel_before = vehicle_velocity_mps,
        .vel_after = vehicle_velocity_mps * intervention.scale_factor
    });
    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    log_loop_latency(latency_ms);

    return intervention;
}

float SafetySupervisor::traversability_to_friction(float trav_score) const {

    float mu = 0.0f;
    //clamp traversability score to [0,1]
    trav_score = std::max(0.0f, std::min(1.0f, trav_score));
    mu = config_.mu_base + config_.mu_trav_scale * trav_score;
    return mu;
}

void SafetySupervisor::update_lidar_timestamp(double timestamp) {
    
    
    last_lidar_timestamp_ = timestamp;
    lidar_initialized_ = true;
    
}

void SafetySupervisor::log_loop_latency(double latency_ms) {
    loop_latencies_ms_.push_back(latency_ms);
    // Optionally, compute and log p50 and p95 latencies here
}

void SafetySupervisor::loop_latency() {
    if(loop_latencies_ms_.empty()) {
        std::cout << "No loop latency data collected yet." << std::endl;
        return;
    }
    std::vector<double> sorted_latencies = loop_latencies_ms_;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    double p50 = sorted_latencies[sorted_latencies.size() / 2];
    double p95 = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.95)];
    std::cout << "Loop Latency - P50: " << p50 << " ms, P95: " << p95 << " ms" << std::endl;
}


