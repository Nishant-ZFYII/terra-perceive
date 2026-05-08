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

// P2-M6: shared sensor-health gate. Returns true and fills `out` if a
// LiDAR-timeout e-stop is forced; both kinematic and CBF paths short-circuit
// on a true return.
bool SafetySupervisor::pre_evaluate_health(double current_timestamp,
                                           SafetyIntervention& out) {
    if (!lidar_initialized_
        || (current_timestamp - last_lidar_timestamp_) * 1000.0
               > config_.lidar_timeout_ms) {
        out.level = SafetyIntervention::EMERGENCY_STOP;
        out.scale_factor = 0.0f;
        out.reason = "LiDAR timeout";
        return true;
    }
    return false;
}

SafetyIntervention SafetySupervisor::evaluate(float vehicle_velocity_mps,
                                              float d_to_nearest_worker,
                                              float worker_approach_speed,
                                              float terrain_traversability,
                                              double current_timestamp) {
    // P2-M6: branch on safety_mode. Per Fork 2 of the M6 plan, CBF gets its
    // own evaluator; the kinematic path stays bit-for-bit unchanged.
    if (config_.safety_mode == "cbf") {
        return evaluate_cbf(vehicle_velocity_mps, d_to_nearest_worker,
                            worker_approach_speed, terrain_traversability,
                            current_timestamp);
    }

    auto start = std::chrono::high_resolution_clock::now();
    SafetyIntervention intervention;
    float mu = traversability_to_friction(terrain_traversability);
    TTCResult ttc = compute_ttc(vehicle_velocity_mps, d_to_nearest_worker,
                                worker_approach_speed, mu);
    //Decision
    if (pre_evaluate_health(current_timestamp, intervention)) {
        // health gate already populated `intervention` with the e-stop.
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

void SafetySupervisor::report_latency_stats() {
      loop_latency();
}

// P2-M6: 1D scalar Control Barrier Function clamp on commanded acceleration.
// Derivation in notes/m6_math_sketch.md §2.3:
//   h(v, d_w) = d_w - [v^2/(2 mu g) + v t_react + cbf_d_safe_min]
//   A         = v/(mu g) + t_react
//   a_safe    = (gamma * h - v_relative) / A
// Convert to the existing scale_factor API:
//   v_safe = max(0, v + a_safe * cbf_dt); scale = clamp(v_safe / v, 0, 1).
SafetyIntervention SafetySupervisor::evaluate_cbf(float vehicle_velocity_mps,
                                                  float d_to_nearest_worker,
                                                  float worker_approach_speed,
                                                  float terrain_traversability,
                                                  double current_timestamp) {
    auto start = std::chrono::high_resolution_clock::now();
    SafetyIntervention intervention;
    if (pre_evaluate_health(current_timestamp, intervention)) {
        events_.push_back(SafetyEvent{
            .timestamp = current_timestamp,
            .rule = intervention.reason,
            .d_worker = d_to_nearest_worker,
            .d_stop = std::numeric_limits<float>::quiet_NaN(),
            .ttc = std::numeric_limits<float>::quiet_NaN(),
            .friction_mu = traversability_to_friction(terrain_traversability),
            .vel_before = vehicle_velocity_mps,
            .vel_after = 0.0f
        });
        return intervention;
    }

    const float mu = traversability_to_friction(terrain_traversability);
    const float v  = std::max(0.0f, vehicle_velocity_mps);
    const float v_relative = v - worker_approach_speed;  // matches kinematic convention

    const float d_stop = model_.compute(v, mu);
    const float h      = d_to_nearest_worker - (d_stop + config_.cbf_d_safe_min);
    const float A      = (mu > 0.0f)
                         ? v / (mu * model_.gravity) + model_.reaction_time_s
                         : std::numeric_limits<float>::infinity();

    float a_safe;
    if (!std::isfinite(A) || A < 1e-6f) {
        // Degenerate: no friction or no leverage. Treat as full stop.
        a_safe = -std::numeric_limits<float>::infinity();
    } else {
        a_safe = (config_.cbf_gamma * h - v_relative) / A;
    }

    const float v_safe = std::max(0.0f, v + a_safe * config_.cbf_dt);
    float scale;
    if (v > 1e-6f) {
        scale = std::clamp(v_safe / v, 0.0f, 1.0f);
    } else {
        // At rest: positive a_safe means "ok to start moving"; we still report
        // scale=1.0 (the upstream commanded velocity stands), since the API
        // is multiplicative on commanded velocity.
        scale = 1.0f;
    }
    intervention.scale_factor = scale;

    if (scale <= 1e-3f) {
        intervention.level = SafetyIntervention::EMERGENCY_STOP;
        intervention.reason = "CBF: v_safe ~ 0";
    } else if (scale < 0.5f) {
        intervention.level = SafetyIntervention::HARD_BRAKE;
        intervention.reason = "CBF: hard clamp";
    } else if (scale < 1.0f) {
        intervention.level = SafetyIntervention::PROPORTIONAL_SCALE;
        intervention.reason = "CBF: smooth clamp";
    } else {
        intervention.level = SafetyIntervention::NONE;
        intervention.reason = "CBF: h > 0 no clamp";
    }

    events_.push_back(SafetyEvent{
        .timestamp = current_timestamp,
        .rule = intervention.reason,
        .d_worker = d_to_nearest_worker,
        .d_stop = d_stop,
        .ttc = h,                // overload ttc column with h(x) for CBF events
        .friction_mu = mu,
        .vel_before = vehicle_velocity_mps,
        .vel_after = vehicle_velocity_mps * scale
    });

    auto end = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    log_loop_latency(latency_ms);
    return intervention;
}


