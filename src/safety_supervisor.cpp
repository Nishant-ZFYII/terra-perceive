// safety_supervisor.cpp
// TODO: Implement kinematic safety supervisor from scratch.
// See include/safety_supervisor.hpp for interface and P1-M5 sub-goals.

#include "safety_supervisor.hpp"

#include <cmath>
#include <limits>

float StoppingDistanceModel::compute(float velocity_mps, float friction_mu) const {
    // TODO (P1-M5.1): d_stop = v^2 / (2 * mu * g) + v * t_react
    //   Handle edge cases: velocity <= 0 -> d_stop = 0
    //   friction_mu clamped to [0.1, 1.0] to avoid division by zero
    //
    // Verification values:
    //   v=2, mu=0.6: d_stop = 4/(2*0.6*9.81) + 2*0.2 = 0.34 + 0.4 = 0.74m
    //   v=5, mu=0.6: d_stop = 25/(2*0.6*9.81) + 5*0.2 = 2.12 + 1.0 = 3.12m
    return 0.0f;
}

SafetySupervisor::SafetySupervisor(const SafetyConfig& config,
                                   const StoppingDistanceModel& model)
    : config_(config), model_(model) {}

TTCResult SafetySupervisor::compute_ttc(float vehicle_velocity, float d_worker,
                                        float worker_speed, float friction_mu) const {
    TTCResult r;
    // TODO (P1-M5.2): TTC computation
    //   v_relative = vehicle_velocity + worker_speed (closing speed)
    //   d_stop = model_.compute(vehicle_velocity, friction_mu)
    //   if v_relative <= 0: TTC = infinity (diverging, safe)
    //   else: TTC = (d_worker - d_stop) / v_relative
    //   if d_worker < d_stop: TTC is negative (already too late)
    //   is_safe = (d_worker > d_stop) && (TTC > config_.ttc_proportional)
    return r;
}

SafetyIntervention SafetySupervisor::evaluate(float vehicle_velocity_mps,
                                              float d_to_nearest_worker,
                                              float worker_approach_speed,
                                              float terrain_traversability) {
    SafetyIntervention intervention;
    // TODO (P1-M5.3 + P1-M5.4): Priority-ordered intervention
    //
    // Step 1: Compute friction from traversability (P1-M5.4)
    //   float mu = traversability_to_friction(terrain_traversability);
    //
    // Step 2: Compute TTC
    //   TTCResult ttc = compute_ttc(vehicle_velocity_mps, d_to_nearest_worker,
    //                               worker_approach_speed, mu);
    //
    // Step 3: Decision (P1-M5.3)
    //   TTC <= 0 or d_worker < d_stop -> EMERGENCY_STOP (scale=0.0)
    //   TTC < 2.0s -> HARD_BRAKE (scale=0.1)
    //   TTC < 5.0s -> PROPORTIONAL_SCALE (scale = (TTC-2)/(5-2))
    //   else -> NONE (scale=1.0)
    //
    // Step 4: Log event (P1-M5.5)
    //   Push SafetyEvent to events_ vector
    return intervention;
}

float SafetySupervisor::traversability_to_friction(float trav_score) const {
    // TODO (P1-M5.4): mu = mu_base + mu_trav_scale * trav_score
    //   trav_score 0.0 -> mu = 0.3 (worst terrain)
    //   trav_score 1.0 -> mu = 0.8 (good terrain)
    return config_.mu_base;
}
