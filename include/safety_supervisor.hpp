// safety_supervisor.hpp
// Kinematic safety supervisor — physics-based, not heuristic.
// Phase 1 upgrade: STOPPING DISTANCE + TTC, not hardcoded distance thresholds.
//
// PDF reference: Part 6 + Critique ("Safety supervisor uses if-else heuristics")
// Read: Nav2 Collision Monitor docs (architectural pattern)
// Read: Ames et al., "Control Barrier Functions" (ECC 2019) — for Phase 2 CBF upgrade
//
// YOUR TASK (P1-M5, Days 9-10):
//   P1-M5.1: Implement stopping distance: d_stop = v^2/(2*mu*g) + v*t_react
//   P1-M5.2: Implement TTC: (d_worker - d_stop) / v_relative
//   P1-M5.3: Priority-ordered interventions based on TTC
//   P1-M5.4: Terrain-aware friction from traversability grid
//   P1-M5.5: Event logging (CSV format)
//   P1-M5.6: Latency self-monitoring
//
// Key insight: a 30-ton bulldozer at 2 m/s needs ~8m to stop on mud, not 3m.
// Safety zones must be dynamic, computed from vehicle kinematics + terrain.

#pragma once

#include <string>
#include <vector>

struct StoppingDistanceModel {
    float gravity = 9.81f;
    float reaction_time_s = 0.2f;  // sensor-to-actuator latency

    // d_stop = v^2 / (2 * mu * g) + v * t_react
    float compute(float velocity_mps, float friction_mu) const;
};

struct TTCResult {
    float ttc_seconds;     // time to collision (infinity if diverging)
    float d_worker;        // current distance to worker
    float d_stop;          // required stopping distance
    float v_relative;      // closing speed
    bool is_safe;          // true if d_worker > d_stop and TTC > threshold
};

struct SafetyIntervention {
    enum Level { NONE, PROPORTIONAL_SCALE, HARD_BRAKE, EMERGENCY_STOP };
    Level level = NONE;
    float scale_factor = 1.0f;  // 1.0 = no change, 0.0 = full stop
    std::string reason;
};

struct SafetyEvent {
    double timestamp;
    std::string rule;
    float d_worker;
    float d_stop;
    float ttc;
    float friction_mu;
    float vel_before;
    float vel_after;
};

struct SafetyConfig {
    // TTC thresholds for intervention levels
    float ttc_emergency_stop = 0.0f;    // TTC <= 0 or d < d_stop
    float ttc_hard_brake = 2.0f;        // TTC < 2s -> scale to 10%
    float ttc_proportional = 5.0f;      // TTC < 5s -> proportional scale

    // Friction model: mu = mu_base + mu_trav_scale * traversability_score
    float mu_base = 0.3f;               // minimum friction (worst terrain)
    float mu_trav_scale = 0.5f;         // added friction per unit traversability

    // Sensor health
    float lidar_timeout_ms = 200.0f;    // trigger e-stop if no LiDAR for this long
    float loop_rate_hz = 50.0f;
    float loop_warn_ms = 20.0f;         // warn if loop exceeds this
};

class SafetySupervisor {
   public:
    SafetySupervisor(const SafetyConfig& config, const StoppingDistanceModel& model);

    // Core computation: given current state, compute safe velocity
    SafetyIntervention evaluate(float vehicle_velocity_mps, float d_to_nearest_worker,
                                float worker_approach_speed, float terrain_traversability);

    // Compute TTC for a single obstacle
    TTCResult compute_ttc(float vehicle_velocity, float d_worker,
                          float worker_speed, float friction_mu) const;

    // Map traversability score to friction coefficient
    float traversability_to_friction(float trav_score) const;

    // Event log
    const std::vector<SafetyEvent>& event_log() const { return events_; }

   private:
    SafetyConfig config_;
    StoppingDistanceModel model_;
    std::vector<SafetyEvent> events_;
};
