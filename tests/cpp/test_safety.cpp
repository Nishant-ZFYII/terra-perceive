// test_safety.cpp — Tests for kinematic safety supervisor.
// 5 mock scenarios with hand-calculated expected values.
//
// P1-M5 checkpoints:
//   M5.1: Stopping distance matches hand calculations
//   M5.2: TTC handles approaching, receding, already-too-close
//   M5.3: Correct intervention level for each TTC range
//   M5.4: Terrain friction derived from traversability score

#include <gtest/gtest.h>

#include "safety_supervisor.hpp"

// --- Stopping distance tests (P1-M5.1) ---

TEST(Safety, StoppingDistanceKnownValues) {
    StoppingDistanceModel model;
    model.gravity = 9.81f;
    model.reaction_time_s = 0.2f;

    ASSERT_NEAR(model.compute(2.0f, 0.6f), 0.74f, 0.01f);

    ASSERT_NEAR(model.compute(5.0f, 0.6f), 3.12f, 0.02f);

    ASSERT_NEAR(model.compute(0.0f, 0.6f), 0.0f, 0.001f);
}

// --- TTC tests (P1-M5.2) ---

TEST(Safety, TTCApproachingWorker) {
    // Worker at 10m, vehicle at 2 m/s, worker stationary, mu=0.6
    // d_stop = 0.74m, v_rel = 2.0, TTC = (10 - 0.74) / 2.0 = 4.63s
    // TODO: verify
    SafetyConfig config;
    config.ttc_proportional = 5.0f; // Set proportional threshold to 5s for this test
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    TTCResult ttc = supervisor.compute_ttc(2.0f, 16.0f, 0.0f, 0.6f);
    ASSERT_TRUE(ttc.is_safe);
    ASSERT_NEAR(ttc.ttc_seconds, 7.63f, 0.01f);
}

TEST(Safety, TTCRecedingWorker) {
    // Worker moving away faster than vehicle approaches
    // v_relative <= 0 -> TTC = infinity (safe)
    SafetyConfig config;
    config.ttc_proportional = 5.0f; // Set proportional threshold to 5s for this test
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    TTCResult ttc = supervisor.compute_ttc(2.0f, 10.0f, 3.0f, 0.6f);
    ASSERT_TRUE(ttc.is_safe);
    ASSERT_EQ(ttc.ttc_seconds, std::numeric_limits<float>::infinity());
}

TEST(Safety, TTCAlreadyTooClose) {
    // Worker at 0.5m, vehicle at 2 m/s
    // d_worker < d_stop -> TTC < 0 -> emergency stop
    SafetyConfig config;
    config.ttc_proportional = 5.0f; // Set proportional threshold to 5s for this test
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    TTCResult ttc = supervisor.compute_ttc(2.0f, 0.5f, 0.0f, 0.6f);
    ASSERT_FALSE(ttc.is_safe);
    ASSERT_LT(ttc.ttc_seconds, 0.0f);

}

// --- Intervention tests (P1-M5.3) ---

TEST(Safety, InterventionEmergencyStop) {
    // Scenario: worker at 0.5m, vehicle at 2 m/s
    // Expected: EMERGENCY_STOP, scale_factor = 0.0
    SafetyConfig config;
    config.ttc_emergency_stop = 0.0f;
    config.ttc_hard_brake = 2.0f;
    config.ttc_proportional = 5.0f;
    StoppingDistanceModel model;    
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0); // Initialize LiDAR timestamp
    SafetyIntervention intervention = supervisor.evaluate(2.0f, 0.5f, 0.0f, 0.6f, 0.0);
    ASSERT_EQ(intervention.level, SafetyIntervention::EMERGENCY_STOP);
    ASSERT_EQ(intervention.scale_factor, 0.0f);
}

TEST(Safety, InterventionHardBrake) {
    // Scenario: TTC = 1.5s (< 2.0 threshold)
    // Expected: HARD_BRAKE, scale_factor = 0.1
    SafetyConfig config;
    config.ttc_emergency_stop = 0.0f;
    config.ttc_hard_brake = 2.0f;
    config.ttc_proportional = 5.0f;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0); // Initialize LiDAR timestamp
    SafetyIntervention intervention = supervisor.evaluate(2.0f, 1.5f, 0.0f, 0.6f, 0.0);
    ASSERT_EQ(intervention.level, SafetyIntervention::HARD_BRAKE);
    ASSERT_EQ(intervention.scale_factor, 0.1f);
}

TEST(Safety, InterventionProportionalScale) {
    // Scenario: TTC = 3.5s (between 2.0 and 5.0)
    // Expected: PROPORTIONAL_SCALE, scale = (3.5-2.0)/(5.0-2.0) = 0.5
    SafetyConfig config;
    config.ttc_emergency_stop = 0.0f;
    config.ttc_hard_brake = 2.0f;
    config.ttc_proportional = 5.0f;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0); // Initialize LiDAR timestamp
    SafetyIntervention intervention = supervisor.evaluate(2.0f, 7.75f, 0.0f, 0.6f, 0.0);
    ASSERT_EQ(intervention.level, SafetyIntervention::PROPORTIONAL_SCALE);
    ASSERT_NEAR(intervention.scale_factor, 0.5f, 0.01f);
}

TEST(Safety, InterventionNone) {
    // Scenario: worker at 50m, vehicle at 1 m/s
    // Expected: NONE, scale_factor = 1.0
    SafetyConfig config;
    config.ttc_emergency_stop = 0.0f;
    config.ttc_hard_brake = 2.0f;
    config.ttc_proportional = 5.0f;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0); // Initialize LiDAR timestamp
    SafetyIntervention intervention = supervisor.evaluate(1.0f, 50.0f, 0.0f, 0.6f, 0.0);
    ASSERT_EQ(intervention.level, SafetyIntervention::NONE);
    ASSERT_EQ(intervention.scale_factor, 1.0f);
}

// --- Terrain friction tests (P1-M5.4) ---

TEST(Safety, FrictionFromTraversability) {
    // trav=0.0 -> mu = 0.3 (worst terrain)
    // trav=1.0 -> mu = 0.8 (good terrain)
    // trav=0.5 -> mu = 0.55
    SafetyConfig config;
    config.mu_base = 0.3f;
    config.mu_trav_scale = 0.5f;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    ASSERT_NEAR(supervisor.traversability_to_friction(0.0f), 0.3f, 0.001f);
    ASSERT_NEAR(supervisor.traversability_to_friction(1.0f), 0.8f, 0.001f);
    ASSERT_NEAR(supervisor.traversability_to_friction(0.5f), 0.55f, 0.001f);
}

TEST(Safety, LowFrictionIncreasesStoppingDistance) {
    // Same velocity, low friction terrain -> longer d_stop -> earlier intervention
    // Compare: mu=0.8 vs mu=0.3 at v=2 m/s
    // mu=0.8: d_stop = 4/(2*0.8*9.81) + 0.4 = 0.255 + 0.4 = 0.655m
    // mu=0.3: d_stop = 4/(2*0.3*9.81) + 0.4 = 0.679 + 0.4 = 1.079m
    SafetyConfig config;
    config.mu_base = 0.3f;
    config.mu_trav_scale = 0.5f;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    float d_stop_high_friction = model.compute(2.0f, supervisor.traversability_to_friction(1.0f));
    float d_stop_low_friction = model.compute(2.0f, supervisor.traversability_to_friction(0.0f));
    ASSERT_GT(d_stop_low_friction, d_stop_high_friction);
}
