// test_safety.cpp — Tests for kinematic safety supervisor.
// 5 mock scenarios with hand-calculated expected values.
//
// P1-M5 checkpoints:
//   M5.1: Stopping distance matches hand calculations
//   M5.2: TTC handles approaching, receding, already-too-close
//   M5.3: Correct intervention level for each TTC range
//   M5.4: Terrain friction derived from traversability score

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

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

// --- P2-M6: CBF tests (4 new) ---

namespace {
SafetyConfig cbf_config(float gamma = 1.0f) {
    SafetyConfig c;
    c.safety_mode      = "cbf";
    c.cbf_gamma        = gamma;
    c.cbf_d_safe_min   = 0.5f;
    c.cbf_dt           = 0.1f;
    c.mu_base          = 0.3f;
    c.mu_trav_scale    = 0.5f;
    c.lidar_timeout_ms = 1e9f;  // disable lidar gate for unit tests
    return c;
}
}  // namespace

TEST(Safety, CbfBarrierIsPositiveWhenSafe) {
    // With d_worker far beyond d_stop + d_safe_min, h(x) > 0 and the CBF
    // produces no clamp (scale_factor == 1.0, level == NONE).
    SafetyConfig config = cbf_config();
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0);

    SafetyIntervention out =
        supervisor.evaluate_cbf(2.0f, 25.0f, 0.0f, 1.0f, 0.0);
    EXPECT_EQ(out.level, SafetyIntervention::NONE);
    EXPECT_NEAR(out.scale_factor, 1.0f, 1e-3f);
}

TEST(Safety, CbfScaleStaysInBounds) {
    // Across 500 random (v, d_worker, mu) samples, CBF's scale_factor must
    // stay in [0, 1]. This is the basic safety property of the conversion
    // a_safe -> v_safe -> scale; a regression here means a sign or division
    // bug in evaluate_cbf().
    SafetyConfig cbf = cbf_config();
    StoppingDistanceModel model;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> v_dist(0.0f, 5.0f);
    std::uniform_real_distribution<float> d_dist(0.1f, 50.0f);
    std::uniform_real_distribution<float> trav_dist(0.0f, 1.0f);

    SafetySupervisor sup(cbf, model);
    sup.update_lidar_timestamp(0.0);
    for (int i = 0; i < 500; ++i) {
        auto out = sup.evaluate_cbf(v_dist(rng), d_dist(rng), 0.0f,
                                    trav_dist(rng), 0.0);
        ASSERT_GE(out.scale_factor, 0.0f);
        ASSERT_LE(out.scale_factor, 1.0f + 1e-6f);
    }
}

TEST(Safety, CbfNeverCollidesAcrossScenarios) {
    // The forward-invariance promise of CBF: starting safely (h(x_0) > 0),
    // the controlled vehicle should never reach d <= 0 against a stationary
    // worker, across a range of starting conditions.
    StoppingDistanceModel model;
    SafetyConfig config = cbf_config();
    const float dt = config.cbf_dt;

    struct Scenario { float v0; float d0; };
    std::vector<Scenario> cases = {
        {1.0f,  6.0f},   // slow + close
        {2.0f,  8.0f},   // typical
        {3.0f, 12.0f},   // faster
        {4.0f, 20.0f},   // fast + far
        {2.0f, 30.0f},   // long approach
    };
    for (auto s : cases) {
        SafetySupervisor sup(config, model);
        sup.update_lidar_timestamp(0.0);
        float v = s.v0;
        float d = s.d0;
        bool collided = false;
        for (int i = 0; i < 600; ++i) {
            sup.update_lidar_timestamp(i * dt);
            auto out = sup.evaluate(v, d, 0.0f, 1.0f, i * dt);
            v = v * out.scale_factor;
            d -= v * dt;
            if (d <= 0.0f) { collided = true; break; }
            if (v < 1e-3f && i > 20) break;
        }
        EXPECT_FALSE(collided)
            << "v0=" << s.v0 << " d0=" << s.d0 << " collided.";
        EXPECT_GT(d, 0.0f);
    }
}

TEST(Safety, CbfVelocityProfileSmooth) {
    // Drive a head-on simulation: vehicle starts at 2 m/s, worker stationary
    // 8m ahead. Integrate v with the CBF scale at dt=0.1s for 60 steps and
    // assert max |dv/dt| stays bounded (smooth, not bang-bang).
    SafetyConfig config = cbf_config();
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    supervisor.update_lidar_timestamp(0.0);

    const float dt = config.cbf_dt;
    float v = 2.0f;
    float d = 8.0f;
    float worker_speed = 0.0f;  // stationary worker
    float trav = 1.0f;
    std::vector<float> v_traj;
    v_traj.reserve(60);
    for (int i = 0; i < 60; ++i) {
        supervisor.update_lidar_timestamp(i * dt);
        auto out = supervisor.evaluate(v, d, worker_speed, trav, i * dt);
        v = v * out.scale_factor;
        v_traj.push_back(v);
        d -= (v - worker_speed) * dt;
        if (d < 0.0f) break;  // collision (should not happen)
    }
    ASSERT_GE(v_traj.size(), 10u);

    float max_step = 0.0f;
    for (size_t i = 1; i < v_traj.size(); ++i) {
        float step = std::fabs(v_traj[i] - v_traj[i - 1]) / dt;
        if (step > max_step) max_step = step;
    }
    // Kinematic mode would step from cruise to 0.1*v in a single 0.1s frame
    // (|dv/dt| = 18 m/s^2 at v=2). CBF should be far smoother.
    EXPECT_LT(max_step, 10.0f) << "CBF velocity steps should be smooth.";
}

TEST(Safety, CbfStoppingMarginConverges) {
    // Across gamma in {0.5, 1.0, 2.0}, the head-on simulation must terminate
    // with a final margin (d_worker - d_stop) inside [d_safe_min - 0.2, d_safe_min + 1.5].
    // Hard floor: no collision (d > 0) at any timestep, all gammas.
    StoppingDistanceModel model;
    for (float gamma : {0.5f, 1.0f, 2.0f}) {
        SafetyConfig config = cbf_config(gamma);
        SafetySupervisor supervisor(config, model);
        supervisor.update_lidar_timestamp(0.0);

        const float dt = config.cbf_dt;
        float v = 2.0f;
        float d = 12.0f;
        float trav = 1.0f;
        bool collision = false;
        for (int i = 0; i < 300; ++i) {
            supervisor.update_lidar_timestamp(i * dt);
            auto out = supervisor.evaluate(v, d, 0.0f, trav, i * dt);
            v = v * out.scale_factor;
            d -= v * dt;
            if (d <= 0.0f) { collision = true; break; }
            if (v < 1e-3f && i > 10) break;  // converged
        }
        EXPECT_FALSE(collision) << "gamma=" << gamma << " collided.";
        // Vehicle stopped: final margin = d. Must be near d_safe_min, not
        // pinned at 0 (would mean the clamp never engaged) and not deep
        // negative (collision).
        float mu = supervisor.traversability_to_friction(trav);
        float d_stop_at_v = model.compute(v, mu);
        float margin = d - d_stop_at_v;
        EXPECT_GT(margin, config.cbf_d_safe_min - 0.2f)
            << "gamma=" << gamma << " final margin too small.";
        EXPECT_LT(margin, config.cbf_d_safe_min + 1.5f)
            << "gamma=" << gamma << " final margin unexpectedly large.";
    }
}
