// test_so3.cpp — Tests for SO(3) Lie group primitives.
// Verifies hat/vee, Exp, Log, Jr against known identities and roundtrips.
//
// P2-M2.0 checkpoints:
//   M2.0.1: hat/vee roundtrip + cross-product identity
//   M2.0.2: Exp correctness on known rotations
//   M2.0.3: Log correctness incl. small-angle and θ→π cases
//   M2.0.4: Exp↔Log roundtrips (the critical correctness test)
//   M2.0.5: Jr basic properties + perturbation consistency (Forster Eq. 7)
//
// References:
//   Forster et al., "IMU Preintegration on Manifold" (RSS 2015), Section II
//   Solà et al., "A micro Lie theory for state estimation" (arXiv 1812.01537)

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "so3.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Absolute tolerances — tune per test if a specific one is too tight/loose
static constexpr double kTolStrict = 1e-12;   // pure algebra (hat/vee)
static constexpr double kTolNumeric = 1e-9;   // involves sin/cos/acos
static constexpr double kTolPerturbation = 1e-6;  // first-order approximations

// -----------------------------------------------------------------------------
// M2.0.1 — hat / vee
// -----------------------------------------------------------------------------

TEST(SO3, HatProducesSkewSymmetric) {
    Eigen::Vector3d omega(1, 2, 3);
    Eigen::Matrix3d Omega = so3::hat(omega);
    EXPECT_NEAR((Omega + Omega.transpose()).norm(), 0.0, kTolStrict);
}

TEST(SO3, VeeHatRoundtrip) {
    // so3::vee(so3::hat(ω)) == ω for arbitrary ω.
    // YOUR TEST:
    //   - try several ω vectors (zero, unit axes, arbitrary)
    //   - assert (so3::vee(so3::hat(omega)) - omega).norm() < kTolStrict

    std::vector<Eigen::Vector3d> omegas = {
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d(1, 0, 0),
        Eigen::Vector3d(0, 1, 0),
        Eigen::Vector3d(0, 0, 1),
        Eigen::Vector3d(1.5, -2.3, 0.7)
    };

    for (const auto& omega : omegas) {
        Eigen::Vector3d result = so3::vee(so3::hat(omega));
        EXPECT_NEAR((result - omega).norm(), 0.0, kTolStrict);
    }
}

TEST(SO3, HatIsCrossProduct) {
    // so3::hat(a) * b should equal a × b.
    // YOUR TEST:
    //   - pick a, b ∈ ℝ³
    //   - compare so3::hat(a) * b against a.cross(b)
    //   - tolerance: kTolStrict
    Eigen::Vector3d a(1, 2, 3);
    Eigen::Vector3d b(4, 5, 6);
    Eigen::Matrix3d Omega = so3::hat(a);
    Eigen::Vector3d result = Omega * b;
    EXPECT_NEAR((result - a.cross(b)).norm(), 0.0, kTolStrict);

}

// -----------------------------------------------------------------------------
// M2.0.2 — Exp
// -----------------------------------------------------------------------------

TEST(SO3, ExpOfZeroIsIdentity) {
    // Exp(0) == I.
    Eigen::Matrix3d result = so3::Exp(Eigen::Vector3d::Zero());
    EXPECT_NEAR((result - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolStrict);
}

TEST(SO3, Exp90DegreeZAxis) {
    // Exp([0, 0, π/2]) should be the 90° rotation around z-axis:
    //   [  0 -1  0 ]
    //   [  1  0  0 ]
    //   [  0  0  1 ]
    // YOUR TEST:
    //   - build phi = Vector3d(0, 0, M_PI / 2)
    //   - build the expected matrix above
    //   - assert (so3::Exp(phi) - expected).norm() < kTolNumeric
    Eigen::Vector3d phi(0, 0, M_PI / 2);
    Eigen::Matrix3d expected;
    expected << 0, -1, 0,
                1,  0, 0,
                0,  0, 1;
    EXPECT_NEAR((so3::Exp(phi) - expected).norm(), 0.0, kTolNumeric);
}

TEST(SO3, ExpOutputIsOrthogonal) {
    // For any phi, Exp(phi) must be a valid rotation: R * Rᵀ == I, det(R) == 1.
    // YOUR TEST:
    //   - pick a few phi values (small, medium, large)
    //   - assert (R * R.transpose() - I).norm() < kTolNumeric
    //   - assert std::abs(R.determinant() - 1.0) < kTolNumeric
    std::vector<Eigen::Vector3d> phis = {
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(0.1, -0.2, 0.3),
        Eigen::Vector3d(-1.5, 2.0, -0.5)
    };
    for (const auto& phi : phis) {
        Eigen::Matrix3d R = so3::Exp(phi);
        EXPECT_NEAR((R * R.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolNumeric);
        EXPECT_NEAR(std::abs(R.determinant() - 1.0), 0.0, kTolNumeric);
    }

}

// -----------------------------------------------------------------------------
// M2.0.3 — Log
// -----------------------------------------------------------------------------

TEST(SO3, LogOfIdentityIsZero) {
    // Log(I) == 0.
    // YOUR TEST:
    //   - assert so3::Log(Matrix3d::Identity()).norm() < kTolNumeric

    EXPECT_NEAR(so3::Log(Eigen::Matrix3d::Identity()).norm(), 0.0, kTolNumeric);
}

TEST(SO3, LogOfKnownRotation) {
    // Log of a 90° z-axis rotation should return [0, 0, π/2].
    // YOUR TEST:
    //   - build the 90° z-rotation matrix
    //   - expected phi = Vector3d(0, 0, M_PI / 2)
    //   - assert (so3::Log(R) - expected).norm() < kTolNumeric
    Eigen::Matrix3d R;
    R << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;
    Eigen::Vector3d expected(0, 0, M_PI / 2);
    EXPECT_NEAR((so3::Log(R) - expected).norm(), 0.0, kTolNumeric);
}

TEST(SO3, LogNearPiDoesNotBlowUp) {
    // A rotation of 179.9° should produce a finite, correct phi with ‖phi‖ ≈ π.
    // YOUR TEST:
    //   - phi_in = Vector3d(0, 0, M_PI - 1e-3)  (close to but not exactly π)
    //   - R = so3::Exp(phi_in)
    //   - phi_out = so3::Log(R)
    //   - assert phi_out is finite (no NaN/Inf)
    //   - assert std::abs(phi_out.norm() - phi_in.norm()) < kTolNumeric
    Eigen::Vector3d phi_in(0, 0, M_PI - 1e-3);
    Eigen::Matrix3d R = so3::Exp(phi_in);
    Eigen::Vector3d phi_out = so3::Log(R);
    EXPECT_TRUE(phi_out.allFinite());
    EXPECT_NEAR(std::abs(phi_out.norm() - phi_in.norm()), 0.0, kTolNumeric);
}

TEST(SO3, LogExactlyPi) {
    // A rotation of exactly 180° triggers the alternative branch (diagonal method).
    // YOUR TEST:
    //   - phi_in = Vector3d(0, 0, M_PI)  (or try other axes)
    //   - R = so3::Exp(phi_in)
    //   - phi_out = so3::Log(R)
    //   - assert (|phi_out| - π) < kTolNumeric
    //   - axis check: phi_out should be parallel to phi_in (possibly with sign flip —
    //     at θ=π, Log is two-to-one: Exp(πn) == Exp(-πn); accept either sign)
    Eigen::Vector3d phi_in(0, 0, M_PI);
    Eigen::Matrix3d R = so3::Exp(phi_in);
    Eigen::Vector3d phi_out = so3::Log(R);
    EXPECT_NEAR(std::abs(phi_out.norm() - M_PI), 0.0, kTolNumeric);
    // Check if phi_out is parallel to phi_in (allowing for sign flip)
    double dot_product = phi_out.dot(phi_in);
    EXPECT_NEAR(std::abs(dot_product) - phi_out.norm() * phi_in.norm(), 0.0, kTolNumeric);
}

// -----------------------------------------------------------------------------
// M2.0.4 — Exp/Log roundtrips (the correctness test)
// -----------------------------------------------------------------------------

TEST(SO3, ExpLogRoundtripSmallAngle) {
    // Log(Exp(phi)) ≈ phi for small phi.
    // YOUR TEST:
    //   - try phi magnitudes at 1e-12, 1e-9, 1e-6, 1e-3 (each multiplied by some axis)
    //   - verify roundtrip within kTolNumeric
    std::vector<Eigen::Vector3d> phis = {
        Eigen::Vector3d(1e-12, 0, 0),
        Eigen::Vector3d(0, -1e-9, 0),
        Eigen::Vector3d(0, 0, 1e-6),
        Eigen::Vector3d(1e-3, -1e-3, 1e-3)
    };
    for (const auto& phi : phis) {
        Eigen::Vector3d result = so3::Log(so3::Exp(phi));
        EXPECT_NEAR((result - phi).norm(), 0.0, kTolNumeric);
    }
}

TEST(SO3, ExpLogRoundtripRandom) {
    // Log(Exp(phi)) ≈ phi for random phi with ‖phi‖ up to (π - ε).
    // YOUR TEST:
    //   - use std::mt19937 with a FIXED seed (determinism — matches Phase 1 style)
    //   - generate N random phi, ‖phi‖ uniform in [0, π - 0.1]
    //   - verify roundtrip within kTolNumeric
    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> angle_dist(0, M_PI - 0.1);
    std::uniform_real_distribution<double> axis_dist(-1, 1);    
    int N = 100;
    for (int i = 0; i < N; ++i) {
        double angle = angle_dist(rng);
        Eigen::Vector3d axis(axis_dist(rng), axis_dist(rng), axis_dist(rng));
        axis.normalize();
        Eigen::Vector3d phi = angle * axis;
        Eigen::Vector3d result = so3::Log(so3::Exp(phi));
        EXPECT_NEAR((result - phi).norm(), 0.0, kTolNumeric);
    }
}

TEST(SO3, LogExpRoundtripRandom) {
    // Exp(Log(R)) ≈ R for random rotations R.
    // YOUR TEST:
    //   - generate random R by: phi = random in ℝ³, R = Exp(phi)
    //   - verify (Exp(Log(R)) - R).norm() < kTolNumeric
    std::mt19937 rng(43);  // different fixed seed
    std::uniform_real_distribution<double> angle_dist(0, M_PI - 0.1);
    std::uniform_real_distribution<double> axis_dist(-1, 1);
    int N = 100;
    for (int i = 0; i < N; ++i) {
        double angle = angle_dist(rng);
        Eigen::Vector3d axis(axis_dist(rng), axis_dist(rng), axis_dist(rng));
        axis.normalize();
        Eigen::Vector3d phi = angle * axis;
        Eigen::Matrix3d R = so3::Exp(phi);
        Eigen::Matrix3d result = so3::Exp(so3::Log(R));
        EXPECT_NEAR((result - R).norm(), 0.0, kTolNumeric);
    }


}

// -----------------------------------------------------------------------------
// M2.0.5 — Jr
// -----------------------------------------------------------------------------

TEST(SO3, JrOfZeroIsIdentity) {
    // Jr(0) == I.
    // YOUR TEST:
    //   - assert (so3::Jr(Vector3d::Zero()) - Matrix3d::Identity()).norm() < kTolStrict
    EXPECT_NEAR((so3::Jr(Eigen::Vector3d::Zero()) - Eigen::Matrix3d::Identity()).norm(), 0.0, kTolStrict);

}

TEST(SO3, JrPerturbationConsistency) {
    // Forster Eq. 7: Exp(phi + delta_phi) ≈ Exp(phi) * Exp(Jr(phi) * delta_phi)
    // for small delta_phi. This is the defining property of the right Jacobian.
    //
    // YOUR TEST:
    //   - phi = Vector3d(0.3, -0.2, 0.5)  (medium magnitude rotation)
    //   - delta_phi = Vector3d::Random() * 1e-5  (must be small for first-order to hold)
    //   - LHS: so3::Exp(phi + delta_phi)
    //   - RHS: so3::Exp(phi) * so3::Exp(so3::Jr(phi) * delta_phi)
    //   - assert (LHS - RHS).norm() < kTolPerturbation
    //   - Note: approximation is first-order, so tolerance scales with ‖delta_phi‖².
    //     If this test is too tight, shrink delta_phi further (1e-6 or 1e-7).
    std::mt19937 rng(44);  
    std::uniform_real_distribution<double> dist(-1, 1);
    Eigen::Vector3d delta_phi(dist(rng), dist(rng), dist(rng));
    delta_phi *= 1e-5;
    Eigen::Vector3d phi(0.3, -0.2, 0.5);
    Eigen::Matrix3d LHS = so3::Exp(phi + delta_phi);
    Eigen::Matrix3d RHS = so3::Exp(phi) * so3::Exp(so3::Jr(phi) * delta_phi);
    EXPECT_NEAR((LHS - RHS).norm(), 0.0, kTolPerturbation);
}
