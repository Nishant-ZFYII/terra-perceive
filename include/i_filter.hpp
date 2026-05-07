// i_filter.hpp
// Polymorphic base for the Track filter slot in SORTTracker.
//
// Why this exists:
//   Phase-3 M12 introduces an IMM (Interacting Multiple Model) Kalman filter
//   alongside the M4 constant-velocity KF. The cleanest way to swap between
//   them at runtime — without templating Track on a filter type, or smearing
//   FilterKind enums through the matching code — is a virtual base.
//
// Cost model:
//   One v-table dispatch per predict()/update() call per track per frame. With
//   ~50 tracks at 10 Hz that's 500 indirect calls/sec. DBSCAN on the same
//   frame is ~10^5 ops. The v-table overhead is invisible.
//
// Lifetime:
//   Track owns its IFilter via std::unique_ptr<IFilter>. SORTTracker prunes
//   and publishes Tracks, so Track is copyable — implemented via IFilter::clone()
//   so we get value semantics over a polymorphic type.
//
// Coordinate frame:
//   2D BEV (x, y in meters). Z is a passthrough on Track, not part of the
//   filter state. A 3D filter would be a separate class deriving from IFilter.

#pragma once
#include <Eigen/Dense>
#include <memory>

namespace tracker {

class IFilter {
   public:
    virtual ~IFilter() = default;

    // Seed the filter at an initial position. Velocity prior is implementation-
    // defined (typical: zero with a large covariance).
    virtual void init(float x, float y) = 0;

    // Propagate the state forward one timestep. Time-invariant dt is baked
    // into the filter at construction.
    virtual void predict() = 0;

    // Fuse a position measurement (z_x, z_y) into the posterior.
    virtual void update(float z_x, float z_y) = 0;

    // Posterior accessors. State layout is (x, y, vx, vy) — the 2D BEV
    // constant-velocity convention shared by every concrete filter we plan to
    // ship in Phase 3.
    virtual Eigen::Vector4f state() const = 0;
    virtual Eigen::Vector2f position() const = 0;
    virtual Eigen::Vector2f velocity() const = 0;

    // Full 4x4 covariance. Needed by IMM mixing (sub-filter init from a
    // weighted blend of the previous-step posteriors) and useful for ablations
    // that gate by trace(P).
    virtual Eigen::Matrix4f covariance() const = 0;
    virtual float covariance_trace() const = 0;

    // 2x2 position covariance for *gating* (Mahalanobis cascade match).
    //
    // Distinct from `covariance().topLeftCorner<2,2>()` because IMM's
    // combined covariance includes the inter-mode spread term
    // `(x_j - x_combined)(x_j - x_combined)^T` — correct for representing
    // marginal posterior uncertainty under model disagreement, but wrong
    // for gating, which asks "is this detection physically the same
    // object?" rather than "where could it be under model uncertainty?"
    //
    // The cascade gate uses the MOST CONFIDENT sub-model's P_position —
    // the tighter prediction. CV-only filters return the same as
    // `covariance().topLeftCorner<2,2>()`; IMMFilter returns the
    // sub-model with the smallest position cov-trace.
    //
    // Found via the post-Mahalanobis audit (2026-04-27): drive-wide
    // revivals at 22–24 m world drift were leaking through because
    // the combined IMM covariance ballooned during pre-Lost misses
    // when CV and CP modes diverged. See docs/m10-debug-log.md
    // "Mahalanobis post-mortem — IMM combined cov inflates the gate".
    virtual Eigen::Matrix2f gating_position_covariance_2x2() const = 0;

    // Deep copy. Lets owners of `unique_ptr<IFilter>` be copyable — Track
    // holds a polymorphic filter and gets copied through SORTTracker's
    // publishing path (update() returns a snapshot vector). Virtual
    // destructor + clone() is the standard "value semantics over a
    // polymorphic type" idiom.
    virtual std::unique_ptr<IFilter> clone() const = 0;
};

}  // namespace tracker
