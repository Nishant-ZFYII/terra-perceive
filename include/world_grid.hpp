#pragma once

// =============================================================================
// WorldGrid — persistent world-frame accumulated traversability map (P2-M3)
// =============================================================================
//
// Takes a stream of single-frame TraversabilityGrid grids (vehicle-local) and
// their corresponding SE(3) poses, and folds them into a persistent
// world-frame grid with three configurable update rules.
//
// Ablation knobs supported:
//   A. Pose source         — driven by caller (which poses.csv is loaded)
//   B. Update rule         — cfg.update_rule: EMA / LogOdds / Overwrite
//   C. Alpha sweep         — cfg.alpha (EMA only)
//   D. Decay ablation      — cfg.decay_rate (confidence only; risk never decays)
//   E. SLAM cov → conf     — caller passes pose_sigma; cfg.pose_uncertainty_k
//
// See p2m3_plan.md for the full ablation matrix.
// =============================================================================

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

#include "traversability.hpp"   // TraversabilityGrid, CellFeatures (global namespace)
#include "pose_graph_slam.hpp"  // pose_graph::Pose (R, t)

// Bring Pose into this translation unit's scope. The pose_graph namespace
// wraps the SLAM types; WorldGrid is not a SLAM type and lives at global scope,
// so we hoist the one symbol we need rather than qualifying every use site.
using Pose = pose_graph::Pose;

// -----------------------------------------------------------------------------
// Update-rule strategy
// -----------------------------------------------------------------------------
enum class UpdateRule {
    EMA,        // exponential moving average on risk (1-pole IIR)
    LogOdds,    // OctoMap-style additive log-odds with clamp
    Overwrite   // no memory; last observation wins
};

// -----------------------------------------------------------------------------
// Configuration struct — everything ablatable lives here
// -----------------------------------------------------------------------------
struct WorldGridConfig {
    // Spatial extent (world frame, meters). Defaults sized for RELLIS-3D 00.
    double x_min      = -100.0;
    double x_max      =  100.0;
    double y_min      = -100.0;
    double y_max      =  100.0;
    double resolution =    0.5;

    // Update-rule selector.
    UpdateRule update_rule = UpdateRule::EMA;

    // EMA parameters.
    double alpha = 0.3;

    // Log-odds parameters (OctoMap defaults from §3.2).
    // NOTE: hit/miss are asymmetric on purpose — see blog-style notes.
    double logodds_hit        =  0.85;
    double logodds_miss       = -0.40;
    double logodds_clamp_min  = -3.50;
    double logodds_clamp_max  =  3.50;

    // Confidence growth per observation, saturating at confidence_max.
    double confidence_growth = 0.10;
    double confidence_max    = 1.00;

    // Temporal decay: confidence -= decay_rate * dt per unobserved second.
    // Set to 0.0 to disable decay (used in ablation D).
    double decay_rate = 0.0;

    // Pose-uncertainty → confidence scaling:
    //   conf_adj = conf_raw * exp(-k * pose_sigma)
    // k=1.0 means a 1m pose sigma drops confidence by 1/e.
    double pose_uncertainty_k = 1.0;
};

// -----------------------------------------------------------------------------
// WorldCell — one cell of the persistent grid
// -----------------------------------------------------------------------------
struct WorldCell {
    float    risk                   = 0.0f;   // [0,1] — EMA or sigmoid(logodds)
    float    logodds                = 0.0f;   // LogOdds rule only
    float    confidence             = 0.0f;   // [0,1] — growth/decay accumulator
    uint32_t obs_count              = 0;
    double   last_update_time       = -1.0;   // seconds; -1 ⇒ never observed
    float    mean_z                 = 0.0f;
    float    pose_sigma_at_last_obs = 0.0f;   // sqrt(trace(P_position)) at obs time
};

// -----------------------------------------------------------------------------
// WorldGrid — the public API
// -----------------------------------------------------------------------------
class WorldGrid {
public:
    explicit WorldGrid(const WorldGridConfig& cfg);

    // Fold one local-frame TraversabilityGrid grid into the world grid.
    //   local_grid: output of TraversabilityGrid::compute() in vehicle-local frame
    //   pose:       T_world_body (body → world)
    //   pose_sigma: sqrt(trace(P_position)); pass 0.0 if unknown
    //   timestamp:  seconds (epoch or monotonic — stay consistent)
    void update(const TraversabilityGrid& local_grid,
                const Pose&               pose,
                double                    pose_sigma,
                double                    timestamp);

    // Apply temporal confidence decay. Call before update() if you want
    // between-frame decay; call from a separate timer in online mode.
    void applyDecay(double current_time);

    // Query the cell containing world (x,y). Returns nullptr if OOB.
    const WorldCell* getCell(double x, double y) const;
    WorldCell*       getCell(double x, double y);

    // Coverage over the trajectory bounding box.
    //   traj: sequence of (x,y) vehicle positions up to now.
    // Returns observed_cells / bbox_cells (axis-aligned BBox v1; convex hull v2).
    double coveragePercent(const std::vector<Eigen::Vector2d>& traj) const;

    // --- Accessors for viz / serialization ---
    int    rows()      const { return rows_; }
    int    cols()      const { return cols_; }
    double resolution() const { return cfg_.resolution; }
    double originX()   const { return cfg_.x_min; }
    double originY()   const { return cfg_.y_min; }
    const  WorldCell& at(int r, int c) const { return cells_[static_cast<size_t>(r) * cols_ + c]; }
    const  WorldGridConfig& config() const { return cfg_; }

    // Snapshot I/O.
    // saveSnapshot writes two files: <stem>.png (risk heatmap) and <stem>.csv (full).
    bool saveSnapshot(const std::string& stem) const;
    bool loadSnapshot(const std::string& stem);

private:
    // world (x,y) → grid (row,col). Returns false if OOB.
    bool worldToGrid(double x, double y, int& row, int& col) const;

    // Per-rule cell updates — dispatched from update().
    void updateCellEMA      (WorldCell& cell, float new_risk, float mean_z);
    void updateCellLogOdds  (WorldCell& cell, float new_risk, float mean_z);
    void updateCellOverwrite(WorldCell& cell, float new_risk, float mean_z);

    // Confidence bookkeeping — common across update rules.
    void growConfidence(WorldCell& cell, float raw_conf, float pose_sigma);

    // logit / sigmoid helpers.
    static float probToLogOdds(float p);
    static float logOddsToProb(float l);

    // Clamp raw confidence using pose uncertainty.
    float adjustConfidence(float raw_conf, float pose_sigma) const;

    WorldGridConfig         cfg_;
    int                     rows_;
    int                     cols_;
    std::vector<WorldCell>  cells_;   // row-major: cells_[r * cols_ + c]
};
