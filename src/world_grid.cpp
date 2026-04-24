// =============================================================================
// WorldGrid implementation — boilerplate with // YOUR CODE: blocks.
// See include/world_grid.hpp for API and ablation mapping.
// =============================================================================

#include "world_grid.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <tinycolormap.hpp>  // for colorizing risk heatmap in saveSnapshot()

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
WorldGrid::WorldGrid(const WorldGridConfig& cfg) : cfg_(cfg), rows_(0), cols_(0) {
    // YOUR CODE:
    //   rows_ = ceil((cfg_.x_max - cfg_.x_min) / cfg_.resolution)
    //   cols_ = ceil((cfg_.y_max - cfg_.y_min) / cfg_.resolution)
    //   Assert rows_ > 0 && cols_ > 0.
    //   Allocate cells_.resize(rows_ * cols_)  — default-constructed WorldCells.
    //   Optional: std::cout << "[WorldGrid] " << rows_ << "x" << cols_ << " cells\n";
    //
    // Convention lock-in (document + stay consistent):
    //   row ↔ x-axis (north/forward), col ↔ y-axis (east/left)
    //   cell (r,c) center at world coords:
    //     x = cfg_.x_min + (r + 0.5) * cfg_.resolution
    //     y = cfg_.y_min + (c + 0.5) * cfg_.resolution
    //   Pick one convention; every other function in this file uses it.

    rows_ = std::max(1, static_cast<int>(std::ceil((cfg_.x_max - cfg_.x_min) / cfg_.resolution)));
    cols_ = std::max(1, static_cast<int>(std::ceil((cfg_.y_max - cfg_.y_min) / cfg_.resolution)));
    cells_.resize(static_cast<size_t>(rows_) * cols_);
    assert(rows_ > 0 && cols_ > 0);
}

// -----------------------------------------------------------------------------
// Coordinate transform
// -----------------------------------------------------------------------------
bool WorldGrid::worldToGrid(double x, double y, int& row, int& col) const {
    // YOUR CODE:
    //   row = static_cast<int>( std::floor((x - cfg_.x_min) / cfg_.resolution) )
    //   col = static_cast<int>( std::floor((y - cfg_.y_min) / cfg_.resolution) )
    //   return (row >= 0 && row < rows_ && col >= 0 && col < cols_);
    //
    // Edge case: x == x_max exactly — floor gives row == rows_ (OOB). That's fine.
    row = static_cast<int>(std::floor((x - cfg_.x_min) / cfg_.resolution));
    col = static_cast<int>(std::floor((y - cfg_.y_min) / cfg_.resolution));
    return (row >= 0 && row < rows_ && col >= 0 && col < cols_);
}

// -----------------------------------------------------------------------------
// logit / sigmoid
// -----------------------------------------------------------------------------
float WorldGrid::probToLogOdds(float p) {
    // YOUR CODE:
    //   const float eps = 1e-6f;
    //   p = std::clamp(p, eps, 1.0f - eps);
    //   return std::log(p / (1.0f - p));
    const float eps = 1e-6f;
    p = std::clamp(p, eps, 1.0f - eps);
    return std::log(p / (1.0f - p));
}

float WorldGrid::logOddsToProb(float l) {
    // YOUR CODE:
    //   return 1.0f / (1.0f + std::exp(-l));   // numerically stable for l in [-50, 50]
    const float eps = 1e-6f;
    l = std::clamp(l, -50.0f, 50.0f);
    return 1.0f / (1.0f + std::exp(-l));
}

// -----------------------------------------------------------------------------
// Confidence math
// -----------------------------------------------------------------------------
float WorldGrid::adjustConfidence(float raw_conf, float pose_sigma) const {
    // YOUR CODE:
    //   if (pose_sigma <= 0.0f) return raw_conf;               // unknown ⇒ pass-through
    //   float scale = std::exp(-cfg_.pose_uncertainty_k * pose_sigma);
    //   return std::clamp(raw_conf * scale, 0.0f, 1.0f);
    
    if (pose_sigma <= 0.0f) return raw_conf;
    float scale = std::exp(-cfg_.pose_uncertainty_k * pose_sigma);
    return std::clamp(raw_conf * scale, 0.0f, 1.0f);
}

void WorldGrid::growConfidence(WorldCell& cell, float raw_conf, float pose_sigma) {
    // YOUR CODE:
    //   float contribution = adjustConfidence(raw_conf, pose_sigma);
    //   cell.confidence = std::min(
    //       cell.confidence + cfg_.confidence_growth * contribution,
    //       static_cast<float>(cfg_.confidence_max));
    //   cell.pose_sigma_at_last_obs = pose_sigma;
    //
    // Design note: confidence grows ADDITIVELY, not as EMA. The single-frame
    // confidence (from TraversabilityGrid) modulates the growth rate — low-quality
    // observations contribute less per frame. This is intentional: confidence
    // is an evidence accumulator, risk is a state estimate.
    float contribution = adjustConfidence(raw_conf, pose_sigma);
    cell.confidence = std::min<float>(
        cell.confidence + cfg_.confidence_growth * contribution,
        static_cast<float>(cfg_.confidence_max));
    cell.pose_sigma_at_last_obs = pose_sigma;
}

// -----------------------------------------------------------------------------
// Per-rule updates
// -----------------------------------------------------------------------------
void WorldGrid::updateCellEMA(WorldCell& cell, float new_risk, float mean_z) {
    // YOUR CODE:
    //   if (cell.obs_count == 0) {
    //       cell.risk   = new_risk;        // no prior → no blending
    //       cell.mean_z = mean_z;
    //   } else {
    //       float a = static_cast<float>(cfg_.alpha);
    //       cell.risk   = a * new_risk + (1.0f - a) * cell.risk;
    //       cell.mean_z = a * mean_z   + (1.0f - a) * cell.mean_z;
    //   }
    if (cell.obs_count == 0) {
        cell.risk   = new_risk;
        cell.mean_z = mean_z;
    } else {
        float a = static_cast<float>(cfg_.alpha);
        cell.risk   = a * new_risk + (1.0f - a) * cell.risk;
        cell.mean_z = a * mean_z   + (1.0f - a) * cell.mean_z;
    }
    
}

void WorldGrid::updateCellLogOdds(WorldCell& cell, float new_risk, float mean_z) {
    // YOUR CODE:
    //   Binary hit/miss form (OctoMap §3.2):
    //     float inc = (new_risk > 0.5f) ? cfg_.logodds_hit : cfg_.logodds_miss;
    //     cell.logodds = std::clamp(
    //         cell.logodds + static_cast<float>(inc),
    //         static_cast<float>(cfg_.logodds_clamp_min),
    //         static_cast<float>(cfg_.logodds_clamp_max));
    //     cell.risk = logOddsToProb(cell.logodds);
    //
    //   (Continuous variant — document and pick one):
    //     float inc = probToLogOdds(new_risk);
    //     ... same clamp ...
    //
    //   mean_z: EMA-smooth it regardless.
    //     cell.mean_z = 0.5f * mean_z + 0.5f * cell.mean_z;   // or reuse cfg_.alpha
    float inc = (new_risk > 0.5f) ? static_cast<float>(cfg_.logodds_hit) : static_cast<float>(cfg_.logodds_miss);
    cell.logodds = std::clamp(
        cell.logodds + inc,
        static_cast<float>(cfg_.logodds_clamp_min),
        static_cast<float>(cfg_.logodds_clamp_max));
    cell.risk = logOddsToProb(cell.logodds);
    if(cell.obs_count > 0) {
        cell.mean_z = static_cast<float>(cfg_.alpha) * mean_z + (1.0f - static_cast<float>(cfg_.alpha)) * cell.mean_z;
    } else {
        cell.mean_z = mean_z;
    }
}

void WorldGrid::updateCellOverwrite(WorldCell& cell, float new_risk, float mean_z) {
    // YOUR CODE:
    //   cell.risk   = new_risk;
    //   cell.mean_z = mean_z;
    //   No blending. No memory. Used as the "flicker baseline" in ablation B.
    cell.risk   = new_risk;
    cell.mean_z = mean_z;
}

// -----------------------------------------------------------------------------
// Temporal decay
// -----------------------------------------------------------------------------
void WorldGrid::applyDecay(double current_time) {
    // YOUR CODE:
    //   if (cfg_.decay_rate <= 0.0) return;                     // ablation D disable path
    //   for (auto& cell : cells_) {
    //       if (cell.last_update_time < 0.0) continue;          // never observed ⇒ skip
    //       double dt = current_time - cell.last_update_time;
    //       if (dt <= 0.0) continue;
    //       cell.confidence = std::max(
    //           0.0f,
    //           cell.confidence - static_cast<float>(cfg_.decay_rate * dt));
    //   }
    //
    // Design note: decay confidence, NEVER risk. A cell we no longer trust
    // becomes "unknown" (confidence → 0), not "safe" (risk → 0). This is a
    // safety invariant — see plan's "No hallucinated terrain."
    if (cfg_.decay_rate <= 0.0) return;
    for (auto& cell : cells_) {
        if (cell.last_update_time < 0.0) continue;
        double dt = current_time - cell.last_update_time;
        if (dt <= 0.0) continue;
        cell.confidence = std::max(0.0f, cell.confidence - static_cast<float>(cfg_.decay_rate * dt));
    }
}

// -----------------------------------------------------------------------------
// Main update
// -----------------------------------------------------------------------------
void WorldGrid::update(const TraversabilityGrid& local_grid,
                       const Pose&               pose,
                       double                    pose_sigma,
                       double                    timestamp) {
    // YOUR CODE:
    //
    // 1) (Optional) Decay before update:
    //      applyDecay(timestamp);
    //    Decide design: per-frame decay here, or separate timer callback.
    //    For offline RELLIS replay, per-frame is fine.
    //
    // 2) Pull the local grid's params once. TraversabilityGrid exposes them via
    //    grid_params() → const GridParams& (see include/traversability.hpp:95).
    //    GridParams has float fields: resolution, x_min, x_max, y_min, y_max
    //    (see include/traversability.hpp:31-44). There is NO xMin()/yMin()/
    //    resolution() getter on the class itself — go through grid_params().
    //    LiDAR-frame convention (header comment line 30): x → forward, y → left.
    //
    //      const GridParams& gp = local_grid.grid_params();
    //      const float res_l   = gp.resolution;
    //      const float x_min_l = gp.x_min;
    //      const float y_min_l = gp.y_min;
    //
    // 3) Iterate over local_grid cells. TraversabilityGrid::at(ix, iy) takes
    //    (ix, iy) — forward-axis index, left-axis index — NOT (row, col).
    //    Use these names to keep the code honest:
    //
    //      for (int ix = 0; ix < local_grid.rows(); ++ix) {
    //        for (int iy = 0; iy < local_grid.cols(); ++iy) {
    //          const CellFeatures& lc = local_grid.at(ix, iy);
    //          if (lc.point_count == 0) continue;     // skip empty local cells
    //
    //          // 3a) Local cell center in vehicle frame (float → double cast).
    //          Eigen::Vector3d p_local(
    //              static_cast<double>(x_min_l) + (ix + 0.5) * res_l,
    //              static_cast<double>(y_min_l) + (iy + 0.5) * res_l,
    //              0.0);
    //
    //          // 3b) Transform to world. pose.R/pose.t are double; p_local is double.
    //          Eigen::Vector3d p_world = pose.R * p_local + pose.t;
    //
    //          // 3c) Map to world-grid indices.
    //          int r_w, c_w;
    //          if (!worldToGrid(p_world.x(), p_world.y(), r_w, c_w)) continue;
    //
    //          // 3d) Update cell. CellFeatures fields are all float.
    //          WorldCell& cell = cells_[static_cast<size_t>(r_w) * cols_ + c_w];
    //          const float new_risk = lc.risk;
    //          const float raw_conf = lc.confidence;
    //          const float mean_z   = lc.mean_z;
    //
    //          switch (cfg_.update_rule) {
    //            case UpdateRule::EMA:       updateCellEMA      (cell, new_risk, mean_z); break;
    //            case UpdateRule::LogOdds:   updateCellLogOdds  (cell, new_risk, mean_z); break;
    //            case UpdateRule::Overwrite: updateCellOverwrite(cell, new_risk, mean_z); break;
    //          }
    //          growConfidence(cell, raw_conf, static_cast<float>(pose_sigma));
    //
    //          cell.obs_count++;
    //          cell.last_update_time = timestamp;
    //        }
    //      }
    //
    // 4) (Optional) Log cells-touched-this-frame for the coverage curve.

    // perframe decay before update 
    applyDecay(timestamp);
    const GridParams& gp = local_grid.grid_params();
    const float res_l   = gp.resolution;
    const float x_min_l = gp.x_min;
    const float y_min_l = gp.y_min;
    for (int ix = 0; ix < local_grid.rows(); ++ix) {
        for (int iy = 0; iy < local_grid.cols(); ++iy) {
            const CellFeatures& lc = local_grid.at(ix, iy);
            if (lc.point_count == 0) continue;

            Eigen::Vector3d p_local(
                static_cast<double>(x_min_l) + (ix + 0.5) * res_l,
                static_cast<double>(y_min_l) + (iy + 0.5) * res_l,
                0.0);

            Eigen::Vector3d p_world = pose.R * p_local + pose.t;

            int r_w, c_w;
            if (!worldToGrid(p_world.x(), p_world.y(), r_w, c_w)) continue;

            WorldCell& cell = cells_[static_cast<size_t>(r_w) * cols_ + c_w];
            const float new_risk = lc.risk;
            const float raw_conf = lc.confidence;
            const float mean_z   = lc.mean_z;

            switch (cfg_.update_rule) {
                case UpdateRule::EMA:       updateCellEMA      (cell, new_risk, mean_z); break;
                case UpdateRule::LogOdds:   updateCellLogOdds  (cell, new_risk, mean_z); break;
                case UpdateRule::Overwrite: updateCellOverwrite(cell, new_risk, mean_z); break;
            }
            growConfidence(cell, raw_conf, static_cast<float>(pose_sigma));

            cell.obs_count++;
            cell.last_update_time = timestamp;
        }
    }   
    
}

// -----------------------------------------------------------------------------
// Convex hull helpers (file-local, used by coveragePercent)
// -----------------------------------------------------------------------------
// Why hull instead of AABB for coverage?
//   AABB over a curving trajectory massively overestimates the area the
//   vehicle could have observed. Convex hull = tighter "mission area" metric.
//   Worth a blog sidebar: AABB denominator inflates → coverage_pct looks
//   artificially low; hull denominator = honest.

namespace {

// Cross product of (A - O) and (B - O). Sign = turn direction at O:
//   > 0  counter-clockwise (left turn)
//   < 0  clockwise         (right turn)
//   = 0  collinear
static double cross2(const Eigen::Vector2d& O,
                     const Eigen::Vector2d& A,
                     const Eigen::Vector2d& B) {
    // YOUR CODE:
    //   return (A.x() - O.x()) * (B.y() - O.y())
    //        - (A.y() - O.y()) * (B.x() - O.x());
    return (A.x() - O.x()) * (B.y() - O.y())
         - (A.y() - O.y()) * (B.x() - O.x());
}

// Graham scan → CCW hull starting from lowest-y point.
// Degenerate: traj.size() <= 2 returned as-is.
static std::vector<Eigen::Vector2d> convexHull(std::vector<Eigen::Vector2d> pts) {
    // YOUR CODE:
    //
    // if (pts.size() <= 2) return pts;
    //
    // 1) Pivot = lowest-y (tie-break: lowest-x). Swap to front.
    //      size_t piv = 0;
    //      for (size_t i = 1; i < pts.size(); ++i)
    //          if (pts[i].y() <  pts[piv].y() ||
    //             (pts[i].y() == pts[piv].y() && pts[i].x() < pts[piv].x()))
    //              piv = i;
    //      std::swap(pts[0], pts[piv]);
    //
    // 2) Sort [1..end] by polar angle around pts[0], using cross product
    //    (robust — avoids atan2 precision issues):
    //      const Eigen::Vector2d O = pts[0];
    //      std::sort(pts.begin() + 1, pts.end(),
    //          [&O](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    //              double c = cross2(O, a, b);
    //              if (c == 0.0)
    //                  return (a - O).squaredNorm() < (b - O).squaredNorm();
    //              return c > 0.0;                 // CCW → a precedes b
    //          });
    //
    // 3) Build hull with monotone stack:
    //      std::vector<Eigen::Vector2d> hull;
    //      for (const auto& p : pts) {
    //          while (hull.size() >= 2 &&
    //                 cross2(hull[hull.size() - 2], hull.back(), p) <= 0.0)
    //              hull.pop_back();
    //          hull.push_back(p);
    //      }
    //      return hull;
    //
    // Complexity: O(n log n) dominated by the sort.
    
    if (pts.size() <= 2) return pts;
    size_t piv = 0;
    for (size_t i = 1; i < pts.size(); ++i)
        if (pts[i].y() <  pts[piv].y() ||
            (pts[i].y() == pts[piv].y() && pts[i].x() < pts[piv].x()))
            piv = i;
    std::swap(pts[0], pts[piv]);

    const Eigen::Vector2d O = pts[0];
    std::sort(pts.begin() + 1, pts.end(),
        [&O](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            double c = cross2(O, a, b);
            if (c == 0.0)
                return (a - O).squaredNorm() < (b - O).squaredNorm();
            return c > 0.0;                 // CCW → a precedes b
        });

    std::vector<Eigen::Vector2d> hull;
    for (const auto& p : pts) {
        while (hull.size() >= 2 &&
               cross2(hull[hull.size() - 2], hull.back(), p) <= 0.0)
            hull.pop_back();
        hull.push_back(p);
    }
    return hull;

}

// Point-in-convex-polygon. Hull must be CCW (as convexHull returns).
// O(|hull|) per query; fine because hull is tiny vs grid-cell scan.
// Binary-search O(log n) exists but not needed here.
static bool pointInConvexHull(const Eigen::Vector2d& p,
                              const std::vector<Eigen::Vector2d>& hull) {
    // YOUR CODE:
    //
    // if (hull.size() < 3) return false;   // line or point has no interior
    //
    // For each edge (hull[i] → hull[(i+1) % n]):
    //   if (cross2(hull[i], hull[(i+1) % n], p) < 0.0)
    //       return false;                  // right of a CCW edge → outside
    // return true;
    //
    // Boundary: cross == 0 treated as inside (>=). For coverage this is
    // correct — cells on the hull edge count as "within mission area."
    if (hull.size() < 3) return false;
    for (size_t i = 0; i < hull.size(); ++i) {
        if (cross2(hull[i], hull[(i + 1) % hull.size()], p) < 0.0)
            return false;
    }
    return true;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Queries
// -----------------------------------------------------------------------------
const WorldCell* WorldGrid::getCell(double x, double y) const {
    int r, c;
    if (!worldToGrid(x, y, r, c)) return nullptr;
    return &cells_[static_cast<size_t>(r) * cols_ + c];
}

WorldCell* WorldGrid::getCell(double x, double y) {
    int r, c;
    if (!worldToGrid(x, y, r, c)) return nullptr;
    return &cells_[static_cast<size_t>(r) * cols_ + c];
}

double WorldGrid::coveragePercent(const std::vector<Eigen::Vector2d>& traj) const {
    // YOUR CODE:
    //   if (traj.empty()) return 0.0;
    //
    // v1 — axis-aligned bounding box:
    //   double xmin = +inf, xmax = -inf, ymin = +inf, ymax = -inf;
    //   for (const auto& p : traj) {
    //     xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
    //     ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
    //   }
    //
    //   Convert bbox corners → grid indices (worldToGrid, clamp to [0, rows_-1] etc.).
    //
    //   int total = 0, observed = 0;
    //   for rows in [r0, r1]: for cols in [c0, c1]:
    //     total++;
    //     if (cells_[r*cols_+c].obs_count > 0) observed++;
    //
    //   return static_cast<double>(observed) / std::max(1, total);
    //
    // v2 — swap AABB for convex hull (Graham scan) for a tighter "mission area" metric.
    //   Worth doing for the blog, not required for M3 exit gate.
    
    // Hull version: AABB used only as a cell-scan pre-filter; the hull is the
    // denominator. Flow:
    //   1. AABB of trajectory → grid-index rectangle (cheap scan bounds)
    //   2. Convex hull of trajectory (the real mission area)
    //   3. For each cell in the AABB: cell-center in hull? count; observed? count.
    //   4. coverage = observed / cells_inside_hull
    if (traj.empty()) return 0.0;

    // (1) AABB for scan bounds
    double xmin = +std::numeric_limits<double>::infinity(), xmax = -std::numeric_limits<double>::infinity();
    double ymin = +std::numeric_limits<double>::infinity(), ymax = -std::numeric_limits<double>::infinity();
    for (const auto& p : traj) {
        xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
        ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
    }

    int r0, c0, r1, c1;
    worldToGrid(xmin, ymin, r0, c0);
    worldToGrid(xmax, ymax, r1, c1);
    r0 = std::clamp(r0, 0, rows_ - 1);
    c0 = std::clamp(c0, 0, cols_ - 1);
    r1 = std::clamp(r1, 0, rows_ - 1);
    c1 = std::clamp(c1, 0, cols_ - 1);

    // (2) Convex hull of trajectory — the tighter denominator
    const std::vector<Eigen::Vector2d> hull = convexHull(traj);

    // (3) YOUR CODE — hull-gated counting.
    //   For each (r, c) in the AABB rectangle:
    //     Compute the cell's world-space center:
    //       Eigen::Vector2d p_cell(
    //           cfg_.x_min + (r + 0.5) * cfg_.resolution,
    //           cfg_.y_min + (c + 0.5) * cfg_.resolution);
    //     if (!pointInConvexHull(p_cell, hull)) continue;   // outside mission area
    //     total++;
    //     if (cells_[static_cast<size_t>(r) * cols_ + c].obs_count > 0) observed++;
    //
    // return static_cast<double>(observed) / std::max(1, total);
    //
    // Fallback if hull degenerates (hull.size() < 3): fall through to AABB
    // counting — a straight-line trajectory has no 2D area, so AABB is the
    // best we can do. Guard explicitly so a tiny traj doesn't divide by zero.
    int total = 0, observed = 0;
    if (hull.size() < 3) {
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                total++;
                if (cells_[static_cast<size_t>(r) * cols_ + c].obs_count > 0) observed++;
            }
        }
    } else {
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                Eigen::Vector2d p_cell(
                    cfg_.x_min + (r + 0.5) * cfg_.resolution,
                    cfg_.y_min + (c + 0.5) * cfg_.resolution);
                if (!pointInConvexHull(p_cell, hull)) continue;
                total++;
                if (cells_[static_cast<size_t>(r) * cols_ + c].obs_count > 0) observed++;
            }
        }
    }
    return static_cast<double>(observed) / std::max(1, total);
}

// -----------------------------------------------------------------------------
// Snapshot I/O
// -----------------------------------------------------------------------------
bool WorldGrid::saveSnapshot(const std::string& stem) const {
    // YOUR CODE:
    //   Two files:
    //
    //   <stem>.csv — full cell data, reloadable:
    //     header: row,col,risk,logodds,confidence,obs_count,last_update_time,mean_z,pose_sigma_at_last_obs
    //     one row per cell with obs_count > 0 (skip unobserved cells for size).
    //
    //   <stem>.png — grayscale risk heatmap (for quick eyeballing).
    //     Options:
    //       a) write PGM (trivially easy, viewable in many tools)
    //       b) shell out to Python (scripts/render_world_grid.py)
    //       c) link stb_image_write.h (single-header)
    //     Pick (a) for M3; upgrade to (c) if the blog needs nicer visuals.
    //
    //   Return true on success.
    
    if(stem.empty()) return false;
    //two types of files to be written: .csv and .png
    std::string csv_filename = stem + ".csv";
    std::string png_filename = stem + ".png";
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << csv_filename << std::endl;
        return false;
    }
    // Write CSV header
    csv_file <<"# rows = " << rows_ << ", cols = " << cols_ << ", resolution = " << cfg_.resolution << "\n";
    csv_file << "row,col,risk,logodds,confidence,obs_count,last_update_time,mean_z,pose_sigma_at_last_obs\n";
    // Write cell data for observed cells
    for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
            const WorldCell& cell = cells_[static_cast<size_t>(r) * cols_ + c];
            if (cell.obs_count > 0) {
                csv_file << r << "," << c << ","
                         << cell.risk << ","
                         << cell.logodds << ","
                         << cell.confidence << ","
                         << cell.obs_count << ","
                         << cell.last_update_time << ","
                         << cell.mean_z << ","
                         << cell.pose_sigma_at_last_obs << "\n";
            }
        }
    }
    csv_file.close();

    // PNG: RGB viridis heatmap of risk. Two conventions baked in:
    //   (1) Vertical flip: we want north/forward (increasing x = increasing row)
    //       to appear UP in the image. stbi writes top row first, so row 0
    //       must be the HIGHEST-x row, i.e. invert r when indexing cells_.
    //   (2) Unobserved cells (obs_count == 0) render as neutral gray, not
    //       bright-green viridis-zero — otherwise "unknown" looks identical
    //       to "very safe," which is a safety-readability bug in the viewer.
    //
    // YOUR CODE:
    //
    //   const int W = cols_;
    //   const int H = rows_;
    //   std::vector<unsigned char> image(static_cast<size_t>(W) * H * 3);   // RGB
    //
    //   for (int r_img = 0; r_img < H; ++r_img) {
    //       // (1) flip: image row 0 = grid row (H-1)
    //       const int r_grid = H - 1 - r_img;
    //       for (int c = 0; c < W; ++c) {
    //           const WorldCell& cell = cells_[static_cast<size_t>(r_grid) * cols_ + c];
    //           unsigned char R, G, B;
    //           if (cell.obs_count == 0) {
    //               R = G = B = 64;                 // neutral dark gray = unknown
    //           } else {
    //               viridisLookup(std::clamp(cell.risk, 0.0f, 1.0f), R, G, B);
    //           }
    //           const size_t px = (static_cast<size_t>(r_img) * W + c) * 3;
    //           image[px + 0] = R;
    //           image[px + 1] = G;
    //           image[px + 2] = B;
    //       }
    //   }
    //
    //   // stbi: (filename, width, height, channels, data, stride_bytes)
    //   stbi_write_png(png_filename.c_str(), W, H, 3, image.data(), W * 3);
    //
    // viridisLookup: 5-stop piecewise-linear approximation of matplotlib viridis.
    //   t ∈ [0,1] where t=0 → dark purple (safe end) and t=1 → yellow (hazard).
    //   If you'd rather have "bright = safe, dark = hazard", pass (1.0 - risk)
    //   into the lookup instead. Pick one and document in the blog.
    //
    //   Suggested control points (R, G, B on [0,255]):
    //     t=0.00  →  ( 68,   1,  84)   // dark purple
    //     t=0.25  →  ( 59,  82, 139)   // blue
    //     t=0.50  →  ( 33, 144, 140)   // teal
    //     t=0.75  →  ( 93, 201,  99)   // green
    //     t=1.00  →  (253, 231,  37)   // yellow
    //   Linearly interpolate between adjacent stops.
    //
    //   For quick iteration you can also skip the colormap entirely and
    //   write a 1-channel grayscale where brightness = (1 - risk) * 255,
    //   but grayscale BEVs are ugly in blog posts.
    
    const int W = cols_;
    const int H = rows_;
    std::vector<unsigned char> image(static_cast<size_t>(W) * H * 3);   // RGB
    //use tinycolormap
    for (int r_img = 0; r_img < H; ++r_img) {
        const int r_grid = H - 1 - r_img;
        for (int c = 0; c < W; ++c) {
            const WorldCell& cell = cells_[static_cast<size_t>(r_grid) * cols_ + c];
            unsigned char R, G, B;
            if (cell.obs_count == 0) {
                R = G = B = 100;
            } else {
                tinycolormap::Color color = tinycolormap::GetColor(cell.risk);
                R = static_cast<unsigned char>(color.r() * 255.0);
                G = static_cast<unsigned char>(color.g() * 255.0);
                B = static_cast<unsigned char>(color.b() * 255.0);
            }
            const size_t px = (static_cast<size_t>(r_img) * W + c) * 3;
            image[px + 0] = R;
            image[px + 1] = G;
            image[px + 2] = B;
        }
    }
    stbi_write_png(png_filename.c_str(), W, H, 3, image.data(), W * 3);



    return true;
}

bool WorldGrid::loadSnapshot(const std::string& stem) {
    // YOUR CODE:
    //   Inverse of saveSnapshot. Validate: rows_/cols_/resolution match cfg_.
    //   If mismatch, log and return false (do NOT silently resize — caller bug).

    //validate rows cols resolution 
    //parse #rows metadata line
    
    
    if(stem.empty()) return false;
    std::string csv_filename = stem + ".csv";
    std::ifstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << csv_filename << std::endl;
        return false;
    }
    int snap_rows, snap_cols;
    double snap_resolution;
    std::string meta_line;
    std::getline(csv_file, meta_line);
    if(std::sscanf(meta_line.c_str(), "# rows = %d, cols = %d, resolution = %lf", &snap_rows, &snap_cols, &snap_resolution) != 3) {
        std::cerr << "Invalid snapshot metadata line: " << meta_line << std::endl;
        return false;
    }

    if(snap_rows != rows_ || snap_cols != cols_ || std::abs(snap_resolution - cfg_.resolution) > 1e-9) {
        std::cerr << "Snapshot config mismatch: expected " << rows_ << "x" << cols_ << " @ " << cfg_.resolution
                  << ", but snapshot has " << snap_rows << "x" << snap_cols << " @ " << snap_resolution << std::endl;
        return false;
    }

    // Consume the column-header line that saveSnapshot() writes between the
    // metadata line and the first data row. Without this, the data loop below
    // would read "row,col,risk,..." as its first row and std::stoi on "row"
    // would throw std::invalid_argument → loadSnapshot would fail on every
    // round-trip.
    // YOUR CODE (optional hardening — skip for M3):
    //   Parse this line and verify column names match the expected schema
    //   ("row,col,risk,logodds,confidence,obs_count,last_update_time,mean_z,
    //   pose_sigma_at_last_obs"). We only load files we wrote, so skipping is
    //   safe for now; upgrade to strict-check if an external tool ever writes
    //   snapshots.
    std::string column_header;
    if (!std::getline(csv_file, column_header)) {
        std::cerr << "Snapshot missing column-header line: " << csv_filename << std::endl;
        return false;
    }

    // Read cell data
    std::string line;
    while (std::getline(csv_file, line)) {
        std::istringstream line_stream(line);
        std::string cell_data;
        std::vector<std::string> cell_values;
        while (std::getline(line_stream, cell_data, ',')) {
            cell_values.push_back(cell_data);        }
        if (cell_values.size() != 9) {
            std::cerr << "Invalid CSV line: " << line << std::endl;
            return false;
        }
        int row, col;
        float risk, logodds, confidence, mean_z, pose_sigma_at_last_obs;
        int obs_count;
        double last_update_time;
        try {
            row = std::stoi(cell_values[0]);
            col = std::stoi(cell_values[1]);
            risk = std::stof(cell_values[2]);
            logodds = std::stof(cell_values[3]);
            confidence = std::stof(cell_values[4]);
            obs_count = std::stoi(cell_values[5]);
            last_update_time = std::stod(cell_values[6]);
            mean_z = std::stof(cell_values[7]);
            pose_sigma_at_last_obs = std::stof(cell_values[8]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing CSV line: " << line << " - " << e.what() << std::endl;
            return false;
        }
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            std::cerr << "Cell indices out of bounds: row=" << row << ", col=" << col << std::endl;
            return false;   
        }
        WorldCell& cell = cells_[static_cast<size_t>(row) * cols_ + col];
        cell.risk = risk;
        cell.logodds = logodds;
        cell.confidence = confidence;
        cell.obs_count = obs_count;
        cell.last_update_time = last_update_time;
        cell.mean_z = mean_z;
        cell.pose_sigma_at_last_obs = pose_sigma_at_last_obs;
    }
    csv_file.close();
    return true;


}

