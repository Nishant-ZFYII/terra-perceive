// =============================================================================
// tracker_runner — CLI for P2-M4 SORT tracker offline runs and ablations.
// =============================================================================
//
// Mirrors slam_runner.cpp / accumulator_runner.cpp flag style. One invocation
// = one run = one results directory.
//
// Usage:
//   ./tracker_runner \
//       --detections   tests/data/crossing.csv  \
//       --solver       munkres                   \
//       --max-dist     3.0                       \
//       --max-misses   3                         \
//       --min-hits     1                         \
//       --process-noise 0.01                     \
//       --meas-noise   0.1                       \
//       --dt           0.1                       \
//       --snapshot-every 1                       \
//       --out          results_m4/ablation_a/munkres/ \
//       --verbose
//
// Detections CSV schema (produced by scripts/synth_detections.py):
//   frame_id,det_id,x,y,class_id,gt_track_id
//   gt_track_id = -1 means spurious / no ground-truth label.
//
// Outputs under <out>/:
//   tracks.csv            frame_id,track_id,x,y,vx,vy,age,cov_trace
//   id_switches.csv       frame_id,gt_track_id,prev_track_id,curr_track_id
//                         (only written when gt_track_id is present and != -1)
//   snapshots/frame_NNNNN.csv (one row per LIVE track at that frame)
//   metrics.json          summary stats + config echo (WRITTEN LAST — atomic
//                         signal that the run completed successfully)
//
// Ablation pre-flight rule mapping (memory: feedback_ablation_lessons_p2m3):
//   #1  --snapshot-every  default=1 (animation-grade), pick higher for scalar runs
//   #2  ID-switch metric is computed live and printed
//   #7  first AND last frame sanity-print: det count, track count, sample (id, pos, vel)
//   #9  metrics.json is written LAST after all CSVs are flushed and closed
// =============================================================================

#include "sort_tracker.hpp"
#include "hungarian.hpp"   // tracker::Solver

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// CLI parsing helpers — same shape as slam_runner.cpp / accumulator_runner.cpp.
// -----------------------------------------------------------------------------
static std::string getArg(int argc, char** argv,
                          const std::string& flag,
                          const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return std::string(argv[i + 1]);
    return def;
}

static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (flag == argv[i]) return true;
    return false;
}

// -----------------------------------------------------------------------------
// Detection record — one row of the input CSV.
// -----------------------------------------------------------------------------
struct DetRow {
    int frame_id;
    int det_id;
    float x;
    float y;
    int class_id;
    int gt_track_id;   // -1 if no ground-truth label
};

// -----------------------------------------------------------------------------
// Load_detections — read the input CSV into a vector of DetRow.
// Skips the header line. Tolerates trailing whitespace.
// -----------------------------------------------------------------------------
static std::vector<DetRow> load_detections(const fs::path& csv_path) {
    // YOUR CODE:
    //   1. std::ifstream in(csv_path);
    //      if (!in) throw std::runtime_error("cannot open " + csv_path.string());
    //   2. std::string header; std::getline(in, header);  // discard
    //   3. std::vector<DetRow> rows;
    //      std::string line;
    //      while (std::getline(in, line)) {
    //          if (line.empty()) continue;
    //          std::stringstream ss(line);
    //          DetRow r{};
    //          char comma;
    //          ss >> r.frame_id >> comma >> r.det_id >> comma
    //             >> r.x >> comma >> r.y >> comma
    //             >> r.class_id >> comma >> r.gt_track_id;
    //          rows.push_back(r);
    //      }
    //   4. return rows;
    //
    // Edge cases to think about:
    //   - Trailing newline → getline returns empty line; skip with `continue`.
    //   - gt_track_id column missing entirely → fall back to -1. (The synth
    //     generator always writes 6 columns, but RELLIS-derived CSVs from
    //     M5+ might have fewer.)
    
    //load file
    std::ifstream in(csv_path);
    if (!in) throw std::runtime_error("cannot open " + csv_path.string());
    //discard header
    std::string header;
    std::getline(in, header);
    //parse rows
    std::vector<DetRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        DetRow r{};
        char comma;
        ss >> r.frame_id >> comma >> r.det_id >> comma
           >> r.x >> comma >> r.y >> comma
           >> r.class_id >> comma >> r.gt_track_id;
        rows.push_back(r);
    }
    return rows;
}

// -----------------------------------------------------------------------------
// Group detections by frame_id. Assumes the CSV is sorted by frame_id; if not,
// you'll need to bucket into a std::map first.
//
// P3-M13: when --features-csv is provided, `features[i]` carries the 8-dim
// hand-crafted feature vector for detection i (parallel to positions[i]).
// When --use-appearance is off, this vector is empty.
// -----------------------------------------------------------------------------
struct FrameDetections {
    int frame_id;
    std::vector<Eigen::Vector2f>      positions;
    std::vector<int>                  class_ids;
    std::vector<int>                  gt_track_ids;
    std::vector<tracker::FeatureVector> features;
};

static std::vector<FrameDetections> group_by_frame(const std::vector<DetRow>& rows) {
    // YOUR CODE:
    //   1. std::map<int, FrameDetections> by_frame;   // ordered by frame_id
    //   2. for (const auto& r : rows) {
    //          auto& f = by_frame[r.frame_id];
    //          f.frame_id = r.frame_id;
    //          f.positions.emplace_back(r.x, r.y);
    //          f.class_ids.push_back(r.class_id);
    //          f.gt_track_ids.push_back(r.gt_track_id);
    //      }
    //   3. std::vector<FrameDetections> out;
    //      out.reserve(by_frame.size());
    //      for (auto& [k, v] : by_frame) out.push_back(std::move(v));
    //   4. return out;
    std::map<int, FrameDetections> by_frame;
    for (const auto& r : rows) {
        auto& f = by_frame[r.frame_id];
        f.frame_id = r.frame_id;
        f.positions.emplace_back(r.x, r.y);
        f.class_ids.push_back(r.class_id);
        f.gt_track_ids.push_back(r.gt_track_id);
    }
    std::vector<FrameDetections> out;
    out.reserve(by_frame.size());
    for (auto& [k, v] : by_frame) out.push_back(std::move(v));
    return out;
}

// -----------------------------------------------------------------------------
// Solver parsing — accepts "greedy" or "munkres" (case-insensitive).
// -----------------------------------------------------------------------------
static tracker::Solver parse_solver(const std::string& s) {
    if (s == "greedy"  || s == "Greedy"  || s == "GREEDY")  return tracker::Solver::Greedy;
    if (s == "munkres" || s == "Munkres" || s == "MUNKRES") return tracker::Solver::Munkres;
    throw std::runtime_error("Unknown solver: " + s + " (expected greedy|munkres)");

}

// -----------------------------------------------------------------------------
// Filter parsing — accepts "cv" (M4 baseline KF) or "imm" (P3-M12 IMM).
// -----------------------------------------------------------------------------
static tracker::FilterKind parse_filter(const std::string& s) {
    if (s == "cv"  || s == "CV")   return tracker::FilterKind::CV;
    if (s == "imm" || s == "IMM")  return tracker::FilterKind::IMM;
    throw std::runtime_error("Unknown filter: " + s + " (expected cv|imm)");
}

// -----------------------------------------------------------------------------
// P3-M13 feature loader — read the parallel features CSV produced by
// scripts/clusters_to_detections.py --features-dir.  Schema:
//     frame_id, det_id, n_log, bbox_x, bbox_y, bbox_z, min_z, eig_r1,
//     eig_r2, range
//
// Returns a map (frame_id, det_id) → FeatureVector. The runner then merges
// this into FrameDetections.features in matching order.
// -----------------------------------------------------------------------------
static std::map<std::pair<int, int>, tracker::FeatureVector>
load_features(const fs::path& csv_path) {
    std::ifstream in(csv_path);
    if (!in) throw std::runtime_error("cannot open features CSV: " + csv_path.string());
    std::string header;
    std::getline(in, header);          // discard
    std::map<std::pair<int, int>, tracker::FeatureVector> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int fid, did;
        char comma;
        tracker::FeatureVector feat;
        ss >> fid >> comma >> did;
        for (int i = 0; i < 8; ++i) {
            ss >> comma >> feat(i);
        }
        out.emplace(std::make_pair(fid, did), feat);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Fix B — load SLAM ego poses keyed by frame_id.
//
// Schema (P2-M2 product, data/poses_slam_full.csv):
//     frame_id, tx, ty, tz, qx, qy, qz, qw
//
// The tracker is BEV-only, so we project SE(3) → SE(2) by extracting yaw
// from the quaternion and dropping z + roll/pitch. This is the same
// projection the rest of the BEV pipeline uses (see slam_runner.cpp).
//
// Returns map<frame_id, T_world_ego_2d>. Frames missing from the CSV are
// absent from the map; the per-frame loop falls back to Identity (which
// makes Fix B a no-op for that frame, equivalent to ego-frame behavior).
// -----------------------------------------------------------------------------
static std::map<int, Eigen::Isometry2f> load_ego_poses(const fs::path& csv_path) {
    std::ifstream in(csv_path);
    if (!in) throw std::runtime_error("cannot open ego-poses CSV: " + csv_path.string());
    std::string header;
    std::getline(in, header);          // discard
    std::map<int, Eigen::Isometry2f> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int fid;
        float tx, ty, tz, qx, qy, qz, qw;
        char comma;
        ss >> fid    >> comma
           >> tx     >> comma >> ty >> comma >> tz >> comma
           >> qx     >> comma >> qy >> comma >> qz >> comma >> qw;
        // Yaw from quaternion (Z-axis rotation):
        //   yaw = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))
        const float yaw = std::atan2(2.0f * (qw * qz + qx * qy),
                                     1.0f - 2.0f * (qy * qy + qz * qz));
        Eigen::Isometry2f T = Eigen::Isometry2f::Identity();
        T.linear()      = Eigen::Rotation2Df(yaw).toRotationMatrix();
        T.translation() = Eigen::Vector2f(tx, ty);
        out.emplace(fid, T);
    }
    return out;
}

// -----------------------------------------------------------------------------
// ID-switch counter — the headline metric for Ablation A.
//
// Definition (Bewley 2016 / MOTChallenge): a frame contributes one ID-switch
// for every ground-truth track whose associated tracker_id this frame differs
// from its tracker_id on its previous appearance. Frames where the gt has no
// matching track contribute nothing (counted as fragmentation, not ID-switch).
//
// State carried across frames:
//   gt_to_track_id[gt] = the most recent tracker_id that was matched to gt.
// -----------------------------------------------------------------------------
class IdSwitchCounter {
   public:
    // Returns the number of NEW ID-switches that happened in this frame.
    // `frame_associations`: list of (gt_track_id, current_tracker_id) pairs
    // for ground-truth labels matched to live tracker output this frame.
    int update(int frame_id,
               const std::vector<std::pair<int, int>>& frame_associations,
               std::ofstream& switches_log) {
        // YOUR CODE:
        //   int new_switches = 0;
        //   for (auto [gt, curr_id] : frame_associations) {
        //       auto it = gt_to_track_id_.find(gt);
        //       if (it != gt_to_track_id_.end() && it->second != curr_id) {
        //           switches_log << frame_id << "," << gt << ","
        //                        << it->second << "," << curr_id << "\n";
        //           new_switches += 1;
        //       }
        //       gt_to_track_id_[gt] = curr_id;
        //   }
        //   total_ += new_switches;
        //   return new_switches;
        int new_switches = 0;
        for (auto [gt, curr_id] : frame_associations) {
            auto it = gt_to_track_id_.find(gt);
            if (it != gt_to_track_id_.end() && it->second != curr_id) {
                switches_log << frame_id << "," << gt << ","
                             << it->second << "," << curr_id << "\n";
                new_switches += 1;
            }
            gt_to_track_id_[gt] = curr_id;
        }
        total_ += new_switches;
        return new_switches;
    }

    int total() const { return total_; }

   private:
    std::unordered_map<int, int> gt_to_track_id_;
    int total_ = 0;
};

// -----------------------------------------------------------------------------
// Match tracker output to ground-truth labels (for ID-switch counting).
//
// Strategy: for each ground-truth-labeled detection in this frame, find the
// LIVE track whose published position is closest. That tracker_id is the
// association for this gt label.
//
// This is independent of the SORT internal matching — we're computing which
// gt label each tracker_id "represents" right now, for evaluation.
// -----------------------------------------------------------------------------
static std::vector<std::pair<int, int>> associate_to_gt(
    const std::vector<tracker::Track>& published,
    const FrameDetections& frame) {
    // YOUR CODE:
    //   std::vector<std::pair<int, int>> assoc;   // (gt_track_id, tracker_id)
    //   for (size_t i = 0; i < frame.gt_track_ids.size(); ++i) {
    //       const int gt = frame.gt_track_ids[i];
    //       if (gt < 0) continue;   // skip spurious
    //       const Eigen::Vector2f& p = frame.positions[i];
    //       int   best_id = -1;
    //       float best_d  = std::numeric_limits<float>::infinity();
    //       for (const auto& t : published) {
    //           const float d = (t.position() - p).norm();
    //           if (d < best_d) { best_d = d; best_id = t.id; }
    //       }
    //       if (best_id != -1) assoc.emplace_back(gt, best_id);
    //   }
    //   return assoc;
                                                                                                                                     
    std::vector<int> gt_idx;                              
    gt_idx.reserve(frame.gt_track_ids.size());
    for (size_t i = 0; i < frame.gt_track_ids.size(); ++i)                                                                                                                                    
        if (frame.gt_track_ids[i] >= 0) gt_idx.push_back(static_cast<int>(i));                                                                                                                
                                                                                                                                                                                            
    if (gt_idx.empty() || published.empty()) return {};                                                                                                                                       
                                                        
    // Cost matrix: rows = gt-labeled dets, cols = published tracks.                                                                                                                          
    Eigen::MatrixXf cost(gt_idx.size(), published.size());
    for (size_t r = 0; r < gt_idx.size(); ++r) {                                                                                                                                              
        const Eigen::Vector2f& p = frame.positions[gt_idx[r]];                                                                                                                                
        for (size_t c = 0; c < published.size(); ++c) {                                                                                                                                       
            cost(r, c) = (published[c].position() - p).norm();                                                                                                                             
        }                                                                                                                                                                                     
    }                                                     
                                                                                                                                                                                            
    // Optimal 1-to-1 assignment (no gating — evaluator should always                                                                                                                         
    // pair every gt with its globally-closest available track).
    const auto pairs = tracker::hungarian_solve(                                                                                                                                              
        cost, tracker::Solver::Munkres,                                                                                                                                                       
        std::numeric_limits<float>::infinity());                                                                                                                                              
                                                                                                                                                                                            
    std::vector<std::pair<int, int>> assoc;                                                                                                                                                   
    assoc.reserve(pairs.size());                          
    for (auto [r, c] : pairs) {                                                                                                                                                               
        const int gt = frame.gt_track_ids[gt_idx[r]];     
        const int tid = published[c].id;                                                                                                                                                      
        assoc.emplace_back(gt, tid);                                                                                                                                                          
    }                                                                                                                                                                                         
    return assoc;                                                                                                                                                                             
  

}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ---- Parse CLI ------------------------------------------------------
    const std::string detections_csv = getArg(argc, argv, "--detections", "");
    const std::string out_dir        = getArg(argc, argv, "--out", "results_m4/run/");
    const std::string solver_str     = getArg(argc, argv, "--solver", "greedy");
    const std::string filter_str     = getArg(argc, argv, "--filter", "cv");
    const float max_dist             = std::stof(getArg(argc, argv, "--max-dist",     "3.0"));
    const int   max_misses           = std::stoi(getArg(argc, argv, "--max-misses",   "3"));
    const int   min_hits             = std::stoi(getArg(argc, argv, "--min-hits",     "1"));
    const float dt                   = std::stof(getArg(argc, argv, "--dt",           "0.1"));
    const float process_noise        = std::stof(getArg(argc, argv, "--process-noise", "0.01"));
    const float meas_noise           = std::stof(getArg(argc, argv, "--meas-noise",   "0.1"));
    const int   snapshot_every       = std::stoi(getArg(argc, argv, "--snapshot-every", "1"));
    const int   max_frames           = std::stoi(getArg(argc, argv, "--frames", "-1"));   // -1 = all
    const bool  verbose              = hasFlag(argc, argv, "--verbose");
    const bool swap_order            = hasFlag(argc, argv, "--swap-order"); // ablation: UpdateThenPredict instead of PredictThenUpdate

    // P3-M13 appearance flags.
    //   --use-appearance              : enable the appearance-augmented cost matrix.
    //   --features-csv FILE           : per-detection 8-dim features
    //                                   (sibling to --detections; produced by
    //                                   scripts/clusters_to_detections.py
    //                                   --features-dir).
    //   --appearance-weight λ          : convex weight in (1-λ)·d_pos + λ·d_emb.
    //   --embedding-alpha α            : running-mean rate for track embeddings.
    const bool        use_appearance     = hasFlag(argc, argv, "--use-appearance");
    const std::string features_csv       = getArg(argc, argv, "--features-csv", "");
    const float       appearance_weight  = std::stof(getArg(argc, argv, "--appearance-weight", "0.4"));
    const float       embedding_alpha    = std::stof(getArg(argc, argv, "--embedding-alpha",   "0.1"));
    // P3.5 cascade matching:
    //   --max-age N  : Lost tracks stay in the retired pool for N frames.
    //                  Default 0 (cascade DISABLED, P3-M13 behavior).
    const int         max_age            = std::stoi(getArg(argc, argv, "--max-age", "0"));
    // Fix B: optional SLAM ego-pose CSV. When provided, the tracker stores
    // Lost-track positions in world frame and the cascade stage transforms
    // them back into current-ego before gating. Without it (default empty),
    // every per-frame T_world_ego stays at Identity and behavior is
    // bit-identical to the pre-Fix-B path.
    const std::string ego_poses_csv      = getArg(argc, argv, "--ego-poses", "");

    if (detections_csv.empty()) {
        std::cerr << "ERROR: --detections is required\n";
        return 1;
    }

    fs::create_directories(out_dir);
    fs::create_directories(fs::path(out_dir) / "snapshots");

    const tracker::Solver solver = parse_solver(solver_str);
    const tracker::FilterKind filter_kind = parse_filter(filter_str);

    if (verbose) {
        std::cerr << "[tracker_runner] config:\n"
                  << "  detections        = " << detections_csv     << "\n"
                  << "  solver            = " << solver_str         << "\n"
                  << "  filter            = " << filter_str         << "\n"
                  << "  max_dist          = " << max_dist           << "\n"
                  << "  max_misses        = " << max_misses         << "\n"
                  << "  min_hits          = " << min_hits           << "\n"
                  << "  dt                = " << dt                 << "\n"
                  << "  process_noise     = " << process_noise      << "\n"
                  << "  meas_noise        = " << meas_noise         << "\n"
                  << "  use_appearance    = " << (use_appearance ? "true" : "false") << "\n"
                  << "  features_csv      = " << features_csv       << "\n"
                  << "  appearance_weight = " << appearance_weight  << "\n"
                  << "  embedding_alpha   = " << embedding_alpha    << "\n"
                  << "  max_age (cascade) = " << max_age            << "\n"
                  << "  ego_poses (Fix B) = " << (ego_poses_csv.empty() ? "(none, Identity)" : ego_poses_csv) << "\n"
                  << "  snapshot_every    = " << snapshot_every     << "\n"
                  << "  out               = " << out_dir            << "\n"
                  << "  swap_order        = " << (swap_order ? "UpdateThenPredict" : "PredictThenUpdate") << "\n";
    }

    if (use_appearance && features_csv.empty()) {
        std::cerr << "ERROR: --use-appearance requires --features-csv\n";
        return 1;
    }

    // ---- Load detections ------------------------------------------------
    auto rows = load_detections(detections_csv);
    auto frames = group_by_frame(rows);
    if (max_frames > 0 && static_cast<int>(frames.size()) > max_frames) {
        frames.resize(max_frames);
    }
    if (frames.empty()) {
        std::cerr << "ERROR: no frames loaded from " << detections_csv << "\n";
        return 1;
    }

    // ---- Load + merge features (P3-M13) ---------------------------------
    if (use_appearance) {
        auto feat_map = load_features(features_csv);
        size_t n_filled = 0, n_zeros = 0;
        for (auto& f : frames) {
            f.features.reserve(f.positions.size());
            for (size_t k = 0; k < f.positions.size(); ++k) {
                auto it = feat_map.find({f.frame_id, static_cast<int>(k)});
                if (it != feat_map.end()) {
                    f.features.push_back(it->second);
                    ++n_filled;
                } else {
                    f.features.push_back(tracker::FeatureVector::Zero());
                    ++n_zeros;
                }
            }
        }
        if (verbose) {
            std::cerr << "[tracker_runner] features: " << n_filled
                      << " matched, " << n_zeros << " zero-filled\n";
        }
    }

    // ---- Load ego poses (Fix B, optional) -------------------------------
    std::map<int, Eigen::Isometry2f> ego_poses;
    if (!ego_poses_csv.empty()) {
        ego_poses = load_ego_poses(ego_poses_csv);
        if (verbose) {
            std::cerr << "[tracker_runner] ego poses loaded: "
                      << ego_poses.size() << " frames\n";
        }
    }

    // ---- Build the tracker ----------------------------------------------
    tracker::SORTTracker tr(max_dist, max_misses, min_hits,
                            solver,
                            swap_order ? tracker::Order::UpdateThenPredict
                                       : tracker::Order::PredictThenUpdate,
                            dt, process_noise, meas_noise,
                            filter_kind,
                            use_appearance,
                            appearance_weight,
                            embedding_alpha,
                            max_age);

    // ---- Open output streams --------------------------------------------
    std::ofstream tracks_csv(fs::path(out_dir) / "tracks.csv");
    tracks_csv << "frame_id,track_id,x,y,vx,vy,age,cov_trace\n";

    std::ofstream switches_csv(fs::path(out_dir) / "id_switches.csv");
    switches_csv << "frame_id,gt_track_id,prev_track_id,curr_track_id\n";

    IdSwitchCounter id_counter;

    // ---- Per-frame loop -------------------------------------------------
    //
    // YOUR CODE: walk frames, drive the tracker, log per-track rows, count
    // ID switches, write per-frame snapshots, and apply the rule-#7 sanity
    // print on the FIRST and LAST frame.
    //
    // Pseudocode skeleton (mirrors accumulator_runner's per-frame loop):
    //
    //   const auto t0 = std::chrono::steady_clock::now();
    //   int total_track_count = 0;     // sum across frames (avg lifetime helper)
    //   std::set<int> distinct_track_ids;
    //
    //   for (size_t fi = 0; fi < frames.size(); ++fi) {
    //       const auto& f = frames[fi];
    //       const auto published = tr.update(f.positions, f.class_ids);
    //
    //       // Per-track CSV rows
    //       for (const auto& t : published) {
    //           const auto p = t.position();
    //           const auto v = t.velocity();
    //           tracks_csv << f.frame_id << "," << t.id << ","
    //                      << p.x() << "," << p.y() << ","
    //                      << v.x() << "," << v.y() << ","
    //                      << t.hits << "," << t.covariance_trace() << "\n";
    //           distinct_track_ids.insert(t.id);
    //           total_track_count += 1;
    //       }
    //
    //       // Snapshot
    //       if (snapshot_every > 0 && (fi % snapshot_every) == 0) {
    //           std::ostringstream name;
    //           name << "snapshots/frame_" << std::setw(5) << std::setfill('0')
    //                << f.frame_id << ".csv";
    //           std::ofstream snap(fs::path(out_dir) / name.str());
    //           snap << "track_id,x,y,vx,vy,age,cov_trace\n";
    //           for (const auto& t : published) { ... write same fields ... }
    //       }
    //
    //       // ID-switch counting (only if gt is present, i.e. any gt_track_id >= 0)
    //       const auto assoc = associate_to_gt(published, f);
    //       const int new_sw = id_counter.update(f.frame_id, assoc, switches_csv);
    //
    //       // Rule #7 sanity print: first and last frame
    //       const bool is_first = (fi == 0);
    //       const bool is_last  = (fi + 1 == frames.size());
    //       if (verbose && (is_first || is_last)) {
    //           std::cerr << "[tracker_runner] frame " << f.frame_id
    //                     << "  dets=" << f.positions.size()
    //                     << "  tracks=" << published.size();
    //           if (!published.empty()) {
    //               const auto& t = published.front();
    //               std::cerr << "  sample(id=" << t.id
    //                         << ", pos=" << t.position().transpose()
    //                         << ", vel=" << t.velocity().transpose() << ")";
    //           }
    //           std::cerr << "  id_switches_so_far=" << id_counter.total() << "\n";
    //       }
    //   }
    //
    //   const auto t1 = std::chrono::steady_clock::now();
    //   const double runtime_ms =
    //       std::chrono::duration<double, std::milli>(t1 - t0).count();
    //
    //   // Flush + close CSVs explicitly BEFORE writing metrics.json (rule #9).
    //   tracks_csv.close();
    //   switches_csv.close();

    // ---- Write metrics.json LAST (rule #9 — atomic completion signal) ---
    //
    // YOUR CODE:
    //   std::ofstream metrics(fs::path(out_dir) / "metrics.json");
    //   metrics << "{\n"
    //           << "  \"frames\": "         << frames.size()           << ",\n"
    //           << "  \"total_track_rows\": " << total_track_count     << ",\n"
    //           << "  \"distinct_track_ids\": " << distinct_track_ids.size() << ",\n"
    //           << "  \"id_switches\": "    << id_counter.total()      << ",\n"
    //           << "  \"runtime_ms\": "     << runtime_ms              << ",\n"
    //           << "  \"solver\": \""       << solver_str              << "\",\n"
    //           << "  \"max_dist\": "       << max_dist                << ",\n"
    //           << "  \"max_misses\": "     << max_misses              << ",\n"
    //           << "  \"min_hits\": "       << min_hits                << ",\n"
    //           << "  \"dt\": "             << dt                      << ",\n"
    //           << "  \"process_noise\": "  << process_noise           << ",\n"
    //           << "  \"meas_noise\": "     << meas_noise              << "\n"
    //           << "}\n";

    // ---- Final sanity line for grep / batch supervisor ------------------
    //
    // YOUR CODE:
    //   std::cerr << "[tracker_runner] DONE  out=" << out_dir
    //             << "  frames=" << frames.size()
    //             << "  id_switches=" << id_counter.total()
    //             << "  runtime_ms=" << runtime_ms << "\n";


    std::unordered_map<int, const FrameDetections*> by_fid;                                                                                                                                       
    for (const auto& f : frames) by_fid[f.frame_id] = &f;                                                                                                                                         
                                                                                                                                                                                                    
    const int min_f = frames.front().frame_id;                                                                                                                                                    
    const int max_f = frames.back().frame_id;                 
                                                                                                                                                                                                    
    const auto t0 = std::chrono::steady_clock::now();                                                                                                                                             
    int total_track_count = 0;
    std::set<int> distinct_track_ids;                                                                                                                                                             
                                                                                                                                                                                                    
    for (int fid = min_f; fid <= max_f; ++fid) {
        auto it = by_fid.find(fid);                                                                                                                                                               
                                                                                                                                                                                                    
        // Empty defaults — used when this fid has no detection rows.                                                                                                                             
        static const FrameDetections kEmpty{};                                                                                                                                                    
        const FrameDetections& f = (it != by_fid.end()) ? *it->second : kEmpty;

        // Fix B: per-frame world←ego transform. Identity if --ego-poses
        // wasn't provided OR this fid is missing from the pose CSV (Lost
        // tracks fall back to ego-frame anchoring for that frame, no worse
        // than pre-Fix-B behavior).
        Eigen::Isometry2f T_world_ego = Eigen::Isometry2f::Identity();
        if (!ego_poses.empty()) {
            auto pit = ego_poses.find(fid);
            if (pit != ego_poses.end()) T_world_ego = pit->second;
        }

        std::vector<tracker::Track> published;
        if (use_appearance) {
            // P3-M13 path: build Detection vector with positions + features.
            std::vector<tracker::Detection> dets;
            dets.reserve(f.positions.size());
            for (size_t k = 0; k < f.positions.size(); ++k) {
                tracker::Detection d;
                d.position = f.positions[k];
                d.features = (k < f.features.size())
                                 ? f.features[k]
                                 : tracker::FeatureVector::Zero();
                dets.push_back(std::move(d));
            }
            published = tr.update_with_features(dets, f.class_ids, T_world_ego);
        } else {
            // M4 / M12 path — pure-position cost matrix.
            published = tr.update(f.positions, f.class_ids, T_world_ego);
        }                                                                                                                               
                                                                                                                                                                                                    
        // Per-track CSV rows                                                                                                                                                                     
        for (const auto& t : published) {                     
            const auto p = t.position();
            const auto v = t.velocity();                                                                                                                                                       
            tracks_csv << fid << "," << t.id << ","
                        << p.x() << "," << p.y() << ","                                                                                                                                            
                        << v.x() << "," << v.y() << ","                                                                                                                                            
                        << t.hits << "," << t.covariance_trace() << "\n";                                                                                                                       
            distinct_track_ids.insert(t.id);                                                                                                                                                      
            total_track_count += 1;                                                                                                                                                               
        }                                                                                                                                                                                         
                                                                                                                                                                                                    
        // Snapshot                                           
        if (snapshot_every > 0 && ((fid - min_f) % snapshot_every) == 0) {
            std::ostringstream name;                                                                                                                                                              
            name << "snapshots/frame_" << std::setw(5) << std::setfill('0')
                << fid << ".csv";                                                                                                                                                                
            std::ofstream snap(fs::path(out_dir) / name.str());
            snap << "track_id,x,y,vx,vy,age,cov_trace\n";                                                                                                                                         
            for (const auto& t : published) {                                                                                                                                                     
                snap << t.id << ","                                                                                                                                                               
                    << t.position().x() << "," << t.position().y() << ","                                                                                                                  
                    << t.velocity().x() << "," << t.velocity().y() << ","
                    << t.hits << "," << t.covariance_trace() << "\n";                                                                                                                         
            }
        }                                                                                                                                                                                         
                                                                
        // ID-switch counter (skips on empty-detection frames automatically since                                                                                                                 
        // associate_to_gt returns {} when frame.gt_track_ids is empty).
        const auto assoc = associate_to_gt(published, f);                                                                                                                                         
        const int new_sw = id_counter.update(fid, assoc, switches_csv);                                                                                                                           
                                                                                                                                                                                                    
        const bool is_first = (fid == min_f);                                                                                                                                                     
        const bool is_last  = (fid == max_f);                                                                                                                                                     
        if (verbose && (is_first || is_last)) {                                                                                                                                                   
            std::cerr << "[tracker_runner] frame " << fid                                                                                                                                         
                        << "  dets=" << f.positions.size()
                        << "  tracks=" << published.size()
                        << "  new_id_switches=" << new_sw                                                                                                                                           
                        << "  order=" << (swap_order ? "UpdateThenPredict" : "PredictThenUpdate");
            if (!published.empty()) {                                                                                                                                                             
                const auto& t = published.front();                                                                                                                                                
                std::cerr << "  sample(id=" << t.id
                            << ", pos=" << t.position().transpose()                                                                                                                              
                            << ", vel=" << t.velocity().transpose() << ")";
            }                                                                                                                                                                                     
            std::cerr << "  id_switches_so_far=" << id_counter.total() << "\n";
        }                                                                                                                                                                                         
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double runtime_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    tracks_csv.close();
    switches_csv.close();
    std::ofstream metrics(fs::path(out_dir) / "metrics.json");
    metrics << "{\n"
            << "  \"frames\": "         << frames.size()           << ",\n"
            << "  \"total_track_rows\": " << total_track_count     << ",\n"
            << "  \"distinct_track_ids\": " << distinct_track_ids.size() << ",\n"
            << "  \"id_switches\": "    << id_counter.total()      << ",\n"
            << "  \"runtime_ms\": "     << runtime_ms              << ",\n"
            << "  \"solver\": \""       << solver_str              << "\",\n"
            << "  \"max_dist\": "       << max_dist                << ",\n"
            << "  \"max_misses\": "     << max_misses              << ",\n"
            << "  \"min_hits\": "       << min_hits                << ",\n"
            << "  \"dt\": "             << dt                      << ",\n"
            << "  \"process_noise\": "  << process_noise           << ",\n"
            << "  \"meas_noise\": "     << meas_noise              << "\n"
            << "}\n";
    metrics.close();
    std::cerr << "[tracker_runner] DONE  out=" << out_dir
            << "  frames=" << frames.size()
            << "  id_switches=" << id_counter.total()
            << "  runtime_ms=" << runtime_ms << "\n";
    return 0;
}
