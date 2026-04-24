// =============================================================================
// accumulator_runner — CLI for P2-M3 world-map accumulation + ablations
// =============================================================================
//
// Mirrors slam_runner.cpp's flag style. Every ablation axis from p2m3_plan.md
// is a flag; one invocation = one run = one results directory.
//
// Usage:
//   ./accumulator_runner \
//     --pose-source slam \
//     --poses        data/poses_slam_manifold.csv \
//     --covariance   data/slam_covariance.csv      # optional
//     --lidar        data/RELLIS-3D/.../os1_cloud_node_kitti_bin/ \
//     --frames       2847 \
//     --update-rule  ema \
//     --alpha        0.3 \
//     --decay-rate   0.0 \
//     --cov-source   g2o \
//     --out          results/m3/slam_ema/ \
//     --publish-nats \
//     --verbose
//
// Produces under <out>/:
//   snapshots/frame_XXXXX.{png,csv}   — every K frames (K=10 default)
//   coverage.csv                      — frame_id, coverage_pct
//   trajectory.csv                    — per-frame pose used
//   pose_sigma.csv                    — per-frame sqrt(trace(P_pos))
//   final_grid.{png,csv}              — last frame snapshot
//   metrics.json                      — summary stats + config echo
//
// Ablation mapping:
//   A  pose-source           --pose-source {gps,icp,slam,carto}  +  --poses
//   B  update rule           --update-rule {ema,logodds,overwrite}
//   C  alpha sweep           --alpha 0.1|0.3|0.5|0.7
//   D  decay                 --decay-rate 0.0|0.01|0.1
//   E  covariance source     --cov-source {none,heuristic,g2o}   +  --covariance
// =============================================================================

#include "world_grid.hpp"
#include "traversability.hpp"
#include "pose_graph_slam.hpp"
#include "point_cloud_loader.hpp"   // provides KITTI .bin reader — header name TBD
#include "ransac_ground_seg.hpp"    // Phase 1 ground segmenter

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// CLI parsing helpers — mirror slam_runner.cpp:getArg / hasFlag
// -----------------------------------------------------------------------------
static std::string getArg(int argc, char** argv,
                          const std::string& flag,
                          const std::string& def) {
    // YOUR CODE:
    //   for (int i = 1; i < argc - 1; ++i)
    //     if (flag == argv[i]) return std::string(argv[i + 1]);
    //   return def;
    
    for(int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) {
            return std::string(argv[i + 1]);
        }
    }
    return def;
}

static bool hasFlag(int argc, char** argv, const std::string& flag) {
    // YOUR CODE:
    //   for (int i = 1; i < argc; ++i)
    //     if (flag == argv[i]) return true;
    //   return false;
    for(int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            return true;
        }
    }

    return false;
}

// -----------------------------------------------------------------------------
// String → enum
// -----------------------------------------------------------------------------
static UpdateRule parseUpdateRule(const std::string& s) {
    // YOUR CODE:
    //   if (s == "ema")       return UpdateRule::EMA;
    //   if (s == "logodds")   return UpdateRule::LogOdds;
    //   if (s == "overwrite") return UpdateRule::Overwrite;
    //   std::cerr << "[warn] unknown update-rule '" << s << "', defaulting to ema\n";
    //   return UpdateRule::EMA;
    if (s == "ema") {
        return UpdateRule::EMA;
    } else if (s == "logodds") {
        return UpdateRule::LogOdds;
    } else if (s == "overwrite") {
        return UpdateRule::Overwrite;
    } else {
        std::cerr << "[warn] unknown update-rule '" << s << "', defaulting to ema\n";
        return UpdateRule::EMA;
    }
    return UpdateRule::EMA;
}

// -----------------------------------------------------------------------------
// Covariance CSV loader (optional — only when --cov-source != none)
// -----------------------------------------------------------------------------
// Schema suggestion:
//   frame_id, P00, P01, ..., P55    (36 values of the 6x6 per row)
// Rotation block = top-left 3x3, translation block = bottom-right 3x3.
// MUST match the convention used by pose_graph_slam's covariance extractor.
static std::vector<Eigen::Matrix<double, 6, 6>>
loadCovariancesFromCSV(const std::string& path) {
    // YOUR CODE:
    //   if (path.empty()) return {};
    //   Open file, skip header, parse comma-separated rows:
    //     row[0] = frame_id (int, discarded or used for alignment)
    //     row[1..36] = 36 doubles, reshape to Matrix6d row-major.
    //   Return vector sized to poses.size(); pad with Identity*1e9 on missing rows.
    if (path.empty()) {
        return {};
    }

    std::vector<Eigen::Matrix<double, 6, 6>> covariances;
    std::ifstream csv_file(path);
    std::string line;
    std::getline(csv_file, line);
    while (std::getline(csv_file, line)) {
        std::istringstream iss(line);
        std::string frame_id_str;
        std::getline(iss, frame_id_str, ',');
        int frame_id = std::stoi(frame_id_str);

        std::vector<double> row_data;
        for (int i = 0; i < 36; ++i) {
            std::string value_str;
            std::getline(iss, value_str, ',');
            row_data.push_back(std::stod(value_str));
        }

        Eigen::Matrix<double, 6, 6> cov;
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                cov(r, c) = row_data[r * 6 + c];
            }
        }

        covariances.push_back(cov);
    }

    return covariances;


}

// Derive per-frame pose_sigma from covariance list.
//   pose_sigma[i] = sqrt(trace(P_translation_block))
// where translation block position depends on your pose_graph_slam convention.
static std::vector<double>
poseSigmaSeries(const std::vector<Eigen::Matrix<double, 6, 6>>& covs) {
    // YOUR CODE:
    //   out.resize(covs.size());
    //   for i: extract the 3x3 translation block (see pose_graph_slam.hpp note),
    //          out[i] = std::sqrt(P_trans.trace());
    //          Clamp to [0, 1e3] to avoid inf contamination downstream.
    
    std::vector<double> out(covs.size());                                                                                                                                 
    for (size_t i = 0; i < covs.size(); ++i) {
        out[i] = pose_graph::poseSigmaFromCovariance(covs[i]);                                                                                                            
    }                                                     
    return out; 
}

// -----------------------------------------------------------------------------
// Small output helpers (fill in as you implement)
// -----------------------------------------------------------------------------
static void writeCoverageRow(std::ofstream& ofs, int frame_id, double pct) {
    // YOUR CODE:
    //   ofs << frame_id << "," << std::fixed << std::setprecision(6) << pct << "\n";
    ofs << frame_id << "," << std::fixed << std::setprecision(6) << pct << "\n";
    
}

static void writeTrajectoryRow(std::ofstream& ofs, int frame_id, const Pose& p) {
    // YOUR CODE:
    //   Match format of savePosesToCSV (frame_id, tx, ty, tz, qx, qy, qz, qw).
    Eigen::Quaterniond q(p.R);                                                                                                                                            
    ofs << frame_id << ","                                                                                                                                                
        << std::fixed << std::setprecision(6)                                                                                                                             
        << p.t.x() << "," << p.t.y() << "," << p.t.z() << ","                                                                                                             
        << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n"; 
}

static void writeMetricsJson(const fs::path&       out_dir,
                             const WorldGrid&      grid,
                             const std::string&    pose_source,
                             const std::string&    update_rule,
                             int                   n_frames,
                             double                final_coverage,
                             double                runtime_sec) {
    // YOUR CODE:
    //   std::ofstream json(out_dir / "metrics.json");
    //   json << "{\n";
    //   json << "  \"pose_source\": \"" << pose_source << "\",\n";
    //   json << "  \"update_rule\": \"" << update_rule << "\",\n";
    //   json << "  \"n_frames\": "      << n_frames << ",\n";
    //   json << "  \"final_coverage\": "<< final_coverage << ",\n";
    //   json << "  \"runtime_sec\": "   << runtime_sec << ",\n";
    //   json << "  \"grid_rows\": "     << grid.rows() << ",\n";
    //   json << "  \"grid_cols\": "     << grid.cols() << ",\n";
    //   json << "  \"resolution\": "    << grid.resolution() << "\n";
    //   json << "}\n";
    //
    // Consider also dumping cfg echo (alpha, decay, clamps, k) so each run's
    // metrics.json is self-describing for later plot_ablation.py runs.
    std::ofstream json(out_dir / "metrics.json");
    json << "{\n";
    json << "  \"pose_source\": \"" << pose_source << "\",\n";
    json << "  \"update_rule\": \"" << update_rule << "\",\n";
    json << "  \"n_frames\": "      << n_frames << ",\n";
    json << "  \"final_coverage\": "<< final_coverage << ",\n";
    json << "  \"runtime_sec\": "   << runtime_sec << ",\n";
    json << "  \"grid_rows\": "     << grid.rows() << ",\n";
    json << "  \"grid_cols\": "     << grid.cols() << ",\n";
    json << "  \"resolution\": "    << grid.resolution() << "\n";
    json << "}\n";
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // YOUR CODE — the full pipeline. Outline below; fill section by section.
    //
    // ──────────── 1) Parse flags ────────────
    //   std::string pose_source = getArg(argc, argv, "--pose-source", "slam");
    //   std::string poses_path  = getArg(argc, argv, "--poses",       "");
    //   std::string cov_path    = getArg(argc, argv, "--covariance",  "");
    //   std::string lidar_dir   = getArg(argc, argv, "--lidar",       "");
    //   int         n_frames    = std::stoi(getArg(argc, argv, "--frames", "0"));
    //   std::string rule_str    = getArg(argc, argv, "--update-rule", "ema");
    //   double      alpha       = std::stod(getArg(argc, argv, "--alpha",      "0.3"));
    //   double      decay_rate  = std::stod(getArg(argc, argv, "--decay-rate", "0.0"));
    //   std::string cov_source  = getArg(argc, argv, "--cov-source",  "none");
    //   std::string out_str     = getArg(argc, argv, "--out",         "results/m3/run/");
    //   int         snapshot_k  = std::stoi(getArg(argc, argv, "--snapshot-every", "10"));
    //   bool        do_nats     = hasFlag (argc, argv, "--publish-nats");
    //   bool        verbose     = hasFlag (argc, argv, "--verbose");
    //
    //   Validate required flags (poses_path, lidar_dir).
    
    std::string pose_source = getArg(argc, argv, "--pose-source", "slam");
    std::string poses_path  = getArg(argc, argv, "--poses",       "");
    std::string cov_path    = getArg(argc, argv, "--covariance",  "");
    std::string lidar_dir   = getArg(argc, argv, "--lidar",       "");
    int n_frames = std::stoi(getArg(argc, argv, "--frames", "0"));
    std::string rule_str = getArg(argc, argv, "--update-rule", "ema");
    double alpha = std::stod(getArg(argc, argv, "--alpha", "0.3"));
    double decay_rate = std::stod(getArg(argc, argv, "--decay-rate", "0.0"));
    std::string cov_source = getArg(argc, argv, "--cov-source", "none");
    std::string out_str = getArg(argc, argv, "--out", "results/m3/run/");
    int snapshot_k = std::stoi(getArg(argc, argv, "--snapshot-every", "10"));
    bool do_nats = hasFlag(argc, argv, "--publish-nats");
    bool verbose = hasFlag(argc, argv, "--verbose");

    if (poses_path.empty()) {
        std::cerr << "Error: --poses is required\n";
        return 1;
    }
    if (lidar_dir.empty()) {
        std::cerr << "Error: --lidar is required\n";
        return 1;
    }

    // ──────────── 2) Build config ────────────
    //   WorldGridConfig cfg;
    //   cfg.update_rule = parseUpdateRule(rule_str);
    //   cfg.alpha       = alpha;
    //   cfg.decay_rate  = decay_rate;
    //   // Override extent from pipeline_config.yaml once config loader lands.
    //   // For now: rely on WorldGridConfig defaults (±100m, 0.5m).
    //
    //   WorldGrid world(cfg);
    //

    WorldGridConfig cfg;
    cfg.update_rule = parseUpdateRule(rule_str);
    cfg.alpha = alpha;
    cfg.decay_rate = decay_rate;
    // Extent sized for full RELLIS sequence 00: final pose at (-208, 119) from
    // origin, so ±250m covers the whole trajectory with margin.
    // 500×500m at 0.5m resolution = 1000×1000 cells ≈ 40 MB of WorldCells.
    cfg.x_min = -250.0;
    cfg.x_max =  250.0;
    cfg.y_min = -250.0;
    cfg.y_max =  250.0;

    WorldGrid world(cfg);
    // ──────────── 3) Load poses + covariance ────────────
    //   auto [poses, timestamps] = loadPosesFromCSV(poses_path);
    //   if (poses.empty()) { std::cerr << "no poses\n"; return 1; }
    //   if (n_frames == 0 || n_frames > (int)poses.size()) n_frames = (int)poses.size();
    //
    //   std::vector<double> pose_sigma(poses.size(), 0.0);
    //   if (cov_source != "none") {
    //     auto covs = loadCovariancesFromCSV(cov_path);
    //     pose_sigma = poseSigmaSeries(covs);
    //   }
    //
    auto [poses, timestamps] = pose_graph::loadPosesFromCSV(poses_path);
    if (poses.empty()) {
        std::cerr << "no poses\n";
        return 1;
    }
    if (n_frames == 0 || n_frames > (int)poses.size()) {
        n_frames = (int)poses.size();
    }

    std::vector<double> pose_sigma(poses.size(), 0.0);
    if (cov_source != "none") {
        auto covs = loadCovariancesFromCSV(cov_path);
        pose_sigma = poseSigmaSeries(covs);
        if (pose_sigma.size() < poses.size()) {
            pose_sigma.resize(poses.size(), 0.0);   // unknown-sigma fallback
        } 
    }

    // ──────────── 4) Prepare output dir ────────────
    //   fs::path out_dir(out_str);
    //   fs::create_directories(out_dir / "snapshots");
    //   std::ofstream coverage_ofs (out_dir / "coverage.csv");    coverage_ofs << "frame_id,coverage_pct\n";
    //   std::ofstream traj_ofs     (out_dir / "trajectory.csv");  traj_ofs     << "frame_id,tx,ty,tz,qx,qy,qz,qw\n";
    //   std::ofstream sigma_ofs    (out_dir / "pose_sigma.csv");  sigma_ofs    << "frame_id,pose_sigma\n";
    //
    fs::path out_dir(out_str);
    fs::create_directories(out_dir / "snapshots");
    std::ofstream coverage_ofs(out_dir / "coverage.csv");
    coverage_ofs << "frame_id,coverage_pct\n";
    std::ofstream traj_ofs(out_dir / "trajectory.csv");
    traj_ofs << "frame_id,tx,ty,tz,qx,qy,qz,qw\n";
    std::ofstream sigma_ofs(out_dir / "pose_sigma.csv");
    sigma_ofs << "frame_id,pose_sigma\n";
    // ──────────── 5) Instantiate per-frame tools ────────────
    //   TraversabilityGrid local_grid(/* params from pipeline_config */);
    //   // PointCloudLoader loader(lidar_dir);
    //   // RansacGroundSeg ground_seg(...);
    //
    TraversabilityGrid local_grid(GridParams{}, VehicleKinematics{});
    RANSACParams ransac_params;
    ransac_params.max_iterations = 200;      // from default 1000 → 5× faster                                                                                                 
    ransac_params.distance_threshold = 0.15f;                                                                                                                                 
    ransac_params.min_inliers = 50;
    auto frame_path = [&](int i) {                                                                                                                                            
        std::ostringstream ss;                                                                                                                                                
        ss << lidar_dir << "/" << std::setw(6) << std::setfill('0') << i << ".bin";                                                                                           
        return ss.str();                                                                                                                                                      
    }; 



    // ──────────── 6) (Optional) NATS bringup ────────────
    //   // if (do_nats) { adapter.connect("nats://localhost:4222"); }
    //   // Adapter is Python — this block likely becomes a ZeroMQ-style shim or
    //   // shells out to a Python worker. Revisit when nats_adapter.py lands.
    //
    // ──────────── 7) Main loop ────────────
    //   std::vector<Eigen::Vector2d> traj_xy;
    //   auto t_start = std::chrono::steady_clock::now();
    //   for (int i = 0; i < n_frames; ++i) {
    //     // (a) Load point cloud
    //     //   std::string bin_path = bin_for_frame(lidar_dir, i);
    //     //   auto cloud = loader.load(bin_path);
    //     //
    //     // (b) Ground segment
    //     //   auto ground_pts = ground_seg.fit(cloud);
    //     //
    //     // (c) Single-frame traversability
    //     //   local_grid.compute(ground_pts);
    //     //
    //     // (d) Accumulate
    //     //   world.update(local_grid, poses[i], pose_sigma[i], timestamps[i]);
    //     //
    //     // (e) Record trajectory + logs
    //     //   traj_xy.emplace_back(poses[i].t.x(), poses[i].t.y());
    //     //   writeTrajectoryRow(traj_ofs, i, poses[i]);
    //     //   sigma_ofs << i << "," << pose_sigma[i] << "\n";
    //     //
    //     // (f) Coverage
    //     //   double cov_pct = world.coveragePercent(traj_xy);
    //     //   writeCoverageRow(coverage_ofs, i, cov_pct);
    //     //
    //     // (g) Snapshot every K frames
    //     //   if (i % snapshot_k == 0) {
    //     //     std::ostringstream name; name << "snapshots/frame_"
    //     //                                   << std::setw(5) << std::setfill('0') << i;
    //     //     world.saveSnapshot((out_dir / name.str()).string());
    //     //   }
    //     //
    //     // (h) NATS publish (optional)
    //     //   if (do_nats) { /* serialize BEVGrid proto + publish */ }
    //     //
    //     // (i) Verbose log
    //     //   if (verbose && i % 100 == 0) {
    //     //     std::cout << "[" << i << "/" << n_frames << "] "
    //     //               << "cov=" << cov_pct << " sigma=" << pose_sigma[i] << "\n";
    //     //   }
    //   }
    //   auto t_end = std::chrono::steady_clock::now();
    //   double runtime = std::chrono::duration<double>(t_end - t_start).count();
    //
    // Accumulate trajectory across frames — used by coveragePercent and by
    // the final metrics call. Scoped OUTSIDE the loop so it grows each iter.
    std::vector<Eigen::Vector2d> traj_xy;

    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n_frames; ++i) {
        // (a) Load point cloud
        // std::string bin_path = bin_for_frame(lidar_dir, i);
        // auto cloud = loader.load(bin_path);

        std::ostringstream path_ss;
        path_ss << lidar_dir << "/" << std::setw(6) << std::setfill('0') << i << ".bin";                                                                                      
        std::vector<Eigen::Vector3f> cloud = load_bin(path_ss.str());                                                                                                         
        if (cloud.empty()) {                                                                                                                                                  
            if (verbose) std::cerr << "[warn] frame " << i << ": empty cloud\n";                                                                                              
            continue;                                                                                                                                                         
        }  
        std::vector<Eigen::Vector3f> subsampled;
        subsampled.reserve(cloud.size() / 5);                                                                                                                                 
        for (size_t k = 0; k < cloud.size(); k += 5) subsampled.push_back(cloud[k]);

        // (b) Ground segment
        // auto ground_pts = ground_seg.fit(cloud);
        SegmentationResult seg = segment_ground(subsampled, ransac_params);

        // (c) Single-frame traversability
        // local_grid.compute(ground_pts);
        local_grid.compute(seg.ground_points);

        // (d) Accumulate
        world.update(local_grid, poses[i], pose_sigma[i], timestamps[i]);

        // (e) Record trajectory + logs
        traj_xy.emplace_back(poses[i].t.x(), poses[i].t.y());
        writeTrajectoryRow(traj_ofs, i, poses[i]);
        sigma_ofs << i << "," << pose_sigma[i] << "\n";

        // (f) Coverage — denominator grows as hull expands each frame.
        double cov_pct = world.coveragePercent(traj_xy);
        writeCoverageRow(coverage_ofs, i, cov_pct);
        coverage_ofs.flush();     // ← flush every frame so Ctrl-C doesn't lose data
        traj_ofs.flush();                                                                                                                                                     
        sigma_ofs.flush();

        // (g) Snapshot every K frames. Guard against snapshot_k == 0 (UB from i%0).
        if (snapshot_k > 0 && i % snapshot_k == 0) {
            std::ostringstream name;
            name << "snapshots/frame_" << std::setw(5) << std::setfill('0') << i;
            world.saveSnapshot((out_dir / name.str()).string());
        }

        // (h) NATS publish (optional)
        // if (do_nats) { /* serialize BEVGrid proto + publish */ }

        // (i) Verbose log
        if (verbose && i % 10 == 0) {
                  int local_cells = 0;
        for (int ix = 0; ix < local_grid.rows(); ++ix)                                                                                                                        
            for (int iy = 0; iy < local_grid.cols(); ++iy)                                                                                                                    
                if (local_grid.at(ix, iy).point_count > 0) ++local_cells;                                                                                                     
                                                                                                                                                                                
        std::cout << "[" << i << "/" << n_frames << "]"       
                    << "  cloud=" << cloud.size()                                                                                                                               
                    << "  subsampled=" << subsampled.size()                                                                                                                     
                    << "  ground_pts=" << seg.ground_points.size()
                    << "  local_cells=" << local_cells                                                                                                                          
                    << "  cov=" << cov_pct                      
                    << "  sigma=" << pose_sigma[i] << "\n";   
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    const double runtime = std::chrono::duration<double>(t_end - t_start).count();
    // ──────────── 8) Final artifacts ────────────
    //   world.saveSnapshot((out_dir / "final_grid").string());
    //   double final_cov = world.coveragePercent(traj_xy);
    //   writeMetricsJson(out_dir, world, pose_source, rule_str,
    //                    n_frames, final_cov, runtime);
    
    world.saveSnapshot((out_dir / "final_grid").string());
    const double final_cov = world.coveragePercent(traj_xy);
    writeMetricsJson(out_dir, world, pose_source, rule_str,
                     n_frames, final_cov, runtime);
    
    // ──────────── 9) Shutdown ────────────
    //   coverage_ofs.close(); traj_ofs.close(); sigma_ofs.close();
    //   // if (do_nats) adapter.close();
    //   std::cout << "done. coverage=" << final_cov
    //             << " runtime=" << runtime << "s\n";
    //   return 0;

    coverage_ofs.close();
    traj_ofs.close();
    sigma_ofs.close();
    // if (do_nats) adapter.close();
    std::cout << "done. coverage=" << final_cov
              << " runtime=" << runtime << "s\n";
    return 0;

}
