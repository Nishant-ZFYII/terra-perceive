// =============================================================================
// traversability_runner — CLI for P2-M6 confidence-mode ablations.
// =============================================================================
//
// Drives the m12 (probabilistic traversability) ablation. Mirrors
// accumulator_runner.cpp's CLI style so existing helper scripts can be reused.
// One invocation = one run = one results directory.
//
// Usage:
//   ./traversability_runner \
//     --lidar              data/RELLIS-3D/.../os1_cloud_node_kitti_bin/ \
//     --frames             2847 \
//     --confidence-mode    probabilistic           # heuristic | probabilistic
//     --sigma-0            0.01 \
//     --sigma-k            0.0001 \
//     --out                results_m6/trav_probabilistic/ \
//     --snapshot-every     50 \
//     --verbose
//
// Per-frame snapshot CSV columns (one row per cell with point_count > 0):
//   ix, iy, x_center, y_center, range_from_sensor, point_count,
//   risk, confidence, sigma_r, lambda_min, lambda_max
//
// metrics.json is written atomically last and is the resume signal for
// run_m6_ablations.sh (skip-if-exists).
// =============================================================================

#include "traversability.hpp"
#include "ransac_ground_seg.hpp"
#include "point_cloud_loader.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// --- CLI helpers — mirror accumulator_runner.cpp:getArg/hasFlag ---
static std::string getArg(int argc, char** argv,
                          const std::string& flag,
                          const std::string& def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::string(argv[i + 1]);
    }
    return def;
}

static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

static ConfidenceMode parseMode(const std::string& s) {
    if (s == "probabilistic") return ConfidenceMode::Probabilistic;
    if (s == "heuristic")     return ConfidenceMode::Heuristic;
    std::cerr << "[warn] unknown confidence-mode '" << s
              << "', defaulting to heuristic\n";
    return ConfidenceMode::Heuristic;
}

static std::vector<fs::path> listLidarFrames(const fs::path& dir, int limit) {
    std::vector<fs::path> out;
    if (!fs::is_directory(dir)) {
        std::cerr << "[error] --lidar path is not a directory: " << dir << "\n";
        return out;
    }
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (limit > 0 && static_cast<int>(out.size()) > limit) {
        out.resize(limit);
    }
    return out;
}

static void writeSnapshotCsv(const fs::path& csv_path,
                             const TraversabilityGrid& grid,
                             const LidarNoiseModel& nm) {
    std::ofstream f(csv_path);
    f << "ix,iy,x_center,y_center,range,point_count,risk,confidence,sigma_r,lambda_min,lambda_max\n";
    f << std::fixed << std::setprecision(6);
    const auto& gp = grid.grid_params();
    for (int ix = 0; ix < grid.rows(); ++ix) {
        for (int iy = 0; iy < grid.cols(); ++iy) {
            const auto& cell = grid.at(ix, iy);
            if (cell.point_count <= 0) continue;
            float x = gp.x_min + (ix + 0.5f) * gp.resolution;
            float y = gp.y_min + (iy + 0.5f) * gp.resolution;
            float sigma_r = lidar_sigma(cell.range_from_sensor, nm);
            // Lambda values are not exported by the public CellFeatures API;
            // emit zeros and rely on the snapshot's risk/confidence/range
            // columns. (Probing the eigenvalues per-cell would require a
            // public accessor; deferred — not load-bearing for the
            // confidence-vs-range plotting scripts.)
            f << ix << "," << iy << "," << x << "," << y << ","
              << cell.range_from_sensor << "," << cell.point_count << ","
              << cell.risk << "," << cell.confidence << ","
              << sigma_r << ",0,0\n";
        }
    }
}

static void writeMetricsJson(const fs::path& out_dir,
                             const std::string& mode_str,
                             const LidarNoiseModel& nm,
                             const std::string& lidar_dir,
                             int n_frames_run,
                             double mean_observed_cells,
                             double mean_confidence,
                             double mean_risk,
                             double runtime_sec) {
    fs::path tmp = out_dir / "metrics.json.tmp";
    {
        std::ofstream j(tmp);
        j << std::fixed << std::setprecision(6);
        j << "{\n";
        j << "  \"confidence_mode\": \"" << mode_str << "\",\n";
        j << "  \"sigma_0\": " << nm.sigma_0 << ",\n";
        j << "  \"sigma_k\": " << nm.k << ",\n";
        j << "  \"lidar_dir\": \"" << lidar_dir << "\",\n";
        j << "  \"n_frames\": " << n_frames_run << ",\n";
        j << "  \"mean_observed_cells\": " << mean_observed_cells << ",\n";
        j << "  \"mean_confidence\": " << mean_confidence << ",\n";
        j << "  \"mean_risk\": " << mean_risk << ",\n";
        j << "  \"runtime_sec\": " << runtime_sec << "\n";
        j << "}\n";
    }
    fs::rename(tmp, out_dir / "metrics.json");
}

int main(int argc, char** argv) {
    if (argc == 1 || hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        std::cout <<
            "traversability_runner — P2-M6 confidence-mode CLI\n"
            "Flags:\n"
            "  --lidar <dir>             directory of *.bin LiDAR frames\n"
            "  --frames N                cap on frames (0 = all)\n"
            "  --confidence-mode MODE    heuristic | probabilistic\n"
            "  --sigma-0 F               LiDAR noise floor (default 0.01)\n"
            "  --sigma-k F               LiDAR noise quadratic factor (default 0.0001)\n"
            "  --out <dir>               results directory\n"
            "  --snapshot-every N        write snapshot every N frames (default 50)\n"
            "  --verbose                 per-frame stdout\n";
        return 0;
    }

    std::string lidar_dir   = getArg(argc, argv, "--lidar", "");
    int         n_frames    = std::stoi(getArg(argc, argv, "--frames", "0"));
    std::string mode_str    = getArg(argc, argv, "--confidence-mode", "heuristic");
    float       sigma_0     = std::stof(getArg(argc, argv, "--sigma-0", "0.01"));
    float       sigma_k     = std::stof(getArg(argc, argv, "--sigma-k", "0.0001"));
    std::string out_str     = getArg(argc, argv, "--out", "results_m6/trav_run/");
    int         snap_k      = std::stoi(getArg(argc, argv, "--snapshot-every", "50"));
    bool        verbose     = hasFlag(argc, argv, "--verbose");

    if (lidar_dir.empty()) {
        std::cerr << "[error] --lidar is required\n";
        return 2;
    }

    fs::path out_dir(out_str);
    fs::create_directories(out_dir / "snapshots");

    GridParams gp;
    gp.confidence_mode  = parseMode(mode_str);
    gp.noise_model.sigma_0 = sigma_0;
    gp.noise_model.k       = sigma_k;
    VehicleKinematics vk;
    TraversabilityGrid grid(gp, vk);

    auto frames = listLidarFrames(lidar_dir, n_frames);
    if (frames.empty()) {
        std::cerr << "[error] no .bin frames found under " << lidar_dir << "\n";
        return 3;
    }

    std::cout << "[traversability_runner] mode=" << mode_str
              << " sigma_0=" << sigma_0 << " sigma_k=" << sigma_k
              << " frames=" << frames.size()
              << " out=" << out_dir.string() << "\n";

    RANSACParams rp;
    auto t_start = std::chrono::high_resolution_clock::now();
    int n_run = 0;
    double sum_observed_cells = 0.0;
    double sum_confidence = 0.0;
    double sum_risk = 0.0;
    long   conf_count = 0;
    long   risk_count = 0;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        auto cloud = load_bin(frames[fi].string());
        if (cloud.empty()) {
            if (verbose) std::cerr << "[skip] empty cloud at " << frames[fi] << "\n";
            continue;
        }
        SegmentationResult seg = segment_ground(cloud, rp);
        // Use ground points for traversability scoring (they fill cells with
        // the trail surface — same convention as P1/M3 pipeline_runner).
        grid.compute(seg.ground_points);

        int observed_cells = 0;
        for (int ix = 0; ix < grid.rows(); ++ix) {
            for (int iy = 0; iy < grid.cols(); ++iy) {
                const auto& cell = grid.at(ix, iy);
                if (cell.point_count > 0) {
                    ++observed_cells;
                    sum_confidence += cell.confidence;
                    ++conf_count;
                    sum_risk += cell.risk;
                    ++risk_count;
                }
            }
        }
        sum_observed_cells += static_cast<double>(observed_cells);

        if (snap_k > 0 && (static_cast<int>(fi) % snap_k == 0
                           || fi + 1 == frames.size())) {
            std::ostringstream name;
            name << "frame_" << std::setw(5) << std::setfill('0') << fi << ".csv";
            writeSnapshotCsv(out_dir / "snapshots" / name.str(), grid, gp.noise_model);
        }

        if (verbose && (fi % 100 == 0 || fi + 1 == frames.size())) {
            std::cout << "[" << fi + 1 << "/" << frames.size() << "] cells="
                      << observed_cells << " mean_conf="
                      << (conf_count > 0 ? sum_confidence / conf_count : 0.0)
                      << "\n";
        }
        ++n_run;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double runtime_sec = std::chrono::duration<double>(t_end - t_start).count();

    double mean_observed = (n_run > 0) ? sum_observed_cells / n_run : 0.0;
    double mean_conf = (conf_count > 0) ? sum_confidence / conf_count : 0.0;
    double mean_risk = (risk_count > 0) ? sum_risk / risk_count : 0.0;

    writeMetricsJson(out_dir, mode_str, gp.noise_model, lidar_dir,
                     n_run, mean_observed, mean_conf, mean_risk, runtime_sec);

    std::cout << "[done] frames=" << n_run
              << " mean_observed_cells=" << mean_observed
              << " mean_confidence=" << mean_conf
              << " mean_risk=" << mean_risk
              << " runtime=" << runtime_sec << "s\n";
    return 0;
}
