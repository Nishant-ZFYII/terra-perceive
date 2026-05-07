// =============================================================================
// dbscan_cli — thin CLI around tracker::dbscan() for the Ablation G sweep.
// =============================================================================
//
// Reads a CSV of 3D points, runs DBSCAN, writes the same points back with
// an appended cluster_id column (-1 for noise). The whole purpose of this
// binary is to drive the parameter sweep from a shell script while keeping
// the actual algorithm in the production C++ library.
//
// Input CSV schema:
//   x,y,z          (one row per point; header line is read and discarded)
//
// Output CSV schema:
//   x,y,z,cluster_id   (-1 = noise; else 0-based cluster index)
//
// Usage:
//   ./dbscan_cli \
//       --in       results_m4/obstacles/obstacles_000050.csv \
//       --eps      0.5 \
//       --min-points 10 \
//       --out      results_m4/ablation_g/eps_0.5_mp_10/clusters_000050.csv
//
// Stdout: one line of summary —
//   "[dbscan_cli] N=<input_count>  K=<num_clusters>  noise=<noise_count>  ms=<runtime>"
// =============================================================================

#include "dbscan.hpp"

#include <Eigen/Core>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string getArg(int argc, char** argv,
                          const std::string& flag, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return std::string(argv[i + 1]);
    return def;
}

static std::vector<Eigen::Vector3f> load_xyz_csv(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    std::vector<Eigen::Vector3f> out;
    std::string header;
    std::getline(f, header);                   // discard header line
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Eigen::Vector3f p;
        char comma;
        ss >> p.x() >> comma >> p.y() >> comma >> p.z();
        out.push_back(p);
    }
    return out;
}

int main(int argc, char** argv) {
    const std::string in_csv     = getArg(argc, argv, "--in", "");
    const std::string out_csv    = getArg(argc, argv, "--out", "");
    const float       eps        = std::stof(getArg(argc, argv, "--eps", "0.5"));
    const int         min_points = std::stoi(getArg(argc, argv, "--min-points", "10"));

    if (in_csv.empty() || out_csv.empty()) {
        std::cerr << "ERROR: --in and --out are required\n";
        return 1;
    }

    fs::create_directories(fs::path(out_csv).parent_path());

    const auto pts = load_xyz_csv(in_csv);
    const auto t0  = std::chrono::steady_clock::now();
    const auto clusters = tracker::dbscan(pts, eps, min_points);
    const auto t1  = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Build per-point cluster_id lookup: -1 default (noise), else cluster index.
    std::vector<int> cluster_id(pts.size(), -1);
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        for (int idx : clusters[c]) cluster_id[idx] = c;
    }

    int noise_count = 0;
    for (int c : cluster_id) if (c < 0) ++noise_count;

    {
        std::ofstream f(out_csv);
        f << "x,y,z,cluster_id\n";
        for (size_t i = 0; i < pts.size(); ++i) {
            f << pts[i].x() << "," << pts[i].y() << "," << pts[i].z()
              << "," << cluster_id[i] << "\n";
        }
    }

    std::cerr << "[dbscan_cli] N=" << pts.size()
              << "  K=" << clusters.size()
              << "  noise=" << noise_count
              << "  ms=" << ms << "\n";
    return 0;
}
