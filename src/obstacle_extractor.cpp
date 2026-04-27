// =============================================================================
// obstacle_extractor — CLI for dumping per-frame obstacle point clouds from
// RELLIS-3D LiDAR. Used as the upstream stage of Ablation G (DBSCAN sweep)
// and the M4 closing-hero MP4 (tracker on RELLIS).
// =============================================================================
//
// Pipeline per frame:
//   1. load_bin(<frame>.bin)              — read KITTI-format LiDAR
//   2. sector_segment_ground(cloud, ...)  — Phase-1 sector RANSAC
//   3. write obstacles_NNNNN.csv          — x,y,z per obstacle point
//
// Reads from --lidar dir, writes per-frame CSVs to --out dir. The output
// schema matches what `dbscan_cli` expects (x,y,z, one row per point) so
// the two binaries chain via the filesystem.
//
// Usage:
//   ./obstacle_extractor \
//       --lidar  data/extracted_frames \
//       --frame-start 0 --frame-end 99 \
//       --out    results_m4/obstacles/ \
//       --verbose
//
// Outputs under <out>/:
//   obstacles_00000.csv ... obstacles_00099.csv   one CSV per processed frame
//   metrics.json                                  summary (count, mean obstacle
//                                                 points per frame, runtime)
//
// Per ablation pre-flight rules:
//   #7 sanity print on FIRST and LAST frame: input cloud size, obstacle count
//   #9 metrics.json LAST after CSVs flushed
// =============================================================================

#include "ransac_ground_seg.hpp"
#include "point_cloud_loader.hpp"

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
// Same CLI helpers as the other runners.
// -----------------------------------------------------------------------------
static std::string getArg(int argc, char** argv,
                          const std::string& flag, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return std::string(argv[i + 1]);
    return def;
}
static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (flag == argv[i]) return true;
    return false;
}

// Frame_id padded to 6 digits — matches RELLIS naming convention.
static std::string frame_filename(int fid, const std::string& ext) {
    std::ostringstream s;
    s << std::setw(6) << std::setfill('0') << fid << ext;
    return s.str();
}

// Write x,y,z CSV for a single obstacle cloud.
static void write_obstacles_csv(const fs::path& out_path,
                                const std::vector<Eigen::Vector3f>& pts) {
    std::ofstream f(out_path);
    f << "x,y,z\n";
    for (const auto& p : pts) {
        f << p.x() << "," << p.y() << "," << p.z() << "\n";
    }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    // ---- Parse CLI ------------------------------------------------------
    const std::string lidar_dir   = getArg(argc, argv, "--lidar", "data/extracted_frames");
    const std::string out_dir     = getArg(argc, argv, "--out",   "results_m4/obstacles/");
    const int frame_start         = std::stoi(getArg(argc, argv, "--frame-start", "0"));
    const int frame_end           = std::stoi(getArg(argc, argv, "--frame-end",   "99"));
    const float dist_thresh       = std::stof(getArg(argc, argv, "--ransac-dist", "0.15"));
    const int   max_iter          = std::stoi(getArg(argc, argv, "--ransac-iter", "200"));
    const int   min_inliers       = std::stoi(getArg(argc, argv, "--ransac-min-inliers", "50"));
    const float sector_size       = std::stof(getArg(argc, argv, "--sector-size", "5.0"));
    const bool  verbose           = hasFlag(argc, argv, "--verbose");

    fs::create_directories(out_dir);

    if (verbose) {
        std::cerr << "[obstacle_extractor] config:\n"
                  << "  lidar         = " << lidar_dir << "\n"
                  << "  frames        = " << frame_start << ".." << frame_end << "\n"
                  << "  ransac_dist   = " << dist_thresh << "\n"
                  << "  ransac_iter   = " << max_iter << "\n"
                  << "  ransac_min    = " << min_inliers << "\n"
                  << "  sector_size   = " << sector_size << "\n"
                  << "  out           = " << out_dir << "\n";
    }

    // ---- Configure RANSAC + sector params ---------------------------------
    RANSACParams ransac_params;
    ransac_params.max_iterations    = max_iter;
    ransac_params.distance_threshold = dist_thresh;
    ransac_params.min_inliers        = min_inliers;

    SectorParams sector_params;
    sector_params.sector_size_x = sector_size;
    sector_params.sector_size_y = sector_size;

    // ---- Per-frame loop ---------------------------------------------------
    //
    // YOUR CODE: walk frame_start..frame_end, load each .bin, run sector
    // RANSAC, write obstacles CSV. Skeleton:
    //
    //   const auto t0 = std::chrono::steady_clock::now();
    //   long total_input_pts    = 0;
    //   long total_obstacle_pts = 0;
    //   int  frames_processed   = 0;
    //
    //   for (int fid = frame_start; fid <= frame_end; ++fid) {
    //       const fs::path bin_path  = fs::path(lidar_dir) / frame_filename(fid, ".bin");
    //       const fs::path csv_path  = fs::path(out_dir)  /
    //                                   ("obstacles_" + frame_filename(fid, ".csv"));
    //
    //       if (!fs::exists(bin_path)) {
    //           if (verbose)
    //               std::cerr << "[obstacle_extractor] skip missing " << bin_path << "\n";
    //           continue;
    //       }
    //
    //       // 1. Load
    //       const auto cloud = load_bin(bin_path.string());
    //       total_input_pts += static_cast<long>(cloud.size());
    //
    //       // 2. Segment
    //       const auto seg = sector_segment_ground(cloud, ransac_params, sector_params);
    //       total_obstacle_pts += static_cast<long>(seg.obstacle_points.size());
    //
    //       // 3. Write
    //       write_obstacles_csv(csv_path, seg.obstacle_points);
    //
    //       // 4. Sanity print on first AND last frame (rule #7)
    //       const bool is_first = (fid == frame_start);
    //       const bool is_last  = (fid == frame_end);
    //       if (verbose && (is_first || is_last)) {
    //           std::cerr << "[obstacle_extractor] frame " << fid
    //                     << "  cloud=" << cloud.size()
    //                     << "  obstacles=" << seg.obstacle_points.size()
    //                     << "  ground=" << seg.ground_points.size()
    //                     << "  reliable_sectors=" << seg.num_reliable_sectors
    //                     << "/" << seg.num_sectors << "\n";
    //       }
    //
    //       frames_processed += 1;
    //   }
    //
    //   const auto t1 = std::chrono::steady_clock::now();
    //   const double runtime_ms =
    //       std::chrono::duration<double, std::milli>(t1 - t0).count();
    //
    // Notes:
    //   - load_bin throws on read failure. The fs::exists guard above catches
    //     the most common case (missing file). For deeper errors, add a
    //     try/catch around the load+segment block per frame.
    //   - sector_segment_ground may emit zero obstacle_points on a frame
    //     where the entire cloud fits one ground plane. That's fine — write
    //     a CSV with just the header and continue.

    // ---- Write metrics.json LAST (rule #9) --------------------------------
    //
    // YOUR CODE:
    //   std::ofstream metrics(fs::path(out_dir) / "metrics.json");
    //   metrics << "{\n"
    //           << "  \"frames_processed\": " << frames_processed     << ",\n"
    //           << "  \"total_input_pts\": "  << total_input_pts      << ",\n"
    //           << "  \"total_obstacle_pts\": " << total_obstacle_pts << ",\n"
    //           << "  \"mean_obstacles_per_frame\": "
    //           << (frames_processed > 0
    //                 ? static_cast<double>(total_obstacle_pts) / frames_processed
    //                 : 0.0)
    //           << ",\n"
    //           << "  \"runtime_ms\": " << runtime_ms << "\n"
    //           << "}\n";
    //
    //   std::cerr << "[obstacle_extractor] DONE  out=" << out_dir
    //             << "  frames=" << frames_processed
    //             << "  mean_obstacles_per_frame="
    //             << (frames_processed > 0
    //                   ? static_cast<double>(total_obstacle_pts) / frames_processed
    //                   : 0.0)
    //             << "  runtime_ms=" << runtime_ms << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    long total_input_pts    = 0;
    long total_obstacle_pts = 0;
    int  frames_processed   = 0;

    for (int fid = frame_start; fid <= frame_end; ++fid) {  
        const fs::path bin_path  = fs::path(lidar_dir) / frame_filename(fid, ".bin");
        const fs::path csv_path  = fs::path(out_dir)  /
                                    ("obstacles_" + frame_filename(fid, ".csv"));

        if (!fs::exists(bin_path)) {
            if (verbose)
                std::cerr << "[obstacle_extractor] skip missing " << bin_path << "\n";
            continue;
        }

        // 1. Load
        const auto cloud = load_bin(bin_path.string());
        total_input_pts += static_cast<long>(cloud.size());

        // 2. Segment
        const auto seg = sector_segment_ground(cloud, ransac_params, sector_params);
        total_obstacle_pts += static_cast<long>(seg.obstacle_points.size());

        // 3. Write
        write_obstacles_csv(csv_path, seg.obstacle_points);

        // 4. Sanity print on first AND last frame (rule #7)
        const bool is_first = (fid == frame_start);
        const bool is_last  = (fid == frame_end);
        if (verbose && (is_first || is_last)) {
            std::cerr << "[obstacle_extractor] frame " << fid
                      << "  cloud=" << cloud.size()
                      << "  obstacles=" << seg.obstacle_points.size()
                      << "  ground=" << seg.ground_points.size()
                      << "  reliable_sectors=" << seg.num_reliable_sectors
                      << "/" << seg.num_sectors << "\n";
        }

        frames_processed += 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double runtime_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::ofstream metrics(fs::path(out_dir) / "metrics.json");
    metrics << "{\n"
            << "  \"frames_processed\": " << frames_processed     << ",\n"
            << "  \"total_input_pts\": "  << total_input_pts      << ",\n"
            << "  \"total_obstacle_pts\": " << total_obstacle_pts << ",\n"
            << "  \"mean_obstacles_per_frame\": "
            << (frames_processed > 0
                  ? static_cast<double>(total_obstacle_pts) / frames_processed
                  : 0.0)
            << ",\n"
            << "  \"runtime_ms\": " << runtime_ms << "\n"
            << "}\n";
    metrics.close();

    std::cerr << "[obstacle_extractor] DONE  out=" << out_dir
              << "  frames=" << frames_processed
              << "  mean_obstacles_per_frame="
              << (frames_processed > 0
                    ? static_cast<double>(total_obstacle_pts) / frames_processed
                    : 0.0)
              << "  runtime_ms=" << runtime_ms << "\n";
    return 0;
}
