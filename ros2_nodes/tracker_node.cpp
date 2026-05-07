// =============================================================================
// tracker_node.cpp — P2-M5.A2 C++ tracker bridge.
// =============================================================================
//
// NOT a ROS2 node yet (M10 territory per V6 master plan). Built as a
// CLI binary that reads length-prefixed `terra_perceive.DetectionList`
// protobuf frames from stdin and writes length-prefixed
// `terra_perceive.TrackList` frames to stdout. The Python sidecar
// `transport/tracker_bridge.py` glues both ends to NATS.
//
// Per-frame algorithm:
//   1. Read 4-byte big-endian length prefix + DetectionList payload from stdin.
//   2. Resolve the frame's LiDAR scan path from --lidar-dir + frame_id.
//   3. Project all LiDAR points to the camera image via CamLidarProjector.
//   4. For each Detection2D: collect LiDAR points whose pixel falls in the
//      bbox, compute median ego-frame (x, y) — that's the tracker input.
//      Drop the detection if zero points fall in the bbox.
//   5. Call SORTTracker::update (the M5 no-feature path) to produce the
//      live track set.
//   6. Build TrackList; write 4-byte length prefix + payload to stdout.
//
// Reuse:
//   - tracker_runner.cpp:1-50  for the CLI flag style and metrics idiom.
//   - cam_lidar_projection.hpp:28  CamLidarProjector::project_point() API.
//   - sort_tracker.hpp:163  SORTTracker constructor signature + update().
//
// Build path (CMake update lands in step 2 of A2):
//   add_executable(tracker_node ros2_nodes/tracker_node.cpp ...)
//   target_link_libraries(tracker_node sort_tracker cam_lidar_projection
//                         protobuf::libprotobuf yaml-cpp)
// =============================================================================

#include "cam_lidar_projection.hpp"
#include "hungarian.hpp"
#include "sort_tracker.hpp"

// Auto-generated C++ proto bindings. Generation lands in step 2 of A2.
#include "perception.pb.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ---------- CLI parsing (mirrors tracker_runner.cpp idiom) ---------- //

struct Args {
    std::string calib_yaml = "config/camera_lidar_calib.yaml";
    std::string lidar_dir =
        "data/RELLIS-3D/Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin";
    float max_dist = 5.0f;
    int max_misses = 10;
    int min_hits = 1;
    float dt = 0.1f;
    float process_noise = 2.0f;
    float meas_noise = 0.3f;
    bool verbose = false;
};

bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

std::string getFlag(int argc, char** argv, const std::string& flag,
                    const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return def;
}

Args parseArgs(int argc, char** argv) {
    Args a;
    a.calib_yaml = getFlag(argc, argv, "--calib", a.calib_yaml);
    a.lidar_dir = getFlag(argc, argv, "--lidar-dir", a.lidar_dir);
    a.max_dist = std::stof(getFlag(argc, argv, "--max-dist", "5.0"));
    a.max_misses = std::stoi(getFlag(argc, argv, "--max-misses", "10"));
    a.min_hits = std::stoi(getFlag(argc, argv, "--min-hits", "1"));
    a.dt = std::stof(getFlag(argc, argv, "--dt", "0.1"));
    a.process_noise = std::stof(getFlag(argc, argv, "--process-noise", "2.0"));
    a.meas_noise = std::stof(getFlag(argc, argv, "--meas-noise", "0.3"));
    a.verbose = hasFlag(argc, argv, "--verbose");
    return a;
}

// ---------- Length-prefixed protobuf framing on stdin/stdout ---------- //
//
// Wire format: 4-byte big-endian unsigned length, then `length` bytes of
// serialized protobuf. EOF on stdin closes the loop cleanly.

bool readLengthPrefixedFrame(std::istream& is, std::string& payload_out) {
    uint8_t hdr[4];
    is.read(reinterpret_cast<char*>(hdr), 4);
    if (is.gcount() < 4) return false;  // EOF
    uint32_t n = (uint32_t(hdr[0]) << 24) | (uint32_t(hdr[1]) << 16) |
                 (uint32_t(hdr[2]) << 8) | uint32_t(hdr[3]);
    payload_out.resize(n);
    is.read(payload_out.data(), n);
    return is.gcount() == static_cast<std::streamsize>(n);
}

bool writeLengthPrefixedFrame(std::ostream& os, const std::string& payload) {
    uint32_t n = static_cast<uint32_t>(payload.size());
    uint8_t hdr[4] = {
        static_cast<uint8_t>((n >> 24) & 0xFF),
        static_cast<uint8_t>((n >> 16) & 0xFF),
        static_cast<uint8_t>((n >> 8) & 0xFF),
        static_cast<uint8_t>(n & 0xFF),
    };
    os.write(reinterpret_cast<const char*>(hdr), 4);
    os.write(payload.data(), payload.size());
    os.flush();
    return os.good();
}

// ---------- LiDAR scan loader (KITTI .bin format) ---------- //
//
// Mirrors point_cloud_loader.cpp's pattern: each point is 4 float32s
// [x, y, z, intensity]; we drop intensity and keep Eigen::Vector3f.

std::vector<Eigen::Vector3f> loadLidarScanByFrameId(const std::string& dir,
                                                    int frame_idx) {
    char fname[32];
    std::snprintf(fname, sizeof(fname), "%06d.bin", frame_idx);
    std::string full = dir + "/" + fname;
    std::ifstream f(full, std::ios::binary | std::ios::ate);
    std::vector<Eigen::Vector3f> pts;
    if (!f.is_open()) return pts;
    auto sz = f.tellg();
    if (sz < 0) return pts;
    f.seekg(0, std::ios::beg);
    size_t n = static_cast<size_t>(sz) / (4 * sizeof(float));
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float xyzi[4];
        f.read(reinterpret_cast<char*>(xyzi), sizeof(xyzi));
        pts.emplace_back(xyzi[0], xyzi[1], xyzi[2]);
    }
    return pts;
}

// Parse "frame_NNNNNN" from header.frame_id; returns -1 on failure.
int parseFrameIdx(const std::string& fid) {
    auto pos = fid.find_last_of('_');
    if (pos == std::string::npos) return -1;
    try {
        return std::stoi(fid.substr(pos + 1));
    } catch (...) {
        return -1;
    }
}

// ---------- Calibration loader (yaml-cpp) ---------- //

struct Calib {
    Eigen::Matrix4f T_cam_lidar = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    int width = 0;
    int height = 0;
};

Calib loadCalib(const std::string& path) {
    Calib c;
    YAML::Node y = YAML::LoadFile(path);
    auto intr = y["camera_intrinsics"];
    c.K(0, 0) = intr["fx"].as<float>();
    c.K(1, 1) = intr["fy"].as<float>();
    c.K(0, 2) = intr["cx"].as<float>();
    c.K(1, 2) = intr["cy"].as<float>();
    c.K(2, 2) = 1.0f;
    c.width = intr["width"].as<int>();
    c.height = intr["height"].as<int>();
    auto T = y["extrinsic_T_cam_lidar"];
    for (int r = 0; r < 4; ++r)
        for (int col = 0; col < 4; ++col)
            c.T_cam_lidar(r, col) = T[r][col].as<float>();
    return c;
}

}  // namespace

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Args args = parseArgs(argc, argv);
    Calib calib = loadCalib(args.calib_yaml);

    CamLidarProjector projector(calib.T_cam_lidar, calib.K, calib.width,
                                calib.height);

    tracker::SORTTracker tracker(args.max_dist, args.max_misses, args.min_hits,
                                 tracker::Solver::Greedy,
                                 tracker::Order::PredictThenUpdate, args.dt,
                                 args.process_noise, args.meas_noise);

    if (args.verbose) {
        std::cerr << "tracker_node: calib=" << args.calib_yaml
                  << "  lidar_dir=" << args.lidar_dir
                  << "  max_dist=" << args.max_dist
                  << "  max_misses=" << args.max_misses << "\n";
    }

    // Per-frame counters surfaced in stderr metrics; useful when debugging
    // the bbox-zero-points drop rule (debug log: option-i drop = clean default).
    uint64_t n_frames = 0;
    uint64_t n_dets_in = 0;
    uint64_t n_dets_dropped_no_depth = 0;
    uint64_t n_tracks_out = 0;

    // ----------------------- main per-frame loop ----------------------- //
    std::string in_payload;
    while (readLengthPrefixedFrame(std::cin, in_payload)) {
        terra_perceive::DetectionList in_msg;
        if (!in_msg.ParseFromString(in_payload)) {
            std::cerr << "tracker_node: parse failure on frame " << n_frames
                      << "; skipping\n";
            continue;
        }

        const std::string& frame_id = in_msg.header().frame_id();
        const int frame_idx = parseFrameIdx(frame_id);
        std::vector<Eigen::Vector3f> lidar_pts;
        if (frame_idx >= 0) {
            lidar_pts = loadLidarScanByFrameId(args.lidar_dir, frame_idx);
        }

        // Project the entire LiDAR scan once per frame; reuse for every
        // bbox lookup. Two parallel arrays: pixel + LiDAR-frame point.
        std::vector<Eigen::Vector2f> pixels;
        std::vector<Eigen::Vector3f> visible_pts;
        pixels.reserve(lidar_pts.size());
        visible_pts.reserve(lidar_pts.size());
        for (const auto& p : lidar_pts) {
            Eigen::Vector2f px;
            if (projector.project_point(p, px)) {
                pixels.push_back(px);
                visible_pts.push_back(p);
            }
        }

        // ---------- Per-bbox 3D localization ---------- //
        std::vector<Eigen::Vector2f> det_positions_xy;
        std::vector<int> det_class_ids;
        det_positions_xy.reserve(in_msg.detections_size());
        det_class_ids.reserve(in_msg.detections_size());

        for (int i = 0; i < in_msg.detections_size(); ++i) {
            const auto& d = in_msg.detections(i);
            ++n_dets_in;

            // 3D localization: median LiDAR (x, y) of all points whose
            // projected pixel falls inside this 2D bbox. Decision: option-i
            // drop when no points fall inside (debug log 2026-05-01).
            //
            // Algorithm:
            //   1) Collect the LiDAR-frame (x, y) of every visible_pts[k]
            //      whose pixels[k] is inside the bbox.
            //   2) If empty -> drop, count, continue.
            //   3) Else median(x), median(y) via nth_element (O(N), N ~ a
            //      few hundred per bbox so std::sort would also be fine).
            //   4) Append to the parallel det_positions_xy / det_class_ids
            //      arrays the SORTTracker consumes.
            //
            // Reference: docs/m4-fusion.md "Projection: The Four-Step Pipeline"
            // for the geometry; we just consume the projector's output here.
            std::vector<float> xs;
            std::vector<float> ys;
            xs.reserve(64);
            ys.reserve(64);
            for (size_t k = 0; k < pixels.size(); ++k) {
                const float u = pixels[k].x();
                const float v = pixels[k].y();
                if (u >= d.x_min() && u < d.x_max() &&
                    v >= d.y_min() && v < d.y_max()) {
                    xs.push_back(visible_pts[k].x());
                    ys.push_back(visible_pts[k].y());
                }
            }
            if (xs.empty()) {
                ++n_dets_dropped_no_depth;
                continue;  // option-i: no LiDAR support inside bbox -> drop.
            }
            const size_t mid = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + mid, xs.end());
            std::nth_element(ys.begin(), ys.begin() + mid, ys.end());
            const float med_x = xs[mid];
            const float med_y = ys[mid];

            det_positions_xy.emplace_back(med_x, med_y);
            det_class_ids.push_back(d.class_id());
        }

        // ---------- SORT update ---------- //
        // M5 path: no-feature update() (FilterKind::CV). update_with_features
        // is the M13.5 appearance entry point - out of scope for M5.
        // T_world_ego defaults to Identity (Phase-3.5 ego-fusion is a separate
        // milestone; replay demo ego at 0.6 m/s is slow enough to ignore).
        std::vector<tracker::Track> tracks =
            tracker.update(det_positions_xy, det_class_ids);

        // ---------- Build TrackList output ---------- //
        terra_perceive::TrackList out_msg;
        out_msg.mutable_header()->CopyFrom(in_msg.header());
        out_msg.mutable_header()->set_source("tracker_node");
        for (const auto& t : tracks) {
            auto* tr = out_msg.add_tracks();
            tr->set_track_id(t.id);
            tr->set_class_id(t.class_id);   // SORT passes through detection class.
            const auto pos = t.position();
            const auto vel = t.velocity();
            tr->set_x(pos.x());
            tr->set_y(pos.y());
            tr->set_vx(vel.x());
            tr->set_vy(vel.y());
            tr->set_z_3d(t.z_3d);
            tr->set_hits(t.hits);
        }
        n_tracks_out += out_msg.tracks_size();

        std::string out_payload;
        out_msg.SerializeToString(&out_payload);
        if (!writeLengthPrefixedFrame(std::cout, out_payload)) {
            std::cerr << "tracker_node: stdout write failed; exiting\n";
            return 2;
        }

        ++n_frames;
        if (args.verbose && (n_frames % 50 == 0)) {
            std::cerr << "tracker_node: frames=" << n_frames
                      << "  dets_in=" << n_dets_in
                      << "  dropped=" << n_dets_dropped_no_depth
                      << "  tracks_out=" << n_tracks_out << "\n";
        }
    }

    if (args.verbose) {
        std::cerr << "tracker_node: end-of-input. frames=" << n_frames
                  << "  dets_in=" << n_dets_in
                  << "  dropped=" << n_dets_dropped_no_depth
                  << "  tracks_out=" << n_tracks_out << "\n";
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
