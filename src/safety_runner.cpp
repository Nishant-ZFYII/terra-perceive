// =============================================================================
// safety_runner — CLI for P1-M5 (kinematic) and P2-M6 (CBF) safety supervisor.
// =============================================================================
//
// Backward-compatible. Two run modes:
//
//   1. Legacy positional (P1-M5 demo, unchanged):
//      ./safety_runner <grid.bin> <vehicle_speed_mps> [output.csv]
//      Loads a flat float32 grid.bin (struct of 6 floats per cell), takes the
//      mean traversability, and simulates 30 steps at dt=0.1 with a single
//      worker approaching at 1.2 m/s from 15 m. Writes events to output.csv.
//
//   2. Scripted-scenario flag mode (P2-M6 ablation runs):
//      ./safety_runner --scenario <csv> [flags...]
//      CSV schema (header expected):
//         frame_id, worker_id, x, y, vx, vy, vehicle_v, vehicle_dir
//      The runner integrates the vehicle's velocity using the supervisor's
//      scale_factor at dt=0.1 and writes per-step events.csv plus an atomic
//      metrics.json. Same scenario can run under either safety_mode.
//
//   Flags:
//      --scenario <csv>          scripted-scenario CSV (mode-2 trigger)
//      --safety-mode MODE        kinematic | cbf (default kinematic)
//      --cbf-gamma F             CBF gain (default 1.0)
//      --cbf-d-safe-min F        minimum margin (default 0.5)
//      --cbf-dt F                control step (default 0.1)
//      --frames N                cap on simulation steps (default = scenario length)
//      --out <dir>               results dir (default ./out_scenario)
//      --verbose                 per-step stdout
// =============================================================================

#include "safety_supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// --- CLI helpers ---
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

// --- Scenario row ---
struct ScenarioRow {
    int frame_id;
    int worker_id;
    float x, y;
    float vx, vy;
    float vehicle_v;     // initial commanded velocity (m/s)
    float vehicle_dir;   // forward direction in radians (0 = +x)
};

static std::vector<ScenarioRow> loadScenario(const fs::path& csv) {
    std::vector<ScenarioRow> out;
    std::ifstream f(csv);
    if (!f) {
        std::cerr << "[error] cannot open scenario: " << csv << "\n";
        return out;
    }
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ScenarioRow r{};
        auto next = [&](float& dst) {
            if (!std::getline(ss, tok, ',')) return false;
            dst = std::stof(tok);
            return true;
        };
        auto nexti = [&](int& dst) {
            if (!std::getline(ss, tok, ',')) return false;
            dst = std::stoi(tok);
            return true;
        };
        if (!nexti(r.frame_id)) continue;
        if (!nexti(r.worker_id)) continue;
        if (!next(r.x)) continue;
        if (!next(r.y)) continue;
        if (!next(r.vx)) continue;
        if (!next(r.vy)) continue;
        if (!next(r.vehicle_v)) continue;
        if (!next(r.vehicle_dir)) continue;
        out.push_back(r);
    }
    return out;
}

// --- Mode-2 main: scripted-scenario integration ---
static int runScenario(const fs::path& scenario_csv, const SafetyConfig& config,
                       const fs::path& out_dir, int frames_cap, bool verbose) {
    auto rows = loadScenario(scenario_csv);
    if (rows.empty()) {
        std::cerr << "[error] empty scenario\n";
        return 4;
    }

    // Group by frame_id to handle multi-worker rows.
    std::map<int, std::vector<ScenarioRow>> by_frame;
    for (const auto& r : rows) by_frame[r.frame_id].push_back(r);

    fs::create_directories(out_dir);
    std::ofstream events(out_dir / "events.csv");
    events << "frame_id,t,d_worker,d_stop,h_or_ttc,mu,vel_before,vel_after,scale,rule\n";
    events << std::fixed << std::setprecision(6);

    StoppingDistanceModel model;
    SafetySupervisor sup(config, model);

    const float dt = config.cbf_dt;  // reuse the same control step in both modes

    // Vehicle state. Take the first frame's vehicle_v and vehicle_dir as initial.
    auto first = by_frame.begin()->second.front();
    float v = first.vehicle_v;
    float dir = first.vehicle_dir;
    float ego_x = 0.0f;
    float ego_y = 0.0f;

    int n_collisions = 0;
    float min_margin = std::numeric_limits<float>::infinity();
    float max_dv_per_dt = 0.0f;
    float v_prev = v;

    int step = 0;
    int max_step = frames_cap > 0 ? frames_cap : static_cast<int>(by_frame.rbegin()->first) + 1;
    for (int frame_id = 0; frame_id < max_step; ++frame_id) {
        sup.update_lidar_timestamp(frame_id * dt);

        // Find this frame's worker rows (or the latest available).
        auto it = by_frame.find(frame_id);
        if (it == by_frame.end()) {
            // Hold last known workers.
            auto lower = by_frame.lower_bound(frame_id);
            if (lower == by_frame.end()) lower = std::prev(by_frame.end());
            else if (lower != by_frame.begin()) --lower;
            it = lower;
        }
        const auto& workers = it->second;

        // Pick nearest worker in the forward arc (60-degree half-arc).
        float best_d = std::numeric_limits<float>::infinity();
        float best_v_close = 0.0f;
        for (const auto& w : workers) {
            float dx = w.x - ego_x;
            float dy = w.y - ego_y;
            float bearing = std::atan2(dy, dx);
            float dbearing = std::fabs(std::atan2(std::sin(bearing - dir),
                                                   std::cos(bearing - dir)));
            if (dbearing > 1.0472f) continue;  // outside +/-60 deg arc
            float d = std::hypot(dx, dy);
            if (d < best_d) {
                best_d = d;
                // Closing speed = -d/dt of distance, projected on forward dir.
                float v_w_along = w.vx * std::cos(dir) + w.vy * std::sin(dir);
                best_v_close = v - v_w_along;  // +ve = closing
            }
        }
        if (!std::isfinite(best_d)) {
            best_d = 100.0f;
            best_v_close = 0.0f;
        }

        // worker_approach_speed in evaluate() is v_worker (along forward axis).
        // From v_relative = v - worker_speed, worker_speed = v - v_relative
        //                                                 = v - best_v_close.
        // For a stationary worker, best_v_close == v, so worker_speed = 0.
        float worker_speed_along = v - best_v_close;

        SafetyIntervention out =
            sup.evaluate(v, best_d, worker_speed_along, 1.0f, frame_id * dt);

        // Apply scale.
        v_prev = v;
        v = std::max(0.0f, v * out.scale_factor);
        ego_x += v * std::cos(dir) * dt;
        ego_y += v * std::sin(dir) * dt;

        // Closing distance: subtract relative-velocity component along forward.
        // (Workers also move; we approximate by re-reading the workers' positions
        // each scenario row.)

        // Diagnostic columns.
        float mu = sup.traversability_to_friction(1.0f);
        float d_stop = model.compute(v, mu);
        float h_or_ttc;
        if (config.safety_mode == "cbf") {
            h_or_ttc = best_d - (d_stop + config.cbf_d_safe_min);
        } else {
            float v_rel = v - worker_speed_along;
            h_or_ttc = (v_rel > 0.0f) ? (best_d - d_stop) / v_rel
                                       : std::numeric_limits<float>::infinity();
        }

        events << frame_id << "," << frame_id * dt << ","
               << best_d << "," << d_stop << "," << h_or_ttc << ","
               << mu << "," << v_prev << "," << v << "," << out.scale_factor << ","
               << out.reason << "\n";

        if (best_d <= 0.0f) ++n_collisions;
        float margin = best_d - d_stop;
        if (margin < min_margin) min_margin = margin;
        float dv_per_dt = std::fabs(v - v_prev) / dt;
        if (dv_per_dt > max_dv_per_dt) max_dv_per_dt = dv_per_dt;

        if (verbose && (step % 20 == 0 || step + 1 == max_step)) {
            std::cout << "[" << step << "/" << max_step << "] v=" << v
                      << " d=" << best_d << " scale=" << out.scale_factor
                      << " rule=" << out.reason << "\n";
        }
        ++step;
        if (v < 1e-3f && step > 20) break;  // converged to stop
    }

    // Atomic metrics.json
    fs::path tmp = out_dir / "metrics.json.tmp";
    {
        std::ofstream j(tmp);
        j << std::fixed << std::setprecision(6);
        j << "{\n";
        j << "  \"safety_mode\": \"" << config.safety_mode << "\",\n";
        j << "  \"cbf_gamma\": " << config.cbf_gamma << ",\n";
        j << "  \"cbf_d_safe_min\": " << config.cbf_d_safe_min << ",\n";
        j << "  \"scenario\": \"" << scenario_csv.string() << "\",\n";
        j << "  \"steps\": " << step << ",\n";
        j << "  \"collisions\": " << n_collisions << ",\n";
        j << "  \"min_margin\": " << min_margin << ",\n";
        j << "  \"max_dv_per_dt\": " << max_dv_per_dt << ",\n";
        j << "  \"final_velocity\": " << v << "\n";
        j << "}\n";
    }
    fs::rename(tmp, out_dir / "metrics.json");

    std::cout << "[done] mode=" << config.safety_mode
              << " steps=" << step
              << " collisions=" << n_collisions
              << " min_margin=" << min_margin
              << " max_dv/dt=" << max_dv_per_dt
              << " final_v=" << v << "\n";
    return 0;
}

// --- Mode-1: legacy positional (P1-M5 unchanged) ---
static int runLegacyPositional(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <grid.bin> <vehicle_speed_mps> [output.csv]\n"
                  << "       " << argv[0]
                  << " --scenario <csv> [flags]\n"
                  << "       " << argv[0] << " --help\n";
        return 1;
    }
    std::string grid_path = argv[1];
    float vehicle_speed_mps = std::stof(argv[2]);
    std::string output_csv = (argc >= 4) ? argv[3] : "output.csv";

    std::ifstream infile(grid_path, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file: " << grid_path << std::endl;
        return 1;
    }
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    size_t num_floats = file_size / sizeof(float);
    std::vector<float> grid_data(num_floats);
    infile.read(reinterpret_cast<char*>(grid_data.data()), file_size);
    infile.close();

    SafetyConfig config;
    StoppingDistanceModel model;
    SafetySupervisor supervisor(config, model);
    float trav_sum = 0.0f;
    int trav_count = 0;
    for (size_t i = 0; i < num_floats / 6; ++i) {
        float confidence = grid_data[i * 6 + 1];
        if (confidence > 0.5f) {
            float risk = grid_data[i * 6 + 0];
            trav_sum += (1.0f - risk);
            trav_count++;
        }
    }
    float mean_trav = (trav_count > 0) ? (trav_sum / trav_count) : 0.0f;
    std::cout << "Mean Traversability: " << mean_trav
              << " (from " << trav_count << " cells)" << std::endl;

    float worker_d = 15.0f;
    supervisor.update_lidar_timestamp(0.0);
    for (int step = 0; step < 30; ++step) {
        double t = step * 0.1;
        supervisor.update_lidar_timestamp(t);
        SafetyIntervention intervention =
            supervisor.evaluate(vehicle_speed_mps, worker_d, -1.2f, mean_trav, t);
        std::cout << "Time: " << t << "s, Worker Distance: " << worker_d
                  << "m, Intervention: "
                  << (intervention.level == SafetyIntervention::EMERGENCY_STOP
                          ? "EMERGENCY_STOP"
                          : (intervention.level == SafetyIntervention::HARD_BRAKE
                                 ? "HARD_BRAKE"
                                 : (intervention.level ==
                                            SafetyIntervention::PROPORTIONAL_SCALE
                                        ? "PROPORTIONAL_SCALE"
                                        : "NONE")))
                  << ", Scale Factor: " << intervention.scale_factor
                  << ", Reason: " << intervention.reason << std::endl;
        worker_d -= 1.2f * 0.1f;
    }

    std::ofstream outfile(output_csv);
    if (!outfile) {
        std::cerr << "Error opening output file: " << output_csv << std::endl;
        return 1;
    }
    outfile << "timestamp,rule,d_worker,d_stop,ttc,mu,vel_before,vel_after\n";
    const auto& events = supervisor.event_log();
    for (const auto& event : events) {
        outfile << event.timestamp << ","
                << event.rule << ","
                << event.d_worker << ","
                << event.d_stop << ","
                << event.ttc << ","
                << event.friction_mu << ","
                << event.vel_before << ","
                << event.vel_after << "\n";
    }
    outfile.close();
    std::cout << "Wrote " << events.size() << " events to " << output_csv << std::endl;
    supervisor.report_latency_stats();
    return 0;
}

int main(int argc, char** argv) {
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        std::cout <<
            "safety_runner — kinematic + CBF safety supervisor CLI.\n"
            "Modes:\n"
            "  Legacy positional: <grid.bin> <vehicle_speed_mps> [output.csv]\n"
            "  Scripted scenario: --scenario <csv> [flags]\n"
            "Flags (scenario mode):\n"
            "  --safety-mode MODE       kinematic | cbf\n"
            "  --cbf-gamma F            CBF gain (default 1.0)\n"
            "  --cbf-d-safe-min F       minimum margin (default 0.5)\n"
            "  --cbf-dt F               control step (default 0.1)\n"
            "  --frames N               cap on simulation steps\n"
            "  --out <dir>              results dir (default ./out_scenario)\n"
            "  --verbose                per-step stdout\n";
        return 0;
    }

    std::string scenario_str = getArg(argc, argv, "--scenario", "");
    if (scenario_str.empty()) {
        return runLegacyPositional(argc, argv);
    }

    SafetyConfig config;
    config.safety_mode      = getArg(argc, argv, "--safety-mode", "kinematic");
    config.cbf_gamma        = std::stof(getArg(argc, argv, "--cbf-gamma", "1.0"));
    config.cbf_d_safe_min   = std::stof(getArg(argc, argv, "--cbf-d-safe-min", "0.5"));
    config.cbf_dt           = std::stof(getArg(argc, argv, "--cbf-dt", "0.1"));
    config.lidar_timeout_ms = 1e9f;  // disable lidar gate in scenario mode
    int frames_cap = std::stoi(getArg(argc, argv, "--frames", "0"));
    fs::path out_dir = getArg(argc, argv, "--out", "./out_scenario");
    bool verbose = hasFlag(argc, argv, "--verbose");

    return runScenario(scenario_str, config, out_dir, frames_cap, verbose);
}
