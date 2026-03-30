---
layout: default
title: "M6: Integration & Docker"
nav_order: 7
---

# Phase 1, Milestone 6: Integration & Docker

M6 closes Phase 1 by wiring all five components into a single runnable pipeline and packaging it in Docker so anyone can evaluate the project without installing ROS2, Eigen, or Python dependencies.

## What was built

**`scripts/run_pipeline.sh`** — chains the full Phase 1 stack on a single frame:
1. Sector RANSAC → ground/obstacle split
2. Traversability grid → BEV risk + confidence map
3. SegFormer cam-LiDAR fusion → semantic risk update
4. Safety supervisor → TTC-based intervention log

**`scripts/smoke_test.sh`** — runs the pipeline on bundled sample data and verifies three output artifacts are produced in under 2 minutes.

**`data/sample/`** — 8 matched RELLIS-3D frames (LiDAR + camera + calibration, 25MB) bundled directly in the Docker image so no data download is needed.

**`src/safety_runner.cpp`** — standalone C++ binary that loads a traversability grid, simulates a synthetic worker scenario (starts at 15m, approaches at 1.2 m/s), and writes the intervention log to CSV.

## Running it

```bash
mkdir -p output
docker pull nishantzfyii/terra-perceive:phase1
docker run --rm \
  -v $(pwd)/output:/ws/src/construction_perception/output \
  nishantzfyii/terra-perceive:phase1
```

## Output

After ~45 seconds:

| Artifact | Description |
|----------|-------------|
| `bev_traversability.png` | Color-coded BEV: green=safe, yellow=caution, red=hazard, gray=unknown |
| `safety_events.csv` | Per-timestep: d_worker, d_stop, TTC, friction_mu, vel_before, vel_after |
| `timing_report.txt` | Per-stage wall-clock time |

## Pipeline in motion

The animation below shows the full pipeline running across 100 RELLIS-3D frames — BEV traversability on the left, all six feature channels on the right.

![BEV Combined Animation](assets/bev_combined_animation.gif)

## Smoke test timing

| Stage | Time |
|-------|------|
| Sector RANSAC + Traversability | ~7s |
| BEV Visualisation | ~3s |
| Cam-LiDAR Fusion | ~16s |
| Safety Supervisor | <1s |
| **Total** | **~27s** |

All 52 unit tests pass inside the container.
