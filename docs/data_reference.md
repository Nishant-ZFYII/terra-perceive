# RELLIS-3D Data Reference — What's Where, What's Used When

All data lives under `data/RELLIS-3D/`. Paths are relative to that directory.
Sequence 00 (`00000/`) is the primary sequence used throughout the project.

---

## Quick Reference: Which Data for Which Milestone

| Milestone | Data Needed | Path | Notes |
|-----------|------------|------|-------|
| P1-M1 to M5 | LiDAR .bin files (Ouster, KITTI format) | `data/extracted_frames/*.bin` | Extracted from zip, 100 frames in use |
| P1-M1 to M5 | Camera images | `Rellis_3D_pylon_camera_node/Rellis-3D/00000/` | Full resolution Basler camera |
| P1-M4 | Camera-LiDAR extrinsics | `Rellis_3D_cam2lidar_20210224/Rellis_3D/00000/transforms.yaml` | SE(3) transform |
| P1-M4 | Camera intrinsics | `Rellis_3D_cam_intrinsic/Rellis-3D/` | K matrix |
| P1-M6 | Bundled sample (8 frames) | `data/sample/` | Subset for smoke test |
| **P2-M1** | **Split Raw rosbags (GPS/IMU)** | **`00000_00.bag` through `00000_04.bag`** | **VectorNav VN-300 topics inside** |
| **P2-M1** | **LiDAR .bin files (for KISS-ICP)** | **`Rellis_3D_os1_cloud_node_kitti_bin.zip` → extract** | **14GB, 2847 frames in seq 00** |
| **P2-M1** | **LiDAR poses (for validation)** | **`Rellis_3D_lidar_poses_20210614/Rellis-3D/00000/poses.txt`** | **2847 lines, 3x4 matrices** |
| P2-M2 | Same rosbags (IMU data for pre-integration) | Same as P2-M1 | `/vectornav/IMU` topic |
| P2-M3 | LiDAR .bin + poses from P2-M1/M2 | Reuses P2-M1 outputs | `data/poses_*.csv` files |
| P2-M4 | LiDAR .bin + camera images | Same as P1 | For YOLO + 3D localization |
| P2-M6 | Same as P1 | Same | For ablation comparison |
| P2-M7 | **nuScenes mini (~4GB)** | **Download separately** | **Not RELLIS-3D** |
| P3-5 | Image annotations (ID format) | `Rellis_3D_pylon_camera_node_label_id/Rellis-3D/` | For SegFormer fine-tuning |
| P3-5 | Image split file | `Rellis_3D_image_split/` | Train/val/test splits |
| P3-6 | LiDAR semantic labels | `Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614/Rellis-3D/` | SemanticKITTI format |

---

## Detailed Data Inventory

### 1. LiDAR Point Clouds — Ouster OS1-64 (KITTI Binary Format)

**Path**: `Rellis_3D_os1_cloud_node_kitti_bin.zip` (14GB)
**Extracted to**: Need to extract — currently only the zip exists
**Also**: `data/extracted_frames/` has 100 .bin files (subset used in Phase 1)

**Format**: Each `.bin` file is packed `float32 [x, y, z, intensity]` per point (4 floats per point). Use first 3 for geometry.

**Structure** (after extraction):
```
Rellis_3D_os1_cloud_node_kitti_bin/Rellis-3D/00000/os1_cloud_node_kitti_bin/
├── 000000.bin
├── 000001.bin
├── ...
└── 002846.bin   (2847 frames total for sequence 00)
```

**Used by**: P1 (all milestones), P2-M1 (KISS-ICP), P2-M2 (SLAM), P2-M3 (accumulated map)

---

### 2. Split Raw Rosbags (GPS/IMU + LiDAR + Camera)

**Path**: `00000_00.bag` through `00000_04.bag` (29GB total, 5 bags)

**Contains** (expected — verify with `rosbag info`):
- `/vectornav/IMU` — orientation quaternion, angular velocity, linear acceleration (VectorNav VN-300, ~100Hz)
- `/vectornav/GPS` — lat/lon/alt (variable rate)
- `/os1_cloud_node/points` — Ouster LiDAR point clouds
- `/pylon_camera_node/image_raw` — Basler camera images

**IMPORTANT**: Topic names may differ. Always run `rosbag info <bag>` first to verify.

**Used by**: P2-M1.1 (GPS/IMU extraction), P2-M1.3 (Cartographer offline), P2-M2 (IMU pre-integration), P2-M10 (ROS2 live replay)

---

### 3. LiDAR Scan Poses

**Path**: `Rellis_3D_lidar_poses_20210614/Rellis-3D/00000/poses.txt`

**Format**: 2847 lines, each line is 12 floats = flattened 3x4 transformation matrix `[R|t]`
```
r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```
These are the poses provided by the dataset authors (not GPS, not ICP — their own registration).

**Used by**: P2-M1 (can be used as an additional reference trajectory for validation)

---

### 4. Camera Images

**Path**: `Rellis_3D_pylon_camera_node/Rellis-3D/00000/`

**Format**: Full resolution images from Basler camera (~11GB total across all sequences)

**Used by**: P1-M4 (projection + fusion), P2-M5 (YOLO detection), P2-M9 (four-pane demo)

---

### 5. Camera-LiDAR Calibration (Extrinsics)

**Path**: `Rellis_3D_cam2lidar_20210224/Rellis_3D/00000/transforms.yaml`

**Contains**: SE(3) transform `T_cam_lidar` — LiDAR frame to camera frame. Per sequence.

**Used by**: P1-M4 (projection), loaded into `config/camera_lidar_calib.yaml`

---

### 6. Camera Intrinsics

**Path**: `Rellis_3D_cam_intrinsic/Rellis-3D/`

**Contains**: Camera intrinsic matrix K (3x3) and distortion parameters.

**Used by**: P1-M4 (projection)

---

### 7. Image Semantic Annotations (for P3 fine-tuning)

**Color format**: `Rellis_3D_pylon_camera_node_label_color/Rellis-3D/` (119MB)
- Human-readable color-coded label images

**ID format**: `Rellis_3D_pylon_camera_node_label_id/Rellis-3D/` (94MB)
- Machine-readable class ID per pixel (uint8)
- 20 classes: sky, grass, tree, bush, concrete, mud, person, puddle, rubble, barrier, log, fence, vehicle, object, pole, water, asphalt, building, dirt, void

**Image split**: `Rellis_3D_image_split/` (44KB)
- Train/val/test split file

**Used by**: P3-5 (SegFormer fine-tuning on RELLIS-3D classes)

---

### 8. LiDAR Semantic Annotations

**Ouster labels**: `Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614/Rellis-3D/` (174MB)
- SemanticKITTI format: per-point uint32 labels matching the .bin files
- Same 20 classes as image annotations

**Used by**: P3-6 (if LiDAR semantic segmentation is needed)

---

### 9. Velodyne LiDAR Data (secondary sensor)

**Point clouds**: `Rellis_3D_vel_cloud_node_kitti_bin/` (5.58GB) — NOT currently downloaded/extracted
**Labels**: `Rellis_3D_vel_cloud_node_semantickitti_label_id/Rellis-3D/`
**Velodyne-to-Ouster transform**: `Rellis_3d_Velodyne2Oster/`

**Used by**: Not currently used. The project uses Ouster OS1-64 as the primary LiDAR. Velodyne data exists for cross-sensor comparison if needed.

---

### 10. Stereo Calibration + Raw Calibration Data

**Stereo**: `RELLIS3D_stereo_calibration/` (3KB)
**Raw calibration**: `Calibration Raw Data/` (774MB)

**Used by**: Not currently used. Available if stereo depth estimation is needed.

---

### 11. Image + LiDAR Examples (small samples)

**Image examples**: `Rellis_3D_image_examples/` (3MB) — annotated sample images
**LiDAR examples**: `Rellis_3D_lidar_example/` (24MB) — annotated sample point clouds

**Used by**: Quick reference / sanity checks

---

### 12. Bundled Sample Data (Phase 1 smoke test)

**Path**: `data/sample/`
**Contains**: 8 matched LiDAR .bin + camera .jpg + calibration files (~25MB)
**Source**: Extracted subset from the full dataset

**Used by**: `scripts/smoke_test.sh`, `docker-compose up`

---

## Data Not Yet Downloaded

| Data | Size | When Needed | Download Link |
|------|------|-------------|---------------|
| nuScenes mini | ~4GB | P2-M7 (Week 9) | nuscenes.org |
| Ouster LiDAR Color PLY | 26GB | Not planned | Available if needed |

---

## RELLIS-3D 20 Semantic Classes

| ID | Class | Traversability Modifier (from pipeline_config.yaml) |
|----|-------|-----------------------------------------------------|
| 0 | void | — |
| 1 | dirt | 0.9 |
| 3 | grass | 1.0 |
| 4 | tree | 0.0 |
| 5 | pole | — |
| 6 | water | 0.1 |
| 7 | sky | — |
| 8 | vehicle | 0.0 |
| 9 | object | — |
| 10 | asphalt | 1.0 |
| 12 | building | 0.0 |
| 15 | log | — |
| 17 | person | 0.0 |
| 18 | fence | 0.0 |
| 19 | bush | 0.3 |
| 23 | concrete | — |
| 27 | barrier | — |
| 29 | puddle | — |
| 30 | mud | 0.5 |
| 31 | rubble | — |

Note: Current pipeline uses ADE20K pretrained SegFormer with approximate class mapping. P3-5 (fine-tuning) would use these native RELLIS-3D class IDs directly.

---

## Key Notes for Future Agents

1. **The LiDAR .bin zip needs extraction** for P2-M1. Currently `Rellis_3D_os1_cloud_node_kitti_bin.zip` exists but `data/extracted_frames/` only has 100 frames (Phase 1 subset). Full sequence 00 has 2847 frames.

2. **Rosbag topic names must be verified** with `rosbag info` before writing extraction scripts. The VectorNav topics may not be exactly `/vectornav/IMU` and `/vectornav/GPS`.

3. **poses.txt** in `Rellis_3D_lidar_poses_20210614/` contains dataset-provided poses (3x4 flattened matrices, 2847 lines for seq 00). These are NOT GPS poses — they are the dataset authors' registration. Can be used as an additional reference trajectory.

4. **Sequence 00** is used throughout. Other sequences (00001-00004) are available but not currently planned.
