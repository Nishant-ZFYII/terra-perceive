

import pytest
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from compare_odometry import umeyama_alignment, compute_ate, compute_rpe, pose_to_se3
from extract_poses_gps import convert_gps_to_enu, associate_data, find_nearest_timestamp




class TestENUConversion:
    """Test GPS lat/lon/alt to ENU conversion."""

    def test_origin_is_zero(self):
        """First GPS fix (origin) should map to (0, 0, 0)."""

        enu_cords = convert_gps_to_enu([(0, 40.0, -105.0, 1600.0)])  
        assert np.allclose(enu_cords[0], [0, 0, 0], atol=1e-3), "Origin GPS fix should convert to (0, 0, 0) in ENU coordinates."

    def test_known_offset(self):
        """A known lat/lon offset should produce expected ENU meters."""
        enu_cords = convert_gps_to_enu([(0, 40.0, -105.0, 1600.0), (0, 41.0, -105.0, 1600.0)])
        assert np.allclose(enu_cords[1], [0, 111320, 0], rtol = 0.01), "1 degree latitude offset should convert to approximately 111320 meters north in ENU coordinates."


class TestPoseCSVFormat:
    """Test that output CSVs have correct structure."""

    def test_gps_csv_header(self):
        """GPS CSV should have the expected 8 columns."""
        with open('data/poses_gps.csv', 'r') as f:
            header = f.readline().strip()
        assert header == 'timestamp,x,y,z,qw,qx,qy,qz', "GPS CSV header should be 'timestamp,x,y,z,qw,qx,qy,qz'"


    def test_csv_row_count_alignment(self):
        """All three CSVs should have similar row counts (within ±5)."""

        with open('data/poses_gps.csv', 'r') as f:
            gps_lines = f.readlines()
        with open('data/poses_icp.csv', 'r') as f:
            icp_lines = f.readlines()
        with open('data/poses_carto.csv', 'r') as f:
            carto_lines = f.readlines()

        assert abs(len(gps_lines) - len(icp_lines)) <= 5, "GPS and ICP CSVs should have similar row counts (within ±5)"
        assert abs(len(gps_lines) - len(carto_lines)) <= 5, "GPS and Carto CSVs should have similar row counts (within ±5)"
        assert abs(len(icp_lines) - len(carto_lines)) <= 5, "ICP and Carto CSVs should have similar row counts (within ±5)"

class TestUmeyamaAlignment:
    """Test Umeyama alignment on synthetic data with known answer."""

    def test_identity(self):
        """Identical point sets → R=I, t=0, s=1."""

        # 100 random points in 3D
        pts = np.random.rand(100, 3)  
        R, t, s = umeyama_alignment(pts, pts)
        assert np.allclose(R, np.eye(3), atol=1e-6), "Rotation matrix should be identity"
        assert np.allclose(t, np.zeros(3), atol=1e-6), "Translation vector should be zero"
        assert np.isclose(s, 1.0, atol=1e-6), "Scale should be 1.0"


    def test_known_rotation(self):
        """Apply a known 90-degree rotation, recover it."""
        # 100 random points in 3D
        pts = np.random.rand(100, 3)  

        # 90-degree rotation around Z-axis
        R_true = Rotation.from_euler('z', 90, degrees=True).as_matrix()  

        # Apply rotation
        pts_rotated = pts @ R_true.T  
        R, t, s = umeyama_alignment(pts, pts_rotated)
        assert np.allclose(R, R_true, atol=1e-6), "Rotation matrix should match the known rotation"

    def test_known_translation(self):
        """Apply a known translation, recover it."""

        # 100 random points in 3D
        pts = np.random.rand(100, 3)  

        #translation 
        t_true = np.array([10, 20, 30]) 
        pts_translated = pts + t_true

        R, t, s = umeyama_alignment(pts, pts_translated)
        assert np.allclose(t, t_true, atol=1e-6), "Translation vector should match the known translation"

    def test_known_scale(self):
        """Apply a known scale factor, recover it."""
        
        # 100 random points in 3D
        pts = np.random.rand(100, 3)  

        #scale
        s_true = 2.0
        pts_scaled = pts * s_true

        R, t, s = umeyama_alignment(pts, pts_scaled)
        assert np.isclose(s, s_true, atol=1e-6), "Scale factor should match the known scale"

N = 50
# poses: [x, y, z, qw, qx, qy, qz]      
poses = np.zeros((N, 7))
poses[:, 0] = np.arange(N) * 0.1
poses[:, 3] = 1.0  # Identity quaternion (qw=1, qx=qy=qz=0)

class TestATEComputation:
    """Test ATE on synthetic trajectories."""



    def test_identical_trajectories_zero_ate(self):
        """Identical src and dst → ATE ≈ 0."""
        #compute ate
        ate_error,_ = compute_ate(poses, poses)
        assert np.isclose(ate_error, 0.0, atol=1e-6), "ATE should be approximately zero for identical trajectories."

    def test_known_offset_ate(self):
        """Trajectory with constant offset → ATE equals that offset after alignment."""
        # Create a second trajectory with a constant offset
        offset = np.array([1.0, 2.0, 3.0])
        poses_offset = poses.copy()

        # Apply constant offset to xyz 
        poses_offset[:, :3] += offset    
        ate_error, _ = compute_ate(poses, poses_offset)
        assert np.isclose(ate_error, 0.0, atol=1e-6), "ATE should be approximately zero after alignment of trajectories with constant offset."


class TestRPEComputation:
    """Test RPE on synthetic trajectories."""

    def test_identical_trajectories_zero_rpe(self):
        """Identical src and dst → RPE ≈ 0."""
        # TODO: same approach as ATE — identical poses → RPE = 0
        rpe_err = compute_rpe(poses, poses)
        assert np.isclose(rpe_err, 0.0, atol=1e-6), "RPE should be approximately zero for identical trajectories."


    def test_constant_velocity_no_rpe(self):
        """Two trajectories with same relative motion → RPE ≈ 0."""
        
        # Create a second trajectory with the same relative motion but different global position
        poses_shifted = poses.copy()
        # Shift the entire trajectory by a constant offset
        poses_shifted[:, :3] += np.array([10.0, 20.0, 30.0])  
        rpe_err = compute_rpe(poses, poses_shifted)
        assert np.isclose(rpe_err, 0.0, atol=1e-6), "RPE should be approximately zero for trajectories with the same relative motion but different global positions."


class TestTimestampAlignment:
    """Test nearest-neighbor timestamp matching."""

    def test_exact_match(self):
        """When timestamps match exactly, index should be correct."""

        ts = np.array([100, 200, 300])
        idx, diff = find_nearest_timestamp(ts, 200)
        assert idx == 1, "Exact timestamp match should return correct index"
        assert diff == 0, "Exact timestamp match should have zero time difference"

    def test_nearest_neighbor(self):
        """Should find closest timestamp within tolerance."""

        ts = np.array([100, 200, 300])
        idx, diff = find_nearest_timestamp(ts, 195)
        assert idx == 1, "Should find nearest timestamp index"
        assert diff == 5, "Time difference should be 5 nanoseconds"

    def test_tolerance_exceeded(self):
        """Should warn/flag when nearest match exceeds 5ms."""

        ts = np.array([100, 200, 300])
        idx, diff = find_nearest_timestamp(ts, 500)
        assert idx == 2, "Should find nearest timestamp index"
        assert diff == 200, "Time difference should be 200 nanoseconds"
