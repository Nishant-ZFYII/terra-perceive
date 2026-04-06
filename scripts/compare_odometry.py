
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

BASE_PATH = 'data/'
rel_poses_gps_path = BASE_PATH + 'poses_gps.csv'
rel_poses_icp_path = BASE_PATH + 'poses_icp.csv'
rel_poses_carto_path = BASE_PATH + 'poses_carto.csv'

def load_poses(poses_gps_path, poses_icp_path, poses_carto_path):
    gps_df = pd.read_csv(poses_gps_path)
    icp_df = pd.read_csv(poses_icp_path)
    carto_df = pd.read_csv(poses_carto_path)

    # Align frame counts
    # same header -> timestamp, x, y, z, qw, qx, qy, qz
    # can't align with timestamp due to different formating 
    # align with frame count
    min_len = min(len(gps_df), len(icp_df), len(carto_df))
    gps_df = gps_df.iloc[:min_len].reset_index(drop=True)
    icp_df = icp_df.iloc[:min_len].reset_index(drop=True)
    carto_df = carto_df.iloc[:min_len].reset_index(drop=True)

    return gps_df, icp_df, carto_df

def umeyama_alignment(src, dst) -> tuple[np.ndarray, np.ndarray, float]:
    """
    src: (N, 3) array — source positions
    dst: (N, 3) array — destination positions
    Returns: R (3,3), t (3,), s (float)
    Such that: dst ≈ s * R @ src + t
    """
    #compute mean
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    #centre
    src_c = src - mu_src; dst_c = dst - mu_dst

    #variance
    sigma2_src = np.mean(np.sum(src_c**2, axis=1))

    #cross-covariance
    H = (dst_c.T @ src_c) / src.shape[0]
    
    #SVD
    U, Sigma, Vt = np.linalg.svd(H)

    #reflection
    S = np.diag([1,1,1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1

    #rotation
    R_mat = U @ S @ Vt

    #scale
    c = 1 / sigma2_src * np.sum(Sigma * np.diag(S))

    #translation
    t = mu_dst - c * R_mat @ mu_src

    return R_mat, t, c

def pose_to_se3(pose):
    #pose: (x, y, z, qw, qx, qy, qz)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

def compute_rpe(src, dst):
    #RPE -> relative pose error between consecutive frames for each source
    rpe_errors = []
    for i in range(1, src.shape[0]):
        P_i = pose_to_se3(src[i])
        P_prev = pose_to_se3(src[i-1])
        Q_i = pose_to_se3(dst[i])
        Q_prev = pose_to_se3(dst[i-1])

        #relative pose error
        F_i = np.linalg.inv(np.linalg.inv(Q_prev) @ Q_i) @ (np.linalg.inv(P_prev) @ P_i)
        rpe_errors.append(np.linalg.norm(F_i[:3, 3]))
    
    rpe_rmse = np.sqrt(np.mean(np.array(rpe_errors)**2))
    return rpe_rmse

def compute_ate(src, dst):
    #Q -> ground truth, P -> projected, S -> rigid body transform
    #F_i = Q_i.inv @ S @ P_i
    #rpe -> RMSE of translational component of F_i

    R_mat , t, scale = umeyama_alignment(src[:, :3], dst[:, :3])

    S = np.eye(4)
    S[:3, :3] = R_mat * scale
    S[:3, 3] = t

    errors = []
    for i in range(src.shape[0]):
        P_i = pose_to_se3(src[i])
        Q_i = pose_to_se3(dst[i])

        #relative pose error
        F_i = np.linalg.inv(Q_i) @ S @ P_i
        errors.append(np.linalg.norm(F_i[:3, 3]))
    
    errors = np.array(errors)
    ate_rmse = np.sqrt(np.mean(errors**2))
    return ate_rmse, errors


def align_trajectory(src, dst):
    """Align src to dst using Umeyama. Returns aligned (N,3) positions."""
    R_mat, t, s = umeyama_alignment(src, dst)
    return (s * R_mat @ src.T).T + t


def plot_trajectories(gps_poses, icp_poses, carto_poses):
    """Three-trajectory overlay plot (2D top-down) with Umeyama alignment."""
    carto_xy = carto_poses[:, :3]
    gps_aligned = align_trajectory(gps_poses[:, :3], carto_xy)
    icp_aligned = align_trajectory(icp_poses[:, :3], carto_xy)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: All three trajectories aligned
    ax = axes[0]
    ax.plot(carto_xy[:, 0], carto_xy[:, 1], 'g-', linewidth=1.2, label='Cartographer (ref)')
    ax.plot(gps_aligned[:, 0], gps_aligned[:, 1], 'b-', linewidth=0.8, alpha=0.7, label='GPS/IMU')
    ax.plot(icp_aligned[:, 0], icp_aligned[:, 1], 'r-', linewidth=0.8, alpha=0.7, label='KISS-ICP')
    ax.scatter(carto_xy[0, 0], carto_xy[0, 1], c='green', s=100, zorder=5, marker='o', label='Start')
    ax.scatter(carto_xy[-1, 0], carto_xy[-1, 1], c='black', s=100, zorder=5, marker='x', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Triple Odometry — Aligned Trajectories')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Per-frame ATE over time
    ax2 = axes[1]
    _, ate_gps_errors = compute_ate(gps_poses, carto_poses)
    _, ate_icp_errors = compute_ate(icp_poses, carto_poses)
    frames = np.arange(len(ate_gps_errors))
    ax2.plot(frames, ate_gps_errors, 'b-', linewidth=0.8, alpha=0.7, label='GPS vs Carto')
    ax2.plot(frames, ate_icp_errors, 'r-', linewidth=0.8, alpha=0.7, label='ICP vs Carto')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('ATE (m)')
    ax2.set_title('Absolute Trajectory Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/triple_odometry_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: output/triple_odometry_comparison.png")


def plot_rpe_over_time(gps_poses, icp_poses, carto_poses):
    """RPE over time for GPS and ICP vs Cartographer reference."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute per-frame RPE (reuse logic from compute_rpe but return per-frame)
    def rpe_per_frame(src, dst):
        errors = []
        for i in range(1, src.shape[0]):
            P_i = pose_to_se3(src[i])
            P_prev = pose_to_se3(src[i-1])
            Q_i = pose_to_se3(dst[i])
            Q_prev = pose_to_se3(dst[i-1])
            F_i = np.linalg.inv(np.linalg.inv(Q_prev) @ Q_i) @ (np.linalg.inv(P_prev) @ P_i)
            errors.append(np.linalg.norm(F_i[:3, 3]))
        return np.array(errors)

    rpe_gps = rpe_per_frame(gps_poses, carto_poses)
    rpe_icp = rpe_per_frame(icp_poses, carto_poses)
    frames = np.arange(len(rpe_gps))

    ax.plot(frames, rpe_gps, 'b-', linewidth=0.6, alpha=0.6, label='GPS vs Carto')
    ax.plot(frames, rpe_icp, 'r-', linewidth=0.6, alpha=0.6, label='ICP vs Carto')
    ax.set_xlabel('Frame')
    ax.set_ylabel('RPE (m)')
    ax.set_title('Relative Pose Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/rpe_over_time.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: output/rpe_over_time.png")

    return rpe_gps, rpe_icp


def print_summary_table(ate_gps_carto, ate_icp_carto, ate_gps_icp,
                        rpe_gps_carto, rpe_icp_carto, rpe_gps_icp):
    """Print the comparison summary table from the plan."""
    print("\n" + "="*85)
    print("TRIPLE ODOMETRY COMPARISON — RELLIS-3D Seq 00000")
    print("="*85)
    print(f"{'Pair':<25} {'ATE RMSE (m)':<15} {'RPE RMSE (m)':<15}")
    print("-"*55)
    print(f"{'GPS vs Cartographer':<25} {ate_gps_carto:<15.4f} {rpe_gps_carto:<15.4f}")
    print(f"{'ICP vs Cartographer':<25} {ate_icp_carto:<15.4f} {rpe_icp_carto:<15.4f}")
    print(f"{'GPS vs ICP':<25} {ate_gps_icp:<15.4f} {rpe_gps_icp:<15.4f}")
    print("-"*55)

    print(f"\n{'Source':<15} {'Type':<20} {'Loop Closure':<15} {'ATE vs Carto':<15} {'RPE vs Carto':<15}")
    print("-"*80)
    print(f"{'GPS/IMU':<15} {'Raw sensor':<20} {'N/A':<15} {ate_gps_carto:<15.4f} {rpe_gps_carto:<15.4f}")
    print(f"{'KISS-ICP':<15} {'Scan-to-scan':<20} {'No':<15} {ate_icp_carto:<15.4f} {rpe_icp_carto:<15.4f}")
    print(f"{'Cartographer':<15} {'Full SLAM':<20} {'Yes':<15} {'(reference)':<15} {'(reference)':<15}")
    print("="*80)


if __name__ == "__main__":
    gps_df, icp_df, carto_df = load_poses(rel_poses_gps_path, rel_poses_icp_path, rel_poses_carto_path)

    gps_poses = gps_df[['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].values
    icp_poses = icp_df[['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].values
    carto_poses = carto_df[['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].values

    # ATE for all 3 pairs
    ate_gps_carto, _ = compute_ate(gps_poses, carto_poses)
    ate_icp_carto, _ = compute_ate(icp_poses, carto_poses)
    ate_gps_icp, _ = compute_ate(gps_poses, icp_poses)

    # RPE for all 3 pairs
    rpe_gps_carto = compute_rpe(gps_poses, carto_poses)
    rpe_icp_carto = compute_rpe(icp_poses, carto_poses)
    rpe_gps_icp = compute_rpe(gps_poses, icp_poses)

    # Summary table
    print_summary_table(ate_gps_carto, ate_icp_carto, ate_gps_icp,
                        rpe_gps_carto, rpe_icp_carto, rpe_gps_icp)

    # Plots
    import os
    os.makedirs('output', exist_ok=True)
    plot_trajectories(gps_poses, icp_poses, carto_poses)
    plot_rpe_over_time(gps_poses, icp_poses, carto_poses)
