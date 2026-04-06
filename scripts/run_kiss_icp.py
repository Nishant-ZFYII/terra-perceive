import glob
import os

import numpy as np
import argparse
from kiss_icp.pipeline import OdometryPipeline
from scipy.spatial.transform import Rotation as R




#https://github.com/PRBonn/kiss-icp/blob/main/python/kiss_icp/datasets/kitti.py

class RELLIS_3D_Dataset:

    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = str(sequence).zfill(2)
        self.rellis_sequence_dir = os.path.join(data_dir, "Rellis-3D", self.sequence_id)
        self.bin_dir = os.path.join(self.rellis_sequence_dir, "os1_cloud_node_kitti_bin/")

        self.scan_files = sorted(glob.glob(self.bin_dir + "*.bin"))
 
    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        # Load the .bin file at the given index
        scan_file = self.scan_files[idx]

        #remove intensity
        point_cloud = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]  
        return point_cloud,np.array([])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KISS-ICP on RELLIS-3D dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence ID to process")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    dataset = RELLIS_3D_Dataset(args.data_dir, args.sequence)
    print(f"Loaded {len(dataset)} scans from sequence {args.sequence} in {args.data_dir}")

    
    pipeline = OdometryPipeline(dataset=dataset, visualize=True)
    results = pipeline.run()
    poses = pipeline.poses #(N,4,4)

    #convert to (x, y, z, qw, qx, qy, qz)
    '''
     2. Missing timestamp column. Your GPS CSV has timestamp, x, y, z, qw, qx, qy, qz. Your ICP CSV header on line 73 only has         
  x,y,z,qw,qx,qy,qz. They need to match for the M1.5 comparison. ICP doesn't have GPS timestamps, so use the frame index as a proxy:
   frame number × 100ms (10 Hz) in nanoseconds, or just the integer frame index. Simplest: use the LiDAR bin filename without       
  extension as the frame ID.   
    '''
    csv_poses = []
    for idx, pose in enumerate(poses):
        x, y, z = pose[:3, 3]
        r = R.from_matrix(pose[:3, :3])
        qx, qy, qz, qw = r.as_quat()  # scipy gives (qx, qy, qz, qw)
        timestamp = idx * 1e8 # Use frame index as a proxy for timestamp in nanoseconds (assuming 10 Hz LiDAR)
        csv_poses.append((timestamp, x, y, z, qw, qx, qy, qz))
    csv_poses = np.array(csv_poses)

    # Save to CSV
    out_csv_path = args.out
    np.savetxt(out_csv_path, csv_poses, delimiter=",", header="timestamp,x,y,z,qw,qx,qy,qz", comments="")
    print(f"Saved poses to {out_csv_path}")
