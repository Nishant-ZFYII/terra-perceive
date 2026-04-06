from rosbags.rosbag1 import Reader
import numpy as np
import csv
import os
import argparse
import pyproj
from rosbags.typesys import Stores, get_typestore

'''
bag_paths = ['/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_00.bag',
             '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_01.bag',
             '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_02.bag',
             '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_03.bag',
             '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_04.bag']
'''



def extract_poses_gps(bag_path): 
    with Reader(bag_path) as bag:
        typestore = get_typestore(Stores.ROS1_NOETIC)
        gps_conns = [c for c in bag.connections if c.topic == "/vectornav/GPS"]
        imu_conns = [c for c in bag.connections if c.topic == "/vectornav/IMU"]
        lidar_conns = [c for c in bag.connections if c.topic == "/os1_cloud_node/points"]

        if not gps_conns:
            print(f"Topic /vectornav/GPS not found in bag.")
            return 0
        if not imu_conns:
            print(f"Topic /vectornav/IMU not found in bag.")
            return 0
        if not lidar_conns:
            print(f"Topic /os1_cloud_node/points not found in bag.")
            return 0
        gps_data = []
        for conn, timestamp, data in bag.messages(connections=gps_conns):
            msg = typestore.deserialize_ros1(data, conn.msgtype)
            gps_data.append((timestamp, msg.latitude, msg.longitude, msg.altitude))
        imu_data = []
        for conn, timestamp, data in bag.messages(connections=imu_conns):
            msg = typestore.deserialize_ros1(data, conn.msgtype)
            imu_data.append((timestamp, msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z))
        lidar_timestamps = []
        for conn, timestamp, data in bag.messages(connections=lidar_conns):
            lidar_timestamps.append(timestamp)

    return gps_data, imu_data, lidar_timestamps

def convert_gps_to_enu(gps_data):
    # pyproj with a proper WGS84 ellipsoid model
    #https://pyproj4.github.io/pyproj/stable/advanced_examples.html#:~:text=tg%20%3D%20TransformerGroup(%22EPSG%3A4326%22%2C%20%22%2Bproj%3Daea%20%2Blat_0%3D50%20%2Blon_0%3D%2D154%20%2Blat_1%3D55%20%2Blat_2%3D65%20%2Bx_0%3D0%20%2By_0%3D0%20%2Bdatum%3DNAD27%20%2Bno_defs%20%2Btype%3Dcrs%20%2Bunits%3Dm%22%2C%20always_xy%3DTrue)
    gps_data = np.array(gps_data)
    lat0, lon0, alt0 = gps_data[0, 1:4]  # Use first GPS fix as origin
    transformer = pyproj.Transformer.from_crs("epsg:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)
    lat, lon, alt = gps_data[:, 1], gps_data[:, 2], gps_data[:, 3]
    x, y = transformer.transform(lon, lat)  
    enu_coords = np.stack([x, y, alt - alt0], axis=1) 
    return enu_coords

#for each lidar timestamp, find nearest GPS + nearest IMU
def associate_data(gps_data, enu_coords, imu_data, lidar_timestamps):

    #timestamp check
    #using dtype = np.int64 
    #check how far apart the timestamps are for GPS and IMU, and LiDAR. 
    #If they are too far apart, we might want to discard those frames or interpolate.
    #<=5ms torelance, check this ;if it is not then warn when the nearest match exceeds it. 
    associated_data = []
    gps_ts = np.array([np.int64(i[0]) for i in gps_data])
    imu_ts = np.array([np.int64(i[0]) for i in imu_data])
    for lidar_ts in lidar_timestamps:
        #we subtract lidar_ts from gps and imu timestamps to find the nearest ones. We take the absolute value to find the closest match regardless of whether it is before or after the LiDAR timestamp.
        nearest_gps_idx = np.searchsorted(gps_ts, np.int64(lidar_ts))
        nearest_gps_idx = min(nearest_gps_idx,len(gps_ts)-1) 
        if nearest_gps_idx > 0 and abs(gps_ts[nearest_gps_idx] - np.int64(lidar_ts)) > abs(gps_ts[nearest_gps_idx - 1] - np.int64(lidar_ts)):
            nearest_gps_idx -= 1
        nearest_gps = gps_data[nearest_gps_idx]
        nearest_imu_idx = np.searchsorted(imu_ts, np.int64(lidar_ts))
        nearest_imu_idx = min(nearest_imu_idx, len(imu_ts) - 1)
        if nearest_imu_idx > 0 and abs(imu_ts[nearest_imu_idx] - np.int64(lidar_ts)) > abs(imu_ts[nearest_imu_idx - 1] - np.int64(lidar_ts)):
            nearest_imu_idx -= 1
        nearest_imu = imu_data[nearest_imu_idx]
        gps_time_diff = abs(np.int64(nearest_gps[0]) - np.int64(lidar_ts))
        imu_time_diff = abs(np.int64(nearest_imu[0]) - np.int64(lidar_ts))
        if gps_time_diff > 5e6:  # 5 milliseconds in nanoseconds
            print(f"Warning: Nearest GPS timestamp is {gps_time_diff/1e6:.2f} milliseconds away from LiDAR timestamp {lidar_ts}")
        if imu_time_diff > 25e6:  # 25 milliseconds in nanoseconds
            print(f"Warning: Nearest IMU timestamp is {imu_time_diff/1e6:.2f} milliseconds away from LiDAR timestamp {lidar_ts}")
        enu_val = enu_coords[nearest_gps_idx]
        associated_data.append((lidar_ts, enu_val[0], enu_val[1], enu_val[2] , nearest_imu[1], nearest_imu[2], nearest_imu[3], nearest_imu[4]))
    return associated_data


        
def write_csv(associated_data, out_csv):
    #if the output directory does not exist, create it
    #if file is present append to it. 
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])  # Write header
        for data in associated_data:
            writer.writerow(data)
    print(f"CSV saved to {out_csv}")

def find_nearest_timestamp(sorted_timestamps, query_ts):                                                                          
    """Find index of nearest timestamp. Returns (index, time_diff_ns)."""                                                       
    idx = np.searchsorted(sorted_timestamps, np.int64(query_ts))                                                                  
    idx = min(idx, len(sorted_timestamps) - 1)                                                                                    
    if idx > 0 and abs(sorted_timestamps[idx-1] - query_ts) < abs(sorted_timestamps[idx] - query_ts):                             
        idx = idx - 1                                                                                                             
    diff = abs(int(sorted_timestamps[idx]) - int(query_ts))                                                                     
    return idx, diff

if __name__ == "__main__":
    #nargs='+ for mulitple bag files.

    parser = argparse.ArgumentParser(description="Extract LiDAR frames from ROS1 bag")
    parser.add_argument("bag_path", nargs='+', help="Path to .bag file")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    all_gps, all_imus, all_lidar_ts = [], [], []
    for bag_path in args.bag_path:
        gps_data, imu_data, lidar_timestamps = extract_poses_gps(bag_path)
        all_gps.extend(gps_data)
        all_imus.extend(imu_data)
        all_lidar_ts.extend(lidar_timestamps)
    enu_coords = convert_gps_to_enu(all_gps)
    associated_data = associate_data(all_gps, enu_coords, all_imus, all_lidar_ts)
    write_csv(associated_data, args.out)
    
