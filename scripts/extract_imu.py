# extract_imu.py                                                                                                                                                          
# Extract raw IMU data from RELLIS-3D rosbags → CSV for C++ preintegration.                                                                                               
#                                                                                                                                                                         
# Output format (consumed by imu::loadIMUFromCSV):
#   t, wx, wy, wz, ax, ay, az                                                                                                                                             
# where t is epoch seconds (double), omega in rad/s, accel in m/s².                                                                                                       

from rosbags.rosbag1 import Reader                                                                                                                                        
from rosbags.typesys import Stores, get_typestore
import numpy as np                                                                                                                                                        
import csv
import argparse                                                                                                                                                           
                
BAG_PATHS = [
    '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_00.bag',
    '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_01.bag',                                                                                                
    '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_02.bag',
    '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_03.bag',                                                                                                
    '/home/nishant/MS_Project/terra-perceive/data/RELLIS-3D/00000_04.bag',                                                                                                
]                                                                                                                                                                         
                                                                                                                                                                        
                                                                                                                                                                        
def extract_imu(bag_path):
    """Read one rosbag, return list of (t_seconds, wx, wy, wz, ax, ay, az)."""                                                                                            
    with Reader(bag_path) as bag:                                                                                                                                         
        typestore = get_typestore(Stores.ROS1_NOETIC)
        imu_conns = [c for c in bag.connections if c.topic == "/vectornav/IMU"]                                                                                           
                                                                                                                                                                        
        if not imu_conns:                                                                                                                                                 
            print(f"Topic /vectornav/IMU not found in {bag_path}")                                                                                                        
            return []                                                                                                                                                     
                
        rows = []                                                                                                                                                         
        for conn, timestamp, data in bag.messages(connections=imu_conns):
            msg = typestore.deserialize_ros1(data, conn.msgtype)                                                                                                          
            wx = msg.angular_velocity.x
            wy = msg.angular_velocity.y
            wz = msg.angular_velocity.z
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            rows.append((timestamp * 1e-9, wx, wy, wz, ax, ay, az))
                                                                                                                                                                        
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/imu_raw.csv",                                                                                                              
                        help="output CSV path")
    args = parser.parse_args()                                                                                                                                            
                                                                                                                                                                        
    all_rows = []
    for bag_path in BAG_PATHS:                                                                                                                                            
        print(f"Reading {bag_path}")
        rows = extract_imu(bag_path)
        print(f"  {len(rows)} IMU messages")                                                                                                                              
        all_rows.extend(rows)
                                                                                                                                                                        
    all_rows.sort(key=lambda row: row[0])  # sort by timestamp
    with open(args.out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "wx", "wy", "wz", "ax", "ay", "az"])
        writer.writerows(all_rows)
                                                                                                                                                                        
    print(f"Wrote {len(all_rows)} rows to {args.out}")                                                                                                                    
                                                                                                                                                                        
                
if __name__ == "__main__":
    main()