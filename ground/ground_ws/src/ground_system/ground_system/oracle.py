import rclpy
from rclpy.node import Node
import threading
import argparse
import time
import random

from pymavlink import mavutil

from ground_system_msgs.msg import SwarmObs, DroneObs

class OracleNode(Node):
    def __init__(self, num_drones, ip, base_port, publish_rate):
        super().__init__('oracle')
        
        self.drone_obs = {}  # Thread-safe storage for latest data
        self.lock = threading.Lock()

        # Start a listener thread for each drone's UDP MAVLink connection
        for i in range(num_drones):
            drone_id = i + 1
            port = base_port + i
            connection_string = f"udp:{ip}:{port}"
            thread = threading.Thread(target=self.mavlink_listener, args=(drone_id, connection_string))
            thread.daemon = True
            thread.start()
            self.get_logger().info(f"Listening MAVLink for drone {drone_id} on {connection_string}")

        # Create the ROS 2 publisher and timer
        self.publisher = self.create_publisher(SwarmObs, '/tracks', 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_swarm_obs)

    def mavlink_listener(self, drone_id, connection_string):
        master = mavutil.mavlink_connection(connection_string)
        
        while rclpy.ok():
            try:
                # Wait for a new message with position and velocity
                msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False) # https://mavlink.io/en/messages/common.html#GPS_RAW_INT
                if not msg:
                    continue

                # MAVLink sends GPS data as integers, convert to standard units
                obs = {
                    'lat': msg.lat / 1e7,
                    'lon': msg.lon / 1e7,
                    'alt': msg.alt / 1000.0, # MSL, alternatively use msg.relative_alt
                    'vx': msg.vx / 100.0, # cm/s to m/s
                    'vy': msg.vy / 100.0, # cm/s to m/s
                    'vz': msg.vz / 100.0  # cm/s to m/s
                }
                
                with self.lock:
                    self.drone_obs[drone_id] = obs

                time.sleep(0.02) # Sleep for 20ms to limit CPU usage
                    
            except Exception as e:
                self.get_logger().error(f"MAVLink listener error for drone {drone_id}: {e}")

    def publish_swarm_obs(self):
        POS_STD_DEV_DEG = 1e-5 # Approx. 1.1 meters of noise for lat, 1.1 or less for lon
        ALT_STD_DEV_M = 0.5 # 0.5 meters of noise for altitude
        VEL_STD_DEV_MS = 0.1 # 0.1 m/s of noise for velocity

        swarm_msg = SwarmObs()
        swarm_msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.lock:
            current_obs = self.drone_obs.copy()

        for drone_id, track in current_obs.items():
            drone_curr_obs = DroneObs()
            drone_curr_obs.id = drone_id
            drone_curr_obs.label = 48 if drone_id == 2 else 0 # Hardcode drone 2 as the talking dead
            # Noisy measurements
            drone_curr_obs.latitude_deg = self.add_noise(track.get('lat', 0.0), POS_STD_DEV_DEG)
            drone_curr_obs.longitude_deg = self.add_noise(track.get('lon', 0.0), POS_STD_DEV_DEG)
            drone_curr_obs.altitude_m = self.add_noise(track.get('alt', 0.0), ALT_STD_DEV_M)
            drone_curr_obs.velocity_n_m_s = self.add_noise(track.get('vx', 0.0), VEL_STD_DEV_MS)
            drone_curr_obs.velocity_e_m_s = self.add_noise(track.get('vy', 0.0), VEL_STD_DEV_MS)
            drone_curr_obs.velocity_d_m_s = self.add_noise(track.get('vz', 0.0), VEL_STD_DEV_MS)
            swarm_msg.tracks.append(drone_curr_obs)
        
        if swarm_msg.tracks:
            self.publisher.publish(swarm_msg)

    def add_noise(self, value, std_dev):
        return value + random.gauss(0, std_dev)

def main(args=None):
    parser = argparse.ArgumentParser(description='Oracle Node')
    parser.add_argument('--num-drones', type=int, required=True, help='Number of drones to listen for.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address to listen on.')
    parser.add_argument('--base-port', type=int, default=14540, help='Base UDP port to start listening from.')
    parser.add_argument('--rate', type=float, default=10.0, help='Publishing rate in Hz.')
    
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    
    node = OracleNode(
        num_drones=cli_args.num_drones,
        ip=cli_args.ip,
        base_port=cli_args.base_port,
        publish_rate=cli_args.rate
    )
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
