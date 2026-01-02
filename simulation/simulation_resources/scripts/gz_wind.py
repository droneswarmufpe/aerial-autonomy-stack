"""
Use as:
    python3 gz_wind.py --from_west 0.0 --from_south 3.0
    python3 gz_wind.py --stop_wind
"""
import os
import time
import argparse
import gz.transport13
from gz.msgs10.wind_pb2 import Wind


def main():
    parser = argparse.ArgumentParser(description='Control Gazebo wind plugin')
    parser.add_argument('--from_west', type=float, default=0.0, help='Number of seconds to advance the simulation')
    parser.add_argument('--from_south', type=float, default=0.0, help='Number of seconds to advance the simulation')
    parser.add_argument('--stop_wind', dest='stop_wind', action='store_true', help='Pause simulation after stepping (default)')
    args = parser.parse_args()
    
    world_name = os.environ.get('WORLD', 'default')

    gz_node = gz.transport13.Node()

    pub = gz_node.advertise(f"/world/{world_name}/wind/", Wind)
    timeout = 0
    while not pub.has_connections():
        if timeout > 100:  # Give up after 10s
            print("No subscribers found! Is Gazebo running?")
            return
        time.sleep(0.1)
        timeout += 1

    wind_msg = Wind()
    if args.stop_wind:
        wind_msg.enable_wind = False
        print("Disabling WindEffects")
    else:
        wind_msg.linear_velocity.x = args.from_west
        wind_msg.linear_velocity.y = args.from_south
        # wind_msg.linear_velocity.z = 0.0 # Unused
        wind_msg.enable_wind = True
        print("Enabling WindEffects")

    pub.publish(wind_msg)
    time.sleep(0.5) # Wait for publication

if __name__ == "__main__":
    main()
