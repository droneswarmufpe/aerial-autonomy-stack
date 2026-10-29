"""Run in an aircraft container while check_bridge.py --gazebo runs in simulation."""
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

rclpy.init()
node = rclpy.create_node('aircraft_vio_delivery_test')
images = [set(), set()]
imu_stamps = set()
def stamp(msg):
    return msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
def image(side, msg):
    assert msg.encoding == 'mono8'
    assert (msg.width, msg.height, msg.step) == (1280, 720, 1280)
    images[side].add(stamp(msg))
subscriptions = [node.create_subscription(Image, f'/camera/infra{i+1}/image_raw',
                  lambda msg, side=i: image(side, msg), 10) for i in range(2)]
subscriptions.append(node.create_subscription(Imu, '/vio/imu',
                    lambda msg: imu_stamps.add(stamp(msg)), qos_profile_sensor_data))
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=.1)
    if len(images[0] & images[1]) >= 10 and len(imu_stamps) >= 50:
        break
assert len(images[0] & images[1]) >= 10, [len(s) for s in images]
assert len(imu_stamps) >= 50, len(imu_stamps)
print(f'Aircraft container received {len(images[0] & images[1])} paired full-resolution mono images and {len(imu_stamps)} distinct IMU samples')
node.destroy_node()
rclpy.shutdown()
