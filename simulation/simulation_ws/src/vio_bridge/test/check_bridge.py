#!/usr/bin/env python3
"""Integration check inside simulation-image after building stereo_bridge.

Use --gazebo to test actual D435 rendering, CameraInfo and the plugin IMU.
Otherwise inject small RGB images to test pairing, pixels and capture stamps.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.camera_info_pb2 import CameraInfo


def find_model(models, name):
    """Return <name>/model.sdf from models/ or one of its subdirectories (e.g. aircraft/vehicles, aircraft/sensors)."""
    for candidate in [models / name, *sorted(models.glob(f'*/{name}'))]:
        if (candidate / 'model.sdf').is_file():
            return candidate / 'model.sdf'
    raise FileNotFoundError(f'Model {name} not found in {models}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--gazebo', action='store_true')
    args = parser.parse_args()
    os.environ['GZ_PARTITION'] = f'aas-vio-test-{os.getpid()}'
    os.environ['ROS_DOMAIN_ID'] = '84'
    prefix = '/world/calibration_vio/model/iris_with_ardupilot_1/model/iris'
    topics = [prefix + f'/model/d435/link/base_footprint/sensor/d435_ired{i}/image' for i in (1, 2)]
    rclpy.init()
    ros = rclpy.create_node('vio_test')
    received = [[], []]
    subscriptions = [ros.create_subscription(Image, f'/camera/infra{i+1}/image_raw', received[i].append, 10) for i in range(2)]
    imu = []
    subscriptions.append(ros.create_subscription(Imu, '/vio/imu', imu.append, qos_profile_sensor_data))
    transport = Node()
    info = []
    transport.subscribe(CameraInfo, topics[0].replace('/image', '/camera_info'), info.append)
    processes = []
    def spin(duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            rclpy.spin_once(ros, timeout_sec=0.05)
    def start(cmd):
        processes.append(subprocess.Popen(cmd))
    def stamps(messages):
        return [(m.header.stamp.sec, m.header.stamp.nanosec) for m in messages]
    try:
        start([args.binary, '--ros-args', '-p', 'gz_topic0:=' + topics[0], '-p', 'gz_topic1:=' + topics[1]])
        if args.gazebo:
            sdf = ET.Element('sdf', version='1.9')
            world = ET.SubElement(sdf, 'world', name='calibration_vio')
            physics = ET.SubElement(world, 'physics', name='2ms', type='ignore')
            ET.SubElement(physics, 'max_step_size').text = '0.002'
            ET.SubElement(physics, 'real_time_factor').text = '1'
            for kind, name in [('physics', 'Physics'), ('sensors', 'Sensors'), ('imu', 'Imu')]:
                plugin = ET.SubElement(world, 'plugin', filename=f'gz-sim-{kind}-system', name=f'gz::sim::systems::{name}')
                if kind == 'sensors': ET.SubElement(plugin, 'render_engine').text = 'ogre2'
            outer = ET.SubElement(world, 'model', name='iris_with_ardupilot_1')
            iris = ET.SubElement(outer, 'model', name='iris')
            source = ET.fromstring(find_model(args.models, 'iris_with_ardupilot').read_text().replace('<!--CAMERA_PITCH_PLACEHOLDER-->', '1.5707'))
            link = source.find('./model/model/link[@name="imu_link"]')
            ET.SubElement(link, 'gravity').text = 'false'
            iris.append(link)
            anchor = ET.SubElement(iris, 'joint', name='test_anchor', type='fixed')
            ET.SubElement(anchor, 'parent').text = 'world'
            ET.SubElement(anchor, 'child').text = 'imu_link'
            camera = ET.parse(find_model(args.models, 'd435')).getroot().find('model')
            # Remove unrelated cameras and meshes; preserve both actual IR sensor definitions.
            camera_link = camera.find('link')
            for child in list(camera_link):
                if child.tag in ('visual', 'collision') or (child.tag == 'sensor' and 'ired' not in child.get('name')):
                    camera_link.remove(child)
            ET.SubElement(camera_link, 'gravity').text = 'false'
            camera.append(source.find('./model/model/include[name="d435"]/pose'))
            iris.append(camera)
            joint = ET.SubElement(iris, 'joint', name='camera_mount', type='fixed')
            ET.SubElement(joint, 'parent').text = 'imu_link'
            ET.SubElement(joint, 'child').text = 'd435::base_footprint'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'world.sdf'
                ET.ElementTree(sdf).write(path)
                imu_topic = prefix + '/link/imu_link/sensor/imu_sensor/imu'
                start(['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge', imu_topic + '@sensor_msgs/msg/Imu[gz.msgs.IMU', '--ros-args', '-r', imu_topic + ':=/vio/imu'])
                start(['gz', 'sim', '-s', '--headless-rendering', '-r', str(path)])
                end = time.monotonic() + 45
                while min(map(len, received)) < 20 and time.monotonic() < end:
                    spin(0.1)
            assert len(info) and len(imu) >= 50, (len(info), len(imu))
            assert abs(info[-1].intrinsics.k[0] - 695.995) < .01, info[-1]
            assert abs(info[-1].intrinsics.k[2] - 640) < .01, info[-1]
            assert abs(info[-1].intrinsics.k[5] - 360) < .01, info[-1]
            assert abs(imu[-1].linear_acceleration.z + 9.8) < .1, imu[-1]
            assert len(set(stamps(imu))) > 40
            print(f'Gazebo: {list(map(len, received))} stereo frames; {len(imu)} IMU samples; CameraInfo intrinsics verified')
        else:
            pubs = [transport.advertise(t, GzImage) for t in topics]
            spin(2)
            def send(side, sec):
                msg = GzImage(width=2, height=1, step=6, pixel_format_type=3, data=bytes([255,0,0,0,255,0]))
                msg.header.stamp.sec = sec
                msg.header.stamp.nsec = 123456789
                pubs[side].publish(msg)
                spin(.2)
            send(0, 10)
            assert not any(received), 'Published before the other camera arrived'
            send(1, 11)
            assert not any(received), 'Paired different capture timestamps'
            send(0, 11)
            assert len(received[0]) == len(received[1]) == 1
            assert stamps(received[0]) == [(11, 123456789)]
            assert list(received[0][0].data) == [76, 150], list(received[0][0].data)
            send(0, 12); send(1, 12)
            assert len(received[0]) == 2
            send(0, 1); send(1, 1)
            assert stamps(received[0])[-1] == (1, 123456789), 'Reset did not recover'
            print('Synthetic: unequal timestamps rejected; equal pairs preserved; RGB-to-mono pixels and reset verified')
        common = set(stamps(received[0])) & set(stamps(received[1]))
        assert len(common) >= (15 if args.gazebo else 3), common
        assert all(m.encoding == 'mono8' and m.step == m.width for stream in received for m in stream)
    finally:
        for process in reversed(processes): process.send_signal(signal.SIGINT)
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        ros.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
