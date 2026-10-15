#!/usr/bin/env python3
"""Generate OpenVINS calibration from the same Iris/D435 SDFs used by Gazebo.

Only the current rigid Iris model is supported. Fail on relative_to poses instead
of silently interpreting a new SDF frame graph incorrectly.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def pose(element):
    result = np.eye(4)
    p = element.find('pose')
    if p is None:
        return result
    if p.get('relative_to') or p.get('rotation_format', 'euler_rpy') != 'euler_rpy':
        raise ValueError('Unsupported SDF pose reference/rotation format')
    x, y, z, r, pitch, yaw = map(float, p.text.split())
    if p.get('degrees', 'false') == 'true':
        r, pitch, yaw = map(math.radians, (r, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]]
    result[:3, 3] = [x, y, z]
    return result


def calibration(models, camera_pitch):
    text = (models / 'iris_with_ardupilot/model.sdf').read_text()
    iris = ET.fromstring(text.replace('<!--CAMERA_PITCH_PLACEHOLDER-->', str(camera_pitch))).find('./model/model[@name="iris"]')
    imu_link = iris.find('./link[@name="imu_link"]')
    imu = imu_link.find('./sensor[@name="imu_sensor"]')
    imu_pose = pose(imu_link) @ pose(imu)
    mount = iris.find('./include[name="d435"]')
    d435 = ET.parse(models / 'd435/model.sdf').getroot().find('model')
    link = d435.find('./link[@name="base_footprint"]')
    # Optical axes (right, down, forward) expressed in Gazebo camera (forward,left,up).
    optical = np.eye(4)
    optical[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cameras = {}
    for i in range(2):
        sensor = link.find(f'./sensor[@name="d435_ired{i+1}"]')
        camera = sensor.find('camera')
        w, h = int(camera.findtext('image/width')), int(camera.findtext('image/height'))
        focal = w / (2 * math.tan(float(camera.findtext('horizontal_fov')) / 2))
        transform = np.linalg.inv(imu_pose) @ pose(mount) @ pose(d435) @ pose(link) @ pose(sensor) @ optical
        cameras[f'cam{i}'] = dict(T_imu_cam=transform.tolist(), cam_overlaps=[1-i],
            camera_model='pinhole', distortion_model='radtan', distortion_coeffs=[0., 0., 0., 0.],
            intrinsics=[focal, focal, w / 2., h / 2.], resolution=[w, h],
            rostopic=f'/camera/infra{i+1}/image_raw', timeshift_cam_imu=0.)
    return cameras


def write_yaml(path, data):
    # OpenCV FileStorage does not accept PyYAML's indentless block sequences.
    class OpenCVDumper(yaml.SafeDumper):
        pass
    OpenCVDumper.add_representer(list, lambda dumper, value:
        dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True))
    path.write_text('%YAML:1.0\n---\n' + yaml.dump(data, Dumper=OpenCVDumper,
                                                sort_keys=False, width=10000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--camera-pitch', type=float, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cameras = calibration(args.models, args.camera_pitch)
    config = yaml.safe_load('\n'.join(args.template.read_text().splitlines()[1:]))
    # Current AAS worlds use SDF's default gravity (0, 0, -9.8).
    config.update(verbosity='INFO', gravity_mag=9.8, use_stereo=True, max_cameras=2, calib_cam_extrinsics=False,
                  calib_cam_intrinsics=False, calib_cam_timeoffset=False,
                  calib_imu_intrinsics=False, calib_imu_g_sensitivity=False,
                  downsample_cameras=False, track_frequency=31.,
                  relative_config_imucam='kalibr_imucam_chain.yaml',
                  relative_config_imu='kalibr_imu_chain.yaml')
    # The plugin's Gazebo IMU has no injected noise. Small nonzero filter noise
    # keeps covariance well conditioned; these are tuning values, not measured hardware noise.
    imu = dict(rostopic='/vio/imu', time_offset=0., update_rate=500., model='kalibr',
               accelerometer_noise_density=0.002, accelerometer_random_walk=0.00001,
               gyroscope_noise_density=0.0002, gyroscope_random_walk=0.000001,
               T_i_b=np.eye(4).tolist(), Tw=np.eye(3).tolist(), Ta=np.eye(3).tolist(),
               R_IMUtoGYRO=np.eye(3).tolist(), R_IMUtoACC=np.eye(3).tolist(),
               Tg=np.zeros((3,3)).tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    write_yaml(args.output / 'kalibr_imucam_chain.yaml', cameras)
    write_yaml(args.output / 'kalibr_imu_chain.yaml', {'imu0': imu})
    write_yaml(args.output / 'estimator_config.yaml', config)
    print(f'Generated SDF-derived VIO calibration in {args.output}')


if __name__ == '__main__':
    main()
