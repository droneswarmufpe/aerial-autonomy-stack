#!/usr/bin/env python3
"""
Generate the Gazebo world and aircraft models for a scenario (see scenario.py).

For each selected drone, the base vehicle model (aircraft/vehicles/<base_model>) is copied to
<output>/<base_model>_<id> with a unique name, topics, and ArduPilot FDM port, its camera pitch, and only the cameras
the drone enables. The scenario's world (worlds/<world>.sdf) is copied to <output>/world.sdf with ArduPilot's
physics settings, the drones at their poses, and the custom objects. The resolved scenario is written to
<output>/scenario.resolved.json. Nothing is written outside <output>.

Usage:
    python3 build_world.py <scenario> [--num-quads N] [--num-vtols N] [--real-time-factor RTF]
                           [--output DIR] [--resources DIR]
"""

import argparse
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latlon_to_xy import latlon_to_xy  # noqa: E402
import scenario as scenario_lib  # noqa: E402

DEFAULT_OUTPUT_DIR = '/tmp/aas_generated'
WORLD_FILE_NAME = 'world.sdf'
RESOLVED_FILE_NAME = 'scenario.resolved.json'

BASE_FDM_PORT = 9002  # ArduPilot JSON FDM port of the first drone, +10 per drone

# ArduPilot SITL physics: 2 ms steps (500 Hz). The vehicles' .params set SCHED_LOOP_RATE 250, so that sensor updates
# are >1.8x faster than the loop rate, as required by the pre-flight checks. See also:
# https://discuss.ardupilot.org/t/multi-vehicle-faster-than-real-time-sitl-with-gazebo-harmonic/141068/3
PHYSICS_BLOCK = '''    <physics name="2ms" type="ignore">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>{rtf}</real_time_factor>
    </physics>'''

# Object meshes are Y-up: +90 deg roll stands them upright in Gazebo's Z-up frame
OBJECT_ROLL_CORRECTION_DEG = 90


def fmt(value):
    """Format a number for SDF: integers without a decimal point, other values with full precision."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _remove_marked_block(text, marker):
    """Remove the lines from '<!-- {marker}_BEGIN -->' to '<!-- {marker}_END -->' (inclusive)."""
    pattern = re.compile(rf'^[^\n]*<!-- {marker}_BEGIN -->.*?<!-- {marker}_END -->[^\n]*\n', re.DOTALL | re.MULTILINE)
    return pattern.sub('', text)


def build_rgbd_camera_model(drone, resources_dir, output_dir):
    """Generate a D435 per drone, omitting disabled sensors. The vehicle's include keeps its name 'd435' for topics."""
    name = f"d435_{drone['id']}"
    source_dir = os.path.join(resources_dir, 'aircraft', 'sensors', 'd435')
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    tree = ET.parse(os.path.join(source_dir, 'model.sdf'))
    tree.getroot().find('model').set('name', name)
    disabled = {sensor for field, sensor in scenario_lib.RGBD_CAMERA_SENSORS.items() if not drone[field]}
    for link in tree.iter('link'):
        for sensor in link.findall('sensor'):
            if sensor.get('name') in disabled:
                link.remove(sensor)
    tree.write(os.path.join(model_dir, 'model.sdf'), encoding='utf-8', xml_declaration=True)
    config = ET.parse(os.path.join(source_dir, 'model.config'))
    config.getroot().find('name').text = name
    config.write(os.path.join(model_dir, 'model.config'), encoding='utf-8', xml_declaration=True)
    return name


def build_vehicle_model(drone, resources_dir, output_dir):
    """Copy the drone's base model to <output>/<model_name>, adapted to the drone. Returns the model directory."""
    base, name, drone_id = drone['base_model'], drone['model_name'], drone['id']
    source_dir = os.path.join(resources_dir, 'aircraft', 'vehicles', base)
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(source_dir, 'model.sdf')) as f:
        text = f.read()
    text = text.replace(f'<model name="{base}">', f'<model name="{name}">')
    text = text.replace(f'<fdm_port_in>{BASE_FDM_PORT}</fdm_port_in>',
                        f'<fdm_port_in>{BASE_FDM_PORT + (drone_id - 1) * 10}</fdm_port_in>')
    text = text.replace('<!--CAMERA_PITCH_PLACEHOLDER-->', str(drone['camera_pitch']))
    if not (drone['camera'] and drone['rgbd_camera']):
        text = _remove_marked_block(text, 'RGBD_CAMERA')
    else:
        camera_model = build_rgbd_camera_model(drone, resources_dir, output_dir)
        text = text.replace('<uri>model://d435</uri>', f'<uri>model://{camera_model}</uri>')
    if not (drone['camera'] and drone['thermal_camera']):
        text = _remove_marked_block(text, 'THERMAL_CAMERA')
    if not drone['lidar']:
        text = _remove_marked_block(text, 'LIDAR')
    else:
        lidar_name = f"sensor_lidar_{drone_id}"
        lidar_dir = os.path.join(output_dir, lidar_name)
        os.makedirs(lidar_dir, exist_ok=True)
        for file_name in ('model.sdf', 'model.config'):
            shutil.copyfile(os.path.join(resources_dir, 'aircraft', 'sensors', 'sensor_lidar', file_name),
                            os.path.join(lidar_dir, file_name))
        text = text.replace('<uri>model://sensor_lidar</uri>', f'<uri>model://{lidar_name}</uri>')
    # Topics of the model, its payload plugins, and the particle emitter, e.g. /iris_with_ardupilot/... -> /iris_with_ardupilot_1/...
    for prefix in ('<topic>/', 'topic="/', '<cmd_topic>/'):
        text = text.replace(f'{prefix}{base}', f'{prefix}{name}')
    text = text.replace(f'/model/{base}/model/', f'/model/{name}/model/')

    with open(os.path.join(model_dir, 'model.sdf'), 'w') as f:
        f.write(text)
    # Loaded only by the Gazebo SITL launch command, after the copied vehicle defaults.
    # Ground-truth startup is an explicit per-vehicle opt-in, independent of VIO.
    with open(os.path.join(model_dir, 'sitl-estimator.params'), 'w') as f:
        f.write('# Generated for Gazebo SITL only; do not load onto a hardware autopilot.\n')
        if drone.get('sitl_ground_truth', False):
            f.write('AHRS_EKF_TYPE 10\n')
            # Read GPS directly from the simulator, avoiding serial receiver detection/configuration.
            f.write('GPS1_TYPE 100\n')
            # One simulated IMU avoids the multi-IMU consistency warm-up; arming checks still run.
            f.write('SIM_IMU_COUNT 1\n')
            # Base profiles contain calibration for extra IMUs. Clear only the unused instances.
            for index in (2, 3):
                f.write(f'INS_ACC{index}_ID 0\n')
                for axis in 'XYZ':
                    f.write(f'INS_ACC{index}OFFS_{axis} 0\n')
                    f.write(f'INS_ACC{index}SCAL_{axis} 1\n')
    # The SITL launch waits for this params copy: write the estimator overlay before it.
    for file_name in ('model.config', 'ardupilot-4.6.params'):
        shutil.copyfile(os.path.join(source_dir, file_name), os.path.join(model_dir, file_name))
    return model_dir


def world_real_time_factor(text):
    match = re.search(r'<physics.*?<real_time_factor>([^<]*)</real_time_factor>.*?</physics>', text, re.DOTALL)
    return match.group(1).strip() if match else '1.0'


def vehicle_include(drone):
    return (f'    <include>\n'
            f'      <uri>model://{drone["model_name"]}</uri>\n'
            f'      <pose degrees="true">{fmt(drone["x"])} {fmt(drone["y"])} {fmt(drone["z"])} 0 0 {fmt(drone["yaw"])}</pose>\n'
            f'    </include>\n')


def object_include(obj, index, origin):
    if 'global_position' in obj:
        if origin is None:
            raise scenario_lib.ScenarioError(
                f"custom object {obj['id']} uses global_position but the world has no <spherical_coordinates>")
        position = obj['global_position']
        x, y = latlon_to_xy(position['lat'], position['lon'], *origin)
        z = position['alt']
    else:
        x, y, z = (obj['local_position'][k] for k in ('x', 'y', 'z'))
    attitude = obj['attitude']
    # SDF pose angles are roll, pitch, yaw; the config yaw is negated
    roll, pitch, yaw = attitude['roll'] + OBJECT_ROLL_CORRECTION_DEG, attitude['pitch'], -attitude['yaw']
    return (f'    <include>\n'
            f'      <uri>model://{obj["model"]}</uri>\n'
            f'      <name>{obj["model"]}_{index}</name>\n'
            f'      <pose degrees="true">{fmt(x)} {fmt(y)} {fmt(z)} {fmt(roll)} {fmt(pitch)} {fmt(yaw)}</pose>\n'
            f'      <static>{"true" if obj["static"] else "false"}</static>\n'
            f'    </include>\n')


def build_world(resolved, resources_dir, output_dir, real_time_factor=None):
    """Write <output>/world.sdf for the resolved scenario. Returns its path."""
    world_file = os.path.join(resources_dir, 'worlds', f"{resolved['world']}.sdf")
    with open(world_file) as f:
        text = f.read()

    rtf = real_time_factor if real_time_factor is not None else world_real_time_factor(text)
    physics = PHYSICS_BLOCK.format(rtf=rtf)
    text, count = re.subn(r'^[^\n]*<physics.*?</physics>[^\n]*$', lambda _: physics, text, flags=re.DOTALL | re.MULTILINE)
    if count == 0:
        text = text.replace('</world>', f'{physics}\n</world>', 1)

    origin = scenario_lib.world_origin(world_file)
    includes = ''.join(vehicle_include(d) for d in resolved['drones'])
    includes += ''.join(object_include(o, i + 1, origin) for i, o in enumerate(resolved['custom_objects']))
    if text.count('</world>') != 1:
        raise scenario_lib.ScenarioError(f'{world_file} must contain exactly one </world>')
    text = text.replace('</world>', f'{includes}  </world>')

    path = os.path.join(output_dir, WORLD_FILE_NAME)
    with open(path, 'w') as f:
        f.write(text)
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate the Gazebo world and aircraft models for a scenario')
    parser.add_argument('scenario', help='scenario name (scenarios/<name>.json) or path to a .json file')
    parser.add_argument('--num-quads', type=scenario_lib._parse_count, default=None, help='spawn only the first N quads')
    parser.add_argument('--num-vtols', type=scenario_lib._parse_count, default=None, help='spawn only the first N VTOLs')
    parser.add_argument('--real-time-factor', default=None, help="Gazebo real time factor (default: the world's own)")
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='output directory')
    parser.add_argument('--resources', default=scenario_lib.RESOURCES_DIR, help='simulation_resources directory')
    args = parser.parse_args()

    try:
        path = scenario_lib.resolve_scenario_path(args.scenario, args.resources)
        resolved = scenario_lib.resolve(path, args.num_quads, args.num_vtols, resources_dir=args.resources)
        os.makedirs(args.output, exist_ok=True)
        for drone in resolved['drones']:
            build_vehicle_model(drone, args.resources, args.output)
        world_path = build_world(resolved, args.resources, args.output, args.real_time_factor)
    except scenario_lib.ScenarioError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(args.output, RESOLVED_FILE_NAME), 'w') as f:
        json.dump(resolved, f, indent=2)
    print(f"Scenario {path}: world '{resolved['world']}', {resolved['num_quads']} quad(s), "
          f"{resolved['num_vtols']} VTOL(s), {len(resolved['custom_objects'])} custom object(s)")
    for drone in resolved['drones']:
        cameras = []
        if drone['camera']:
            if drone['rgbd_camera']:
                cameras.extend(field for field in scenario_lib.RGBD_CAMERA_SENSORS if drone[field])
            if drone['thermal_camera']:
                cameras.append('thermal_camera')
        print(f"  drone {drone['id']}: {drone['model_name']} at ({fmt(drone['x'])}, {fmt(drone['y'])}, {fmt(drone['z'])}), "
              f"yaw {fmt(drone['yaw'])}, cameras: {', '.join(cameras) or 'none'}"
              f"{', lidar' if drone['lidar'] else ''}{', vio_bridge' if drone['vio_bridge'] else ''}")
    print(f'World written to {world_path}')


if __name__ == '__main__':
    main()
