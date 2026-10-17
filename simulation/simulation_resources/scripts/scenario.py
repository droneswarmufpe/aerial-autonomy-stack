#!/usr/bin/env python3
"""
Load, validate, and resolve a simulation scenario (Python standard library only, so it runs on the host and in the
simulation container).

A scenario is a JSON file with the world to load, the quads and VTOLs to spawn, and the custom objects to place:

    {
      "world": "esefex_fbx",
      "quads": [{"x": 0, "y": 0}],
      "vtols": [],
      "custom_objects": []
    }

Each vehicle accepts camera, rgbd_camera, thermal_camera, vio_bridge (quads only), camera_pitch (radians), x, y
(required), z, and yaw (degrees); missing fields take the defaults in VEHICLE_DEFAULTS. Each custom object has a
model, a global_position (lat, lon, alt) or a local_position (x, y, z), and optionally an id, an attitude (roll,
pitch, yaw in degrees), and static.

The first num_quads quads and num_vtols VTOLs are selected (all of them by default). Drone IDs are assigned to the
selected quads first, then to the selected VTOLs, starting from 1.

CLI usage:
    python3 scenario.py <scenario> [--num-quads N] [--num-vtols N] [--rcpilot true|false] [--jetson true|false]
                        [--resources DIR] [--format json|shell]

<scenario> is a scenario name (scenarios/<name>.json) or a path to a .json file. The resolved scenario is printed as
JSON, or as shell variable assignments for sim_run.sh. Validation errors are printed to stderr with exit code 1.
"""

import argparse
import json
import math
import os
import re
import shlex
import sys

RESOURCES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

VEHICLE_DEFAULTS = {
    'camera': True,
    'rgbd_camera': True,
    'thermal_camera': False,
    'vio_bridge': False,
    'camera_pitch': 1.5707,
    'z': 0.75,
    'yaw': 0.0,
}
VEHICLE_BOOLEANS = ('camera', 'rgbd_camera', 'thermal_camera', 'vio_bridge')
VEHICLE_NUMBERS = ('camera_pitch', 'x', 'y', 'z', 'yaw')
VEHICLE_REQUIRED = ('x', 'y')

VEHICLE_TYPES = {
    # type: (scenario key, base model in aircraft/vehicles)
    'quad': ('quads', 'iris_with_ardupilot'),
    'vtol': ('vtols', 'alti_transition_quad'),
}

OBJECT_DEFAULT_MODEL = 'ime-target'
OBJECT_MODEL_DIRS = ('objects', 'world_models')  # Searched in this order

NAME_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')


class ScenarioError(Exception):
    pass


def resolve_scenario_path(scenario, resources_dir=RESOURCES_DIR):
    """Return the path of a scenario given as a name (scenarios/<name>.json) or as a path to a .json file."""
    if scenario.endswith('.json') or os.sep in scenario:
        path = os.path.abspath(scenario)
    else:
        path = os.path.join(resources_dir, 'scenarios', f'{scenario}.json')
    if not os.path.isfile(path):
        available = ', '.join(list_names(os.path.join(resources_dir, 'scenarios'), '.json')) or 'none'
        raise ScenarioError(f"Scenario '{scenario}' not found at {path}. Available scenarios: {available}")
    return path


def list_names(directory, suffix=None):
    if not os.path.isdir(directory):
        return []
    entries = sorted(os.listdir(directory))
    if suffix:
        return [e[:-len(suffix)] for e in entries if e.endswith(suffix)]
    return [e for e in entries if os.path.isdir(os.path.join(directory, e))]


def find_object_model(model, resources_dir=RESOURCES_DIR):
    """Return the directory of a custom object model, or None."""
    for subdir in OBJECT_MODEL_DIRS:
        path = os.path.join(resources_dir, subdir, model)
        if os.path.isdir(path):
            return path
    return None


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _resolve_vehicle(vehicle, vehicle_type, index, errors):
    label = f'{VEHICLE_TYPES[vehicle_type][0]}[{index}]'
    if not isinstance(vehicle, dict):
        errors.append(f'{label} must be an object')
        return None
    unknown = set(vehicle) - set(VEHICLE_DEFAULTS) - set(VEHICLE_REQUIRED)
    if unknown:
        errors.append(f'{label} has unknown field(s): {", ".join(sorted(unknown))}')
    resolved = dict(VEHICLE_DEFAULTS)
    resolved.update(vehicle)
    for field in VEHICLE_REQUIRED:
        if field not in vehicle:
            errors.append(f'{label} is missing required field "{field}"')
    for field in VEHICLE_BOOLEANS:
        if not isinstance(resolved[field], bool):
            errors.append(f'{label}.{field} must be true or false')
    for field in VEHICLE_NUMBERS:
        if field in resolved and not _is_number(resolved[field]):
            errors.append(f'{label}.{field} must be a number')
    if vehicle_type == 'vtol' and resolved['vio_bridge'] is True:
        errors.append(f'{label}.vio_bridge is only supported on quads')
    if resolved['vio_bridge'] is True and not (resolved['camera'] is True and resolved['rgbd_camera'] is True):
        errors.append(f'{label}.vio_bridge requires camera and rgbd_camera to be true')
    return resolved


def _resolve_object(obj, index, resources_dir, errors):
    label = f'custom_objects[{index}]'
    if not isinstance(obj, dict):
        errors.append(f'{label} must be an object')
        return None
    unknown = set(obj) - {'id', 'model', 'global_position', 'local_position', 'attitude', 'static'}
    if unknown:
        errors.append(f'{label} has unknown field(s): {", ".join(sorted(unknown))}')
    resolved = {
        'id': obj.get('id', index + 1),
        'model': obj.get('model', OBJECT_DEFAULT_MODEL),
        'static': obj.get('static', False),
    }
    if not isinstance(resolved['model'], str) or not NAME_PATTERN.match(resolved['model']):
        errors.append(f'{label}.model must be a model folder name')
    elif find_object_model(resolved['model'], resources_dir) is None:
        errors.append(f"{label}.model '{resolved['model']}' not found in {' or '.join(d + '/' for d in OBJECT_MODEL_DIRS)}")
    if not isinstance(resolved['static'], bool):
        errors.append(f'{label}.static must be true or false')

    if 'global_position' in obj:
        position = obj['global_position']
        if not isinstance(position, dict) or not all(_is_number(position.get(k)) for k in ('lat', 'lon')):
            errors.append(f'{label}.global_position needs numeric lat and lon')
        elif 'alt' in position and not _is_number(position['alt']):
            errors.append(f'{label}.global_position.alt must be a number')
        else:
            resolved['global_position'] = {'lat': position['lat'], 'lon': position['lon'], 'alt': position.get('alt', 0)}
    elif 'local_position' in obj:
        position = obj['local_position']
        if not isinstance(position, dict) or not all(_is_number(position.get(k)) for k in ('x', 'y', 'z')):
            errors.append(f'{label}.local_position needs numeric x, y, and z')
        else:
            resolved['local_position'] = {k: position[k] for k in ('x', 'y', 'z')}
    else:
        errors.append(f'{label} needs a global_position or a local_position')

    attitude = obj.get('attitude', {})
    if not isinstance(attitude, dict):
        errors.append(f'{label}.attitude must be an object')
        attitude = {}
    resolved['attitude'] = {}
    for angle in ('roll', 'pitch', 'yaw'):
        value = attitude.get(angle, 0)
        if not _is_number(value):
            errors.append(f'{label}.attitude.{angle} must be a number')
        resolved['attitude'][angle] = value
    return resolved


def resolve(path, num_quads=None, num_vtols=None, rcpilot=None, jetson=None, resources_dir=RESOURCES_DIR):
    """Load the scenario at path and return it resolved: defaults filled in, vehicles selected, IDs assigned."""
    try:
        with open(path) as f:
            scenario = json.load(f)
    except json.JSONDecodeError as e:
        raise ScenarioError(f'{path} is not valid JSON: {e}')
    if not isinstance(scenario, dict):
        raise ScenarioError(f'{path} must contain a JSON object')

    errors = []
    unknown = set(scenario) - {'world', 'quads', 'vtols', 'custom_objects'}
    if unknown:
        errors.append(f'unknown top-level field(s): {", ".join(sorted(unknown))}')

    world = scenario.get('world')
    if not isinstance(world, str) or not NAME_PATTERN.match(world):
        errors.append('"world" must be the name of a world file in worlds/ (without .sdf)')
    elif not os.path.isfile(os.path.join(resources_dir, 'worlds', f'{world}.sdf')):
        available = ', '.join(list_names(os.path.join(resources_dir, 'worlds'), '.sdf'))
        errors.append(f"world '{world}' not found in worlds/. Available worlds: {available}")

    drones = []
    counts = {}
    for vehicle_type, limit in (('quad', num_quads), ('vtol', num_vtols)):
        key, base_model = VEHICLE_TYPES[vehicle_type]
        vehicles = scenario.get(key, [])
        if not isinstance(vehicles, list):
            errors.append(f'"{key}" must be a list')
            vehicles = []
        if limit is None:
            limit = len(vehicles)
        elif limit < 0:
            errors.append(f'the number of {key} cannot be negative')
            limit = 0
        elif limit > len(vehicles):
            errors.append(f'{limit} {key} requested but the scenario only defines {len(vehicles)}')
            limit = len(vehicles)
        counts[vehicle_type] = limit
        for index, vehicle in enumerate(vehicles[:limit]):
            resolved = _resolve_vehicle(vehicle, vehicle_type, index, errors)
            if resolved is not None:
                resolved.update({'type': vehicle_type, 'base_model': base_model})
                drones.append(resolved)

    for sitl_instance, drone in enumerate(drones):
        drone['id'] = sitl_instance + 1
        drone['sitl_instance'] = sitl_instance
        drone['model_name'] = f"{drone['base_model']}_{drone['id']}"
        if drone['vio_bridge'] and rcpilot is False:
            errors.append(f"drone {drone['id']} uses vio_bridge, which requires RCPILOT=true")
        if drone['vio_bridge'] and jetson is True:
            errors.append(f"drone {drone['id']} uses vio_bridge, which is not supported with JETSON=true")

    objects = scenario.get('custom_objects', [])
    if not isinstance(objects, list):
        errors.append('"custom_objects" must be a list')
        objects = []
    resolved_objects = [_resolve_object(obj, i, resources_dir, errors) for i, obj in enumerate(objects)]

    if errors:
        raise ScenarioError(f'Invalid scenario {path}:\n  - ' + '\n  - '.join(errors))

    return {
        'scenario_file': os.path.abspath(path),
        'world': world,
        'num_quads': counts['quad'],
        'num_vtols': counts['vtol'],
        'drones': drones,
        'custom_objects': resolved_objects,
    }


def _shell_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return shlex.quote(str(value))


def to_shell(resolved):
    """Shell variable assignments for sim_run.sh (eval'd there). Per-drone values are space-separated lists in drone ID order."""
    drones = resolved['drones']
    lines = [
        f"SCENARIO_FILE={_shell_value(resolved['scenario_file'])}",
        f"WORLD={_shell_value(resolved['world'])}",
        f"NUM_QUADS={resolved['num_quads']}",
        f"NUM_VTOLS={resolved['num_vtols']}",
        f"NUM_DRONES={len(drones)}",
        f"ANY_VIO_BRIDGE={_shell_value(any(d['vio_bridge'] for d in drones))}",
        "DRONE_TYPES=(" + ' '.join(d['type'] for d in drones) + ")",
        "DRONE_CAMERA_PITCHES=(" + ' '.join(_shell_value(d['camera_pitch']) for d in drones) + ")",
        "DRONE_VIO_BRIDGES=(" + ' '.join(_shell_value(d['vio_bridge']) for d in drones) + ")",
    ]
    return '\n'.join(lines)


def _parse_bool(value):
    if value not in ('true', 'false'):
        raise argparse.ArgumentTypeError('expected true or false')
    return value == 'true'


def _parse_count(value):
    # An empty value (e.g. an unset NUM_QUADS in sim_run.sh) means "all vehicles in the scenario"
    return None if value == '' else int(value)


def main():
    parser = argparse.ArgumentParser(description='Validate and resolve a simulation scenario')
    parser.add_argument('scenario', help='scenario name (scenarios/<name>.json) or path to a .json file')
    parser.add_argument('--num-quads', type=_parse_count, default=None, help='spawn only the first N quads')
    parser.add_argument('--num-vtols', type=_parse_count, default=None, help='spawn only the first N VTOLs')
    parser.add_argument('--rcpilot', type=_parse_bool, default=None)
    parser.add_argument('--jetson', type=_parse_bool, default=None)
    parser.add_argument('--resources', default=RESOURCES_DIR, help='simulation_resources directory')
    parser.add_argument('--format', choices=('json', 'shell'), default='json')
    args = parser.parse_args()

    try:
        path = resolve_scenario_path(args.scenario, args.resources)
        resolved = resolve(path, args.num_quads, args.num_vtols, args.rcpilot, args.jetson, args.resources)
    except ScenarioError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    print(to_shell(resolved) if args.format == 'shell' else json.dumps(resolved, indent=2))


if __name__ == '__main__':
    main()
