# Scenarios

A scenario is a JSON file that describes what to simulate: the Gazebo world, the quads and VTOLs (each with its own cameras and spawn pose), and the custom objects placed in the world. Every simulation run loads exactly one scenario.

Scenario files live in [`simulation/simulation_resources/scenarios/`](/simulation/simulation_resources/scenarios), but a scenario can also be any `.json` file on your computer.

## Running a scenario

```sh
cd aerial-autonomy-stack/scripts
./sim_run.sh                                                    # ESEFEX: the default scenario, one quad and the EVAAT GCS
SCENARIO=imav_hotspots ./sim_run.sh                             # A scenario in scenarios/ (name without .json)
SCENARIO=/path/to/my_scenario.json ./sim_run.sh                 # Any scenario file
SCENARIO=imav_vehicles NUM_QUADS=0 ./sim_run.sh                 # Only the world and its objects, no drones
```

`NUM_QUADS` and `NUM_VTOLS` spawn only the first N quads or VTOLs listed in the scenario. When they are not set, every vehicle in the scenario is spawned. Asking for more vehicles than the scenario defines is an error.

Give each vehicle a distinct spawn position with enough clearance for its body and payload. Identical resolved `(x, y, z)` positions are rejected before launch because overlapping collision meshes can abort Gazebo. For Iris quads, a 3-metre spacing is a useful starting point; distinct positions alone do not guarantee clearance for every vehicle model.

The other `sim_run.sh` variables (`RTF`, `HEADLESS`, `CENTRALIZED`, `RCPILOT`, `JETSON`, `JETSON_IPS`, `GND_CONTAINER`, `INSTANCE`, `START_AS_PAUSED`) are about how the simulation runs, not what it contains, so they stay environment variables. See [`DOCKER_PARAMS.md`](DOCKER_PARAMS.md).

In the Gymnasium environment, pass the scenario to `AASEnv` (a name of a scenario baked into `simulation-image`, or a path to a `.json` file on the host), with the same optional limits:

```python
env = AASEnv(scenario="gym", num_quads=1)
```

## File format

```json
{
  "world": "imav",
  "quads": [
    {"x": 0, "y": 0, "yaw": 90},
    {"x": 3, "y": 0, "rgbd_camera": false, "thermal_camera": true},
    {"global_position": {"lat": 48.80255, "lon": 7.85420, "alt": 1}}
  ],
  "vtols": [
    {"x": 0, "y": 4, "camera": false}
  ],
  "custom_objects": [
    {"id": 1, "model": "construction_cone", "global_position": {"lat": 48.810124, "lon": 7.853343, "alt": 4}},
    {"id": 2, "model": "ime-target", "local_position": {"x": 15, "y": -25, "z": 0}, "static": true}
  ]
}
```

| Field | Required | Description |
| --- | --- | --- |
| `world` | yes | Name of a world file in [`worlds/`](/simulation/simulation_resources/worlds), without `.sdf` (e.g. `esefex_fbx` loads `worlds/esefex_fbx.sdf`) |
| `quads` | no | List of quadcopters (ArduPilot Iris, `iris_with_ardupilot`) |
| `vtols` | no | List of VTOLs (ArduPilot Alti Transition, `alti_transition_quad`) |
| `custom_objects` | no | List of objects to place in the world, see [`CUSTOM_OBJECTS.md`](CUSTOM_OBJECTS.md) |

Unknown fields are rejected, so a typo (e.g. `"quad"` instead of `"quads"`) is reported instead of being silently ignored.

### Vehicles

Quads and VTOLs accept the same fields:

| Field | Default | Description |
| --- | --- | --- |
| `x`, `y` | required, or `global_position` | Spawn position in meters, in the world frame |
| `global_position` | — | Spawn position as `lat` and `lon` in degrees (and optionally `alt` in meters, used as `z`), instead of `x` and `y` |
| `z` | `0.75` | Spawn height in meters |
| `yaw` | `0` | Spawn heading in degrees |
| `camera` | `true` | Master camera switch: `false` disables all cameras, whatever their individual values |
| `rgbd_camera` | `true` | Master switch for all four Intel RealSense D435 sensors below |
| `rgb_camera` | `true` | D435 RGB color sensor and stream |
| `infrared_1_camera` | `true` | D435 left infrared sensor and stream |
| `infrared_2_camera` | `true` | D435 right infrared sensor and stream |
| `depth_camera` | `true` | D435 depth sensor and stream |
| `thermal_camera` | `false` | Thermal camera stream |
| `camera_pitch` | `1.5707` | Camera pitch in radians (`1.5707` = 90 deg, pointing down) |
| `vio_bridge` | `false` | Stereo visual-inertial odometry (quads only), see [Stereo VIO](#stereo-vio) |

So `{"x": 0, "y": 0}` is a complete vehicle: a downward-facing RGB-D camera, no thermal camera, at the world origin.

Set the individual camera fields on each vehicle in the scenario JSON. For RGB only:

```json
{
  "x": 0,
  "y": 0,
  "rgb_camera": true,
  "infrared_1_camera": false,
  "infrared_2_camera": false,
  "depth_camera": false,
  "thermal_camera": false
}
```

Disabled D435 sensors are omitted from the generated model, and their video bridges are not started. `camera` and `rgbd_camera` must both be `true` for any individual D435 switch to take effect. Omitted fields retain their defaults, so existing scenarios keep all four D435 sensors. Restart `sim_run.sh` after changing cameras.

The world frame is ENU, like Gazebo's: `x` points east, `y` points north, and `yaw` is measured counter-clockwise from `x`, so `"yaw": 90` faces north.

A `global_position` is converted to `x` and `y` with the world's origin, the `<spherical_coordinates>` of its `.sdf` file, the same way as for custom objects. Its `alt` is the spawn height relative to that origin, like `z`, not an altitude above sea level; without `alt`, `z` (default `0.75`) is used. A vehicle has either `x`/`y` or a `global_position`, and `alt` and `z` can't both be set. `scenario.py` prints the resulting `x` and `y`:

```sh
python3 simulation/simulation_resources/scripts/scenario.py my_scenario
```

### Custom objects

Each object places one model from [`objects/`](/simulation/simulation_resources/objects) (or [`world_models/`](/simulation/simulation_resources/world_models)), at a local position (meters) or at a global position (latitude/longitude, converted with the world's origin). The fields, the orientation conventions, and how to add new models are described in [`CUSTOM_OBJECTS.md`](CUSTOM_OBJECTS.md).

## Drone IDs

Drones are numbered from 1 in the order of the file: first the selected quads, then the selected VTOLs. With 2 quads and 1 VTOL:

| Drone ID | Vehicle | Gazebo model | SITL instance | `ROS_DOMAIN_ID` | Aircraft container |
| --- | --- | --- | --- | --- | --- |
| 1 | `quads[0]` | `iris_with_ardupilot_1` | 0 | 1 | `aircraft-container-inst0_1` |
| 2 | `quads[1]` | `iris_with_ardupilot_2` | 1 | 2 | `aircraft-container-inst0_2` |
| 3 | `vtols[0]` | `alti_transition_quad_3` | 2 | 3 | `aircraft-container-inst0_3` |

With `JETSON=true`, `JETSON_IPS` lists one Jetson per drone in this same order.

## Validation

`sim_run.sh` validates the scenario before starting any container, and stops with a list of every problem it finds. It checks that:

- the world file exists (the error lists the available worlds)
- every field has the right type and every vehicle has `x` and `y`
- every custom object has a model that exists and a position
- `NUM_QUADS`/`NUM_VTOLS` are not larger than the scenario's lists
- `vio_bridge` is only used on quads with `camera`, `rgbd_camera`, and both infrared cameras enabled, with `RCPILOT=true` and `JETSON=false`

You can run the same check without starting the simulation. It prints the resolved scenario, with every default filled in and the drone IDs assigned:

```sh
python3 simulation/simulation_resources/scripts/scenario.py imav_hotspots
python3 simulation/simulation_resources/scripts/scenario.py /path/to/my_scenario.json --num-quads 1
```

## What happens at startup

1. `sim_run.sh` resolves the scenario with [`scenario.py`](/simulation/simulation_resources/scripts/scenario.py) and mounts the `scenarios/` folder into the simulation container at `/aas/simulation_resources/scenarios` (a scenario file outside that folder is mounted on its own at `/aas/scenario.json`). The `SCENARIO_FILE` environment variable holds the scenario's path inside the simulation container. Each aircraft container gets its own drone's type, camera pitch, and VIO setting.
2. In the simulation container, [`build_world.py`](/simulation/simulation_resources/scripts/build_world.py) writes everything Gazebo needs to `/tmp/aas_generated/`:
    - one model per drone (`iris_with_ardupilot_1/`, ...) with its own name, topics, ArduPilot port, camera pitch, and only the cameras it enables; each enabled D435 assembly gets its own `d435_<id>/` model with disabled sensors removed and its nested name kept as `d435` to preserve topic paths
    - `world.sdf`: the scenario's world with ArduPilot's physics settings, the drones at their poses, and the custom objects
    - `scenario.resolved.json`: the resolved scenario, useful for debugging
3. Gazebo, one ArduPilot SITL per drone, and each drone's camera streams are started.

Nothing under `simulation_resources/` is modified, so every run starts from the same files.

### SITL estimator

Gazebo SITL uses simulator ground truth (`AHRS_EKF_TYPE=10`) by default. `build_world.py` writes a separate `sitl-estimator.params` in each generated vehicle directory, and `simulation.yml.erb` loads it after that vehicle's defaults. This avoids EKF convergence delay without changing RTF. Vehicles with `vio_bridge: true` use EKF3 (`AHRS_EKF_TYPE=3`) for estimator validation.

The ground-truth profile also uses direct simulated GPS (`GPS1_TYPE=100`) and one simulated IMU (`SIM_IMU_COUNT=1`). This removes GPS serial receiver probing and the multi-IMU consistency wait before arming. Calibration records for unused accelerometers are cleared in the generated overlay to match the simulated sensor count. Arming checks remain enabled; VIO scenarios retain their original GPS and IMU setup.

The override exists only in `/tmp/aas_generated/` inside the simulation container and is passed only to `sim_vehicle.py`. Source parameter files, EVAAT configs, and hardware autopilots are unchanged. Other pre-arm checks still apply.

**What needs an image rebuild?** With `sim_run.sh`, the `scenarios/`, resource `scripts/`, `objects/` and `world_models/` folders, the selected world file and the launch template are mounted from your computer. Changes apply on the next run without rebuilding. Aircraft models (`aircraft/`) are part of `simulation-image`: rebuild it after changing them (`CONTAINER=simulation ./sim_build.sh`). In the Gymnasium environment, a scenario given by name is read from the image, so pass a file path to use an edited scenario without rebuilding. New scenario fields require rebuilding the Gymnasium simulation image to include the updated scripts.

## Running the world again

To try changes to a scenario without restarting the containers, use the `gz_sim` window of the `Simulation` terminal. Its pane runs three separate commands, so each one is in the shell history:

1. Stop Gazebo with `Ctrl + c`.
2. Edit the scenario file on your computer (the `scenarios/` folder is mounted into the simulation container).
3. Press the up arrow twice and `Enter` to run `build_world.py` again: it regenerates `/tmp/aas_generated/world.sdf` from the edited scenario.
4. Press the up arrow twice and `Enter` to run `gz sim` again with the new world.

ArduPilot SITL and the camera streams keep running in the other windows and reconnect to the new Gazebo. This works for changes to the custom objects and to the vehicles' poses. SITL may report a GPS glitch when a vehicle's position jumps, which clears by itself; restart its `ardu_sitl_N` window if you need a clean autopilot state.

Changes to the number of vehicles, their cameras, `vio_bridge`, or the `world` need a new run of `sim_run.sh`: the SITL instances, camera streams, and ArduPilot's home location are set up from the scenario when the containers start.

## Stereo VIO

Set `"vio_bridge": true` on a quad to publish its D435 infrared stereo pair and the Gazebo IMU in its ROS domain and start OpenVINS in its aircraft container. It requires `camera`, `rgbd_camera`, `infrared_1_camera`, and `infrared_2_camera` on that quad, `RCPILOT=true`, and `JETSON=false`. RGB and depth can be disabled independently. Quads without `vio_bridge` in the same scenario keep their selected camera streams.

```sh
SCENARIO=calibration_vio RCPILOT=true RTF=1 ./sim_run.sh
```

See the Stereo VIO section of the [README](/README.md) for the topics and the calibration.

## Available scenarios

| Scenario | World | Vehicles | Custom objects |
| --- | --- | --- | --- |
| `default` | `esefex_fbx` | 1 quad (default of `sim_run.sh`) | none |
| `esefex_fbx` | `esefex_fbx` | 1 quad | none |
| `gym` | `impalpable_greyness` | 1 quad | none (default of the Gymnasium environment) |
| `calibration_vio` | `calibration_vio` | 1 quad with `vio_bridge` | none |
| `imav_hotspots` | `imav` | 1 quad | 2 construction cones and an ArUco marker |
| `imav_objects` | `imav` | 1 quad | 3 `rescue_randy` dummies, 2 vehicles, and an ArUco marker |
| `imav_vehicles` | `imav` | 1 quad | 5 vehicles (`panhard_vbl`, `renault_ccfm_a`, `renault_gbc180`, `toyota_hilux`, `arquus_vt4`) |
| `esefex_targets` | `esefex_fbx` | 1 quad | 4 `ime-target` targets |
| `esefex_targets_1`, `_2`, `_3` | `esefex_fbx` | 1 quad | 30 `ime-target` targets each |

[`generate_target_set.py`](/simulation/simulation_resources/scripts/generate_target_set.py) generates a new scenario with 30 random `ime-target` targets in `esefex_fbx` (`scenarios/esefex_targets_random.json`).

## Creating a scenario

1. Copy an existing scenario (e.g. `scenarios/default.json`) to a new file in `scenarios/`.
2. Set `world` to the world you want. To add a new world, see [`WORLD_GEN.md`](WORLD_GEN.md).
3. List the quads and VTOLs with their positions, and override only the camera fields that differ from the defaults.
4. Add custom objects, see [`CUSTOM_OBJECTS.md`](CUSTOM_OBJECTS.md).
5. Validate it with `scenario.py`, then run it with `SCENARIO=<name> ./sim_run.sh`.

Spawn the vehicles at least 2 m apart, and above the ground of the world: the default `z` of 0.75 m assumes flat ground at height 0 around the spawn point.
