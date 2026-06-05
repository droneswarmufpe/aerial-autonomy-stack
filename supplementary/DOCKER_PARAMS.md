# Docker parameters

This document lists the environment variables supported by the Docker helper scripts under `scripts/`.

## scripts/sim_run.sh

Starts the simulation, ground, and aircraft containers (SITL by default, HITL when enabled).

| Variable | Default | Description |
| --- | --- | --- |
| `AUTOPILOT` | `ardupilot` | Autopilot stack (`px4` is removed). |
| `HEADLESS` | `false` | Disable GUI when `true`. |
| `CAMERA` | `true` | Enable camera sensors. |
| `LIDAR` | `false` | Enable LiDAR sensors. |
| `CAMERA_PITCH` | `1.5707` | Camera pitch in radians (90 deg down). |
| `SIM_SUBNET` | `10.42` | Base simulation subnet (overridden when `INSTANCE != 0`). |
| `AIR_SUBNET` | `10.22` | Base inter-vehicle subnet (overridden when `INSTANCE != 0`). |
| `SIM_ID` | `100` | Last byte of the simulation container IP. |
| `GROUND_ID` | `101` | Last byte of the ground container IP. |
| `NUM_QUADS` | `3` | Number of quadrotor containers. |
| `NUM_VTOLS` | `0` | Number of VTOL containers. |
| `WORLD` | `esefex_fbx` | Gazebo world name. |
| `CENTRALIZED` | `true` | When `true`, all aircraft data (including MAVLink) is streamed to the ground container so processing runs externally; `false` keeps streams local to each aircraft container. |
| `X_OFFSET` | `5` | X offset used to place drones. |
| `Y_OFFSET` | `-110` | Y offset used to place drones. |
| `DEV` | `false` | Mount workspaces/resources from host and use `/bin/bash` as entrypoint. |
| `HITL` | `false` | Use host networking and skip SITL networks when `true`. |
| `GND_CONTAINER` | `true` | Start the ground container when `true`. |
| `RTF` | `5.0` | Real-time factor (`<= 0.0` runs as fast as possible). |
| `START_AS_PAUSED` | `false` | Start simulation paused when `true`. |
| `INSTANCE` | `0` | Unique instance ID; offsets subnets and container/network names. |

## scripts/sim_build.sh

Clones external repositories and builds the simulation, ground, and aircraft images.

| Variable | Default | Description |
| --- | --- | --- |
| `CLEAN_BUILD` | `false` | Remove `github_clones`, delete images, and prune the builder cache before rebuilding. |
| `CLONE_ONLY` | `false` | Clone repositories without building Docker images. |

## scripts/deploy_run.sh

Runs the aircraft container on deployment/HITL setups and can also start the ground container.

| Variable | Default | Description |
| --- | --- | --- |
| `AUTOPILOT` | `ardupilot` | Autopilot stack (`px4` is removed). |
| `HEADLESS` | `true` | Disable GUI when `true`. |
| `CAMERA` | `true` | Enable camera sensors. |
| `LIDAR` | `true` | Enable LiDAR sensors. |
| `SIM_SUBNET` | `10.42` | Simulation subnet base. |
| `AIR_SUBNET` | `10.22` | Inter-vehicle subnet base. |
| `SIM_ID` | `100` | Last byte of the simulation container IP. |
| `GROUND_ID` | `101` | Last byte of the ground container IP. |
| `DRONE_TYPE` | `quad` | Drone type (`quad` or `vtol`). |
| `DRONE_ID` | `1` | Drone ID inside the container. |
| `NUM_QUADS` | `1` | Number of quadrotors (used by ground container). |
| `NUM_VTOLS` | `0` | Number of VTOLs (used by ground container). |
| `DEV` | `false` | Mount workspaces/resources from host and use `/bin/bash` as entrypoint. |
| `HITL` | `false` | Run in HITL mode when `true`. |
| `GND_CONTAINER` | `true` | Inform aircraft container whether a ground container is present. |
| `GROUND` | `false` | When `true`, launch the ground container instead of aircraft. |
