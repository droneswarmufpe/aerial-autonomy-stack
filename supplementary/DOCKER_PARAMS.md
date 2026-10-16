# Docker parameters

This document lists the environment variables supported by the Docker helper scripts under `scripts/`.

## scripts/sim_run.sh

Starts the simulation, ground, and aircraft containers (SITL by default, Jetson-in-the-loop with `JETSON=true`).

| Variable | Default | Description |
| --- | --- | --- |
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
| `JETSON` | `false` | Jetson-in-the-loop: the simulation container uses host networking and streams each drone's SITL MAVLink (UDP `14550`) and cameras (UDP `5600`, `5700`, `5710`, `5720`, `5800`) to its own Jetson in `JETSON_IPS`. Aircraft containers are not started. Not compatible with `CENTRALIZED=true` or `VIO_BRIDGE=true`. |
| `JETSON_IPS` | `192.168.55.1` | Comma-separated Jetson IPs on the same LAN as the PC, one per drone in drone ID order (quads first, then VTOLs), e.g. `192.168.1.50,192.168.1.51`. Used when `JETSON=true`. |
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
