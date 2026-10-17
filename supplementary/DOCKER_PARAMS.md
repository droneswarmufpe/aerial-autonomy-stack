# Docker parameters

This document lists the environment variables supported by the Docker helper scripts under `scripts/`.

## scripts/sim_run.sh

Starts the simulation, ground, and aircraft containers (SITL by default, Jetson-in-the-loop with `JETSON=true`).

| Variable | Default | Description |
| --- | --- | --- |
| `SCENARIO` | `default` | Scenario to load: a name in `simulation/simulation_resources/scenarios/` (without `.json`) or a path to a `.json` file. It sets the world, the quads and VTOLs (cameras, VIO bridge, camera pitch, pose), and the custom objects. See [`SCENARIOS.md`](SCENARIOS.md). |
| `NUM_QUADS` | all | Spawn only the first `NUM_QUADS` quads of the scenario. |
| `NUM_VTOLS` | all | Spawn only the first `NUM_VTOLS` VTOLs of the scenario. |
| `HEADLESS` | `false` | Disable GUI when `true`. |
| `SIM_SUBNET` | `10.42` | Base simulation subnet (overridden when `INSTANCE != 0`). |
| `AIR_SUBNET` | `10.22` | Base inter-vehicle subnet (overridden when `INSTANCE != 0`). |
| `SIM_ID` | `100` | Last byte of the simulation container IP. |
| `GROUND_ID` | `101` | Last byte of the ground container IP. |
| `CENTRALIZED` | `false` | When `true`, all aircraft data (including MAVLink) is streamed to the ground container so processing runs externally; `false` keeps streams local to each aircraft container. |
| `RCPILOT` | `false` | Use the rcpilot images (`rc-ground-image`, `rc-aircraft-image`) and start rcpilot. Required by quads with `vio_bridge`. |
| `JETSON` | `false` | Jetson-in-the-loop: the simulation container uses host networking and streams each drone's SITL MAVLink (UDP `14550`) and cameras (UDP `5600`, `5700`, `5710`, `5720`, `5800`) to its own Jetson in `JETSON_IPS`. Aircraft containers are not started. Not compatible with `CENTRALIZED=true` or with quads using `vio_bridge`. |
| `JETSON_IPS` | `192.168.55.1` | Comma-separated Jetson IPs on the same LAN as the PC, one per drone in drone ID order (quads first, then VTOLs), e.g. `192.168.1.50,192.168.1.51`. Used when `JETSON=true`. |
| `GND_CONTAINER` | `true` | Start the ground container when `true`. |
| `RTF` | `1.0` | Real-time factor (`<= 0.0` runs as fast as possible). |
| `START_AS_PAUSED` | `false` | Start simulation paused when `true`. |
| `INSTANCE` | `0` | Unique instance ID; offsets subnets and container/network names. |

## scripts/sim_build.sh

Clones external repositories and builds the simulation, ground, and aircraft images.

| Variable | Default | Description |
| --- | --- | --- |
| `CLEAN_BUILD` | `false` | Remove `github_clones`, delete images, and prune the builder cache before rebuilding. |
| `CLONE_ONLY` | `false` | Clone repositories without building Docker images. |
