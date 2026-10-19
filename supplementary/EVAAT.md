# EVAAT in the simulation

With `EVAAT=true`, `sim_run.sh` runs the [evaat_embedded_py](http://172.20.2.28:3000/EVAAT-GCN/evaat_embedded_py) stack in the simulation: the evaat drone stack in each aircraft container and the evaat GCS in the ground container, connected to the simulated ArduPilot drones.

The aerial autonomy stack does not depend on evaat to build or run. The images only contain evaat's dependencies; the evaat code is your own checkout, mounted into the containers when `EVAAT=true`. Without `EVAAT=true`, nothing about evaat is used.

## Setup

1. Clone evaat_embedded_py. By default it is expected in `github_clones/evaat_embedded_py`; set `EVAAT_DIR` to use a checkout elsewhere.
2. Build the aircraft and ground images, which include evaat's dependencies (see [Dependencies](#dependencies)):

```sh
cd aerial-autonomy-stack/scripts
CONTAINER=aircraft ./sim_build.sh
CONTAINER=ground ./sim_build.sh
```

You don't need to install evaat itself (`./evaatctl install`), nor a virtual environment: the containers run it with their system Python.

## Running

```sh
cd aerial-autonomy-stack/scripts
EVAAT=true ./sim_run.sh                                                 # The default scenario, with evaat's SITL configs
EVAAT=true SCENARIO=imav_hotspots NUM_QUADS=2 ./sim_run.sh             # Any scenario (up to 3 drones with evaat's SITL configs)
EVAAT=true EVAAT_DIR=~/evaat_embedded_py ./sim_run.sh                   # A checkout elsewhere
EVAAT=true EVAAT_DRONE_CONFIG=config/ufpe/drone{id}.yml ./sim_run.sh   # Other evaat configs
```

| Variable | Default | Description |
| --- | --- | --- |
| `EVAAT` | `false` | Run the evaat drone stack in each aircraft container and the evaat GCS in the ground container |
| `EVAAT_DIR` | `github_clones/evaat_embedded_py` | The evaat_embedded_py checkout on your computer |
| `EVAAT_DRONE_CONFIG` | `config/sitl/drone{id}.yml` | Drone config, relative to `EVAAT_DIR`. `{id}` is replaced by the drone ID (1, 2, ...), so each drone gets its own config |
| `EVAAT_GCS_CONFIG` | `config/sitl/gcs.yml` | GCS config, relative to `EVAAT_DIR` |

`sim_run.sh` checks the checkout and every config before starting any container. evaat ships SITL configs for 3 drones (`drone1.yml` to `drone3.yml`), so add configs for more drones. Each drone config's `autopilot` must be `127.0.0.1:14550` to receive the simulated drone's MAVLink, see [What runs where](#what-runs-where).

`EVAAT=true` is not supported with `RCPILOT=true`, `JETSON=true` (evaat then runs on the Jetson itself), `CENTRALIZED=true` (the aircraft containers have no MAVLink router), or `GND_CONTAINER=false` (the GCS runs in the ground container).

## What runs where

- **Aircraft containers**: an `evaat` tmux window runs `./evaatctl run drone <config> -y --here` from `/aas/evaat`, so evaat's own tmux session (its nodes, one per pane) is shown in that window.
- **MAVLink**: evaat's MAVLink bridge listens on its config's `autopilot.ip:autopilot.port`. Each aircraft container's `mavlink-routerd` sends the drone's MAVLink to fixed addresses: `127.0.0.1:14550`, which is free for evaat, and the container's own `<SIM_SUBNET>.90.<drone ID>:14572` and `:14571`, used by rcpilot and Projeto-Enxame-Drones. So the evaat drone configs must set:

```yaml
autopilot:
  ip: 127.0.0.1
  port: 14550
```
- **Ground container**: an `evaat_gcs` tmux window runs `./evaatctl run gcs <config> -y --here`; the GCS window opens on your screen.
- **Drone to GCS link**: evaat's SITL configs use UDP broadcast (port 50053), which reaches the ground and aircraft containers on the simulation network.
- **Camera**: evaat's SITL perception reads the GStreamer RTP stream on port 5600, where the simulation streams each drone's camera.

The checkout is mounted read-write at `/aas/evaat`: evaat writes its node logs to `logs/` and, the first time, builds its codec (`native/evaat_codec/libevaat_codec.so`) there. Both are created by the containers' root user, so use `sudo` to delete them from your computer. Since the code is mounted, changes to your checkout apply on the next run, without rebuilding the images.

Inside the `evaat` windows, evaat's tmux session is nested in the container's: press `Ctrl + b` twice to send a key to evaat's session (e.g. `Ctrl + b`, `Ctrl + b`, `n` for its next window).

## Dependencies

The images install evaat's third-party dependencies (from evaat's `requirements/`), not evaat itself:

| Image | evaat profile | Packages |
| --- | --- | --- |
| `aircraft-image` (`Dockerfile.aircraft`) | drone | pyyaml, pyserial, pymavlink, utm, pyzmq, dearpygui, TensorRT, pycuda; Chorus, MAVKit, rcpilot.mavsys, and OpenCV are part of the image already. The image uses the CUDA *devel* base, which pycuda is built against |
| `ground-image` (`Dockerfile.ground`) | gcs | pyyaml, pyserial, PySide6 6.11.0 (with QtWebEngine), and the system libraries its Qt needs. PySide6 6.12.0 crashes the GCS on the image's Python 3.10 (`Fatal Python error: none_dealloc`) |

The rcpilot images (`Dockerfile.rc_aircraft`, `Dockerfile.rc_ground`) don't include them, as `EVAAT=true` is not used with `RCPILOT=true`. evaat's optional `pyrealsense2` is not installed.

If evaat adds a dependency, add it to the matching Dockerfile: the ground image's build fails if PySide6's QtWebEngine can't be imported.
