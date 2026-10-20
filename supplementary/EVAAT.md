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

`EVAAT=true` is not supported with `RCPILOT=true`, `CENTRALIZED=true` (the aircraft containers have no MAVLink router), or `GND_CONTAINER=false` (the GCS runs in the ground container).

### With Jetsons (`JETSON=true`)

```sh
EVAAT=true JETSON=true JETSON_IPS=192.168.1.50,192.168.1.51 ./sim_run.sh
```

The evaat drone stack runs on the Jetsons themselves, one per drone, so no aircraft containers are started (as always with `JETSON=true`) and `EVAAT_DRONE_CONFIG` is not used; only the GCS runs in the simulation, in the ground container.

- The simulation streams each drone's SITL MAVLink (UDP `14550`) and cameras (UDP `5600`, ...) to its Jetson, as without EVAAT. The MAVLink arrives at the Jetson's LAN address, so the evaat drone config on each Jetson must listen there: `autopilot.ip` `0.0.0.0` (or the Jetson's IP) and `autopilot.port` `14550`, not `127.0.0.1`.
- The ground container runs on the host network, like the simulation container: the evaat GCS links to the drones on the Jetsons by UDP broadcast (port 50053), which can't leave a docker network. The PC and the Jetsons must be on the same LAN.
- SITL sends the ground's QGroundControl MAVLink to `127.0.0.1` instead of the ground container's docker IP, and the ground container isn't connected to the inter-vehicle docker network.

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

## Onboard perception

evaat's perception node runs on each drone when its config enables it (`perception.enabled: true`, and `enabled: true` on a pipeline, see evaat's `docs/perception.md`). In the simulation, its `gstreamer` provider on port 5600 receives the drone's simulated RGB camera.

The `yolo_detector` pipeline needs a TensorRT engine (`assets/models/<model_name>.engine`). A TensorRT engine only runs with the TensorRT version and GPU it was built with, so export it from the YOLO `.pt` model inside `aircraft-image`, on the computer that runs the simulation (e.g. for `assets/models/yolov11-ime-target.pt`):

```sh
docker run --rm --gpus all -v <EVAAT_DIR>:/aas/evaat -w /aas/evaat/assets/models --entrypoint yolo aircraft-image \
  export model=yolov11-ime-target.pt format=engine device=0
sudo chown $USER: <EVAAT_DIR>/assets/models/yolov11-ime-target.*    # The container writes them as root
```

Then set the pipeline's `model_name` to the engine's name (`yolov11-ime-target`). Re-export it after rebuilding `aircraft-image` with another TensorRT version, and on a different GPU. The intermediate `.onnx` file can be deleted.

With `display: true`, the pipeline's annotated image opens in a window on your screen. The model detects the target from above (e.g. with the default `camera_pitch` pointing down, from a few meters up), not from the ground.

## Dependencies

The images install evaat's third-party dependencies (from evaat's `requirements/`), not evaat itself:

| Image | evaat profile | Packages |
| --- | --- | --- |
| `aircraft-image` (`Dockerfile.aircraft`) | drone | pyyaml, pyserial, pymavlink, utm, pyzmq, dearpygui, TensorRT, pycuda, and onnx / onnxslim (to export YOLO models to TensorRT engines); Chorus, MAVKit, rcpilot.mavsys, ultralytics, and OpenCV are part of the image already. The image uses the CUDA *devel* base, which pycuda is built against, and the system OpenCV, which has GStreamer (MAVProxy's pip OpenCV, without it, is removed) |
| `ground-image` (`Dockerfile.ground`) | gcs | pyyaml, pyserial, PySide6 6.11.0 (with QtWebEngine), and the system libraries its Qt needs. PySide6 6.12.0 crashes the GCS on the image's Python 3.10 (`Fatal Python error: none_dealloc`) |

The rcpilot images (`Dockerfile.rc_aircraft`, `Dockerfile.rc_ground`) don't include them, as `EVAAT=true` is not used with `RCPILOT=true`. evaat's optional `pyrealsense2` is not installed (the simulated camera comes through GStreamer).

If evaat adds a dependency, add it to the matching Dockerfile: the ground image's build fails if PySide6's QtWebEngine can't be imported.
