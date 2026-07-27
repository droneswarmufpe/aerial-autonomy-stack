# ArUco Controller

ROS 2 package que detecta o marcador `aruco_5x5_0` na camera inferior e publica velocidade em MAVROS para centralizar e descer.

## Rodar

No host, suba a simulacao com um ArUco embaixo do drone:

```bash
cd /home/pcgr/Code/pds/aerial-autonomy-stack/scripts

DEV=false \
CENTRALIZED=false \
GND_CONTAINER=false \
CUSTOM_OBJECTS=true \
ARUCO_PLACEMENT=under-drones \
NUM_QUADS=1 \
NUM_VTOLS=0 \
WORLD=esefex_fbx \
X_OFFSET=5 \
Y_OFFSET=-110 \
ARUCO_Z=-1.43225 \
./sim_run.sh
```
Se o MAVRos não estiver rodando:
```
docker exec -d aircraft-container-inst0_1 bash -lc \
'source /opt/ros/humble/setup.bash; ros2 launch mavros apm.launch fcu_url:=udp://:14573@127.0.0.1:14573 >/tmp/mavros_aruco.log 2>&1'
```

Dentro do aircraft container:

```bash
docker exec -it aircraft-container-inst0_1 bash
```

```bash
cd /aas/aircraft_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ROS_DOMAIN_ID=1 ros2 run aruco_controller aruco_controller \
  --ros-args --params-file /aas/aircraft_ws/src/aruco_controller/config/aruco_controller_params.yaml
```

O executavel se chama `aruco_controller`.

## Build

Se o pacote ainda nao estiver buildado:

```bash
cd /aas/aircraft_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select aruco_controller
source install/setup.bash
```

## Antes De Rodar

A simulacao precisa estar com:

- MAVROS conectado no aircraft container
- stream da camera chegando na porta `5600`
- ArUco `aruco_0` spawnado no Gazebo
- drone ja em takeoff/GUIDED

Checks rapidos:

```bash
ros2 topic echo /mavros/state --once
ros2 topic info /mavros/setpoint_velocity/cmd_vel_unstamped -v
```

No simulation container:

```bash
gz model --list | grep aruco
```

## Logs Esperados

```text
Opened GStreamer camera stream.
Marker detected! Starting centering phase.
Marker centered! Starting descent.
MAVROS land service accepted.
```
