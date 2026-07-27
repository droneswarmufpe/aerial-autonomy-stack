# Landing Controller

Controller ROS 2 para pouso sobre ArUco usando MAVROS.

## Simulacao

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

## Controller

Dentro do aircraft container:

```bash
cd /aas/aircraft_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select landing_controller
source install/setup.bash

ROS_DOMAIN_ID=1 ros2 run landing_controller landing_controller \
  --ros-args --params-file /aas/aircraft_ws/src/landing_controller/config/landing_controller_params.yaml
```

O node se chama `/landing_controller`.

## Checks

```bash
ros2 topic echo /mavros/state --once
ros2 param get /mavros/setpoint_velocity mav_frame
gz model --list | grep aruco
```
