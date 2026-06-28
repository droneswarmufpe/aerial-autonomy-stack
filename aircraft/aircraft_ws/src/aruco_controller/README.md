# ArUco Centered Descent Controller

A ROS2 node that enables a drone to autonomously track and center on an ArUco marker in the downward-facing camera and descend towards it.

## Features

- **Real-time ArUco Detection**: Uses OpenCV's ArUco detector to identify and track markers
- **Proportional Control**: Centers the drone in the camera frame using XY velocity commands
- **Constant Descent**: Maintains constant downward velocity after centering
- **Auto-Landing**: Triggers land when marker is too close or disappears
- **Configurable Parameters**: All control gains and thresholds are parameterized
- **Debug Visualization**: Optional publishing of annotated camera frames
- **State Machine**: Clear state transitions for initialization, centering, descending, and landing

## Prerequisites

- ROS 2 (tested with Humble)
- Python 3.10+
- OpenCV (cv2.aruco)
- cv_bridge
- autopilot_interface_msgs

## Installation

1. Clone or copy the `aruco_controller` package to your aircraft workspace:
```bash
cp -r aruco_controller <aircraft_ws>/src/
```

2. Build the package:
```bash
cd <aircraft_ws>
colcon build --packages-select aruco_controller
```

3. Source the setup script:
```bash
source install/setup.bash
```

## Parameters

Configure behavior via `config/aruco_controller_params.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_topic` | str | `/camera` | Downward-facing camera topic |
| `command_topic` | str | `/mavros/setpoint_velocity/cmd_vel_unstamped` | Velocity command topic |
| `aruco_marker_id` | int | `0` | ID of ArUco marker to track |
| `aruco_marker_size` | float | `0.05` | Physical marker size (meters) |
| `kp_x` | float | `0.3` | X-axis centering proportional gain |
| `kp_y` | float | `0.3` | Y-axis centering proportional gain |
| `descent_velocity` | float | `-0.5` | Descent speed (m/s, negative = down) |
| `centering_tolerance_px` | int | `20` | Acceptable centering error (pixels) |
| `max_marker_area_px` | int | `10000` | Marker area threshold for landing (too close) |
| `min_marker_area_px` | int | `100` | Minimum marker area to track (too far) |
| `marker_lost_timeout_sec` | float | `2.0` | Time before landing if marker disappears |
| `max_horizontal_velocity` | float | `1.0` | Max XY velocity (m/s) |
| `max_descent_velocity` | float | `1.0` | Max descent speed (m/s) |
| `initial_alignment_timeout_sec` | float | `10.0` | Max time for centering phase (seconds) |
| `descent_timeout_sec` | float | `30.0` | Max time for descent phase (seconds) |
| `publish_debug_images` | bool | `false` | Publish annotated camera frames |

## Usage

### In Simulation

1. **Spawn drone and aruco marker** (in simulation container):
```bash
# Open simulation container terminal
docker exec -it aerial_autonomy_stack_sim bash

# Start simulation with appropriate world
# (Set WORLD_NAME, DRONE_ID, etc. as needed)

# Spawn ArUco marker below drone (in another sim container terminal)
gz service -s /world/$WORLD_NAME/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'sdf_filename: "/aas/simulation_resources/simulation_worlds/aruco_5x5_0/model.sdf", name: "aruco_0", allow_renaming: true, pose: {position: {x: 10, y: -20, z: 2.0}}'
```

2. **Run the mission** (in aircraft container):
```bash
ros2 launch mission mission_launch.py conops:=/aas/aircraft_resources/missions/aruco_tracking_mission.yaml
```

3. **Start ArUco controller** (in aircraft container, separate terminal):
```bash
ros2 run aruco_controller aruco_controller --ros-args --params-file /aas/aircraft_ws/src/aruco_controller/config/aruco_controller_params.yaml
```

Or pass parameters directly:
```bash
ros2 run aruco_controller aruco_controller \
  --ros-args \
  -p camera_topic:=/camera \
  -p aruco_marker_id:=0 \
  -p descent_velocity:=-0.5 \
  -p publish_debug_images:=true
```

Use `--ros-args --log-level DEBUG` for verbose node logs.

## Control Flow

```
┌─────────────────────┐
│   INITIALIZING      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WAITING_FOR_MARKER  │  ◄── Marker lost timeout
└──────────┬──────────┘
           │ (Marker detected)
           ▼
┌─────────────────────┐
│   CENTERING         │  ◄── Align XY, zero Z velocity
└──────────┬──────────┘     Timeout → DESCENDING
           │ (Aligned or timeout)
           ▼
┌─────────────────────┐
│   DESCENDING        │  ◄── Soft centering + constant descent
└──────────┬──────────┘     Marker too close → LANDING
           │ (Marker too close or lost)
           ▼
┌─────────────────────┐
│    LANDING          │
└──────────┬──────────┘
           │ (Send LAND action)
           ▼
┌─────────────────────┐
│   COMPLETED         │
└─────────────────────┘
```

## Topics

### Subscribed

- **`/camera`** (sensor_msgs/Image): Downward-facing camera stream

### Published

- **`/mavros/setpoint_velocity/cmd_vel_unstamped`** (geometry_msgs/Twist): Velocity commands (linear.x/y/z)
- **`/aruco_debug`** (sensor_msgs/Image): Annotated camera frames (if `publish_debug_images: true`)

### Actions Used

- **`/land_action`** (autopilot_interface_msgs/Land): Land the drone when conditions met

## Velocity Command Structure

The node publishes `geometry_msgs/Twist` commands with:

- **`linear.x`**: Forward velocity (proportional to X-axis error, clamped)
- **`linear.y`**: Lateral velocity (proportional to Y-axis error, clamped)
- **`linear.z`**: Vertical velocity (constant descent or 0, clamped)
- **`angular.*`**: All zero (rotation not controlled)

## Tuning Guide

### Centering (XY)

- **Increase `kp_x`/`kp_y`**: Faster response, risk of oscillation
- **Decrease `kp_x`/`kp_y`**: Slower, smoother, may not center in time
- **Adjust `centering_tolerance_px`**: Tighter = stricter centering requirement
- **`initial_alignment_timeout_sec`**: Give controller more time to center

### Descent

- **Make `descent_velocity` more negative**: Faster descent
- **Adjust `max_marker_area_px`**: Higher = land closer to ground, lower = land higher

### Robustness

- **Increase `marker_lost_timeout_sec`**: More tolerance for marker detection gaps
- **Decrease `min_marker_area_px`**: Track even when marker is small (farther away)

## Troubleshooting

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| "Marker not detected" | Camera/detector issue | Check camera topic, verify marker is visible, check marker ID |
| Oscillating around center | Gains too high | Reduce `kp_x`, `kp_y` |
| Won't center | Gains too low or timeout too short | Increase `kp_x`, `kp_y`, or `initial_alignment_timeout_sec` |
| Lands too high/low | Marker area threshold wrong | Adjust `max_marker_area_px` |
| Frequent lost marker | Lighting/contrast issue | Verify camera feed, adjust lighting, check marker contrast |

## Example Debug Output

```
[INFO] [aruco_controller_node]: ArUco Controller initialized. Target marker ID: 0
[INFO] [aruco_controller_node]: Camera topic: /camera
[INFO] [aruco_controller_node]: Marker detected! Starting centering phase.
[INFO] [aruco_controller_node]: Marker centered! Starting descent.
[INFO] [aruco_controller_node]: Marker too close (area: 12500px). Landing.
[INFO] [aruco_controller_node]: Sending LAND action.
[INFO] [aruco_controller_node]: Land goal accepted.
[INFO] [aruco_controller_node]: Mission completed. Shutting down.
```

## Future Enhancements

- [ ] Multi-marker support
- [ ] 6-DOF pose estimation from marker
- [ ] Adaptive descent based on marker visibility confidence
- [ ] Dynamic gain adjustment based on error magnitude
- [ ] Integration with lidar for true altitude measurement
- [ ] Estimator for marker depth using pinhole camera model

## Citation

Uses OpenCV ArUco implementation: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
