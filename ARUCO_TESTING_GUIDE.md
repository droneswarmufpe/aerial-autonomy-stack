# ArUco Controller - Testing and Setup Guide

## Quick Start (Follow the root README flow)

### Step 1: Start the stack the same way as the root README

```bash
cd aerial-autonomy-stack/scripts
DEV=true AUTOPILOT=ardupilot NUM_QUADS=1 NUM_VTOLS=0 WORLD=swiss_town RTF=3.0 ./sim_run.sh
```

This opens the `Simulation`, `Ground`, and `QUAD` xterm terminals. In `DEV=true`, the source folders are mounted from your host, so changes here are visible immediately.

### Step 2: Build the controller inside the `QUAD` terminal

```bash
cd /aas/aircraft_ws
colcon build --symlink-install --packages-select aruco_controller
source install/setup.bash
```

### Step 3: Start the tmux sessions from the root README

In the `Simulation` terminal:

```bash
tmuxinator start -p /aas/simulation.yml.erb
```

In the `QUAD` terminal:

```bash
tmuxinator start -p /aas/aircraft.yml.erb
```

Then, inside the `QUAD` panes/windows, start the mission and controller:

```bash
source /aas/aircraft_ws/install/setup.bash
ros2 run mission mission --ros-args -r __ns:=/Drone1 -p use_sim_time:=true
```

```bash
ros2 run aruco_controller aruco_controller --ros-args \
  --params-file /aas/aircraft_ws/src/aruco_controller/config/aruco_controller_params.yaml
```

### Step 4: Spawn the ArUco marker

Run this in the `Simulation` terminal after the world is up:

```bash
gz service -s /world/$WORLD/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'sdf_filename: "/aas/simulation_resources/simulation_worlds/aruco_5x5_0/model.sdf", name: "aruco_0", allow_renaming: true, pose: {position: {x: 10, y: -20, z: 2.0}}'
```

### Step 5: Monitor topics

```bash
ros2 topic echo /mavros/setpoint_velocity/cmd_vel_unstamped
ros2 topic list | grep camera
ros2 node list | grep aruco
ros2 node info /aruco_controller_node
```

## Verification Checklist

- [ ] Stack started with `DEV=true ./sim_run.sh`
- [ ] ArUco package builds with `colcon build --symlink-install`
- [ ] `tmuxinator start -p /aas/simulation.yml.erb` is running
- [ ] Drone spawns successfully
- [ ] ArUco marker visible in simulation
- [ ] Camera topic publishes frames (`ros2 topic list | grep camera`)
- [ ] Velocity commands published to `/mavros/setpoint_velocity/cmd_vel_unstamped`
- [ ] Controller state transitions from WAITING → CENTERING → DESCENDING
- [ ] Drone centers on marker
- [ ] Drone descends after centering
- [ ] Landing triggered when marker too close

## Debugging

### 1. Check if Marker is Detected

```bash
# Manually test ArUco detection
python3 << 'EOF'
import cv2
import numpy as np

# Load a test image from the camera
# This assumes you can publish an image to a file

# Test ArUco detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params = (
    cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, 'DetectorParameters')
    else cv2.aruco.DetectorParameters_create()
)
detector = (
    cv2.aruco.ArucoDetector(aruco_dict, params)
    if hasattr(cv2.aruco, 'ArucoDetector')
    else None
)

# Load test image
img = cv2.imread('test_frame.png')
if detector is not None:
    corners, ids, rejected = detector.detectMarkers(img)
else:
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)

if ids is not None:
    print(f"Detected marker IDs: {ids.flatten()}")
    for i, corner in enumerate(corners):
        print(f"Marker {i}: corners shape = {corner.shape}")
else:
    print("No markers detected")
EOF
```

### 2. View Camera Feed

```bash
# Stream camera to file and analyze
ros2 bag record /camera -o camera_bag &
sleep 5
pkill ros2

# Then extract and view frames
```

### 3. Check Parameters Are Loaded

```bash
ros2 param get /aruco_controller_node camera_topic
ros2 param get /aruco_controller_node aruco_marker_id
ros2 param get /aruco_controller_node kp_x
```

### 4. Enable Verbose Logging

```bash
ros2 run aruco_controller aruco_controller \
  --ros-args \
  --log-level DEBUG \
  --params-file /aas/aircraft_ws/src/aruco_controller/config/aruco_controller_params.yaml
```

### 5. Common Issues

**Issue**: "Camera topic not found"
- **Solution**: Check camera topic name in simulation (might be `/simple_camera/image_raw`)
- Update `aruco_controller_params.yaml` with correct topic

**Issue**: "No markers detected"
- **Solution**: 
  - Verify marker is visible in simulation
  - Check marker is ArUco DICT_5X5_50
  - Verify marker ID matches `aruco_marker_id` parameter
  - Increase brightness/contrast if needed

**Issue**: "Drone oscillates around center"
- **Solution**: 
  - Reduce `kp_x` and `kp_y` gains
  - Increase `centering_tolerance_px`
  - Try values: `kp_x: 0.1`, `kp_y: 0.1`

**Issue**: "Drone doesn't land"
- **Solution**:
  - Increase `max_marker_area_px` to trigger landing sooner
  - Verify LAND action server exists: `ros2 action list | grep land`
  - Check autopilot interface logs

## Performance Tuning

### Recommended Parameter Sets

**Conservative (slower, more stable):**
```yaml
kp_x: 0.15
kp_y: 0.15
centering_tolerance_px: 30
descent_velocity: -0.3
max_marker_area_px: 8000
```

**Aggressive (faster, less stable):**
```yaml
kp_x: 0.5
kp_y: 0.5
centering_tolerance_px: 10
descent_velocity: -1.0
max_marker_area_px: 12000
```

**Balanced (recommended starting point):**
```yaml
kp_x: 0.3
kp_y: 0.3
centering_tolerance_px: 20
descent_velocity: -0.5
max_marker_area_px: 10000
```

## Expected Behavior

1. **Takeoff phase** (2-3 seconds): Drone takes off to 5m
2. **Waiting phase** (varies): Controller waits for marker detection
3. **Centering phase** (5-10 seconds): Drone centers on marker using XY velocity
4. **Descending phase** (5-15 seconds): Drone descends at constant velocity while maintaining center
5. **Landing phase** (2-5 seconds): When marker too close or disappears, LAND action is triggered

**Total expected duration**: 15-35 seconds depending on marker detection timing

## Real Hardware Deployment Checklist

- [ ] Camera calibration performed
- [ ] Marker physically attached below target location
- [ ] Marker size matches parameter
- [ ] Marker orientation downward
- [ ] Lighting sufficient for marker detection
- [ ] Test in stationary hover first
- [ ] Validate velocity command mapping (NED vs FLU convention)
- [ ] Emergency landing procedure tested
- [ ] Battery fully charged
- [ ] Flight approved by operations

## Next Steps

After successful simulation:

1. **Validate on real hardware** with safety pilot
2. **Collect performance data** (landing accuracy, timing)
3. **Tune gains** based on real flight characteristics
4. **Add safety features**: 
   - Minimum altitude cutoff
   - Timeout with auto-land
   - Manual override capability
5. **Integrate with mission planning** system
6. **Document operational procedures**

## Support

For issues or questions:
1. Check README.md in aruco_controller package
2. Review controller logs with `ros2 node info /aruco_controller_node`
3. Verify marker detection with debug images (`publish_debug_images: true`)
4. Check simulation setup matches documentation
