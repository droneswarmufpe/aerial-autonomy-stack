#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set up the simulation
HEADLESS="${HEADLESS:-false}" # Options: true, false (default)
CAMERA="${CAMERA:-true}" # Options: true (default), false
THERMAL_CAMERA="${THERMAL_CAMERA:-false}" # Options: true (default), false
RGBD_CAMERA="${RGBD_CAMERA:-true}" # Options: true (default), false
VIO_BRIDGE="${VIO_BRIDGE:-false}" # ROS stereo + Gazebo IMU and OpenVINS (ArduPilot Iris)
CAMERA_PITCH="${CAMERA_PITCH:-1.5707}" # Camera pitch in radians (default = 1.5707, 90 deg down)
#
SIM_SUBNET="${SIM_SUBNET:-10.42}" # Simulation subnet (default = 10.42) Note: this is overridden if INSTANCE != 0
AIR_SUBNET="${AIR_SUBNET:-10.22}" # Inter-vehicle subnet (default = 10.22) Note: this is overridden if INSTANCE != 0
SIM_ID="${SIM_ID:-100}" # Last byte of the simulation container IP (default = 100)
GROUND_ID="${GROUND_ID:-101}" # Last byte of the simulation container IP (default = 101)
#
NUM_QUADS="${NUM_QUADS:-2}" # Number of quadcopters (default = 1)
NUM_VTOLS="${NUM_VTOLS:-0}" # Number of VTOLs (default = 0)
WORLD="${WORLD:-esefex_fbx}" # Options: impalpable_greyness (default), apple_orchard, shibuya_crossing, swiss_town
CENTRALIZED="${CENTRALIZED:-false}" # Options: true, false (default) - If true, all cameras will stream to the ground container. If false, each camera will stream to its own IP (useful for testing network conditions and scalability)
X_OFFSET="${X_OFFSET:-0}" # X offset for drone placement in the world (default = 0)
Y_OFFSET="${Y_OFFSET:-0}" # Y offset for drone placement in the world (default = 0)
#
RCPILOT="${RCPILOT:-false}" # Options: true, false (default) - If true, the rcpilot repo will be cloned and built. This is useful for testing the rcpilot SDK and its integration with the AAS.
JETSON="${JETSON:-false}" # Options: true, false (default) - If true, SITL and cameras are streamed to an external Jetson (on the same LAN as this PC) instead of aircraft containers
JETSON_IPS="${JETSON_IPS:-192.168.55.1}" # Comma-separated Jetson IPs, one per drone in drone ID order, used when JETSON=true (default = 192.168.55.1)
GND_CONTAINER="${GND_CONTAINER:-true}" # Options: true (default), false
RTF="${RTF:-1.0}" # Real-time factor (default = 1.0), set to <=0.0 for as fast as possible execution
START_AS_PAUSED="${START_AS_PAUSED:-false}" # Options: true, false (default)
INSTANCE="${INSTANCE:-0}" # Integer ID to make docker network/container names unique as well as offsetting the second byte of the subnets (default = 0)
# Set unique subnets and container/network names based on INSTANCE
SIM_BYTE_1=$(echo "$SIM_SUBNET" | cut -d'.' -f1)
SIM_BYTE_2=$(echo "$SIM_SUBNET" | cut -d'.' -f2)
SIM_SUBNET="${SIM_BYTE_1}.$((SIM_BYTE_2 + INSTANCE))"
AIR_BYTE_1=$(echo "$AIR_SUBNET" | cut -d'.' -f1)
AIR_BYTE_2=$(echo "$AIR_SUBNET" | cut -d'.' -f2)
AIR_SUBNET="${AIR_BYTE_1}.$((AIR_BYTE_2 + INSTANCE))"
SIM_NET_NAME="aas-sim-network-inst${INSTANCE}"
AIR_NET_NAME="aas-air-network-inst${INSTANCE}"
SIM_CONT_NAME="simulation-container-inst${INSTANCE}"
GND_CONT_NAME="ground-container-inst${INSTANCE}"
CUSTOM_OBJECTS="${CUSTOM_OBJECTS:-false}"

if [[ "$VIO_BRIDGE" != "true" && "$VIO_BRIDGE" != "false" ]]; then
  echo "VIO_BRIDGE must be true or false" >&2
  exit 1
fi
if [[ "$JETSON" == "true" && "$CENTRALIZED" == "true" ]]; then
  echo "JETSON=true and CENTRALIZED=true are mutually exclusive (streams go either to the Jetson or to the ground container)" >&2
  exit 1
fi
if [[ "$JETSON" == "true" ]]; then
  IFS=',' read -ra JETSON_IP_LIST <<< "$JETSON_IPS"
  if [[ ${#JETSON_IP_LIST[@]} -ne $((NUM_QUADS + NUM_VTOLS)) ]]; then
    echo "JETSON_IPS must list one IP per drone: got ${#JETSON_IP_LIST[@]} IP(s) for $((NUM_QUADS + NUM_VTOLS)) drone(s)" >&2
    exit 1
  fi
fi
if [[ "$VIO_BRIDGE" == "true" ]]; then
  if [[ "$NUM_VTOLS" != "0" || "$CAMERA" != "true" || "$RGBD_CAMERA" != "true" || "$JETSON" == "true" || "$RCPILOT" != "true" ]]; then
    echo "VIO_BRIDGE requires RCPILOT=true, CAMERA=true, RGBD_CAMERA=true, NUM_VTOLS=0, JETSON=false" >&2
    exit 1
  fi
fi

# Detect the environment (Ubuntu/GNOME, WSL, etc.)
if command -v gnome-terminal >/dev/null 2>&1 && [ -n "$XDG_CURRENT_DESKTOP" ]; then
  DESK_ENV="gnome"
elif grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
  DESK_ENV="wsl"
else
  echo "Unsupported environment" 
  exit 1
fi
echo "Desktop environment: $DESK_ENV"
# Terminal emulator used to launch the containers
if [[ "$DESK_ENV" == "gnome" ]]; then
  TERMINAL="${TERMINAL:-gnome}" # Options: gnome (default on GNOME), xterm
else
  TERMINAL="${TERMINAL:-xterm}" # Options: xterm (default on WSL), gnome
fi
echo "Terminal: $TERMINAL"

# Create docker networks
docker network inspect $SIM_NET_NAME >/dev/null 2>&1 || docker network create --subnet=${SIM_SUBNET}.0.0/16 $SIM_NET_NAME
docker network inspect $AIR_NET_NAME >/dev/null 2>&1 || docker network create --subnet=${AIR_SUBNET}.0.0/16 $AIR_NET_NAME

# Grant access to the X server
if command -v xhost >/dev/null 2>&1; then 
  xhost +local:docker
fi

# WSL-specific options
WSL_OPTS="--env WAYLAND_DISPLAY=$WAYLAND_DISPLAY --env PULSE_SERVER=$PULSE_SERVER --volume /usr/lib/wsl:/usr/lib/wsl \
--env MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA --env LD_LIBRARY_PATH=/usr/lib/wsl/lib --env LIBGL_ALWAYS_SOFTWARE=0 --env __GLX_VENDOR_LIBRARY_NAME=nvidia"

# Get display dimensions
resolution=$(xrandr 2>/dev/null | grep " connected primary" | grep -oE '[0-9]+x[0-9]+' | head -1)
if [[ ! "$resolution" =~ ^[0-9]+x[0-9]+$ ]]; then
  resolution=$(xrandr 2>/dev/null | grep " connected" | grep -oE '[0-9]+x[0-9]+' | head -1) # Fallback
fi
if [[ "$resolution" =~ ^[0-9]+x[0-9]+$ ]]; then
  SCREEN_WIDTH=$(echo "$resolution" | cut -d'x' -f1)
  SCREEN_HEIGHT=$(echo "$resolution" | cut -d'x' -f2)
  echo "Detected display: ${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
else
  SCREEN_WIDTH=1920
  SCREEN_HEIGHT=1080
  echo "Fallback resolution to ${SCREEN_WIDTH}x${SCREEN_HEIGHT} default"
fi

# Function to calculate terminal position based on ID
calculate_terminal_position() {
  local term_id=$1
  SCREEN_SCALE=$((SCREEN_HEIGHT * 100 / 1080)) # Full HD = 100%
  X_POS=$(( (50 + term_id * 50) * SCREEN_SCALE / 100 ))
  Y_POS=$(( (term_id * 125) * SCREEN_SCALE / 100 ))
}

# Setup terminal dimensions and enable Shift+Ctrl+c, Shift+Ctrl+v copy-paste in Xterm (built into GNOME Terminal)
TERM_COLS=160
TERM_ROWS=32
FONT_SIZE=10 # Xterm only
TERM_ZOOM=0.9 # GNOME Terminal only (uses the default profile)
XTERM_CONFIG_ARGS=(
  -xrm 'XTerm*selectToClipboard: true'
  -xrm 'XTerm*VT100.Translations: #override \
    Ctrl Shift <Key>C: copy-selection(CLIPBOARD) \n\
    Ctrl Shift <Key>V: insert-selection(CLIPBOARD)'
)

# Function to run a command in a new terminal window: launch_terminal <title> <position_id> <command>
launch_terminal() {
  local title=$1
  local term_id=$2
  local cmd=$3
  calculate_terminal_position $term_id
  if [[ "$TERMINAL" == "gnome" ]]; then
    # --wait keeps the client alive until the command exits, the final read emulates xterm's -hold
    # Note: window positioning is ignored on Wayland, only the size is applied
    gnome-terminal --wait --title="$title" --zoom=$TERM_ZOOM \
      --geometry="${TERM_COLS}x${TERM_ROWS}+${X_POS}+${Y_POS}" \
      -- bash -c "$cmd; echo; read -n 1 -s -p 'Process exited, press any key to close...'" &
  else
    xterm "${XTERM_CONFIG_ARGS[@]}" -title "$title" -fa Monospace -fs $FONT_SIZE -bg black -fg white \
      -geometry "${TERM_COLS}x${TERM_ROWS}+${X_POS}+${Y_POS}" -hold -e bash -c "$cmd" &
  fi
}

PARENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VIO_SIM_OPTS=""
VIO_AIR_OPTS=""
if [[ "$VIO_BRIDGE" == "true" ]]; then
  # Use identical source geometry for Gazebo and calibration, even with older images.
  VIO_SIM_OPTS="--volume ${PARENT_DIR}/simulation/simulation_resources/aircraft_models/d435:/aas/simulation_resources/aircraft_models/d435:ro --volume ${PARENT_DIR}/simulation/simulation_resources/aircraft_models/iris_with_ardupilot:/aas/simulation_resources/aircraft_models/iris_with_ardupilot:ro"
  VIO_AIR_OPTS="--volume ${PARENT_DIR}/simulation/simulation_resources/aircraft_models:/aas/vio_models:ro"
fi

# Launch the simulation container
DOCKER_CMD="docker run -it --rm \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
  --volume ${PARENT_DIR}/simulation/simulation_resources/custom_objects:/aas/simulation_resources/custom_objects \
  --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR --env GST_DEBUG=3 \
  --env HEADLESS=$HEADLESS --env CAMERA=$CAMERA --env THERMAL_CAMERA=$THERMAL_CAMERA --env RGBD_CAMERA=$RGBD_CAMERA --env CAMERA_PITCH=$CAMERA_PITCH \
  --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=$NUM_VTOLS --env WORLD=$WORLD \
  --env VIO_BRIDGE=$VIO_BRIDGE $VIO_SIM_OPTS \
  --env SIMULATED_TIME=true --env RTF=$RTF --env START_AS_PAUSED=$START_AS_PAUSED \
  --env JETSON=$JETSON --env JETSON_IPS=$JETSON_IPS \
  --env SIM_SUBNET=$SIM_SUBNET --env GROUND_ID=$GROUND_ID \
  --env GND_CONTAINER=$GND_CONTAINER --env CENTRALIZED=$CENTRALIZED \
  --env X_OFFSET=$X_OFFSET --env Y_OFFSET=$Y_OFFSET --env CUSTOM_OBJECTS=$CUSTOM_OBJECTS \
  --env ROS_DOMAIN_ID=$SIM_ID \
  --privileged \
  --name $SIM_CONT_NAME"
# With JETSON=true, use host networking so the simulation can reach the Jetson on the LAN (and vice versa)
if [[ "$JETSON" == "true" ]]; then
  DOCKER_CMD="$DOCKER_CMD --net=host"
else
  DOCKER_CMD="$DOCKER_CMD --net=$SIM_NET_NAME --ip=${SIM_SUBNET}.90.${SIM_ID}"
fi
# Add WSL-specific options and complete the command
if [[ "$DESK_ENV" == "wsl" ]]; then
  DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
fi
DOCKER_CMD="$DOCKER_CMD simulation-image"
launch_terminal "Simulation" 0 "$DOCKER_CMD"

if [[ "$GND_CONTAINER" == "true" ]]; then
  sleep 1.0 # Limit resource usage
  # Launch the ground container
  DOCKER_CMD="docker run -it --rm \
    --ipc=host \
    --volume ${PARENT_DIR}/github_clones/Projeto-Enxame-Drones:/aas/Projeto-Enxame-Drones \
    --volume ${PARENT_DIR}/simulation/simulation_resources/custom_objects:/aas/custom_objects \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
    --volume ${PARENT_DIR}/github_clones/rcpilot/src:/workspaces/rcpilot/src:cached \
    --volume ${PARENT_DIR}/github_clones/rcpilot/log:/workspaces/rcpilot/log:cached \
    --volume ${PARENT_DIR}/github_clones/rcpilot/sdk:/workspaces/rcpilot/sdk:cached \
    --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR --env GST_DEBUG=3 \
    --env HEADLESS=$HEADLESS --env RCPILOT=$RCPILOT \
    --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=$NUM_VTOLS \
    --env SIMULATED_TIME=true --env CENTRALIZED=$CENTRALIZED \
    --env ROS_DOMAIN_ID=$GROUND_ID \
    --net=$SIM_NET_NAME --ip=${SIM_SUBNET}.90.${GROUND_ID} \
    --privileged \
    --name $GND_CONT_NAME"
  # Add WSL-specific options and complete the command
  if [[ "$DESK_ENV" == "wsl" ]]; then
    DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
  fi
  if [[ "$RCPILOT" == "true" ]]; then
    DOCKER_CMD="$DOCKER_CMD rc-ground-image"
  else
    DOCKER_CMD="$DOCKER_CMD ground-image"
  fi
  launch_terminal "Ground" 1 "$DOCKER_CMD"
fi

# Initialize a counter for the drone IDs
DRONE_ID=1 # 1, 2, .., N drones

# Function to launch the aircraft containers
launch_aircraft_containers() {
  local drone_type=$1
  local num_drones=$2
  
  for i in $(seq 1 $num_drones); do
    sleep 1.0 # Limit resource usage
    local NAME_AIRCRAFT_CNT="aircraft-container-inst${INSTANCE}_${DRONE_ID}"
    DOCKER_CMD="docker run -it --rm \
      --ipc=host \
      --volume ${PARENT_DIR}/github_clones/Projeto-Enxame-Drones:/aas/Projeto-Enxame-Drones \
      --volume ${PARENT_DIR}/github_clones/rcpilot/src:/workspaces/rcpilot/src:cached \
      --volume ${PARENT_DIR}/github_clones/rcpilot/sdk:/workspaces/rcpilot/sdk:cached \
      --volume ${PARENT_DIR}/github_clones/rcpilot/log:/workspaces/rcpilot/log:cached \
      --volume ${PARENT_DIR}/simulation/simulation_resources/custom_objects:/aas/custom_objects \
      --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
      --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR --env GST_DEBUG=3 \
      --env HEADLESS=$HEADLESS --env CAMERA=$CAMERA \
      --env DRONE_TYPE=$drone_type --env DRONE_ID=$DRONE_ID \
      --env VIO_BRIDGE=$VIO_BRIDGE --env CAMERA_PITCH=$CAMERA_PITCH $VIO_AIR_OPTS \
      --env SIMULATED_TIME=true --env CENTRALIZED=$CENTRALIZED \
      --env JETSON=$JETSON \
      --env SIM_SUBNET=$SIM_SUBNET --env AIR_SUBNET=$AIR_SUBNET --env SIM_ID=$SIM_ID --env GROUND_ID=$GROUND_ID \
      --env GND_CONTAINER=$GND_CONTAINER --env RCPILOT=$RCPILOT \
      --env ROS_DOMAIN_ID=$DRONE_ID \
      --net=$SIM_NET_NAME --ip=${SIM_SUBNET}.90.$DRONE_ID \
      --privileged \
      --name $NAME_AIRCRAFT_CNT"
    # Add WSL-specific options and complete the command
    if [[ "$DESK_ENV" == "wsl" ]]; then
      DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
    fi
    if [[ "$RCPILOT" == "true" ]]; then
      DOCKER_CMD="$DOCKER_CMD rc-aircraft-image"
    else
      DOCKER_CMD="$DOCKER_CMD aircraft-image"
    fi
    launch_terminal "${drone_type^^} $DRONE_ID" $(($DRONE_ID + 1)) "$DOCKER_CMD"
    DRONE_ID=$((DRONE_ID + 1))
  done
}
# With JETSON=true, the aircraft software runs on the Jetson instead of in aircraft containers
if [[ "$JETSON" == "false" ]]; then
  # Launch the Quad containers
  launch_aircraft_containers "quad" $NUM_QUADS
  # Launch the VTOL containers
  launch_aircraft_containers "vtol" $NUM_VTOLS
fi  

if [[ "$GND_CONTAINER" == "true" ]]; then
  sleep 2.0 # Once all containers are up, connect ground and aircraft containers to the air network
  docker network connect --ip=${AIR_SUBNET}.90.$GROUND_ID $AIR_NET_NAME $GND_CONT_NAME
  if [[ "$JETSON" == "false" ]]; then
    for i in $(seq 1 $((NUM_QUADS + NUM_VTOLS))); do
      docker network connect --ip=${AIR_SUBNET}.90.$i $AIR_NET_NAME "aircraft-container-inst${INSTANCE}_${i}"
    done
  fi
fi

echo "Fly, my pretties, fly!"
echo "Press any key to stop all containers and close the terminals..."
read -n 1 -s # Wait for user input

# Cleanup function
cleanup() {
  DOCKER_PIDS=$(pgrep -f "docker run.*inst${INSTANCE}" 2>/dev/null || true)
  # Record the bash wrappers before stopping the containers (GNOME Terminal windows close when their wrapper exits)
  WRAPPER_PIDS=""
  for dpid in $DOCKER_PIDS; do
    wpid=$(ps -o ppid= -p $dpid 2>/dev/null | tr -d ' ')
    if [[ -n "$wpid" && "$(ps -o comm= -p $wpid 2>/dev/null)" == "bash" ]]; then # Never kill gnome-terminal-server itself
      WRAPPER_PIDS+="$wpid "
    fi
  done
  CONTAINER_NAMES=("${SIM_CONT_NAME}" "${GND_CONT_NAME}" "aircraft-container-inst${INSTANCE}")
  CONTAINERS_TO_STOP=""
  for name in "${CONTAINER_NAMES[@]}"; do
      CONTAINERS_TO_STOP+=$(docker ps -a -q --filter name="${name}" 2>/dev/null || true)
      CONTAINERS_TO_STOP+=" "
  done
  echo "Stopping Docker containers (this will take a few seconds)..."
  if [ -n "$CONTAINERS_TO_STOP" ]; then
      echo "$CONTAINERS_TO_STOP" | xargs docker stop
  fi
  docker network rm $SIM_NET_NAME 2>/dev/null && echo "Removed $SIM_NET_NAME" || echo "Network $SIM_NET_NAME not found or already removed"
  docker network rm $AIR_NET_NAME 2>/dev/null && echo "Removed $AIR_NET_NAME" || echo "Network $AIR_NET_NAME not found or already removed"
  if [ -n "$DOCKER_PIDS" ]; then
    for dpid in $DOCKER_PIDS; do
      PARENT_PID=$(ps -o ppid= -p $dpid 2>/dev/null | tr -d ' ') # Determine process pids with a parent pid
      if [ -n "$PARENT_PID" ]; then
        echo "Killing terminal process $dpid"
        kill $dpid 2>/dev/null || true
      fi
    done
  fi
  if [[ "$TERMINAL" == "gnome" ]]; then
    for wpid in $WRAPPER_PIDS; do
      kill $wpid 2>/dev/null && echo "Closing terminal window $wpid" || true
    done
  fi
  echo "All-clear"
}
# Set trap to cleanup on script interruption (Ctrl+C, etc.)
trap cleanup EXIT INT TERM
