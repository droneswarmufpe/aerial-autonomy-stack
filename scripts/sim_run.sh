#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set up the simulation
AUTOPILOT="${AUTOPILOT:-px4}" # Options: px4 (default), ardupilot
HEADLESS="${HEADLESS:-false}" # Options: true, false (default)
CAMERA="${CAMERA:-true}" # Options: true (default), false
LIDAR="${LIDAR:-true}" # Options: true (default), false 
#
SIM_SUBNET="${SIM_SUBNET:-10.42}" # Simulation subnet (default = 10.42)
AIR_SUBNET="${AIR_SUBNET:-10.22}" # Inter-vehicle subnet (default = 10.22)
SIM_ID="${SIM_ID:-100}" # Last byte of the simulation container IP (default = 100)
GROUND_ID="${GROUND_ID:-101}" # Last byte of the simulation container IP (default = 101)
#
NUM_QUADS="${NUM_QUADS:-1}" # Number of quadcopters (default = 1)
NUM_VTOLS="${NUM_VTOLS:-0}" # Number of VTOLs (default = 0)
WORLD="${WORLD:-impalpable_greyness}" # Options: impalpable_greyness (default), apple_orchard, shibuya_crossing, swiss_town
#
DEV="${DEV:false}" # Options: true, false (default)
HITL="${HITL:-false}" # Options: true, false (default)
GND_CONTAINER="${GND_CONTAINER:-true}" # Options: true (default), false

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

# Cleanup function
cleanup() {
  DOCKER_PIDS=$(pgrep -f "docker run.*(simulation|ground|aircraft)-image" 2>/dev/null || true)
  CONTAINER_NAMES=("simulation-container" "ground-container" "aircraft-container")
  CONTAINERS_TO_STOP=""
  for name in "${CONTAINER_NAMES[@]}"; do
      CONTAINERS_TO_STOP+=$(docker ps -a -q --filter name="${name}" 2>/dev/null || true)
      CONTAINERS_TO_STOP+=" "
  done
  echo "Stopping Docker containers (this will take a few seconds)..."
  if [ -n "$CONTAINERS_TO_STOP" ]; then
      echo "$CONTAINERS_TO_STOP" | xargs docker stop
  fi
  docker network rm aas-sim-network 2>/dev/null && echo "Removed aas-sim-network" || echo "Network aas-sim-network not found or already removed"
  docker network rm aas-air-network 2>/dev/null && echo "Removed aas-air-network" || echo "Network aas-air-network not found or already removed"
  if [ -n "$DOCKER_PIDS" ]; then
    for dpid in $DOCKER_PIDS; do
      PARENT_PID=$(ps -o ppid= -p $dpid 2>/dev/null | tr -d ' ') # Determine process pids with a parent pid
      if [ -n "$PARENT_PID" ]; then
        echo "Killing terminal process $dpid"
        kill $dpid
      fi
    done
  fi
  echo "All-clear"
}
# Set trap to cleanup on script interruption (Ctrl+C, etc.)
trap cleanup EXIT INT TERM

# Grant access to the X server
if command -v xhost >/dev/null 2>&1; then 
  xhost +local:docker
fi

# Create docker network for SITL
if [[ "$HITL" == "false" ]]; then
  docker network inspect aas-sim-network >/dev/null 2>&1 || docker network create --subnet=${SIM_SUBNET}.0.0/16 aas-sim-network
  docker network inspect aas-air-network >/dev/null 2>&1 || docker network create --subnet=${AIR_SUBNET}.0.0/16 aas-air-network
fi

# WSL-specific options
WSL_OPTS="--env WAYLAND_DISPLAY=$WAYLAND_DISPLAY --env PULSE_SERVER=$PULSE_SERVER --volume /usr/lib/wsl:/usr/lib/wsl \
--env LD_LIBRARY_PATH=/usr/lib/wsl/lib --env LIBGL_ALWAYS_SOFTWARE=0 --env __GLX_VENDOR_LIBRARY_NAME=nvidia"

# Get primary display dimensions
get_primary_display_info() {
  local resolution=$(xrandr 2>/dev/null | grep " connected primary" | grep -oE '[0-9]+x[0-9]+' | head -1)
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
}
get_primary_display_info

# Setup terminal dimensions
TERM_COLS=80
TERM_ROWS=32
FONT_SIZE=10
calculate_terminal_position() {
  local drone_id=$1
  SCREEN_SCALE=$((SCREEN_HEIGHT * 100 / 1080)) # Full HD = 100%
  X_POS=$(( (50 + drone_id * 50) * SCREEN_SCALE / 100 ))
  Y_POS=$(( (drone_id * 125) * SCREEN_SCALE / 100 ))
}

# Enable Shift+Ctrl+c, Shift+Ctrl+v copy-paste in xterm
XTERM_CONFIG_ARGS=(
  -xrm 'XTerm*selectToClipboard: true'
  -xrm 'XTerm*VT100.Translations: #override \
    Ctrl Shift <Key>C: copy-selection(CLIPBOARD) \n\
    Ctrl Shift <Key>V: insert-selection(CLIPBOARD)'
)

# In dev mode, resources and workspaces are mounted from the host
if [[ "$DEV" == "true" ]]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  #
  DEV_SIM_OPTS="--entrypoint /bin/bash"
  DEV_SIM_OPTS+=" -v ${SCRIPT_DIR}/../simulation/simulation_resources/:/aas/simulation_resources:cached"
  #
  DEV_GND_OPTS="--entrypoint /bin/bash"
  DEV_GND_OPTS+=" -v ${SCRIPT_DIR}/../ground/ground_resources/:/aas/ground_resources:cached"
  DEV_GND_OPTS+=" -v ${SCRIPT_DIR}/../ground/ground_ws/src:/aas/ground_ws/src:cached"
  #
  DEV_AIR_OPTS="--entrypoint /bin/bash"
  DEV_AIR_OPTS+=" -v ${SCRIPT_DIR}/../aircraft/aircraft_resources/:/aas/aircraft_resources:cached"
  DEV_AIR_OPTS+=" -v ${SCRIPT_DIR}/../aircraft/aircraft_ws/src:/aas/aircraft_ws/src:cached"
  DEV_AIR_OPTS+=" -v ${SCRIPT_DIR}/../ground/ground_ws/src/ground_system_msgs:/aas/aircraft_ws/src/ground_system_msgs:cached"
fi

# Launch the simulation container
DOCKER_CMD="docker run -it --rm \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
  --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  --env AUTOPILOT=$AUTOPILOT --env HEADLESS=$HEADLESS --env CAMERA=$CAMERA --env LIDAR=$LIDAR \
  --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=$NUM_VTOLS --env WORLD=$WORLD \
  --env SIMULATED_TIME=true \
  --env SIM_SUBNET=$SIM_SUBNET --env GROUND_ID=$GROUND_ID \
  --env GND_CONTAINER=$GND_CONTAINER \
  --env ROS_DOMAIN_ID=$SIM_ID \
  --privileged \
  --name simulation-container"
# Configure network for HITL or SITL
if [[ "$HITL" == "true" ]]; then
  DOCKER_CMD="$DOCKER_CMD --net=host"
else
  DOCKER_CMD="$DOCKER_CMD --net=aas-sim-network --ip=${SIM_SUBNET}.90.${SIM_ID}"
fi
# Add WSL-specific options and complete the command
if [[ "$DESK_ENV" == "wsl" ]]; then
  DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
fi
DOCKER_CMD="$DOCKER_CMD ${DEV_SIM_OPTS} simulation-image"
calculate_terminal_position 0
xterm "${XTERM_CONFIG_ARGS[@]}" -title "Simulation" -fa Monospace -fs $FONT_SIZE -bg black -fg white -geometry "${TERM_COLS}x${TERM_ROWS}+${X_POS}+${Y_POS}" -hold -e bash -c "$DOCKER_CMD" &

if [[ "$HITL" == "false" ]]; then

  if [[ "$GND_CONTAINER" == "true" ]]; then
    sleep 0.5 # Limit resource usage
    # Launch the ground container
    DOCKER_CMD="docker run -it --rm \
      --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
      --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
      --env HEADLESS=$HEADLESS \
      --env NUM_QUADS=$NUM_QUADS --env NUM_VTOLS=$NUM_VTOLS \
      --env SIMULATED_TIME=true \
      --env ROS_DOMAIN_ID=$GROUND_ID \
      --net=aas-sim-network --ip=${SIM_SUBNET}.90.${GROUND_ID} \
      --privileged \
      --name ground-container"
    # Add WSL-specific options and complete the command
    if [[ "$DESK_ENV" == "wsl" ]]; then
      DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
    fi
    DOCKER_CMD="$DOCKER_CMD ${DEV_GND_OPTS} ground-image"
    calculate_terminal_position 1
    xterm "${XTERM_CONFIG_ARGS[@]}" -title "Ground" -fa Monospace -fs $FONT_SIZE -bg black -fg white -geometry "${TERM_COLS}x${TERM_ROWS}+${X_POS}+${Y_POS}" -hold -e bash -c "$DOCKER_CMD" &
  fi

  # Initialize a counter for the drone IDs
  DRONE_ID=1 # 1, 2, .., N drones

  # Function to launch the aircraft containers
  launch_aircraft_containers() {
    local drone_type=$1
    local num_drones=$2
    
    for i in $(seq 1 $num_drones); do
      sleep 1.5 # Limit resource usage
      DOCKER_CMD="docker run -it --rm \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
        --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        --env AUTOPILOT=$AUTOPILOT --env HEADLESS=$HEADLESS --env CAMERA=$CAMERA --env LIDAR=$LIDAR \
        --env DRONE_TYPE=$drone_type --env DRONE_ID=$DRONE_ID \
        --env SIMULATED_TIME=true \
        --env SIM_SUBNET=$SIM_SUBNET --env AIR_SUBNET=$AIR_SUBNET --env SIM_ID=$SIM_ID --env GROUND_ID=$GROUND_ID \
        --env GND_CONTAINER=$GND_CONTAINER \
        --env ROS_DOMAIN_ID=$DRONE_ID \
        --net=aas-sim-network --ip=${SIM_SUBNET}.90.$DRONE_ID \
        --privileged \
        --name aircraft-container_$DRONE_ID"
      # Add WSL-specific options and complete the command
      if [[ "$DESK_ENV" == "wsl" ]]; then
        DOCKER_CMD="$DOCKER_CMD $WSL_OPTS"
      fi
      DOCKER_CMD="$DOCKER_CMD ${DEV_AIR_OPTS} aircraft-image"
      calculate_terminal_position $(($DRONE_ID + 1))
      xterm "${XTERM_CONFIG_ARGS[@]}" -title "${drone_type^^} $DRONE_ID" -fa Monospace -fs $FONT_SIZE -bg black -fg white -geometry "${TERM_COLS}x${TERM_ROWS}+${X_POS}+${Y_POS}" -hold -e bash -c "$DOCKER_CMD" &
      DRONE_ID=$((DRONE_ID + 1))
    done
  }
  # Launch the Quad containers
  launch_aircraft_containers "quad" $NUM_QUADS
  # Launch the VTOL containers
  launch_aircraft_containers "vtol" $NUM_VTOLS

  if [[ "$GND_CONTAINER" == "true" ]]; then
    # Finally, connect ground and aircraft containers to the air network
    sleep 1
    docker network connect --ip=${AIR_SUBNET}.90.$GROUND_ID aas-air-network ground-container
    for i in $(seq 1 $((NUM_QUADS + NUM_VTOLS))); do
      docker network connect --ip=${AIR_SUBNET}.90.$i aas-air-network aircraft-container_$i
    done
  fi
fi

echo "Fly, my pretties, fly!"
echo "Press any key to stop all containers and close the terminals..."
read -n 1 -s # Wait for user input
