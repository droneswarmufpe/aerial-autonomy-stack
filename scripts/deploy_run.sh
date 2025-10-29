#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set up the aircraft
DRONE_TYPE="${DRONE_TYPE:-quad}" # Options: quad (default), vtol
AUTOPILOT="${AUTOPILOT:-px4}" # Options: px4 (default), ardupilot
DRONE_ID="${DRONE_ID:-1}" # Id of aircraft (default = 1)
HEADLESS="${HEADLESS:-true}" # Options: true (default), false 
CAMERA="${CAMERA:-true}" # Options: true (default), false
LIDAR="${LIDAR:-true}" # Options: true (default), false
MODE="${MODE:-}" # Options: empty (default), dev, ...
SUBNET_PREFIX="${SUBNET_PREFIX:-42.42}" # Subnet prefix, e.g., 42.42 (default), 192.168, etc.
HITL="${HITL:-false}" # Options: true, false (default)

# Initialize an empty variable for the flags
MODE_OPTS=""
case "$MODE" in
  dev)
    # In dev mode, resources and workspaces are mounted from the host
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    MODE_OPTS="--entrypoint /bin/bash"
    MODE_OPTS+=" -v ${SCRIPT_DIR}/../aircraft/aircraft_resources/:/aas/aircraft_resources:cached"
    MODE_OPTS+=" -v ${SCRIPT_DIR}/../aircraft/aircraft_ws/src:/aas/aircraft_ws/src:cached"
    MODE_OPTS+=" -v ${SCRIPT_DIR}/../ground/ground_ws/src/ground_system_msgs:/aas/aircraft_ws/src/ground_system_msgs:cached"
    ;;
  *)
    MODE_OPTS=""
    ;;
esac

if [ "$HEADLESS" = "false" ]; then
  # Grant access to the X server
  xhost +local:docker # Avoid this when building the TensorRT cache for the first time
fi

if [ "$HITL" = "true" ]; then
  DOCKER_RUN_FLAGS="-it --rm" # Interactive mode with auto-remove
else
  DOCKER_RUN_FLAGS="-d -t" # Detached mode
  if [[ "$MODE" == "dev" ]]; then
    echo ""
    echo "With MODE=dev, attach directly to the bash shell:"
    echo ""
    echo -e "\t docker exec -it aircraft-container_$DRONE_ID bash"
  else
    echo ""
    echo "Attach to the Tmux session in the running 'aircraft-container':"
    echo ""
    echo -e "\t docker exec -it aircraft-container_$DRONE_ID tmux attach"
  fi
  echo ""
  echo "To stop all containers and remove stopped containers:"
  echo ""
  echo -e '\t docker stop $(docker ps -q) && docker container prune'
  echo ""
fi

# Launch the aircraft container
docker run $DOCKER_RUN_FLAGS \
  --runtime nvidia \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri --gpus all \
  --volume /tmp/argus_socket:/tmp/argus_socket \
  --volume ~/tensorrt_cache/:/tensorrt_cache \
  --device=/dev/ttyTHS1:/dev/ttyTHS1 \
  --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  --env ROS_DOMAIN_ID=$DRONE_ID --env AUTOPILOT=$AUTOPILOT --env DRONE_TYPE=$DRONE_TYPE \
  --env DRONE_ID=$DRONE_ID --env HEADLESS=$HEADLESS --env CAMERA=$CAMERA --env LIDAR=$LIDAR --env GST_DEBUG=3 \
  --env HITL=$HITL --env SIMULATED_TIME=$HITL \
  --env SUBNET_PREFIX=$SUBNET_PREFIX \
  --net=host \
  --privileged \
  --name aircraft-container_$DRONE_ID \
  ${MODE_OPTS} \
  aircraft-image

# Check ONNX runtimes
# MODE=dev HEADLESS=false ./deploy_run.sh
# docker exec -it aircraft-container bash
# python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# tmuxinator start -p /aas/aircraft.yml.erb
