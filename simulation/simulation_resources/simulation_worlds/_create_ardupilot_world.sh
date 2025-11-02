#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <num_quads> <num_vtols> <full_path_to_empty_world>"
  echo "Example: ./_create_ardupilot_world.sh 2 1 /aas/simulation_resources/simulation_worlds/impalpable_greyness.sdf"
  exit 1
fi

NUM_QUADS=$1
NUM_VTOLS=$2
BASE_WORLD_WITH_PATH=$3

# Resolve the path relative to the script's directory if it's not absolute
if [[ "$BASE_WORLD_WITH_PATH" != /* ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BASE_WORLD_WITH_PATH="${SCRIPT_DIR}/${BASE_WORLD_WITH_PATH}"
fi

# Create a copy of the template to work on
BASE_WORLD_DIR=$(dirname "$BASE_WORLD_WITH_PATH")
OUTPUT_FILE="${BASE_WORLD_DIR}/populated_ardupilot.sdf"
cp "$BASE_WORLD_WITH_PATH" "$OUTPUT_FILE"

# IMPORTANT: this replaces the whole <physics> block with Ardupilot's SITL settings
ARDUPILOT_PHYSICS="    <physics name=\"1ms\" type=\"ignore\">\n      <max_step_size>0.001<\/max_step_size>\n      <real_time_factor>1.0<\/real_time_factor>\n    <\/physics>"
sed -i -e "/<physics/,/<\/physics>/c\\
${ARDUPILOT_PHYSICS}" "$OUTPUT_FILE"

# This loop builds a single string containing all the <include> blocks
ALL_MODELS_XML=""
DRONE_ID=0

# Loop for quads
for i in $(seq 1 $NUM_QUADS); do
  DRONE_ID=$((DRONE_ID + 1))
  MODEL_XML="    <include>\n      <uri>model://iris_with_ardupilot_${DRONE_ID}</uri>\n      <pose degrees=\"true\">$(( (i-1) * 2 )) $(( (i-1) * 2 )) 0.20 0 0 0</pose>\n    </include>\n"
  ALL_MODELS_XML+=$MODEL_XML
done

# Loop for VTOLs
for i in $(seq 1 $NUM_VTOLS); do
  DRONE_ID=$((DRONE_ID + 1))
  MODEL_XML="    <include>\n      <uri>model://alti_transition_quad_${DRONE_ID}</uri>\n      <pose degrees=\"true\">$(( (i-1) * 2 )) $(( 2 + (i-1) * 2 )) 0.35 0 0 0</pose>\n    </include>\n"
  ALL_MODELS_XML+=$MODEL_XML
done

# Read the file, replace the tag, and write the content back out
WORLD_CONTENT=$(cat "$OUTPUT_FILE")
echo "${WORLD_CONTENT//'</world>'/"$ALL_MODELS_XML</world>"}" > "$OUTPUT_FILE"
