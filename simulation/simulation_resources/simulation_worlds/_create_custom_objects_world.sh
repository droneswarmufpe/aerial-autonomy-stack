#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <full_path_to_world>"
  echo "Example: ./_create_custom_targets_world.sh /aas/simulation_resources/simulation_worlds/populated_ardupilot.sdf"
  exit 1
fi

WORLD_FILE_PATH=$1
echo "WORLD_FILE_PATH: $WORLD_FILE_PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the path relative to the script's directory if it's not absolute
if [[ "$WORLD_FILE_PATH" != /* ]]; then
  WORLD_FILE_PATH="${SCRIPT_DIR}/${WORLD_FILE_PATH}"
fi

CUSTOM_OBJECTS_CONFIG_FILE="${SCRIPT_DIR}/../patches/custom_objects_config.json"

if [[ ! -f "$CUSTOM_OBJECTS_CONFIG_FILE" ]]; then
  echo "WARNING: custom_objects_config.json not found at $CUSTOM_OBJECTS_CONFIG_FILE. No objects will be added."
  exit 0
fi

MODELS_DIR="$(dirname "$WORLD_FILE_PATH")"

model_exists() {
  local model_name="$1"

  if [[ ! -d "$MODELS_DIR" ]]; then
    echo "WARNING: models directory not found at $MODELS_DIR" >&2
    return 1
  fi

  [[ -d "${MODELS_DIR}/${model_name}" ]]
}

echo "Adding custom objects from $CUSTOM_OBJECTS_CONFIG_FILE"
ALL_MODELS_XML=""
MODEL_COUNT=0

for row in $(jq -c '.objects[]' "$CUSTOM_OBJECTS_CONFIG_FILE"); do
  MODEL_COUNT=$((MODEL_COUNT + 1))
  LAT=$(echo "$row" | jq '.lat')
  LON=$(echo "$row" | jq '.lon')
  Z=$(echo "$row" | jq '.z')
  MODEL=$(echo "$row" | jq -r '.model')
  STATIC=$(echo "$row" | jq -r '.static')

  # Do not modify (IMAV world reference coordinates)
  LAT_REF=48.802462
  LON_REF=7.854108

  X=$(python3 -c "import sys, math; lat, lon, lat_ref, lon_ref = map(float, sys.argv[1:5]); r = 6371000; d_lon = math.radians(lon - lon_ref); d_lat = math.radians(lat - lat_ref); print(r * d_lon * math.cos(math.radians(lat_ref)))" "$LAT" "$LON" "$LAT_REF" "$LON_REF")
  Y=$(python3 -c "import sys, math; lat, lon, lat_ref, lon_ref = map(float, sys.argv[1:5]); r = 6371000; d_lat = math.radians(lat - lat_ref); print(r * d_lat)" "$LAT" "$LON" "$LAT_REF" "$LON_REF")

  # If there is no model field, set ime-target as default
  if [[ -z "$MODEL" || "$MODEL" == "null" ]]; then
    echo "WARNING: No model specified for object with id $MODEL_COUNT. Using default model 'ime-target'."
    MODEL=ime-target
  fi

  if ! model_exists "$MODEL"; then
    echo "WARNING: Model '$MODEL' not found in $MODELS_DIR. Skipping this object."
    continue
  fi

  # If there is no model field, set ime-target as default
  if [[ -z "$STATIC" || "$STATIC" == "null" ]]; then
    echo "WARNING: No static value specified for object with id $MODEL_COUNT. Using default static value 'false'."
    STATIC=false
  fi

  echo "Adding model '$MODEL' at position ($X, $Y, $Z) with static=$STATIC"
  MODEL_XML="    <include>\n        <uri>model://${MODEL}</uri>\n        <name>${MODEL}_${MODEL_COUNT}</name>\n        <pose degrees=\"true\">${X} ${Y} ${Z} 90 0 0</pose>\n        <static>${STATIC}</static>\n    </include>\n"
  ALL_MODELS_XML+=$MODEL_XML
done

# Read the file, replace the tag, and write the content back out
WORLD_CONTENT=$(cat "$WORLD_FILE_PATH")
echo "${WORLD_CONTENT//'</world>'/"$ALL_MODELS_XML</world>"}" > "$WORLD_FILE_PATH"