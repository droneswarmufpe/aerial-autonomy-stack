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

TARGET_CONFIG_FILE="${SCRIPT_DIR}/../patches/target_config_1.json"

if [[ ! -f "$TARGET_CONFIG_FILE" ]]; then
  echo "Warning: target_config.json not found at $TARGET_CONFIG_FILE. No targets will be added."
  exit 0
fi

echo "Adding random targets from $TARGET_CONFIG_FILE"
ALL_MODELS_XML=""
TARGET_COUNT=0
for row in $(jq -c '.targets[]' "$TARGET_CONFIG_FILE"); do
  TARGET_COUNT=$((TARGET_COUNT + 1))
  X=$(echo "$row" | jq '.x')
  Y=$(echo "$row" | jq '.y')
  Z=$(echo "$row" | jq '.z')
  MODEL_XML="    <include>\n        <uri>model://ime_target</uri>\n        <name>ime_target_${TARGET_COUNT}</name>\n        <pose degrees=\"true\">${X} ${Y} ${Z} 90 0 0</pose>\n        <static>true</static>\n    </include>\n"
  ALL_MODELS_XML+=$MODEL_XML
done

# Read the file, replace the tag, and write the content back out
WORLD_CONTENT=$(cat "$WORLD_FILE_PATH")
echo "${WORLD_CONTENT//'</world>'/"$ALL_MODELS_XML</world>"}" > "$WORLD_FILE_PATH"
