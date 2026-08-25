#!/bin/bash
#
# Populates an SDF world file with custom objects described in
# custom_objects_config.json, converting global (lat/lon/alt) positions to
# local (x/y/z) using the world's origin, or using local positions directly.
#
# Usage: ./_create_custom_targets_world.sh <full_path_to_world>

# Exit immediately if a command exits with a non-zero status
set -e

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

# Warnings go to stderr so they never get swallowed by command substitution
# (several functions below return values via stdout).
log_warn() {
    echo "WARNING: $*" >&2
}

log_error() {
    echo "ERROR: $*" >&2
}

# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

# Extract a scalar field from a JSON object. Missing/null becomes "".
# Usage: get_field '<json>' '.some.path'
get_field() {
    local json="$1" path="$2"
    jq -r "${path} // empty" <<< "$json"
}

# Extract a boolean field/expression as-is (no "// empty", since that would
# turn a real `false` into an empty string).
# Usage: get_bool '<json>' 'has("some_key")'
get_bool() {
    local json="$1" expr="$2"
    jq -r "$expr" <<< "$json"
}

# Fill in a default value (with a warning) when a field is empty/null.
# Prints the resolved value on stdout.
# Usage: value=$(default_if_empty "$value" "<default>" "$obj_label" "<field name for the warning>")
default_if_empty() {
    local value="$1" default="$2" obj_label="$3" field_name="$4"
    if [[ -z "$value" || "$value" == "null" ]]; then
        log_warn "No $field_name value specified for object with id $obj_label. Using default $field_name value '$default'."
        echo "$default"
    else
        echo "$value"
    fi
}

# --------------------------------------------------------------------------
# Per-object resolvers
# --------------------------------------------------------------------------

# Resolve an object's position to local x/y/z.
# Prefers global_position (converted via ORIGIN_LAT/ORIGIN_LON and the
# latlon_to_xy.py helper) over local_position when both are present.
# Prints "X Y Z" on stdout, or nothing (with a warning) if the object should
# be skipped.
resolve_position() {
    local row="$1" obj_label="$2"
    local has_global has_local

    has_global=$(get_bool "$row" 'has("global_position")')
    has_local=$(get_bool "$row" 'has("local_position")')

    if [[ "$has_global" == "true" ]]; then
        resolve_global_position "$row" "$obj_label"
    elif [[ "$has_local" == "true" ]]; then
        resolve_local_position "$row" "$obj_label"
    else
        log_warn "Object id $obj_label has neither global_position nor local_position. Skipping this object."
        return 1
    fi
}

resolve_global_position() {
    local row="$1" obj_label="$2"
    local lat lon alt x y

    lat=$(get_field "$row" '.global_position.lat')
    lon=$(get_field "$row" '.global_position.lon')
    alt=$(get_field "$row" '.global_position.alt')

    if [[ -z "$lat" || -z "$lon" ]]; then
        log_warn "Object id $obj_label has a global_position but is missing lat/lon. Skipping this object."
        return 1
    fi

    if [[ -z "$ORIGIN_LAT" || -z "$ORIGIN_LON" ]]; then
        log_warn "Object id $obj_label uses global_position but origin.lat/origin.lon are not set in $CUSTOM_OBJECTS_CONFIG_FILE. Skipping this object."
        return 1
    fi

    alt=$(default_if_empty "$alt" "0" "$obj_label" "alt")

    read -r x y <<< "$(python3 "$LATLON_TO_XY_SCRIPT" "$lat" "$lon" "$ORIGIN_LAT" "$ORIGIN_LON")"
    echo "$x $y $alt"
}

resolve_local_position() {
    local row="$1" obj_label="$2"
    local x y z

    x=$(get_field "$row" '.local_position.x')
    y=$(get_field "$row" '.local_position.y')
    z=$(get_field "$row" '.local_position.z')

    if [[ -z "$x" || -z "$y" || -z "$z" ]]; then
        log_warn "Object id $obj_label has an incomplete local_position. Skipping this object."
        return 1
    fi

    echo "$x $y $z"
}

# Resolve an object's attitude, applying defaults and the IMAV pitch offset.
# Prints "ROLL PITCH YAW" on stdout.
resolve_attitude() {
    local row="$1" obj_label="$2"
    local roll pitch yaw

    roll=$(get_field "$row" '.attitude.roll')
    pitch=$(get_field "$row" '.attitude.pitch')
    yaw=$(get_field "$row" '.attitude.yaw')

    roll=$(default_if_empty "$roll" "0" "$obj_label" "ROLL")
    yaw=$(default_if_empty "$yaw" "0" "$obj_label" "YAW")

    if [[ -z "$pitch" || "$pitch" == "null" ]]; then
        log_warn "No PITCH value specified for object with id $obj_label. Using default PITCH value '90'."
        pitch=90
    else
        pitch=$((pitch + 90)) # For the IMAV world
    fi

    echo "$roll $pitch $yaw"
}

# Resolve and validate an object's model name against MODELS_DIR.
# Prints the model name on stdout, or nothing (with a warning) if the
# object should be skipped.
resolve_model() {
    local row="$1" obj_label="$2"
    local model

    model=$(get_field "$row" '.model')
    model=$(default_if_empty "$model" "ime-target" "$obj_label" "model")

    if ! model_exists "$model"; then
        log_warn "Model '$model' not found in $MODELS_DIR. Skipping this object."
        return 1
    fi

    echo "$model"
}

# Prints "true"/"false" on stdout.
resolve_static() {
    local row="$1" obj_label="$2"
    local static

    static=$(get_field "$row" '.static')
    default_if_empty "$static" "false" "$obj_label" "static"
}

model_exists() {
    local model_name="$1"

    if [[ ! -d "$MODELS_DIR" ]]; then
        log_warn "models directory not found at $MODELS_DIR"
        return 1
    fi

    [[ -d "${MODELS_DIR}/${model_name}" ]]
}

# --------------------------------------------------------------------------
# SDF generation
# --------------------------------------------------------------------------

# Build the <include> block for one object.
build_include_xml() {
    local model="$1" index="$2" x="$3" y="$4" z="$5" roll="$6" pitch="$7" yaw="$8" static="$9"
    echo "    <include>\n        <uri>model://${model}</uri>\n        <name>${model}_${index}</name>\n        <pose degrees=\"true\">${x} ${y} ${z} ${pitch} ${roll} -${yaw}</pose>\n        <static>${static}</static>\n    </include>\n"
}

# Insert the accumulated <include> blocks just before </world> and write the
# world file back out.
write_world_file() {
    local world_file_path="$1" models_xml="$2"
    local world_content
    world_content=$(cat "$world_file_path")
    echo "${world_content//'</world>'/"$models_xml</world>"}" > "$world_file_path"
}

# --------------------------------------------------------------------------
# Per-object pipeline
# --------------------------------------------------------------------------

# Resolves and validates a single object, printing its <include> XML on
# stdout. Returns non-zero (with no output) if the object should be skipped.
process_object() {
    local row="$1" index="$2"
    local obj_id obj_label position attitude model static
    local x y z roll pitch yaw

    obj_id=$(get_field "$row" '.id')
    obj_label="${obj_id:-$index}"

    position=$(resolve_position "$row" "$obj_label") || return 1
    read -r x y z <<< "$position"

    attitude=$(resolve_attitude "$row" "$obj_label")
    read -r roll pitch yaw <<< "$attitude"

    model=$(resolve_model "$row" "$obj_label") || return 1
    static=$(resolve_static "$row" "$obj_label")

    echo "Adding model '$model' at position ($x, $y, $z), attitude (R=$roll, P=$pitch, Y=$yaw) with static=$static" >&2
    build_include_xml "$model" "$index" "$x" "$y" "$z" "$roll" "$pitch" "$yaw" "$static"
}

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

parse_args() {
    if [[ "$#" -ne 2 ]]; then
        echo "Usage: $0 <full_path_to_world> <custom_objects_config>"
        echo "Example: ./_create_custom_targets_world.sh /aas/simulation_resources/simulation_worlds/populated_ardupilot.sdf /path/to/custom_objects_config.json"
        exit 1
    fi

    echo "$1 $2"
}

# Resolve WORLD_FILE_PATH relative to the script's directory if it isn't absolute.
resolve_world_file_path() {
    local path="$1"
    if [[ "$path" != /* ]]; then
        path="${SCRIPT_DIR}/${path}"
    fi
    echo "$path"
}

main() {
    read -r WORLD_FILE_PATH OBJECTS_FILE < <(parse_args "$@")

    echo "WORLD_FILE_PATH: $WORLD_FILE_PATH"
    echo "OBJECTS_FILE: $OBJECTS_FILE"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WORLD_FILE_PATH="$(resolve_world_file_path "$WORLD_FILE_PATH")"

    CUSTOM_OBJECTS_CONFIG_FILE="${SCRIPT_DIR}/../custom_objects/${OBJECTS_FILE}.json"
    LATLON_TO_XY_SCRIPT="${SCRIPT_DIR}/../custom_objects/latlon_to_xy.py"
    MODELS_DIR="$(dirname "$WORLD_FILE_PATH")"

    if [[ ! -f "$CUSTOM_OBJECTS_CONFIG_FILE" ]]; then
        log_warn "${OBJECTS_FILE}.json not found at $CUSTOM_OBJECTS_CONFIG_FILE. No objects will be added."
        exit 0
    fi

    if [[ ! -f "$LATLON_TO_XY_SCRIPT" ]]; then
        log_error "latlon_to_xy.py not found at $LATLON_TO_XY_SCRIPT."
        exit 1
    fi

    echo "Adding custom objects from $CUSTOM_OBJECTS_CONFIG_FILE"

    # World origin (lat/lon), used to convert global positions to local x/y.
    # Only required if at least one object below uses global_position; if
    # none do, the config doesn't need an "origin" field at all.
    ORIGIN_LAT=$(jq -r '.origin.lat // empty' "$CUSTOM_OBJECTS_CONFIG_FILE")
    ORIGIN_LON=$(jq -r '.origin.lon // empty' "$CUSTOM_OBJECTS_CONFIG_FILE")

    local all_models_xml="" model_count=0 include_xml

    for row in $(jq -c '.objects[]' "$CUSTOM_OBJECTS_CONFIG_FILE"); do
        model_count=$((model_count + 1))

        if include_xml=$(process_object "$row" "$model_count"); then
            all_models_xml+="$include_xml"
        fi
    done

    write_world_file "$WORLD_FILE_PATH" "$all_models_xml"
}

main "$@"