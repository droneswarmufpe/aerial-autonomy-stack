#!/bin/bash

# This script allows to analyze ArduPilot logs using MAVExplorer

NUM_DRONES=$((NUM_QUADS + NUM_VTOLS))
if [[ -n "$NUM_DRONES" && "$NUM_DRONES" =~ ^[0-9]+$ ]]; then
    
    for i in $(seq 1 $NUM_DRONES); do
        LOGS_BASE_DIR="/aas/ardu_sitl_${i}/logs"
        LASTLOG_PATH="${LOGS_BASE_DIR}/LASTLOG.TXT"
        log_index=$(cat "$LASTLOG_PATH" | tr -d '\r') # Remove potential carriage return characters
        filename=$(printf "%08d.BIN" "$log_index") # Create filenames like 00000001.BIN, 00000002.BIN, etc.
        logfile="${LOGS_BASE_DIR}/$filename"
        if [ -f "$logfile" ]; then
            echo "Opening log for drone $i: $logfile"
            MAVExplorer.py "$logfile"  &
        else
            echo "Log file not found for drone $i: $logfile"
        fi
    done

else
    echo "Error: NUM_DRONES environment variable is not set or is not a valid number."
    exit 1
fi