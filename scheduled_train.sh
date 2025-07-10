#!/bin/bash

# Default values
PARENT_DIR="/data1/qzq/CustomTrain"
DATE=""
DEVICES="0,1,2,3,4,5,6,7"  # Default devices

# Function to display usage
usage() {
    echo "Usage: $0 --user USER --date YYYYMMDD [--devices DEVICES]"
    echo
    echo "Required parameters:"
    echo "  --user         User name"
    echo "  --date         Date in YYYYMMDD format"
    echo
    echo "Optional parameters:"
    echo "  --devices  Comma-separated list of device IDs (default: 0,1,2,3,4,5,6,7)"
    echo "  -h, --help         Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --user ma-user --date 20250707"
    echo "  $0 --user ma-user --date 20250707 --devices 0,1,2,3,4,5,6,7"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER="$2"
            shift 2
            ;;
        --date)
            DATE="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[ERROR] Unknown parameter: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$USER" ]; then
    echo "[ERROR] User parameter is required"
    usage
fi
if [ -z "$DATE" ]; then
    echo "[ERROR] Date parameter is required"
    usage
fi

# Validate date format
if ! [[ $DATE =~ ^[0-9]{8}$ ]]; then
    echo "[ERROR] Date must be in YYYYMMDD format"
    exit 1
fi

# Define directories
SCHEDULE_DIR="$PARENT_DIR/examples/CoEditor/$USER/$DATE/schedule"
LOGS_DIR="$PARENT_DIR/examples/CoEditor/$USER/$DATE/logs"

# Check if schedule directory exists
if [ ! -d "$SCHEDULE_DIR" ]; then
    echo "[ERROR] Directory $SCHEDULE_DIR does not exist"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"
if [ ! -d "$LOGS_DIR" ]; then
    echo "[ERROR] Failed to create logs directory $LOGS_DIR"
    exit 1
fi

echo "[INFO] Using user: $USER"
echo "[INFO] Using date: $DATE"
echo "[INFO] Using devices: $DEVICES"

# Process each YAML file
echo "[INFO] Processing YAML files in $SCHEDULE_DIR..."
for yaml_file in "$SCHEDULE_DIR"/*.yaml; do
    # Skip if no files match the pattern
    [ -e "$yaml_file" ] || continue
    
    # Get the basename without extension
    basename=$(basename "$yaml_file")
    filename="${basename%.*}"
    
    echo "[INFO] Running: $basename"
    
    # Remove json cache
    # rm -rf /home/ma-user/.cache/huggingface/datasets/json/

    # Run the command and wait for it to complete
    nohup env ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
        llamafactory-cli train "$yaml_file" > "$LOGS_DIR/$filename.log" 2>&1
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        echo "[INFO] Successfully completed: $basename"
    else
        echo "[ERROR] Error running: $basename"
    fi
done

echo "[INFO] All YAML files processed."
exit 0