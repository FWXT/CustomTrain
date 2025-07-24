#!/bin/bash

# Default values
PARENT_DIR="/data/mnt_bucket/qzq/CustomTrain"
DATE=""
DEVICES="0,1,2,3,4,5,6,7"  # Default devices
KILL_SCRIPT="$PARENT_DIR/kill_npu_process.sh"
AUTO_RECOVERY=false  # Whether to automatically run recovery yaml

# Function to find the latest valid checkpoint in a directory
find_latest_checkpoint() {
    local output_dir="$1"
    if [ ! -d "$output_dir" ]; then
        echo ""
        return
    fi
    
    # Find all checkpoint directories and sort them by number (descending)
    local checkpoints=($(find "$output_dir" -maxdepth 1 -type d -name "checkpoint-*" | \
        sed 's/.*checkpoint-//' | \
        sort -nr))
    
    # Check each checkpoint from newest to oldest
    for checkpoint_num in "${checkpoints[@]}"; do
        local checkpoint_dir="$output_dir/checkpoint-$checkpoint_num"
        local global_step_dir="$checkpoint_dir/global_step$checkpoint_num"
        
        # Check if the global_step subdirectory exists
        if [ -d "$global_step_dir" ]; then
            echo "$checkpoint_dir"
            return
        else
            echo "[WARNING] Checkpoint $checkpoint_dir is incomplete (missing global_step$checkpoint_num), checking previous checkpoint..."
        fi
    done
    
    # No valid checkpoint found
    echo ""
}

# Function to create recovery yaml from failed yaml
create_recovery_yaml() {
    local failed_yaml="$1"
    local recovery_yaml="$2"
    local latest_checkpoint="$3"
    
    # Extract current output_dir from failed yaml
    local current_output_dir=$(grep "^output_dir:" "$failed_yaml" | sed 's/output_dir: *//')
    
    # Create new output_dir by appending _recovery_TIMESTAMP
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local new_output_dir="${current_output_dir}_recovery_${timestamp}"
    
    # Copy the failed yaml and modify it
    cp "$failed_yaml" "$recovery_yaml"
    
    # Update resume_from_checkpoint
    if grep -q "^resume_from_checkpoint:" "$recovery_yaml"; then
        sed -i "s|^resume_from_checkpoint:.*|resume_from_checkpoint: $latest_checkpoint|" "$recovery_yaml"
    else
        # Add resume_from_checkpoint after ddp_timeout line
        sed -i "/^ddp_timeout:/a resume_from_checkpoint: $latest_checkpoint" "$recovery_yaml"
    fi
    
    # Update output_dir
    sed -i "s|^output_dir:.*|output_dir: $new_output_dir|" "$recovery_yaml"
    
    echo "[INFO] Created recovery yaml: $recovery_yaml"
    echo "[INFO] New output_dir: $new_output_dir"
    echo "[INFO] Resume from checkpoint: $latest_checkpoint"
}

# Function to validate checkpoint
validate_checkpoint() {
    local checkpoint_dir="$1"
    
    # Check if checkpoint directory exists
    if [ ! -d "$checkpoint_dir" ]; then
        return 1
    fi
    
    # Extract checkpoint number from directory name
    local checkpoint_num=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
    local global_step_dir="$checkpoint_dir/global_step$checkpoint_num"
    
    # Check if global_step subdirectory exists
    if [ ! -d "$global_step_dir" ]; then
        return 1
    fi
    
    # Check if required files exist in checkpoint
    # Check if trainer_state.json exists (indicates training state)
    if [ -f "$checkpoint_dir/trainer_state.json" ]; then
        return 0
    fi
    
    return 1
}

# Function to display usage
usage() {
    echo "Usage: $0 --user USER --date YYYYMMDD [--devices DEVICES] [--auto-recovery]"
    echo
    echo "Required parameters:"
    echo "  --user         User name"
    echo "  --date         Date in YYYYMMDD format"
    echo
    echo "Optional parameters:"
    echo "  --devices      Comma-separated list of device IDs (default: 0,1,2,3,4,5,6,7)"
    echo "  --auto-recovery    Automatically run recovery yaml when training fails"
    echo "  -h, --help         Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --user ma-user --date 20250707"
    echo "  $0 --user ma-user --date 20250707 --devices 0,1,2,3,4,5,6,7"
    echo "  $0 --user ma-user --date 20250707 --auto-recovery"
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
        --auto-recovery)
            AUTO_RECOVERY=true
            shift
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
# if ! [[ $DATE =~ ^[0-9]{8}$ ]]; then
#     echo "[ERROR] Date must be in YYYYMMDD format"
#     exit 1
# fi

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
echo "[INFO] Auto-recovery: $AUTO_RECOVERY"
if [ "$AUTO_RECOVERY" = true ]; then
    echo "[INFO] Failed training will be automatically recovered from the latest checkpoint"
else
    echo "[INFO] Failed training will create recovery yaml files for manual execution"
fi

# Process each YAML file
echo "[INFO] Processing YAML files in $SCHEDULE_DIR..."
for yaml_file in "$SCHEDULE_DIR"/*.yaml; do
    # Skip if no files match the pattern
    [ -e "$yaml_file" ] || continue
    
    # Get the basename without extension
    basename=$(basename "$yaml_file")
    filename="${basename%.*}"
    
    echo "[INFO] Running: $basename"
    
    # ！ Remove json cache 看你自己电脑需不需要
    rm -rf /home/ma-user/.cache/huggingface/datasets/json/

    # Run the command and wait for it to complete
    nohup env ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
        llamafactory-cli train "$yaml_file" > "$LOGS_DIR/$filename.log" 2>&1
    
    # Check if command was successful
    if [ $? -eq 0 ]; then
        echo "[INFO] Successfully completed: $basename"
    else
        echo "[ERROR] Error running: $basename"
        
        # Execute kill script to clean up processes
        echo "[INFO] Executing kill script to clean up processes..."
        if [ -f "$KILL_SCRIPT" ]; then
            bash "$KILL_SCRIPT"
        else
            echo "[WARNING] Kill script not found: $KILL_SCRIPT"
        fi
        
        # Extract output_dir from failed yaml
        failed_output_dir=$(grep "^output_dir:" "$yaml_file" | sed 's/output_dir: *//')
        
        if [ -n "$failed_output_dir" ]; then
            echo "[INFO] Looking for checkpoints in: $failed_output_dir"
            
            # Find the latest checkpoint
            latest_checkpoint=$(find_latest_checkpoint "$failed_output_dir")
            
            if [ -n "$latest_checkpoint" ]; then
                echo "[INFO] Found latest checkpoint: $latest_checkpoint"
                
                # Validate the checkpoint
                if validate_checkpoint "$latest_checkpoint"; then
                    echo "[INFO] Checkpoint validation passed"
                    
                    # Create recovery yaml filename
                    recovery_filename="${filename}_recovery_$(date +"%Y%m%d_%H%M%S").yaml"
                    recovery_yaml="$SCHEDULE_DIR/$recovery_filename"
                    
                    # Create recovery yaml
                     create_recovery_yaml "$yaml_file" "$recovery_yaml" "$latest_checkpoint"
                     
                     if [ "$AUTO_RECOVERY" = true ]; then
                         echo "[INFO] Auto-recovery enabled. Running recovery yaml: $recovery_filename"
                         
                         # Remove json cache before recovery run
                         rm -rf /home/ma-user/.cache/huggingface/datasets/json/
                         
                         # Run the recovery yaml
                         recovery_log="$LOGS_DIR/${filename}_recovery_$(date +"%Y%m%d_%H%M%S").log"
                         nohup env ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
                             llamafactory-cli train "$recovery_yaml" > "$recovery_log" 2>&1
                         
                         if [ $? -eq 0 ]; then
                             echo "[INFO] Recovery training completed successfully: $recovery_filename"
                         else
                             echo "[ERROR] Recovery training failed: $recovery_filename"
                             echo "[INFO] Check recovery log: $recovery_log"
                         fi
                     else
                         echo "[INFO] Recovery yaml created: $recovery_filename"
                         echo "[INFO] You can manually run it or use --auto-recovery flag for automatic execution."
                     fi
                else
                    echo "[WARNING] Checkpoint validation failed: $latest_checkpoint"
                    echo "[WARNING] Checkpoint may be corrupted or incomplete"
                fi
            else
                echo "[WARNING] No valid checkpoints found in: $failed_output_dir"
            fi
        else
            echo "[WARNING] Could not extract output_dir from failed yaml: $yaml_file"
        fi
        
        echo "[INFO] Failure handling completed for: $basename"
    fi
done

echo "[INFO] All YAML files processed."
exit 0