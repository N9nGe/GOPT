#!/bin/bash

# Evaluation script with table output
# Usage: ./eval_all_configs_table.sh <checkpoint_path> [output_file]

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: No checkpoint path provided"
    echo "Usage: ./eval_all_configs_table.sh <checkpoint_path> [output_file]"
    echo "Example: ./eval_all_configs_table.sh logs/my_run/policy_step_best.pth"
    exit 1
fi

CHECKPOINT="$1"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Set output file
if [ -z "$2" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_FILE="eval_table_${TIMESTAMP}.txt"
else
    OUTPUT_FILE="$2"
fi

# List of test configurations
CONFIGS=(
    "cfg/config_test_10x10x10.yaml"
    "cfg/config_test_30x30x30.yaml"
    "cfg/config_test_50x50x50.yaml"
    "cfg/config_test_100x100x100.yaml"
    "cfg/config_test_noncubic_5x8x10.yaml"
    "cfg/config_test_noncubic_8x18x15.yaml"
    "cfg/config_test_noncubic_12x15x20.yaml"
    "cfg/config_test_noncubic_15x25x30.yaml"
    "cfg/config_test_noncubic_20x35x40.yaml"
)

# Print header
{
    echo "=========================================================================="
    echo "                         Evaluation Results"
    echo "=========================================================================="
    echo "Checkpoint: $CHECKPOINT"
    echo "Date: $(date)"
    echo "=========================================================================="
    echo ""
    printf "%-25s | %-15s | %-15s | %-15s\n" "Configuration" "Utilization" "Items Packed" "Std Dev"
    echo "--------------------------------------------------------------------------"
} | tee "$OUTPUT_FILE"

# Loop through each configuration
for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        printf "%-25s | %-15s | %-15s | %-15s\n" "$(basename $CONFIG .yaml)" "FILE NOT FOUND" "-" "-" | tee -a "$OUTPUT_FILE"
        continue
    fi

    # Extract bin size from config name
    CONFIG_NAME=$(basename "$CONFIG" .yaml | sed 's/config_test_//')

    echo "Testing $CONFIG_NAME..." >&2

    # Run evaluation and capture output
    RESULT=$(python ts_test.py --config "$CONFIG" --ckp "$CHECKPOINT" 2>&1)

    # Extract metrics
    RATIO=$(echo "$RESULT" | grep "average space utilization" | awk '{print $4}')
    NUM=$(echo "$RESULT" | grep "average put item number" | awk '{print $5}')
    STD=$(echo "$RESULT" | grep "standard variance" | awk '{print $3}')

    # Print results in table format
    if [ -n "$RATIO" ]; then
        printf "%-25s | %-15s | %-15s | %-15s\n" "$CONFIG_NAME" "$RATIO" "$NUM" "$STD" | tee -a "$OUTPUT_FILE"
    else
        printf "%-25s | %-15s | %-15s | %-15s\n" "$CONFIG_NAME" "ERROR" "-" "-" | tee -a "$OUTPUT_FILE"
    fi
done

# Print footer
{
    echo "=========================================================================="
    echo "Evaluation Complete! Results saved to: $OUTPUT_FILE"
    echo "=========================================================================="
} | tee -a "$OUTPUT_FILE"
