#!/bin/bash
set -e

# Define the expected model directory
# Matches the default save_dir in download_model.py relative to WORKDIR /app
MODEL_DIR="/app/models/gemma-2-2b-it"

echo "Checking for model in $MODEL_DIR..."

# Check for a critical file to ensure the model is actually present
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Model not found or incomplete (config.json missing). Downloading..."
    
    # Run download script
    if python scripts/download_model.py; then
        echo "Download successful."
    else
        echo "Download failed!"
        # Clean up the directory so we retry next time
        echo "Cleaning up incomplete download..."
        rm -rf "$MODEL_DIR"
        exit 1
    fi
else
    echo "Model found (config.json present). Skipping download."
fi

# Execute the main command
exec "$@"
