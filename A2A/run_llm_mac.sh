#!/bin/bash

# Configuration
export MODEL_PATH=${MODEL_PATH:-"./models/gemma-2-2b-it"}
export PYTHONUNBUFFERED=1

echo "------------------------------------------------"
echo "Starting LLM Server natively on Mac"
echo "Model Path: $MODEL_PATH"
echo "------------------------------------------------"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model directory $MODEL_PATH not found."
    echo "The server will attempt to download the model if needed."
fi

# Run the server
# Pytorch with MPS requires specific env vars sometimes, but usually defaults are fine
# unless you want to force it. Our llm_server.py already checks for MPS.
python3 llm_server.py
