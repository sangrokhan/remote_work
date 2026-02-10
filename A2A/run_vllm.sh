#!/bin/bash

# Configuration
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}

echo "------------------------------------------------"
echo "Starting vLLM OpenAI-Compatible Server"
echo "Model Path: $MODEL_PATH"
echo "------------------------------------------------"

# Check if vllm is installed
if ! python3 -c "import vllm" &> /dev/null; then
    echo "Error: vllm is not installed. Please run: pip install vllm"
    exit 1
fi

# Run vLLM
# Note: On Mac (MPS), vLLM might have specific limitations or require specific build flags.
# Ensure you have the latest vLLM installed.
python3 -m vllm.entrypoints.openai.api_server 
    --model "$MODEL_PATH" 
    --host 0.0.0.0 
    --port 8000 
    --trust-remote-code
