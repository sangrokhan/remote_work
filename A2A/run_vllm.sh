#!/bin/bash

# Configuration
export MODEL_PATH=${MODEL_PATH:-"models/gemma-2-2b-it"}

echo "------------------------------------------------"
echo "Starting vLLM OpenAI-Compatible Server"
echo "Model Path: $MODEL_PATH"
echo "------------------------------------------------"

# Run vLLM
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
