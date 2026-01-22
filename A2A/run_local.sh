#!/bin/bash

# Setup script for local execution

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the script directory
cd "$SCRIPT_DIR" || exit

# 1. Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate venv
source venv/bin/activate

# 3. Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Check for Model
MODEL_DIR="models/gemma-2b-it"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found in $MODEL_DIR. Downloading..."
    python scripts/download_model.py
else
    echo "Model found in $MODEL_DIR."
fi

# 6. Run the application
echo "Running A2A Manager..."
python main.py
