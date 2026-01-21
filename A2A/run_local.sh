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

# 4. Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama could not be found. Please install Ollama first."
    exit 1
fi

# 5. Check if Model exists, if not pull it
echo "Checking for Gemma 4B model..."
if ! ollama list | grep -q "gemma:4b"; then
    echo "Pulling gemma:4b model (this may take a while)..."
    ollama pull gemma:4b
fi

# 6. Run the application
echo "Running A2A Manager..."
python main.py
