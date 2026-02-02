#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate and install requirements
echo "Installing requirements..."
source venv/bin/activate
pip install -r requirements.txt

echo "Setup complete. To use the environment, run 'source venv/bin/activate'"
