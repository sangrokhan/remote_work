#!/bin/bash
set -e

# Resolve the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$SCRIPT_DIR"
VENV="$REPO_DIR/venv/bin/python3"

mkdir -p "$REPO_DIR/docs/raw" "$REPO_DIR/docs/md"

# 1. Download document if not exists
if [ ! -f "$REPO_DIR/docs/raw/38331-j10.docx" ]; then
    echo "[*] Downloading 38.331 spec..."
    cd "$REPO_DIR/docs/raw"
    curl -s -O https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/38331-j10.zip
    unzip -o 38331-j10.zip
    rm 38331-j10.zip
fi

# 2. Convert to Markdown
echo "[*] Converting docx to markdown..."
"$VENV" "$REPO_DIR/preprocessing/converter.py" "$REPO_DIR/docs/raw/38331-j10.docx" "$REPO_DIR/docs/md/"

# 3. Generate FSM
echo "[*] Discovering FSM structure..."
"$VENV" "$REPO_DIR/fsm_core/fsm_generator.py"

# 4. Update Visualization
echo "[*] Updating HTML visualizer..."
"$VENV" "$REPO_DIR/validation/visualizer.py"

echo "[Success] Pipeline complete."
