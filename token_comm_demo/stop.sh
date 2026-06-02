#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found. Server not running."
  exit 1
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  rm "$PID_FILE"
  echo "Server stopped (PID $PID)"
else
  echo "Process $PID not found. Cleaning up PID file."
  rm "$PID_FILE"
fi
