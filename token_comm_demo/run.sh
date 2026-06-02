#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/.server.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server already running (PID $(cat "$PID_FILE"))"
  exit 1
fi

nohup node "$SCRIPT_DIR/server.js" > "$SCRIPT_DIR/server.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Server started (PID $!). Log: $SCRIPT_DIR/server.log"
