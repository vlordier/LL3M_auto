#!/usr/bin/env bash
# Launch Blender with MCP server (background) with logging and PID file
set -euo pipefail

echo "🚀 Starting Blender MCP Server..."

# Defaults and args
PORT=${1:-${BLENDER_MCP_SERVER_PORT:-3001}}
LOG_PATH=${BLENDER_MCP_LOG_PATH:-"$(pwd)/blender_mcp.log"}
PID_FILE=${BLENDER_MCP_PID_FILE:-"$(pwd)/blender_mcp.pid"}

# Determine Blender executable
BLENDER_BIN=${BLENDER_PATH:-${BLENDER_EXECUTABLE:-"/Applications/Blender.app/Contents/MacOS/Blender"}}

if [ ! -x "$BLENDER_BIN" ]; then
  echo "❌ Blender not found at: $BLENDER_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

echo "Port: $PORT"
echo "Log:  $LOG_PATH"
echo "PID:  $PID_FILE"

# Start in background and write PID
"$BLENDER_BIN" --background --python setup/simple_blender_server.py -- "$PORT" >> "$LOG_PATH" 2>&1 &
MCP_PID=$!
echo "$MCP_PID" > "$PID_FILE"

echo "✅ MCP started (pid=$MCP_PID). Tail logs: tail -f \"$LOG_PATH\""
exit 0
