#!/bin/bash
# Launch Blender with MCP server

echo "🚀 Starting Blender with MCP Server..."

# Set default port
PORT=${1:-3001}

# Start Blender with the simple server script
/Applications/Blender.app/Contents/MacOS/Blender --background --python setup/blender_mcp_server.py -- $PORT

echo "Blender MCP Server stopped"
