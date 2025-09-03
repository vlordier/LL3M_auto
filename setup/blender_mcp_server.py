"""Blender MCP Server for LL3M integration."""

import json
import sys
import time
from pathlib import Path

import bpy
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str
    timeout: int = 300


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""
    success: bool
    result: str | None = None
    error: str | None = None
    logs: list[str] = []


class BlenderMCPServer:
    """Blender MCP server for remote code execution."""

    def __init__(self, port: int = 3001):
        self.port = port
        self.app = FastAPI(title="Blender MCP Server", version="1.0.0")
        self.setup_routes()
        self.logs = []

    def setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "blender_version": bpy.app.version_string}

        @self.app.post("/execute", response_model=CodeExecutionResponse)
        async def execute_code(request: CodeExecutionRequest):
            """Execute Python code in Blender."""
            try:
                # Clear previous logs
                self.logs.clear()
                
                # Redirect stdout to capture output
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                # Execute code with output capture
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    # Create a safe execution environment
                    safe_globals = {
                        'bpy': bpy,
                        'print': print,
                        '__builtins__': __builtins__,
                    }
                    
                    exec(request.code, safe_globals)
                
                # Get captured output
                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()
                
                logs = []
                if stdout_content:
                    logs.append(f"STDOUT: {stdout_content}")
                if stderr_content:
                    logs.append(f"STDERR: {stderr_content}")
                
                return CodeExecutionResponse(
                    success=True,
                    result=stdout_content,
                    logs=logs
                )
                
            except Exception as e:
                error_msg = str(e)
                return CodeExecutionResponse(
                    success=False,
                    error=error_msg,
                    logs=[f"ERROR: {error_msg}"]
                )

        @self.app.get("/scene/info")
        async def get_scene_info():
            """Get current Blender scene information."""
            try:
                scene_info = {
                    "name": bpy.context.scene.name,
                    "frame_start": bpy.context.scene.frame_start,
                    "frame_end": bpy.context.scene.frame_end,
                    "frame_current": bpy.context.scene.frame_current,
                    "objects": [obj.name for obj in bpy.context.scene.objects],
                    "active_object": bpy.context.active_object.name if bpy.context.active_object else None,
                }
                return scene_info
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/scene/save")
        async def save_scene(filepath: str):
            """Save the current Blender scene."""
            try:
                bpy.ops.wm.save_as_mainfile(filepath=filepath)
                return {"success": True, "filepath": filepath}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        """Run the MCP server."""
        print(f"Starting Blender MCP Server on port {self.port}")
        print(f"Access at: http://localhost:{self.port}")
        print("Press Ctrl+C to stop")
        
        # Run uvicorn server
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )


def main():
    """Main entry point."""
    # Default port
    port = 3001
    
    # Check command line arguments for port
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 3001")
    
    # Create and run server
    server = BlenderMCPServer(port=port)
    server.run()


if __name__ == "__main__":
    main()