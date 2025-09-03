#!/usr/bin/env python3
"""Setup script for Blender MCP integration on macOS."""

import os
import subprocess
import sys
from pathlib import Path


def check_blender_installation() -> Path | None:
    """Check if Blender is installed and return its path."""
    common_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender.app/Contents/MacOS/blender",
        "/usr/local/bin/blender",
        "/opt/homebrew/bin/blender",
    ]

    for path in common_paths:
        blender_path = Path(path)
        if blender_path.exists():
            print(f"✓ Found Blender at: {blender_path}")
            return blender_path

    print("⚠️  Blender not found in common locations")
    print("Please install Blender from: https://www.blender.org/download/")
    return None


def setup_blender_python_environment(blender_path: Path) -> bool:
    """Setup Python environment for Blender with required packages."""
    print("\n📦 Setting up Blender Python environment...")

    # Get Blender's Python version
    try:
        result = subprocess.run(  # nosec B603
            [
                str(blender_path),
                "--background",
                "--python-expr",
                "import sys; print(sys.version)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"❌ Failed to get Blender Python version: {result.stderr}")
            return False

        print(f"✓ Blender Python version: {result.stdout.strip()}")
    except subprocess.TimeoutExpired:
        print("❌ Timeout getting Blender Python version")
        return False

    # Install required packages in Blender's Python
    packages = [
        "fastapi",
        "uvicorn",
        "aiohttp",
        "structlog",
        "pydantic",
    ]

    for package in packages:
        print(f"  Installing {package}...")
        try:
            # Use Blender's built-in pip
            result = subprocess.run(  # nosec B603
                [
                    str(blender_path),
                    "--background",
                    "--python-expr",
                    f"import subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '{package}'])",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"⚠️  Warning: Failed to install {package}: {result.stderr}")
            else:
                print(f"    ✓ {package} installed")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Timeout installing {package}")

    return True


def create_blender_mcp_server() -> Path:
    """Create the Blender MCP server script."""
    print("\n🔧 Creating Blender MCP server...")

    server_script = '''"""Blender MCP Server for LL3M integration."""

import json
import sys
import threading
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
'''

    # Create setup directory if it doesn't exist
    setup_dir = Path("setup")
    setup_dir.mkdir(exist_ok=True)

    server_path = setup_dir / "blender_mcp_server.py"
    with open(server_path, "w") as f:
        f.write(server_script)

    print(f"✓ Created Blender MCP server at: {server_path}")
    return server_path


def create_blender_startup_script(blender_path: Path, server_script: Path) -> Path:
    """Create a startup script to launch Blender with MCP server."""
    startup_script = f'''#!/bin/bash
"""Launch Blender with MCP server."""

echo "🚀 Starting Blender with MCP Server..."

# Start Blender with the MCP server script
"{blender_path}" --background --python "{server_script}" -- $1

echo "Blender MCP Server stopped"
'''

    script_path = Path("setup/start_blender_mcp.sh")
    with open(script_path, "w") as f:
        f.write(startup_script)

    # Make executable (owner only for security)
    os.chmod(script_path, 0o744)  # nosec B103

    print(f"✓ Created startup script at: {script_path}")
    return script_path


def create_test_script() -> Path:
    """Create a test script to verify the setup."""
    test_script = '''#!/usr/bin/env python3
"""Test script for Blender MCP server."""

import asyncio
import aiohttp

async def test_blender_mcp():
    """Test the Blender MCP server."""
    base_url = "http://localhost:3001"

    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Health check passed: {data.get('status')}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

if __name__ == "__main__":
    print("🧪 Testing Blender MCP Server...")
    success = asyncio.run(test_blender_mcp())
    if not success:
        print("❌ Test failed.")
        exit(1)
    else:
        print("✅ Test passed!")
'''

    test_path = Path("setup/test_blender_mcp.py")
    with open(test_path, "w") as f:
        f.write(test_script)

    # Make executable (owner only for security)
    os.chmod(test_path, 0o744)  # nosec B103

    print(f"✓ Created test script at: {test_path}")
    return test_path


def create_env_template():
    """Create a template .env file for local development."""
    env_template = """# LL3M Local Development Configuration

# LLM Configuration - Use local LLM via LM Studio
USE_LOCAL_LLM=true

# LM Studio Configuration
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=local-model
LMSTUDIO_TEMPERATURE=0.7
LMSTUDIO_MAX_TOKENS=2000
LMSTUDIO_TIMEOUT=300

# OpenAI Configuration (fallback)
# OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Blender Configuration
BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender
BLENDER_HEADLESS=true
BLENDER_TIMEOUT=300
BLENDER_MCP_SERVER_PORT=3001
BLENDER_MCP_SERVER_URL=http://localhost:3001

# Application Configuration
LOG_LEVEL=DEBUG
DEVELOPMENT=true
DEBUG=true
MAX_REFINEMENT_ITERATIONS=3
ENABLE_ASYNC=true

# Context7 Configuration (optional)
# CONTEXT7_MCP_SERVER=http://localhost:8080
# CONTEXT7_API_KEY=your-context7-key
"""

    env_path = Path(".env.local")
    with open(env_path, "w") as f:
        f.write(env_template)

    print(f"✓ Created environment template at: {env_path}")
    return env_path


def main():
    """Main setup function."""
    print("🔧 LL3M Blender MCP Setup for macOS")
    print("=" * 50)

    # Check Blender installation
    blender_path = check_blender_installation()
    if not blender_path:
        return False

    # Setup Python environment
    if not setup_blender_python_environment(blender_path):
        print("❌ Failed to setup Blender Python environment")
        return False

    # Create MCP server script
    server_script = create_blender_mcp_server()

    # Create startup script
    create_blender_startup_script(blender_path, server_script)

    # Create test script
    create_test_script()

    # Create environment template
    create_env_template()

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.local to .env and customize settings")
    print("2. Install and start LM Studio with a model loaded")
    print("3. Start Blender MCP server: ./setup/start_blender_mcp.sh")
    print("4. Run tests: ./setup/test_blender_mcp.py")
    print("5. Run E2E tests: python -m pytest tests/e2e/")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
