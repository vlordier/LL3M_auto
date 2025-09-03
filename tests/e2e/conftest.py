"""E2E test configuration and fixtures."""

import asyncio
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

import aiohttp
import pytest

from src.utils.config import get_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def lm_studio_server():
    """Ensure LM Studio server is running."""
    settings = get_settings()

    if not settings.app.use_local_llm:
        pytest.skip("Local LLM not enabled, skipping LM Studio tests")

    # Check if LM Studio server is accessible
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.lmstudio.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        models = await response.json()
                        available_models = [m["id"] for m in models.get("data", [])]
                        if available_models:
                            print(f"✓ LM Studio connected, models: {available_models}")
                            yield settings.lmstudio.base_url
                            return
                        else:
                            print("⚠️  LM Studio running but no models loaded")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: LM Studio not accessible: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                pytest.skip(
                    "LM Studio server not accessible. Please start LM Studio "
                    "with a model loaded."
                )


@pytest.fixture(scope="session")
async def blender_mcp_server():
    """Start Blender MCP server for testing."""
    settings = get_settings()

    # Check if Blender MCP server is already running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.blender.mcp_server_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ Blender MCP server already running: {data}")
                    yield settings.blender.mcp_server_url
                    return
    except Exception:
        pass  # Server not running, we'll start it

    # Start Blender MCP server
    print("🚀 Starting Blender MCP server...")

    # Check if Blender exists
    blender_path = Path(settings.blender.path)
    if not blender_path.exists():
        pytest.skip(
            f"Blender not found at {blender_path}. Please install Blender "
            f"or update BLENDER_PATH."
        )

    # Start the server
    server_script = Path("setup/blender_mcp_server.py")
    if not server_script.exists():
        pytest.skip("Blender MCP server script not found. Run setup first.")

    process = subprocess.Popen(
        [
            str(blender_path),
            "--background",
            "--python",
            str(server_script),
            "--",
            str(settings.blender.mcp_server_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    max_retries = 30
    for _attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.blender.mcp_server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response:
                    if response.status == 200:
                        print(
                            f"✓ Blender MCP server started on port "
                            f"{settings.blender.mcp_server_port}"
                        )
                        yield settings.blender.mcp_server_url
                        # Cleanup
                        process.terminate()
                        process.wait(timeout=10)
                        return
        except Exception:
            await asyncio.sleep(1)

    # Cleanup on failure
    process.terminate()
    process.wait(timeout=10)

    # Check if there were any startup errors
    stdout, stderr = process.communicate()
    error_msg = f"Failed to start Blender MCP server after {max_retries} seconds"
    if stderr:
        error_msg += f"\nStderr: {stderr.decode()}"
    if stdout:
        error_msg += f"\nStdout: {stdout.decode()}"

    pytest.skip(error_msg)


@pytest.fixture
async def http_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Provide an HTTP session for tests."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Provide a temporary output directory for tests."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_prompt() -> str:
    """Provide a simple test prompt."""
    return "Create a simple red cube at the origin"


@pytest.fixture
def complex_test_prompt() -> str:
    """Provide a more complex test prompt."""
    return """Create a scene with:
1. A blue cube at position (0, 0, 0)
2. A red sphere at position (2, 0, 0)
3. A yellow cylinder at position (-2, 0, 0)
4. Add a camera positioned to view all objects
5. Add basic lighting to the scene"""


@pytest.mark.asyncio
async def test_prerequisites():
    """Test that all prerequisites are met for E2E testing."""
    settings = get_settings()

    # Check environment configuration
    assert settings.app.use_local_llm, "Local LLM should be enabled for E2E tests"
    assert settings.app.development, "Development mode should be enabled for E2E tests"

    # Check Blender path
    blender_path = Path(settings.blender.path)
    assert blender_path.exists(), f"Blender not found at {blender_path}"

    print("✓ All prerequisites met for E2E testing")
