"""E2E test configuration and fixtures."""

import asyncio
import os
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio

from src.utils.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def setup_e2e_environment(request):
    """Set up environment variables for E2E testing."""
    # Set test environment to allow mock API keys
    os.environ["ENVIRONMENT"] = "test"
    # Set mock API keys for services that require them
    os.environ["OPENAI_API_KEY"] = "sk-test-mock-e2e-key"
    os.environ["CONTEXT7_API_KEY"] = "test-context7-key"
    # Ensure development mode is enabled for E2E tests
    os.environ["DEVELOPMENT"] = "true"
    os.environ["USE_LOCAL_LLM"] = "true"
    # Set real Blender path for E2E tests
    os.environ["BLENDER_PATH"] = "/Applications/Blender.app/Contents/MacOS/Blender"

    yield

    # Cleanup is not strictly necessary as pytest isolates test runs


@pytest.fixture(autouse=True)
def mock_settings():
    """Override mock_settings for E2E tests - don't mock, use real settings."""
    # This fixture overrides the main conftest.py mock_settings
    # to allow E2E tests to use real configuration
    yield


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def lm_studio_server():
    """Ensure LM Studio server is running."""
    settings = get_settings()

    if not settings.app.use_local_llm:
        pytest.skip("Local LLM not enabled, skipping LM Studio tests")

    # Check if LM Studio server is accessible
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{settings.lmstudio.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response,
            ):
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
                pytest.skip("LM Studio server not accessible. Please start LM Studio with a model loaded.")


async def _check_existing_server(settings) -> bool:
    """Check if Blender MCP server is already running."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{settings.blender.mcp_server_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response,
        ):
            if response.status == 200:
                data = await response.json()
                print(f"✓ Blender MCP server already running: {data}")
                return True
    except Exception as e:
        print(f"⚠️  Blender MCP server not running: {e}")
    return False


def _validate_blender_environment(settings):
    """Validate Blender installation and server script."""
    blender_path = Path(settings.blender.path)
    if not blender_path.exists():
        pytest.skip(f"Blender not found at {blender_path}. Please install Blender or update BLENDER_PATH.")

    server_script = Path("setup/simple_blender_server.py")
    if not server_script.exists():
        pytest.skip("Blender MCP server script not found. Run setup first.")

    return blender_path, server_script


async def _wait_for_server_startup(settings, max_retries=10) -> bool:
    """Wait for server to start with health check polling."""
    for attempt in range(max_retries):
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{settings.blender.mcp_server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=1),
                ) as response,
            ):
                if response.status == 200:
                    print(f"✓ Blender MCP server started on port {settings.blender.mcp_server_port}")
                    return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Server not ready: {e}")
            await asyncio.sleep(1)
    return False


def _cleanup_process(process):
    """Cleanup server process."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest_asyncio.fixture(scope="session")
async def blender_mcp_server():
    """Start Blender MCP server for testing."""
    settings = get_settings()

    # Check if server is already running
    if await _check_existing_server(settings):
        yield settings.blender.mcp_server_url
        return

    print("🚀 Starting Blender MCP server...")

    # Validate environment
    blender_path, server_script = _validate_blender_environment(settings)

    # Start the server process
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

    # Wait for server startup
    server_started = await _wait_for_server_startup(settings)

    if not server_started:
        _cleanup_process(process)
        pytest.skip("Blender MCP server failed to start within timeout")

    # Server started successfully
    try:
        yield settings.blender.mcp_server_url
    finally:
        _cleanup_process(process)


@pytest_asyncio.fixture
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


@pytest.mark.e2e
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
