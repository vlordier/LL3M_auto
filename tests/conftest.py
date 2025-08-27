"""Shared test fixtures for LL3M."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from src.utils.types import (
    AgentResponse,
    AgentType,
    AssetMetadata,
    ExecutionResult,
    SubTask,
    TaskType,
    WorkflowState,
)


@pytest.fixture
def sample_workflow_state() -> WorkflowState:
    """Sample workflow state for testing."""
    return WorkflowState(
        prompt="Create a red cube",
        original_prompt="Create a red cube",
        subtasks=[],
        documentation="",
        generated_code="",
        user_feedback=None,
        execution_result=None,
        refinement_request="",
        asset_metadata=None,
        error_message=None,
    )


@pytest.fixture
def sample_subtask() -> SubTask:
    """Sample subtask for testing."""
    return SubTask(
        id="task-1",
        type=TaskType.GEOMETRY,
        description="Create a red cube",
        priority=1,
        dependencies=[],
        parameters={"shape": "cube", "color": [0.8, 0.2, 0.2], "location": [0, 0, 0]},
    )


@pytest.fixture
def sample_agent_response() -> AgentResponse:
    """Sample agent response for testing."""
    return AgentResponse(
        agent_type=AgentType.PLANNER,
        success=True,
        data=["test_data"],
        message="Test successful",
        execution_time=0.5,
        metadata={"test": True},
    )


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """Sample execution result for testing."""
    return ExecutionResult(
        success=True,
        errors=[],
        asset_path="/test/asset.blend",
        screenshot_path="/test/screenshot.png",
        execution_time=1.0,
    )


@pytest.fixture
def sample_asset_metadata() -> AssetMetadata:
    """Sample asset metadata for testing."""
    return AssetMetadata(
        id="test-asset-1",
        prompt="Create a red cube",
        file_path="/test/asset.blend",
        screenshot_path="/test/screenshot.png",
        subtasks=[],
        quality_score=8.0,
    )


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock(spec=AsyncOpenAI)

    # Mock successful response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.total_tokens = 100

    # Make the create method properly async
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_blender_executor() -> MagicMock:
    """Mock Blender executor."""
    executor = MagicMock()
    executor.execute_code = AsyncMock()

    # Default successful execution
    executor.execute_code.return_value = ExecutionResult(
        success=True,
        errors=[],
        asset_path="/test/asset.blend",
        screenshot_path="/test/screenshot.png",
        execution_time=1.0,
    )

    return executor


@pytest.fixture
def mock_context7_service() -> MagicMock:
    """Mock Context7 retrieval service."""
    service = MagicMock()
    service.retrieve_documentation = AsyncMock()

    # Default successful retrieval
    mock_response = MagicMock()
    mock_response.success = True
    mock_response.data = "Sample Blender documentation"
    service.retrieve_documentation.return_value = mock_response

    return service


@pytest.fixture
def agent_config() -> dict[str, Any]:
    """Sample agent configuration."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "max_retries": 3,
        "timeout": 30.0,
    }


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary output directory for tests."""
    return Path(tmp_path_factory.mktemp("ll3m_test_outputs"))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings configuration."""
    settings = MagicMock()
    settings.get_agent_config.return_value = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "max_retries": 3,
        "timeout": 30.0,
    }
    return settings
