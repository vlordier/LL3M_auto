"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.mcts import MCTSNode
from src.utils.types import ExecutionResult, SubTask, TaskType, WorkflowState


@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch):
    """Set test environment variable."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-for-testing")
    monkeypatch.setenv("BLENDER_PATH", "/usr/bin/blender")
    # Ensure test defaults (override .env)
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("DEVELOPMENT", "false")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("USE_LOCAL_LLM", "false")


@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """Mock settings to avoid need for API keys."""
    with patch("src.utils.config.Settings") as mock_settings_class:
        mock_settings_instance = MagicMock()
        mock_settings_instance.openai.api_key = "test-key"
        mock_settings_instance.context7.api_key = "test-key"
        mock_settings_instance.blender.path = str(tmp_path / "fake-blender")
        mock_settings_class.return_value = mock_settings_instance
        yield


@pytest.fixture
def sample_workflow_state() -> WorkflowState:
    """Return a sample workflow state for testing."""
    return WorkflowState(
        prompt="Create a red cube",
        user_feedback=None,
        documentation="",
        generated_code="",
        execution_result=None,
        refinement_request="",
        original_prompt="Create a red cube",
        asset_metadata=None,
        error_message=None,
        previous_screenshot_path=None,
        critic_analysis=None,
        verification_result=None,
    )


@pytest.fixture
def agent_config() -> dict:
    """Return a sample agent config for testing."""
    return {"model": "gpt-4", "temperature": 0.7}


@pytest.fixture
def sample_subtask() -> SubTask:
    """Return a sample subtask for testing."""
    return SubTask(
        id="task-1",
        type=TaskType.GEOMETRY,
        description="Create a red cube",
        priority=1,
        dependencies=[],
        parameters={"shape": "cube", "color": [0.8, 0.2, 0.2]},
    )


@pytest.fixture
def sample_execution_result(tmp_path) -> ExecutionResult:
    """Return a sample execution result for testing."""
    return ExecutionResult(
        success=True,
        asset_path=str(tmp_path / "test_asset.blend"),
        screenshot_path=str(tmp_path / "test_screenshot.png"),
        logs=["Blender output log"],
        errors=[],
        execution_time=1.0,
    )


@pytest.fixture
def mcts_config():
    """Basic MCTS agent configuration."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1500,
        "max_iterations": 20,
        "max_depth": 3,
        "exploration_weight": 1.414,
    }


@pytest.fixture
def sample_mcts_node():
    """Sample MCTS node for testing."""
    return MCTSNode(
        task_description="Create a red cube",
        subtasks=[
            SubTask(
                id="test-geometry",
                type=TaskType.GEOMETRY,
                description="Create cube geometry",
                priority=5,
                dependencies=[]
            ),
            SubTask(
                id="test-material",
                type=TaskType.MATERIAL,
                description="Apply red material",
                priority=4,
                dependencies=["test-geometry"]
            )
        ]
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response for MCTS decomposition."""
    response_content = {
        "strategies": [
            {
                "strategy": "geometry_first",
                "subtasks": [
                    {
                        "task_type": "geometry",
                        "description": "Create cube geometry",
                        "priority": 5,
                        "dependencies": []
                    },
                    {
                        "task_type": "material",
                        "description": "Apply red material",
                        "priority": 4,
                        "dependencies": []
                    }
                ],
                "is_terminal": True,
                "reasoning": "Create geometry first, then apply material"
            }
        ]
    }
    import json
    return json.dumps(response_content)
