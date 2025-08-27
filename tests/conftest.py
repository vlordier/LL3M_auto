"""Shared test fixtures for LL3M."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set test environment for all tests
os.environ["ENVIRONMENT"] = "test"

from src.utils.types import WorkflowState


@pytest.fixture
def sample_workflow_state() -> WorkflowState:
    """Sample workflow state for testing."""
    return WorkflowState(
        prompt="Create a red cube",
        user_feedback=None,
        subtasks=[],
        documentation="",
        generated_code="",
        execution_result=None,
        asset_metadata=None,
        error_message=None,
    )


@pytest.fixture
def mock_blender_executor() -> MagicMock:
    """Mock Blender executor."""
    executor = MagicMock()
    executor.execute_code = AsyncMock()
    return executor


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary output directory for tests."""
    return Path(tmp_path_factory.mktemp("ll3m_test_outputs"))
