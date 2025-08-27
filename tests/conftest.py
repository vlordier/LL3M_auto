"""Shared test fixtures for LL3M."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.types import WorkflowState


@pytest.fixture
def sample_workflow_state() -> WorkflowState:
    """Sample workflow state for testing."""
    return WorkflowState(
        prompt="Create a red cube",
        subtasks=[],
        documentation="",
        generated_code="",
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
    return tmp_path_factory.mktemp("ll3m_test_outputs")
