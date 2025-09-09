"""Unit tests for the workflow orchestration."""

from unittest.mock import patch

import pytest

from src.utils.types import WorkflowState
from src.workflow.orchestrator import LL3MOrchestrator
from tests.mocks import mock_services


@pytest.mark.asyncio
@patch("src.workflow.workflow.services", new=mock_services)
async def test_generate_asset_e2e():
    """Test the full asset generation workflow with mock services."""
    orchestrator = LL3MOrchestrator()
    orchestrator.enable_caching = False  # Disable caching for predictable testing
    prompt = "Create a simple red cube."

    # Configure mock services to succeed
    mock_services.blender_service.should_succeed = True
    mock_services.knowledge_service.should_succeed = True

    result = await orchestrator.generate_asset(prompt)

    assert result is not None
    assert "final_state" in result
    final_state: WorkflowState = result["final_state"]

    assert final_state.execution_result is not None
    assert final_state.execution_result.success
    assert final_state.asset_metadata is not None
    # Verify that an asset file path was created (could be mock or real)
    assert final_state.asset_metadata.file_path is not None
    assert final_state.asset_metadata.file_path.endswith(".blend")
