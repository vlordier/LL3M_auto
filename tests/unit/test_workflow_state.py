"""Test workflow state module."""

from src.workflow.state import WorkflowState


class TestWorkflowStateModule:
    """Test workflow state module functionality."""

    def test_workflow_state_import(self) -> None:
        """Test that WorkflowState can be imported from workflow.state."""
        # Test that the import works
        assert WorkflowState is not None

        # Test that we can create an instance
        state = WorkflowState(
            prompt="Test prompt",
            user_feedback=None,
            documentation="",
            generated_code="",
            execution_result=None,
            asset_metadata=None,
            error_message=None,
            refinement_request="",
            original_prompt="Test prompt",
        )
        assert state.prompt == "Test prompt"
        assert state.subtasks == []

    def test_workflow_state_all_export(self) -> None:
        """Test that __all__ exports are correct."""
        from src.workflow.state import __all__

        assert __all__ == ["WorkflowState"]
