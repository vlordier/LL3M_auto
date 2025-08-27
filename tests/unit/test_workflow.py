"""Test workflow state definitions."""

from src.utils.types import AssetMetadata, ExecutionResult, Issue, IssueType, SubTask, TaskType
from src.workflow.state import LL3MState


class TestWorkflowState:
    """Test workflow state functionality."""

    def test_ll3m_state_creation(self) -> None:
        """Test basic LL3MState creation."""
        # Create a minimal state
        state: LL3MState = {
            "prompt": "Create a red cube",
            "user_feedback": None,
            "subtasks": [],
            "documentation": "",
            "generated_code": "",
            "execution_result": None,
            "asset_path": None,
            "screenshot_path": None,
            "detected_issues": [],
            "refinement_count": 0,
            "max_refinements": 3,
            "asset_metadata": None,
            "should_continue": True,
            "error_message": None,
        }
        
        assert state["prompt"] == "Create a red cube"
        assert state["user_feedback"] is None
        assert state["subtasks"] == []
        assert state["refinement_count"] == 0

    def test_ll3m_state_with_data(self) -> None:
        """Test LL3MState with actual data."""
        subtask = SubTask(
            id="task-1",
            type=TaskType.GEOMETRY,
            description="Create a cube"
        )
        
        issue = Issue(
            id="issue-1",
            type=IssueType.SCALE_ISSUE,
            description="Object too small",
            severity=3,
            suggested_fix="Scale up by factor of 2"
        )
        
        execution_result = ExecutionResult(
            success=True,
            asset_path="/path/to/asset.blend",
            screenshot_path="/path/to/screenshot.png",
            logs=["Asset created"],
            errors=[],
            execution_time=2.5
        )
        
        metadata = AssetMetadata(
            id="asset-123",
            prompt="Create a red cube",
            file_path="/path/to/asset.blend",
            subtasks=[subtask]
        )
        
        state: LL3MState = {
            "prompt": "Create a red cube",
            "user_feedback": "Make it larger",
            "subtasks": [subtask],
            "documentation": "# Blender docs...",
            "generated_code": "bpy.ops.mesh.primitive_cube_add()",
            "execution_result": execution_result,
            "asset_path": "/path/to/asset.blend",
            "screenshot_path": "/path/to/screenshot.png",
            "detected_issues": [issue],
            "refinement_count": 1,
            "max_refinements": 3,
            "asset_metadata": metadata,
            "should_continue": True,
            "error_message": None,
        }
        
        assert len(state["subtasks"]) == 1
        assert len(state["detected_issues"]) == 1
        assert state["execution_result"] is not None
        assert state["asset_metadata"] is not None
        assert state["refinement_count"] == 1