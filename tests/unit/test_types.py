"""Test type definitions."""

from datetime import datetime

from src.utils.types import (
    AgentResponse,
    AgentType,
    AssetMetadata,
    ExecutionResult,
    SubTask,
    TaskType,
    WorkflowState,
)


class TestSubTask:
    """Test SubTask model."""

    def test_subtask_creation(self) -> None:
        """Test basic SubTask creation."""
        subtask = SubTask(
            id="test-1", type=TaskType.GEOMETRY, description="Create a cube"
        )

        assert subtask.id == "test-1"
        assert subtask.type == TaskType.GEOMETRY
        assert subtask.description == "Create a cube"
        assert subtask.priority == 1
        assert subtask.dependencies == []
        assert subtask.parameters == {}

    def test_subtask_with_params(self) -> None:
        """Test SubTask with parameters."""
        subtask = SubTask(
            id="test-2",
            type=TaskType.MATERIAL,
            description="Apply red material",
            priority=2,
            dependencies=["test-1"],
            parameters={"color": "red", "metallic": 0.5},
        )

        assert subtask.priority == 2
        assert subtask.dependencies == ["test-1"]
        assert subtask.parameters == {"color": "red", "metallic": 0.5}


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_execution_result_success(self) -> None:
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            asset_path="/path/to/asset.blend",
            screenshot_path="/path/to/screenshot.png",
            logs=["Asset created successfully"],
            errors=[],
            execution_time=2.5,
        )

        assert result.success is True
        assert result.asset_path == "/path/to/asset.blend"
        assert result.screenshot_path == "/path/to/screenshot.png"
        assert result.logs == ["Asset created successfully"]
        assert result.errors == []
        assert result.execution_time == 2.5

    def test_execution_result_failure(self) -> None:
        """Test failed execution result."""
        result = ExecutionResult(
            success=False, logs=[], errors=["Blender crashed"], execution_time=1.0
        )

        assert result.success is False
        assert result.asset_path is None
        assert result.screenshot_path is None
        assert result.errors == ["Blender crashed"]


class TestWorkflowState:
    """Test WorkflowState model."""

    def test_workflow_state_creation(self) -> None:
        """Test basic WorkflowState creation."""
        state = WorkflowState(prompt="Create a red cube")

        assert state.prompt == "Create a red cube"
        assert state.user_feedback is None
        assert state.subtasks == []
        assert state.documentation == ""
        assert state.generated_code == ""
        assert state.refinement_count == 0
        assert state.max_refinements == 3
        assert state.should_continue is True


class TestAgentResponse:
    """Test AgentResponse model."""

    def test_agent_response_success(self) -> None:
        """Test successful agent response."""
        response = AgentResponse(
            agent_type=AgentType.PLANNER,
            success=True,
            data=["subtask1", "subtask2"],
            message="Successfully planned tasks",
            execution_time=1.5,
            metadata={"task_count": 2},
        )

        assert response.agent_type == AgentType.PLANNER
        assert response.success is True
        assert response.data == ["subtask1", "subtask2"]
        assert response.message == "Successfully planned tasks"
        assert response.execution_time == 1.5
        assert response.metadata == {"task_count": 2}


class TestAssetMetadata:
    """Test AssetMetadata model."""

    def test_asset_metadata_creation(self) -> None:
        """Test AssetMetadata creation."""
        metadata = AssetMetadata(
            id="asset-123", prompt="Create a sphere", file_path="/path/to/sphere.blend"
        )

        assert metadata.id == "asset-123"
        assert metadata.prompt == "Create a sphere"
        assert metadata.file_path == "/path/to/sphere.blend"
        assert isinstance(metadata.creation_time, datetime)
        assert metadata.refinement_count == 0
        assert metadata.quality_score is None
        assert metadata.subtasks == []
