"""Tests for the main workflow graph."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.types import (
    AgentResponse,
    AssetMetadata,
    ExecutionResult,
    SubTask,
    TaskType,
    WorkflowState,
)
from src.workflow.graph import (
    _load_checkpoint,
    _save_checkpoint,
    coding_node,
    create_initial_workflow,
    execution_node,
    planner_node,
    refinement_node,
    retrieval_node,
    should_continue,
    should_refine,
    validation_node,
)


@pytest.fixture
def sample_workflow_state() -> WorkflowState:
    """Return a sample workflow state for testing."""
    return WorkflowState(prompt="Create a simple cube")


@pytest.fixture
def mock_planner_response() -> AgentResponse:
    """Mock response for the planner agent."""
    return AgentResponse(
        success=True,
        data=[
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create a cube",
                priority=1,
                dependencies=[],
                parameters={"shape": "cube"},
            )
        ],
        message="Planner executed successfully",
    )


@pytest.fixture
def mock_retrieval_response() -> AgentResponse:
    """Mock response for the retrieval agent."""
    return AgentResponse(
        success=True,
        data="Blender documentation for cube creation",
        message="Documentation retrieved successfully",
    )


@pytest.fixture
def mock_coding_response() -> AgentResponse:
    """Mock response for the coding agent."""
    return AgentResponse(
        success=True,
        data="import bpy\nbpy.ops.mesh.primitive_cube_add()",
        message="Code generated successfully",
    )


@pytest.fixture
def mock_execution_result(tmp_path) -> ExecutionResult:
    """Mock execution result from Blender executor."""
    return ExecutionResult(
        success=True,
        asset_path=str(tmp_path / "test_asset.blend"),
        screenshot_path=str(tmp_path / "test_screenshot.png"),
        logs=["Blender output"],
        errors=[],
        execution_time=1.0,
    )


class TestWorkflowNodes:
    """Test individual nodes of the workflow graph."""

    @pytest.mark.asyncio
    async def test_planner_node(self, sample_workflow_state, mock_planner_response):
        """Test planner node execution."""
        with (
            patch(
                "src.agents.planner.PlannerAgent.process",
                AsyncMock(return_value=mock_planner_response),
            ) as mock_process,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            state = await planner_node(sample_workflow_state)

            mock_process.assert_called_once()
            assert state.subtasks == mock_planner_response.data
            assert state.original_prompt == sample_workflow_state.prompt

    @pytest.mark.asyncio
    async def test_retrieval_node(self, sample_workflow_state, mock_retrieval_response):
        """Test retrieval node execution."""
        with (
            patch(
                "src.agents.retrieval.RetrievalAgent.process",
                AsyncMock(return_value=mock_retrieval_response),
            ) as mock_process,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            state = await retrieval_node(sample_workflow_state)

            mock_process.assert_called_once()
            assert state.documentation == mock_retrieval_response.data

    @pytest.mark.asyncio
    async def test_coding_node(self, sample_workflow_state, mock_coding_response):
        """Test coding node execution."""
        sample_workflow_state.subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create a cube",
                parameters={"shape": "cube"},
            )
        ]
        sample_workflow_state.documentation = "Blender docs"

        with (
            patch(
                "src.agents.coding.CodingAgent.process",
                AsyncMock(return_value=mock_coding_response),
            ) as mock_process,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            state = await coding_node(sample_workflow_state)

            mock_process.assert_called_once()
            assert state.generated_code == mock_coding_response.data

    @pytest.mark.asyncio
    async def test_execution_node(self, sample_workflow_state, mock_execution_result):
        """Test execution node execution."""
        sample_workflow_state.generated_code = "import bpy"

        with (
            patch(
                "src.blender.enhanced_executor.EnhancedBlenderExecutor.execute_code",
                AsyncMock(return_value=mock_execution_result),
            ) as mock_execute_code,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            state = await execution_node(sample_workflow_state)

            mock_execute_code.assert_called_once()
            assert state.execution_result == mock_execution_result
            assert state.asset_metadata.file_path == mock_execution_result.asset_path

    @pytest.mark.asyncio
    async def test_validation_node_success(
        self, sample_workflow_state, mock_execution_result, tmp_path
    ):
        """Test validation node with successful execution."""
        sample_workflow_state.execution_result = mock_execution_result
        sample_workflow_state.asset_metadata = AssetMetadata(
            id="test",
            prompt="test",
            file_path=str(tmp_path / "test.blend"),
            screenshot_path=str(tmp_path / "test.png"),
        )

        state = await validation_node(sample_workflow_state)

        assert state.needs_refinement is False
        assert state.should_continue is False
        assert state.refinement_request is None

    @pytest.mark.asyncio
    async def test_validation_node_failure(
        self, sample_workflow_state, mock_execution_result
    ):
        """Test validation node with failed execution."""
        mock_execution_result.success = False
        mock_execution_result.errors = ["Blender crashed"]
        sample_workflow_state.execution_result = mock_execution_result

        state = await validation_node(sample_workflow_state)

        assert state.needs_refinement is True
        assert state.should_continue is True
        assert "Blender crashed" in state.refinement_request

    @pytest.mark.asyncio
    async def test_refinement_node(self, sample_workflow_state):
        """Test refinement node execution."""
        sample_workflow_state.refinement_request = "Improve lighting"
        sample_workflow_state.refinement_iterations = 0
        sample_workflow_state.original_prompt = sample_workflow_state.prompt

        state = await refinement_node(sample_workflow_state)

        assert state.refinement_iterations == 1
        assert "Improve lighting" in state.prompt
        assert state.should_continue is True


class TestWorkflowConditions:
    """Test conditional edges of the workflow graph."""

    def test_should_continue(self):
        """Test should_continue condition."""
        state = WorkflowState(should_continue=True)
        assert should_continue(state) == "continue"

        state.should_continue = False
        assert should_continue(state) == "end"

        state.should_continue = True
        state.error_message = "Error"
        assert should_continue(state) == "end"

    def test_should_refine(self):
        """Test should_refine condition."""
        state = WorkflowState(needs_refinement=True, refinement_iterations=0)
        assert should_refine(state) == "refine"

        state.refinement_iterations = 3
        assert should_refine(state) == "complete"

        state.needs_refinement = False
        assert should_refine(state) == "complete"

        state.error_message = "Error"
        assert should_refine(state) == "end"


class TestWorkflowGraph:
    """Test the overall workflow graph structure and execution flow."""

    @pytest.mark.asyncio
    async def test_create_initial_workflow(self):
        """Test creation of the initial workflow graph."""
        workflow = create_initial_workflow()
        assert workflow is not None

        # Verify nodes and edges (basic check)
        nodes = workflow.nodes
        assert "planner" in nodes
        assert "retrieval" in nodes
        assert "coding" in nodes
        assert "execution" in nodes
        assert "validation" in nodes
        assert "refinement" in nodes

    @pytest.mark.asyncio
    async def test_workflow_execution_flow(self, sample_workflow_state, tmp_path):
        """Test the complete flow of the initial workflow."""
        workflow = create_initial_workflow()

        with (
            patch(
                "src.agents.planner.PlannerAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data=[], message="")
                ),
            ),
            patch(
                "src.agents.retrieval.RetrievalAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.agents.coding.CodingAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.blender.enhanced_executor.EnhancedBlenderExecutor.execute_code",
                AsyncMock(
                    return_value=ExecutionResult(
                        success=True,
                        asset_path=str(tmp_path / "test.blend"),
                        screenshot_path=str(tmp_path / "test.png"),
                    )
                ),
            ),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            result = await workflow.ainvoke(sample_workflow_state)

            assert result["should_continue"] is False
            assert result["error_message"] == ""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, sample_workflow_state, _tmp_path):
        """Test saving workflow checkpoint."""
        # Mock Path.mkdir to avoid actual directory creation
        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", MagicMock()) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):
            await _save_checkpoint(sample_workflow_state, "test_checkpoint")

            mock_mkdir.assert_called_once()
            mock_open.assert_called_once()
            mock_json_dump.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, sample_workflow_state, _tmp_path):
        """Test loading workflow checkpoint."""
        mock_checkpoint_data = {
            "checkpoint_name": "test_checkpoint",
            "timestamp": time.time(),
            "state": sample_workflow_state.model_dump(),
        }

        with (
            patch("builtins.open", MagicMock()) as mock_open,
            patch("json.load", return_value=mock_checkpoint_data) as mock_json_load,
        ):
            loaded_state = await _load_checkpoint("test_checkpoint.json")

            mock_open.assert_called_once()
            mock_json_load.assert_called_once()
            assert loaded_state.prompt == sample_workflow_state.prompt

    @pytest.mark.asyncio
    async def test_workflow_with_config_refinement_enabled(
        self, sample_workflow_state, tmp_path
    ):
        """Test workflow with refinement enabled via config."""
        workflow = create_initial_workflow()

        with (
            patch(
                "src.agents.planner.PlannerAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data=[], message="")
                ),
            ),
            patch(
                "src.agents.retrieval.RetrievalAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.agents.coding.CodingAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.blender.enhanced_executor.EnhancedBlenderExecutor.execute_code",
                AsyncMock(
                    return_value=ExecutionResult(
                        success=False,
                        errors=["test error"],
                        asset_path=str(tmp_path / "test.blend"),
                        screenshot_path=str(tmp_path / "test.png"),
                    )
                ),
            ),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            result = await workflow.ainvoke(sample_workflow_state)

            assert result["should_continue"] is True  # Should enter refinement
            assert result["refinement_iterations"] == 1

    @pytest.mark.asyncio
    async def test_workflow_with_config_refinement_disabled(
        self, sample_workflow_state, tmp_path
    ):
        """Test workflow with refinement disabled via config."""
        workflow = create_initial_workflow()

        with (
            patch(
                "src.agents.planner.PlannerAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data=[], message="")
                ),
            ),
            patch(
                "src.agents.retrieval.RetrievalAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.agents.coding.CodingAgent.process",
                AsyncMock(
                    return_value=AgentResponse(success=True, data="", message="")
                ),
            ),
            patch(
                "src.blender.enhanced_executor.EnhancedBlenderExecutor.execute_code",
                AsyncMock(
                    return_value=ExecutionResult(
                        success=False,
                        errors=["test error"],
                        asset_path=str(tmp_path / "test.blend"),
                        screenshot_path=str(tmp_path / "test.png"),
                    )
                ),
            ),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            result = await workflow.ainvoke(sample_workflow_state)

            assert result["should_continue"] is True  # Should enter refinement
            assert result["refinement_iterations"] == 1
