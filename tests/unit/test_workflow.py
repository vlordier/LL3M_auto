"""Test workflow functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.types import WorkflowState
from src.workflow.graph import (
    _load_checkpoint,
    _save_checkpoint,
    coding_node,
    create_initial_workflow,
    create_workflow_with_config,
    execution_node,
    planner_node,
    refinement_node,
    retrieval_node,
    should_continue_main,
    should_refine,
    validation_node,
)


class TestWorkflowNodes:
    """Test workflow node functions."""

    @pytest.mark.asyncio
    async def test_planner_node_success(
        self, sample_workflow_state, mock_settings, sample_subtask
    ):
        """Test successful planner node execution."""
        # Mock the planner agent
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = [sample_subtask]
        mock_agent.process.return_value = mock_response

<<<<<<< HEAD
        with patch("src.workflow.graph.PlannerAgent", return_value=mock_agent), \
             patch("src.workflow.graph.settings", mock_settings), \
             patch("src.workflow.graph._save_checkpoint", AsyncMock()):

=======
        with (
            patch("src.workflow.graph.PlannerAgent", return_value=mock_agent),
            patch("src.workflow.graph.settings", mock_settings),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
>>>>>>> origin/master
            result = await planner_node(sample_workflow_state)

            assert result.subtasks == [sample_subtask]
            assert result.error_message is None
            assert result.original_prompt == "Create a red cube"

    @pytest.mark.asyncio
    async def test_planner_node_failure(self, sample_workflow_state, mock_settings):
        """Test planner node failure handling."""
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.message = "Planning failed"
        mock_agent.process.return_value = mock_response

<<<<<<< HEAD
        with patch("src.workflow.graph.PlannerAgent", return_value=mock_agent), \
             patch("src.workflow.graph.settings", mock_settings):

=======
        with (
            patch("src.workflow.graph.PlannerAgent", return_value=mock_agent),
            patch("src.workflow.graph.settings", mock_settings),
        ):
>>>>>>> origin/master
            result = await planner_node(sample_workflow_state)

            assert result.error_message == "Planning failed"
            assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_retrieval_node_success(self, sample_workflow_state, mock_settings):
        """Test successful retrieval node execution."""
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = "Retrieved documentation"
        mock_agent.process.return_value = mock_response

<<<<<<< HEAD
        with patch("src.workflow.graph.RetrievalAgent", return_value=mock_agent), \
             patch("src.workflow.graph.settings", mock_settings), \
             patch("src.workflow.graph._save_checkpoint", AsyncMock()):

=======
        with (
            patch("src.workflow.graph.RetrievalAgent", return_value=mock_agent),
            patch("src.workflow.graph.settings", mock_settings),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
>>>>>>> origin/master
            result = await retrieval_node(sample_workflow_state)

            assert result.documentation == "Retrieved documentation"
            assert result.error_message is None

    @pytest.mark.asyncio
    async def test_coding_node_success(self, sample_workflow_state, mock_settings):
        """Test successful coding node execution."""
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.data = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
        mock_agent.process.return_value = mock_response

<<<<<<< HEAD
        with patch("src.workflow.graph.CodingAgent", return_value=mock_agent), \
             patch("src.workflow.graph.settings", mock_settings), \
             patch("src.workflow.graph._save_checkpoint", AsyncMock()):

            result = await coding_node(sample_workflow_state)

            assert result.generated_code == "import bpy\nbpy.ops.mesh.primitive_cube_add()"
=======
        with (
            patch("src.workflow.graph.CodingAgent", return_value=mock_agent),
            patch("src.workflow.graph.settings", mock_settings),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
            result = await coding_node(sample_workflow_state)

            assert (
                result.generated_code == "import bpy\nbpy.ops.mesh.primitive_cube_add()"
            )
>>>>>>> origin/master
            assert result.error_message is None

    @pytest.mark.asyncio
    async def test_execution_node_success(
        self, sample_workflow_state, sample_execution_result
    ):
        """Test successful execution node."""
        mock_executor = AsyncMock()
        mock_executor.execute_code.return_value = sample_execution_result
<<<<<<< HEAD

        sample_workflow_state.generated_code = "import bpy\nbpy.ops.mesh.primitive_cube_add()"

        with patch("src.workflow.graph.BlenderExecutor", return_value=mock_executor), \
             patch("src.workflow.graph._save_checkpoint", AsyncMock()):

=======

        sample_workflow_state.generated_code = (
            "import bpy\nbpy.ops.mesh.primitive_cube_add()"
        )

        with (
            patch("src.workflow.graph.BlenderExecutor", return_value=mock_executor),
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
        ):
>>>>>>> origin/master
            result = await execution_node(sample_workflow_state)

            assert result.execution_result == sample_execution_result
            assert result.asset_metadata is not None
            assert result.error_message is None

    @pytest.mark.asyncio
    async def test_execution_node_failure(self, sample_workflow_state):
        """Test execution node failure handling."""
        mock_executor = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Blender error"]
        mock_executor.execute_code.return_value = mock_result

        sample_workflow_state.generated_code = "invalid code"

        with patch("src.workflow.graph.BlenderExecutor", return_value=mock_executor):
<<<<<<< HEAD

            result = await execution_node(sample_workflow_state)

=======
            result = await execution_node(sample_workflow_state)

            assert result.error_message is not None
>>>>>>> origin/master
            assert "Execution failed" in result.error_message
            assert "Blender error" in result.error_message

    @pytest.mark.asyncio
    async def test_validation_node_success(
        self, sample_workflow_state, sample_execution_result
    ):
        """Test validation node with successful execution."""
        sample_workflow_state.execution_result = sample_execution_result

        result = await validation_node(sample_workflow_state)

        assert result.needs_refinement is False
        assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_validation_node_needs_refinement(self, sample_workflow_state):
        """Test validation node that identifies need for refinement."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Code execution failed"]
        sample_workflow_state.execution_result = mock_result

        result = await validation_node(sample_workflow_state)

        assert result.needs_refinement is True
        assert "Fix issues: Code execution failed" in result.refinement_request

    @pytest.mark.asyncio
    async def test_refinement_node(self, sample_workflow_state):
        """Test refinement node processing."""
        sample_workflow_state.refinement_request = "Fix the cube color"
        sample_workflow_state.refinement_count = 0
        sample_workflow_state.original_prompt = "Create a red cube"

        result = await refinement_node(sample_workflow_state)

<<<<<<< HEAD
        assert result.refinement_iterations == 1
=======
        assert result.refinement_count == 1
>>>>>>> origin/master
        assert result.should_continue is True
        assert "Fix the cube color" in result.prompt

    @pytest.mark.asyncio
    async def test_refinement_node_max_iterations(self, sample_workflow_state):
        """Test refinement node with max iterations reached."""
        sample_workflow_state.refinement_request = "Fix something"
<<<<<<< HEAD
        sample_workflow_state.refinement_iterations = 3
=======
        sample_workflow_state.refinement_count = 3
>>>>>>> origin/master

        result = await refinement_node(sample_workflow_state)

        assert result.should_continue is False


class TestWorkflowLogic:
    """Test workflow decision logic."""

    def test_should_continue_main_success(self, sample_workflow_state):
        """Test should_continue_main with successful state."""
        result = should_continue_main(sample_workflow_state)
        assert result == "continue"

    def test_should_continue_main_error(self, sample_workflow_state):
        """Test should_continue_main with error state."""
        sample_workflow_state.error_message = "Some error"
        result = should_continue_main(sample_workflow_state)
        assert result == "end"

    def test_should_continue_main_stop(self, sample_workflow_state):
        """Test should_continue_main with stop flag."""
        sample_workflow_state.should_continue = False
        result = should_continue_main(sample_workflow_state)
        assert result == "end"

    def test_should_refine_complete(self, sample_workflow_state):
        """Test should_refine when no refinement needed."""
        sample_workflow_state.needs_refinement = False
        result = should_refine(sample_workflow_state)
        assert result == "complete"

    def test_should_refine_needs_refinement(self, sample_workflow_state):
        """Test should_refine when refinement needed."""
        sample_workflow_state.needs_refinement = True
        sample_workflow_state.refinement_count = 1
        result = should_refine(sample_workflow_state)
        assert result == "refine"

    def test_should_refine_max_iterations(self, sample_workflow_state):
        """Test should_refine with max iterations reached."""
        sample_workflow_state.needs_refinement = True
        sample_workflow_state.refinement_count = 3
        result = should_refine(sample_workflow_state)
        assert result == "complete"

    def test_should_refine_error(self, sample_workflow_state):
        """Test should_refine with error state."""
        sample_workflow_state.error_message = "Some error"
        result = should_refine(sample_workflow_state)
        assert result == "end"


class TestWorkflowGraph:
    """Test complete workflow graph."""

    def test_create_initial_workflow(self):
        """Test initial workflow graph creation."""
        graph = create_initial_workflow()
        assert graph is not None

    def test_create_workflow_with_config_refinement_enabled(self):
        """Test workflow creation with refinement enabled."""
        config = {"enable_refinement": True}
        graph = create_workflow_with_config(config)
        assert graph is not None

    def test_create_workflow_with_config_refinement_disabled(self):
        """Test workflow creation with refinement disabled."""
        config = {"enable_refinement": False}
        graph = create_workflow_with_config(config)
        assert graph is not None


class TestCheckpointing:
    """Test workflow checkpointing functionality."""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, sample_workflow_state, tmp_path):
        """Test saving workflow checkpoint."""
        with patch("src.workflow.graph.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
<<<<<<< HEAD
            mock_path.return_value.__truediv__.return_value = tmp_path / "test_checkpoint.json"
=======
            mock_path.return_value.__truediv__.return_value = (
                tmp_path / "test_checkpoint.json"
            )
>>>>>>> origin/master

            # Mock open to avoid actual file operations in test
            with patch("builtins.open", MagicMock()):
                await _save_checkpoint(sample_workflow_state, "test_checkpoint")

    @pytest.mark.asyncio
    async def test_save_checkpoint_error_handling(self, sample_workflow_state):
        """Test checkpoint save error handling."""
        with patch("src.workflow.graph.Path") as mock_path:
            mock_path.side_effect = Exception("Disk full")

            # Should not raise exception, just print warning
            await _save_checkpoint(sample_workflow_state, "test_checkpoint")

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, sample_workflow_state, tmp_path):  # noqa: ARG002
        """Test loading workflow checkpoint."""
        checkpoint_data = {
            "checkpoint_name": "test",
            "timestamp": 123456789,
            "state": sample_workflow_state.model_dump(),
        }

<<<<<<< HEAD
        with patch('builtins.open', MagicMock()) as mock_open, \
             patch('json.load', return_value=checkpoint_data):

=======
        with (
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=checkpoint_data),
        ):
>>>>>>> origin/master
            result = await _load_checkpoint("test_checkpoint.json")

            assert isinstance(result, WorkflowState)
            assert result.prompt == sample_workflow_state.prompt
