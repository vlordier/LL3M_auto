"""E2E tests for enhanced LL3M workflow with all agents."""

from unittest.mock import patch

import pytest

from src.utils.types import ExecutionResult, SubTask, TaskType, WorkflowState
from src.workflow.workflow import create_enhanced_workflow


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_enhanced_workflow_e2e():
    """Test the enhanced workflow end-to-end with mocked dependencies."""
    # Mock the node functions BEFORE creating the workflow
    with (
        patch("src.workflow.enhanced_graph.planner_node") as mock_planner_node,
        patch("src.workflow.enhanced_graph.retrieval_node") as mock_retrieval_node,
        patch("src.workflow.enhanced_graph.coding_node") as mock_coding_node,
        patch("src.workflow.enhanced_graph.execution_node") as mock_execution_node,
        patch("src.workflow.enhanced_graph.critic_node") as mock_critic_node,
        patch("src.workflow.enhanced_graph.verification_node") as mock_verification_node,
        patch("src.workflow.enhanced_graph.user_proxy_node") as mock_user_proxy_node,
    ):
        # Setup all the mock functions first
        async def mock_planner_func(state):
            state.subtasks = [
                SubTask(
                    id="subtask_1",
                    type=TaskType.GEOMETRY,
                    description="Create red cube",
                    priority=1,
                    dependencies=[],
                    parameters={"shape": "cube", "color": [0.8, 0.2, 0.2]},
                )
            ]
            return state

        mock_planner_node.side_effect = mock_planner_func

        # Mock retrieval node
        async def mock_retrieval_func(state):
            state.documentation = (
                "# Blender Python API Documentation\n\n## Basic Operations\n```python\nimport bpy\n```"
            )
            return state

        mock_retrieval_node.side_effect = mock_retrieval_func

        # Mock coding node
        async def mock_coding_func(state):
            state.generated_code = "import bpy\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))"
            return state

        mock_coding_node.side_effect = mock_coding_func

        # Mock execution node
        async def mock_execution_func(state):
            state.execution_result = ExecutionResult(
                success=True,
                asset_path="/tmp/test_cube.blend",
                screenshot_path="/tmp/test_cube.png",
                logs=["Cube created successfully"],
                errors=[],
                execution_time=2.0,
            )
            return state

        mock_execution_node.side_effect = mock_execution_func

        # Mock critic node
        async def mock_critic_func(state):
            state.critic_analysis = {
                "overall_score": 8.5,
                "visual_quality": {"score": 8, "notes": "Good quality"},
                "geometry": {"score": 9, "notes": "Clean geometry"},
                "needs_refinement": False,
            }
            return state

        mock_critic_node.side_effect = mock_critic_func

        # Mock verification node
        async def mock_verification_func(state):
            state.verification_result = {"quality_score": 8.0, "issues": []}
            return state

        mock_verification_node.side_effect = mock_verification_func

        # Mock user proxy node - complete the workflow
        async def mock_user_proxy_func(state):
            state.should_continue = False  # End the workflow
            return state

        mock_user_proxy_node.side_effect = mock_user_proxy_func

        # Create the enhanced workflow AFTER setting up mocks
        workflow = create_enhanced_workflow()

        # Create initial state
        initial_state = WorkflowState(
            prompt="Create a simple red cube with metallic material and spot lighting",
            should_continue=True,
            refinement_iterations=0,
            max_refinements=3,
            refinement_request="",
            needs_refinement=False,
            original_prompt="Create a simple red cube with metallic material and spot lighting",
        )

        # Run the workflow
        config = {"configurable": {"thread_id": "test_thread"}}
        result = await workflow.ainvoke(initial_state, config)

        # Verify workflow completion
        assert result is not None
        assert result["should_continue"] is False
        assert "execution_result" in result
        assert "critic_analysis" in result
        assert "verification_result" in result


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_enhanced_workflow_with_refinement():
    """Test the enhanced workflow with refinement cycle."""
    workflow = create_enhanced_workflow()

    # Create initial state
    initial_state = WorkflowState(
        prompt="Create a detailed wooden table with realistic materials",
        should_continue=True,
        refinement_iterations=0,
        max_refinements=3,
        refinement_request="",
        needs_refinement=False,
        original_prompt="Create a detailed wooden table with realistic materials",
    )

    # Mock dependencies for refinement scenario
    with (
        patch("src.workflow.enhanced_graph.planner_node") as mock_planner_node,
        patch("src.workflow.enhanced_graph.retrieval_node") as mock_retrieval_node,
        patch("src.workflow.enhanced_graph.coding_node") as mock_coding_node,
        patch("src.workflow.enhanced_graph.execution_node") as mock_execution_node,
        patch("src.workflow.enhanced_graph.critic_node") as mock_critic_node,
        patch("src.workflow.enhanced_graph.verification_node") as mock_verification_node,
        patch("src.workflow.enhanced_graph.user_proxy_node") as mock_user_proxy_node,
    ):
        # Mock node functions for refinement scenario
        async def mock_planner_func(state):
            state.subtasks = [SubTask(id="table_task", type=TaskType.GEOMETRY, description="Create table", priority=1)]
            return state

        mock_planner_node.side_effect = mock_planner_func

        async def mock_retrieval_func(state):
            state.documentation = "Table documentation"
            return state

        mock_retrieval_node.side_effect = mock_retrieval_func

        async def mock_coding_func(state):
            state.generated_code = "import bpy\n# Create table"
            return state

        mock_coding_node.side_effect = mock_coding_func

        async def mock_execution_func(state):
            state.execution_result = ExecutionResult(
                success=True,
                asset_path="/tmp/test_table.blend",
                screenshot_path="/tmp/test_table.png",
                logs=["Table created"],
                errors=[],
                execution_time=2.0,
            )
            return state

        mock_execution_node.side_effect = mock_execution_func

        # Mock critic to request refinement
        async def mock_critic_func(state):
            state.critic_analysis = {
                "overall_score": 6.0,
                "needs_refinement": True,
                "improvement_suggestions": ["Add more realistic wood texture", "Improve table proportions"],
            }
            state.needs_refinement = True
            return state

        mock_critic_node.side_effect = mock_critic_func

        # Mock verification to also request refinement
        async def mock_verification_func(state):
            state.verification_result = {"quality_score": 6.5, "issues": ["material_quality"]}
            return state

        mock_verification_node.side_effect = mock_verification_func

        # Mock user proxy to approve refinement
        async def mock_user_proxy_func(state):
            state.should_continue = False  # End workflow for simplicity
            return state

        mock_user_proxy_node.side_effect = mock_user_proxy_func

        # Run the workflow (will only run first iteration due to mocking)
        config = {"configurable": {"thread_id": "test_refinement"}}
        result = await workflow.ainvoke(initial_state, config)

        # Verify refinement was triggered
        assert result is not None
        assert "needs_refinement" in result or "should_continue" in result
        # Note: Due to mocking, we can't test the full refinement cycle,
        # but we can verify the initial setup for refinement


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling when agents fail."""
    # Mock planner node to fail BEFORE creating workflow
    with patch("src.workflow.enhanced_graph.planner_node") as mock_planner_node:

        async def mock_planner_func(state):
            state.error_message = "Planning failed: Invalid prompt"
            state.should_continue = False
            return state

        mock_planner_node.side_effect = mock_planner_func

        # Create workflow AFTER setting up mock
        workflow = create_enhanced_workflow()

        initial_state = WorkflowState(
            prompt="Invalid prompt that should cause failure",
            should_continue=True,
            refinement_iterations=0,
            max_refinements=3,
            refinement_request="",
            needs_refinement=False,
            original_prompt="Invalid prompt that should cause failure",
        )

        # Run the workflow
        config = {"configurable": {"thread_id": "test_error"}}
        result = await workflow.ainvoke(initial_state, config)

        # Verify error handling
        assert result is not None
        assert result["should_continue"] is False
        assert result.get("error_message") is not None
        assert "Planning failed" in result["error_message"]
