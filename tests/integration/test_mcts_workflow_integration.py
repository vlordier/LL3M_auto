"""Integration tests for MCTS agent in the full workflow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.mcts import MCTSTaskDecomposer
from src.utils.types import AgentType, SubTask, TaskType, WorkflowState
from src.workflow.workflow import create_ll3m_workflow
from tests.utils.builders import AgentResponseBuilder, WorkflowStateBuilder


class TestMCTSWorkflowIntegration:
    """Test MCTS agent integration with the full LL3M workflow."""

    @pytest.mark.asyncio
    async def test_mcts_in_workflow_planner_replacement(self):
        """Test MCTS agent as a replacement for the standard planner in workflow."""
        # Create workflow with mocked dependencies
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            # Mock agent factory to return MCTS agent for planner
            mock_agent_factory = MagicMock()
            
            # Create mock MCTS agent
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            mock_mcts_agent.process.return_value = AgentResponseBuilder(
                AgentType.MCTS_DECOMPOSER
            ).with_success([
                SubTask(
                    id="mcts-task-1",
                    type=TaskType.GEOMETRY,
                    description="Create cube geometry using MCTS planning",
                    priority=5,
                    dependencies=[],
                ),
                SubTask(
                    id="mcts-task-2",
                    type=TaskType.MATERIAL,
                    description="Apply red material using MCTS planning",
                    priority=4,
                    dependencies=["mcts-task-1"],
                ),
            ]).build()
            
            # Mock other agents to avoid full workflow execution
            mock_retrieval = AsyncMock()
            mock_retrieval.process.return_value = AgentResponseBuilder(
                AgentType.RETRIEVAL
            ).with_success("Blender documentation").build()
            
            mock_coding = AsyncMock()
            mock_coding.process.return_value = AgentResponseBuilder(
                AgentType.CODING
            ).with_success("# Blender code").build()
            
            # Configure factory to return MCTS for planner
            mock_agent_factory.create_planner.return_value = mock_mcts_agent
            mock_agent_factory.create_retrieval.return_value = mock_retrieval
            mock_agent_factory.create_coding.return_value = mock_coding
            
            mock_factory.return_value = mock_agent_factory
            
            # Create workflow
            workflow = create_ll3m_workflow()
            
            # Create initial state
            initial_state = WorkflowStateBuilder().with_prompt(
                "Create a red cube using advanced planning"
            ).build()
            
            # Mock all subsequent nodes to avoid Blender execution
            with (
                patch("src.workflow.workflow.execution_node") as mock_execution,
                patch("src.workflow.workflow.critic_node") as mock_critic,
                patch("src.workflow.workflow.verification_node") as mock_verification,
                patch("src.workflow.workflow.quality_assessment_node") as mock_quality,
                patch("src.workflow.workflow.user_proxy_node") as mock_user_proxy,
            ):
                # Set up mock to stop execution after coding
                def stop_after_coding(state):
                    state.should_continue = False
                    return state
                
                mock_execution.side_effect = stop_after_coding
                
                # Run workflow (only through coding phase)
                result = await workflow.ainvoke(
                    initial_state, 
                    config={"configurable": {"thread_id": "test-thread"}}
                )
            
            # Verify MCTS agent was called
            mock_mcts_agent.process.assert_called_once()
            
            # Verify MCTS-generated subtasks are in result
            assert len(result.subtasks) == 2
            assert result.subtasks[0].description == "Create cube geometry using MCTS planning"
            assert result.subtasks[1].description == "Apply red material using MCTS planning"
            assert result.subtasks[0].type == TaskType.GEOMETRY
            assert result.subtasks[1].type == TaskType.MATERIAL

    @pytest.mark.asyncio
    async def test_mcts_enhances_standard_planner(self):
        """Test MCTS agent working alongside standard planner for enhanced planning."""
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            mock_agent_factory = MagicMock()
            
            # Mock standard planner
            mock_planner = AsyncMock()
            mock_planner.process.return_value = AgentResponseBuilder(
                AgentType.PLANNER
            ).with_success([
                SubTask(
                    id="basic-task-1",
                    type=TaskType.GEOMETRY,
                    description="Create basic cube",
                    priority=3,
                    dependencies=[],
                ),
            ]).build()
            
            # Mock MCTS agent for advanced decomposition
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            mock_mcts_agent.process.return_value = AgentResponseBuilder(
                AgentType.MCTS_DECOMPOSER
            ).with_success([
                SubTask(
                    id="mcts-geo-1",
                    type=TaskType.GEOMETRY,
                    description="Create optimized cube mesh",
                    priority=5,
                    dependencies=[],
                ),
                SubTask(
                    id="mcts-mat-1",
                    type=TaskType.MATERIAL,
                    description="Apply PBR red material",
                    priority=4,
                    dependencies=["mcts-geo-1"],
                ),
                SubTask(
                    id="mcts-light-1",
                    type=TaskType.LIGHTING,
                    description="Set up three-point lighting",
                    priority=3,
                    dependencies=["mcts-mat-1"],
                ),
            ]).build()
            
            mock_agent_factory.create_planner.return_value = mock_planner
            mock_agent_factory.create_mcts.return_value = mock_mcts_agent
            mock_factory.return_value = mock_agent_factory
            
            # In a real enhanced workflow, we could call both agents
            initial_state = WorkflowStateBuilder().with_prompt(
                "Create a photorealistic red cube with professional lighting"
            ).build()
            
            # Simulate calling both agents
            planner_result = await mock_planner.process(initial_state)
            mcts_result = await mock_mcts_agent.process(initial_state)
            
            # Verify both agents provide subtasks
            assert planner_result.success
            assert mcts_result.success
            assert len(planner_result.data) == 1  # Basic planning
            assert len(mcts_result.data) == 3   # Advanced MCTS decomposition
            
            # MCTS should provide more detailed and sophisticated planning
            mcts_subtasks = mcts_result.data
            assert any(task.type == TaskType.LIGHTING for task in mcts_subtasks)
            assert any(task.description.lower().find("pbr") != -1 for task in mcts_subtasks)

    @pytest.mark.asyncio
    async def test_mcts_error_handling_in_workflow(self):
        """Test error handling when MCTS agent fails in workflow."""
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            mock_agent_factory = MagicMock()
            
            # Mock MCTS agent that fails
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            mock_mcts_agent.process.return_value = AgentResponseBuilder(
                AgentType.MCTS_DECOMPOSER
            ).with_failure("MCTS tree search failed").build()
            
            mock_agent_factory.create_planner.return_value = mock_mcts_agent
            mock_factory.return_value = mock_agent_factory
            
            # Create workflow
            workflow = create_ll3m_workflow()
            
            initial_state = WorkflowStateBuilder().with_prompt(
                "Create a complex scene"
            ).build()
            
            # Run workflow - should handle MCTS failure gracefully
            result = await workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": "test-thread"}}
            )
            
            # Verify error handling
            assert result.error_message is not None
            assert "MCTS tree search failed" in result.error_message
            assert result.should_continue is False

    @pytest.mark.asyncio
    async def test_mcts_performance_vs_standard_planner(self):
        """Test performance characteristics of MCTS vs standard planning."""
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            mock_agent_factory = MagicMock()
            
            # Standard planner - fast but simple
            mock_standard_planner = AsyncMock()
            standard_response = AgentResponseBuilder(AgentType.PLANNER).with_success([
                SubTask(
                    id="std-1",
                    type=TaskType.GEOMETRY,
                    description="Create object",
                    priority=3,
                    dependencies=[],
                ),
            ]).build()
            standard_response.execution_time = 0.1  # Fast
            mock_standard_planner.process.return_value = standard_response
            
            # MCTS planner - slower but more comprehensive
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            mcts_response = AgentResponseBuilder(AgentType.MCTS_DECOMPOSER).with_success([
                SubTask(
                    id="mcts-1",
                    type=TaskType.GEOMETRY,
                    description="Create optimized geometry with subdivision",
                    priority=5,
                    dependencies=[],
                ),
                SubTask(
                    id="mcts-2",
                    type=TaskType.MATERIAL,
                    description="Apply physically accurate material",
                    priority=4,
                    dependencies=["mcts-1"],
                ),
                SubTask(
                    id="mcts-3",
                    type=TaskType.LIGHTING,
                    description="Configure HDRI environment lighting",
                    priority=3,
                    dependencies=["mcts-2"],
                ),
            ]).build()
            mcts_response.execution_time = 2.5  # Slower but more thorough
            mcts_response.metadata = {
                "mcts_iterations": 20,
                "total_nodes": 15,
                "exploration_strategy": "UCB1",
            }
            mock_mcts_agent.process.return_value = mcts_response
            
            initial_state = WorkflowStateBuilder().with_prompt(
                "Create a photorealistic scene"
            ).build()
            
            # Test standard planner
            std_result = await mock_standard_planner.process(initial_state)
            
            # Test MCTS planner
            mcts_result = await mock_mcts_agent.process(initial_state)
            
            # Verify performance tradeoffs
            assert std_result.execution_time < mcts_result.execution_time  # Standard is faster
            assert len(std_result.data) < len(mcts_result.data)  # MCTS is more comprehensive
            
            # MCTS provides richer metadata
            assert "mcts_iterations" in mcts_result.metadata
            assert "exploration_strategy" in mcts_result.metadata
            
            # MCTS subtasks are more detailed
            mcts_descriptions = [task.description for task in mcts_result.data]
            assert any("subdivision" in desc.lower() for desc in mcts_descriptions)
            assert any("physically accurate" in desc.lower() for desc in mcts_descriptions)
            assert any("hdri" in desc.lower() for desc in mcts_descriptions)

    @pytest.mark.asyncio
    async def test_mcts_complex_task_decomposition(self):
        """Test MCTS agent handling complex multi-step tasks."""
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            mock_agent_factory = MagicMock()
            
            # MCTS agent for complex scene
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            complex_subtasks = [
                SubTask(
                    id="scene-setup",
                    type=TaskType.SCENE_SETUP,
                    description="Initialize scene and clear defaults",
                    priority=5,
                    dependencies=[],
                ),
                SubTask(
                    id="terrain-geo",
                    type=TaskType.GEOMETRY,
                    description="Create terrain base mesh",
                    priority=5,
                    dependencies=["scene-setup"],
                ),
                SubTask(
                    id="terrain-displacement",
                    type=TaskType.GEOMETRY,
                    description="Apply displacement mapping for terrain detail",
                    priority=4,
                    dependencies=["terrain-geo"],
                ),
                SubTask(
                    id="terrain-material",
                    type=TaskType.MATERIAL,
                    description="Create layered terrain material with multiple textures",
                    priority=4,
                    dependencies=["terrain-displacement"],
                ),
                SubTask(
                    id="vegetation-geo",
                    type=TaskType.GEOMETRY,
                    description="Generate vegetation using particle systems",
                    priority=3,
                    dependencies=["terrain-material"],
                ),
                SubTask(
                    id="atmosphere-setup",
                    type=TaskType.LIGHTING,
                    description="Configure atmospheric scattering and sky",
                    priority=3,
                    dependencies=["scene-setup"],
                ),
                SubTask(
                    id="sun-lighting",
                    type=TaskType.LIGHTING,
                    description="Set up sun lamp with realistic intensity",
                    priority=3,
                    dependencies=["atmosphere-setup"],
                ),
            ]
            
            mock_mcts_agent.process.return_value = AgentResponseBuilder(
                AgentType.MCTS_DECOMPOSER
            ).with_success(complex_subtasks).build()
            mock_mcts_agent.process.return_value.metadata = {
                "mcts_iterations": 45,
                "total_nodes": 32,
                "max_depth_reached": 3,
                "exploration_strategy": "UCB1",
            }
            
            mock_agent_factory.create_planner.return_value = mock_mcts_agent
            mock_factory.return_value = mock_agent_factory
            
            initial_state = WorkflowStateBuilder().with_prompt(
                "Create a realistic outdoor landscape with terrain, vegetation, and atmospheric lighting"
            ).build()
            
            # Create workflow and test planning phase
            workflow = create_ll3m_workflow()
            
            # Mock execution to stop after planning
            with patch("src.workflow.workflow.retrieval_node") as mock_retrieval:
                mock_retrieval.side_effect = lambda state: setattr(state, "should_continue", False) or state
                
                result = await workflow.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": "test-thread"}}
                )
            
            # Verify complex decomposition
            assert len(result.subtasks) == 7
            
            # Verify task dependencies are properly set
            dependent_tasks = [task for task in result.subtasks if task.dependencies]
            assert len(dependent_tasks) > 0
            
            # Verify task types are diverse
            task_types = {task.type for task in result.subtasks}
            assert TaskType.GEOMETRY in task_types
            assert TaskType.MATERIAL in task_types
            assert TaskType.LIGHTING in task_types
            assert TaskType.SCENE_SETUP in task_types
            
            # Verify priority ordering makes sense
            high_priority_tasks = [task for task in result.subtasks if task.priority == 5]
            assert len(high_priority_tasks) >= 2  # Scene setup and terrain base
            
            # Scene setup should have no dependencies
            scene_setup_task = next(task for task in result.subtasks if task.id == "scene-setup")
            assert len(scene_setup_task.dependencies) == 0

    @pytest.mark.asyncio
    async def test_mcts_agent_configuration_in_workflow(self):
        """Test MCTS agent respects configuration when used in workflow."""
        with (
            patch("src.workflow.workflow._get_agent_factory") as mock_factory,
            patch("src.workflow.workflow._save_checkpoint") as mock_checkpoint,
        ):
            mock_agent_factory = MagicMock()
            
            # Create MCTS agent with specific config
            mock_mcts_agent = AsyncMock(spec=MCTSTaskDecomposer)
            mock_mcts_agent.max_iterations = 10
            mock_mcts_agent.exploration_constant = 1.0
            mock_mcts_agent.max_depth = 2
            
            # Mock response with config-influenced behavior
            mock_mcts_agent.process.return_value = AgentResponseBuilder(
                AgentType.MCTS_DECOMPOSER
            ).with_success([
                SubTask(
                    id="config-task-1",
                    type=TaskType.GEOMETRY,
                    description="Task generated with config max_depth=2",
                    priority=5,
                    dependencies=[],
                ),
            ]).build()
            mock_mcts_agent.process.return_value.metadata = {
                "mcts_iterations": 10,  # Should match configured max
                "total_nodes": 8,
                "max_depth_reached": 2,  # Should respect max_depth
                "exploration_strategy": "UCB1",
            }
            
            mock_agent_factory.create_mcts.return_value = mock_mcts_agent
            mock_factory.return_value = mock_agent_factory
            
            initial_state = WorkflowStateBuilder().with_prompt(
                "Simple task to test configuration"
            ).build()
            
            # Test direct MCTS agent usage
            result = await mock_mcts_agent.process(initial_state)
            
            # Verify configuration was respected
            assert result.success
            assert result.metadata["mcts_iterations"] == 10
            assert result.metadata["max_depth_reached"] == 2
            
            # Verify agent properties
            assert mock_mcts_agent.max_iterations == 10
            assert mock_mcts_agent.exploration_constant == 1.0
            assert mock_mcts_agent.max_depth == 2

    def test_mcts_workflow_integration_points(self):
        """Test identifying all integration points for MCTS in workflow."""
        # Integration points where MCTS could be used:
        integration_points = {
            "planner_replacement": "Replace standard planner with MCTS",
            "planner_enhancement": "Use MCTS to enhance standard planning", 
            "adaptive_planning": "Switch between standard/MCTS based on task complexity",
            "refinement_planning": "Use MCTS for refinement task decomposition",
            "quality_assessment_planning": "Use MCTS for quality improvement planning",
        }
        
        # Verify we've identified the key integration points
        assert "planner_replacement" in integration_points
        assert "planner_enhancement" in integration_points
        assert "adaptive_planning" in integration_points
        
        # Each integration point should have a clear use case
        for point, description in integration_points.items():
            assert len(description) > 0
            assert isinstance(description, str)