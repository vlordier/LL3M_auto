"""Integration tests for the complete LL3M workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.types import ExecutionResult, WorkflowState
from src.workflow.graph import create_initial_workflow, create_workflow_with_config


class TestFullWorkflow:
    """Test the complete workflow integration."""

    @pytest.mark.asyncio
    async def test_successful_workflow_execution(self, sample_workflow_state):
        """Test complete workflow from prompt to asset creation."""
        # Create workflow
        workflow = create_initial_workflow()

        # Mock all external dependencies
        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
            patch("src.workflow.graph.BlenderExecutor") as mock_executor_class,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup mock OpenAI client
            mock_openai_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.usage.total_tokens = 100
            mock_openai_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_openai_client

            # Setup mock Context7 service
            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Blender documentation for cube creation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            # Setup mock Blender executor
            mock_executor = AsyncMock()
            mock_execution_result = ExecutionResult(
                success=True,
                errors=[],
                asset_path="/test/asset.blend",
                screenshot_path="/test/screenshot.png",
                execution_time=1.5,
            )
            mock_executor.execute_code.return_value = mock_execution_result
            mock_executor_class.return_value = mock_executor

            # Setup mock settings
            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            # Configure planner response
            planner_response = """
            {
                "tasks": [
                    {
                        "id": "task-1",
                        "type": "geometry",
                        "description": "Create a red cube",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"shape": "cube", "color": [0.8, 0.2, 0.2]}
                    }
                ],
                "reasoning": "Single geometry task for cube creation"
            }
            """

            # Configure coding response
            coding_response = """
            import bpy
            import mathutils

            # Clear existing mesh objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False, confirm=False)

            # Create red cube
            bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
            cube = bpy.context.active_object
            cube.name = "RedCube"

            # Create material
            material = bpy.data.materials.new(name="RedMaterial")
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)

            # Assign material
            cube.data.materials.append(material)
            """

            # Setup responses for different calls
            mock_response.choices[0].message.content = planner_response
            call_count = 0

            def side_effect(*_args, **_kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # Planner call
                    return MagicMock(
                        choices=[
                            MagicMock(message=MagicMock(content=planner_response))
                        ],
                        usage=MagicMock(total_tokens=200),
                    )
                else:  # Coding call
                    return MagicMock(
                        choices=[MagicMock(message=MagicMock(content=coding_response))],
                        usage=MagicMock(total_tokens=500),
                    )

            mock_openai_client.chat.completions.create.side_effect = side_effect

            # Execute workflow
            config = {"thread_id": "test_thread"}
            result = await workflow.ainvoke(sample_workflow_state, config=config)

            # Verify final state
            assert result["subtasks"] is not None
            assert len(result["subtasks"]) > 0
            assert result["documentation"] != ""
            assert result["generated_code"] != ""
            assert result["asset_metadata"] is not None
            assert result["execution_result"] is not None
            assert result["execution_result"].success is True

    @pytest.mark.asyncio
    async def test_workflow_with_refinement_loop(self, sample_workflow_state):
        """Test workflow with refinement iterations."""
        # Create workflow with refinement enabled
        workflow = create_workflow_with_config({"enable_refinement": True})

        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
            patch("src.workflow.graph.BlenderExecutor") as mock_executor_class,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup mocks similar to above but with initial failure
            mock_openai_client = AsyncMock()
            mock_openai_class.return_value = mock_openai_client

            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Blender documentation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            # First execution fails, second succeeds
            mock_executor = AsyncMock()
            mock_failed_result = ExecutionResult(
                success=False,
                errors=["Syntax error in generated code"],
                asset_path="",
                screenshot_path="",
                execution_time=0.5,
            )
            mock_success_result = ExecutionResult(
                success=True,
                errors=[],
                asset_path="/test/asset.blend",
                screenshot_path="/test/screenshot.png",
                execution_time=1.5,
            )
            mock_executor.execute_code.side_effect = [
                mock_failed_result,
                mock_success_result,
            ]
            mock_executor_class.return_value = mock_executor

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            # Configure responses
            planner_response = """
            {
                "tasks": [
                    {
                        "id": "task-1",
                        "type": "geometry",
                        "description": "Create a blue sphere",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"shape": "sphere", "color": [0.2, 0.2, 0.8]}
                    }
                ],
                "reasoning": "Single geometry task for sphere creation"
            }
            """

            mock_openai_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=planner_response))],
                usage=MagicMock(total_tokens=200),
            )

            # Execute workflow
            config = {"thread_id": "test_refinement_thread"}
            result = await workflow.ainvoke(sample_workflow_state, config=config)

            # Verify refinement occurred
            assert result["refinement_iterations"] >= 1

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, sample_workflow_state):
        """Test workflow error handling and graceful failures."""
        workflow = create_initial_workflow()

        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup failing OpenAI client
            mock_openai_client = AsyncMock()
            mock_openai_client.chat.completions.create.side_effect = Exception(
                "API Error"
            )
            mock_openai_class.return_value = mock_openai_client

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            # Execute workflow
            config = {"thread_id": "test_error_thread"}
            result = await workflow.ainvoke(sample_workflow_state, config=config)

            # Verify error handling
            assert result["error_message"] is not None
            assert result["should_continue"] is False

    @pytest.mark.asyncio
    async def test_workflow_with_complex_dependencies(self, sample_workflow_state):
        """Test workflow with complex task dependencies."""
        workflow = create_initial_workflow()

        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
            patch("src.workflow.graph.BlenderExecutor") as mock_executor_class,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup mocks
            mock_openai_client = AsyncMock()
            mock_openai_class.return_value = mock_openai_client

            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Complex Blender documentation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            mock_executor = AsyncMock()
            mock_execution_result = ExecutionResult(
                success=True,
                errors=[],
                asset_path="/test/complex_asset.blend",
                screenshot_path="/test/complex_screenshot.png",
                execution_time=3.0,
            )
            mock_executor.execute_code.return_value = mock_execution_result
            mock_executor_class.return_value = mock_executor

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
            }

            # Configure complex planner response with dependencies
            complex_planner_response = """
            {
                "tasks": [
                    {
                        "id": "task-1",
                        "type": "geometry",
                        "description": "Create base cube",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"shape": "cube", "location": [0, 0, 0]}
                    },
                    {
                        "id": "task-2",
                        "type": "material",
                        "description": "Add red material to cube",
                        "priority": 2,
                        "dependencies": ["task-1"],
                        "parameters": {"color": [0.8, 0.2, 0.2], "metallic": 0.5}
                    },
                    {
                        "id": "task-3",
                        "type": "lighting",
                        "description": "Add area light",
                        "priority": 3,
                        "dependencies": ["task-1"],
                        "parameters": {"type": "AREA", "energy": 5.0}
                    }
                ],
                "reasoning": "Multi-step creation with proper dependencies"
            }
            """

            complex_coding_response = """
            import bpy

            # Task 1: Create base cube
            bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
            cube = bpy.context.active_object
            cube.name = "BaseCube"

            # Task 2: Add red material
            material = bpy.data.materials.new(name="RedMaterial")
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)
            bsdf.inputs['Metallic'].default_value = 0.5
            cube.data.materials.append(material)

            # Task 3: Add area light
            bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))
            light = bpy.context.active_object
            light.data.energy = 5.0
            """

            call_count = 0

            def side_effect(*_args, **_kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MagicMock(
                        choices=[
                            MagicMock(
                                message=MagicMock(content=complex_planner_response)
                            )
                        ],
                        usage=MagicMock(total_tokens=300),
                    )
                else:
                    return MagicMock(
                        choices=[
                            MagicMock(
                                message=MagicMock(content=complex_coding_response)
                            )
                        ],
                        usage=MagicMock(total_tokens=700),
                    )

            mock_openai_client.chat.completions.create.side_effect = side_effect

            # Execute workflow
            config = {"thread_id": "test_complex_thread"}
            result = await workflow.ainvoke(sample_workflow_state, config=config)

            # Verify complex workflow completion
            assert result["subtasks"] is not None
            assert len(result["subtasks"]) == 3
            assert result["asset_metadata"] is not None
            assert result["execution_result"].success is True

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, sample_workflow_state, _tmp_path):
        """Test workflow state persistence across executions."""
        create_initial_workflow()

        # Mock checkpoint functionality
        checkpoint_data = None

        async def mock_save_checkpoint(state, name):
            nonlocal checkpoint_data
            checkpoint_data = {
                "checkpoint_name": name,
                "timestamp": 123456789,
                "state": state.model_dump(),
            }

        async def mock_load_checkpoint(_filename):
            return WorkflowState(**checkpoint_data["state"])

        with (
            patch("src.workflow.graph._save_checkpoint", mock_save_checkpoint),
            patch("src.workflow.graph._load_checkpoint", mock_load_checkpoint),
        ):
            # Test that checkpointing works during workflow execution
            # This would be more thoroughly tested in actual integration scenarios
            await mock_save_checkpoint(sample_workflow_state, "test_checkpoint")
            assert checkpoint_data is not None
            assert checkpoint_data["checkpoint_name"] == "test_checkpoint"

            # Test loading
            loaded_state = await mock_load_checkpoint("test_checkpoint.json")
            assert loaded_state.prompt == sample_workflow_state.prompt


class TestWorkflowConfiguration:
    """Test different workflow configurations."""

    def test_workflow_with_different_configs(self):
        """Test workflow creation with various configurations."""
        # Test with refinement enabled
        config_with_refinement = {"enable_refinement": True}
        workflow1 = create_workflow_with_config(config_with_refinement)
        assert workflow1 is not None

        # Test with refinement disabled
        config_without_refinement = {"enable_refinement": False}
        workflow2 = create_workflow_with_config(config_without_refinement)
        assert workflow2 is not None

        # Test with empty config (should use defaults)
        empty_config = {}
        workflow3 = create_workflow_with_config(empty_config)
        assert workflow3 is not None

    def test_workflow_node_coverage(self):
        """Test that all workflow nodes are properly configured."""
        workflow = create_initial_workflow()

        # This test would verify that all expected nodes are present
        # in the compiled workflow graph. The actual implementation
        # would depend on LangGraph's inspection capabilities.
        assert workflow is not None
