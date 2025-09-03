"""Performance benchmarks and monitoring for LL3M system."""

import asyncio
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coding import CodingAgent
from src.agents.planner import PlannerAgent
from src.agents.retrieval import RetrievalAgent
from src.utils.monitoring import PerformanceMonitor
from src.utils.types import SubTask, TaskType, WorkflowState
from src.workflow.graph import create_initial_workflow


@pytest.fixture(autouse=True)
def mock_blender_executor():
    """Auto-mock BlenderExecutor for all tests."""
    # Patch at the workflow module level where it's imported
    with patch("src.workflow.graph.BlenderExecutor") as mock_class:
        # Create a mock instance
        mock_instance = AsyncMock()

        # Default successful execution result
        from src.utils.types import ExecutionResult

        default_result = ExecutionResult(
            success=True,
            errors=[],
            asset_path="/test/perf_asset.blend",
            screenshot_path="/test/perf_screenshot.png",
            execution_time=0.1,
        )
        mock_instance.execute_code.return_value = default_result

        # Return the mock instance when BlenderExecutor() is called
        mock_class.return_value = mock_instance

        yield mock_class


def create_test_workflow_state(
    prompt: str,
    *,
    original_prompt: str | None = None,
    subtasks: list[SubTask] | None = None,
    documentation: str = "",
) -> WorkflowState:
    """Create a test WorkflowState with all required fields."""
    return WorkflowState(
        prompt=prompt,
        original_prompt=original_prompt or prompt,
        user_feedback=None,
        subtasks=subtasks or [],
        documentation=documentation,
        generated_code="",
        execution_result=None,
        asset_metadata=None,
        error_message=None,
        refinement_request="",
    )


@pytest.fixture
def performance_monitor():
    """Performance monitor fixture."""
    return PerformanceMonitor()


class TestAgentPerformance:
    """Test individual agent performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_planner_agent_performance(self, agent_config, performance_monitor):
        """Benchmark planner agent performance."""
        with patch("src.agents.base.AsyncOpenAI") as mock_openai_class:
            # Setup fast mock response
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = """
            {
                "tasks": [
                    {
                        "id": "task-1",
                        "type": "geometry",
                        "description": "Create cube",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"shape": "cube"}
                    }
                ],
                "reasoning": "Simple geometry task"
            }
            """
            mock_response.usage.total_tokens = 150
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            planner = PlannerAgent(agent_config)

            # Run multiple iterations for statistical significance
            execution_times = []
            for i in range(10):
                state = create_test_workflow_state(prompt=f"Create object {i}")

                start_time = time.time()
                response = await planner.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_execution("planner", execution_time)
                performance_monitor.record_token_usage("planner", 150)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 1.0, (
                f"Planner avg time {avg_time}s exceeds 1.0s threshold"
            )
            assert max(execution_times) < 2.0, "Max planner time exceeds 2.0s threshold"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_retrieval_agent_performance(self, agent_config, performance_monitor):
        """Benchmark retrieval agent performance."""
        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
        ):
            # Setup mocks
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_context7 = MagicMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Fast retrieval response"
            mock_context7.retrieve_documentation = AsyncMock(
                return_value=mock_retrieval_response
            )
            mock_context7_class.return_value = mock_context7

            retrieval = RetrievalAgent(agent_config)

            # Benchmark retrieval performance
            execution_times = []
            for i in range(10):
                subtask = SubTask(
                    id=f"task-{i}",
                    type=TaskType.GEOMETRY,
                    description=f"Create object {i}",
                )
                state = create_test_workflow_state(
                    prompt=f"Create object {i}", subtasks=[subtask]
                )

                start_time = time.time()
                response = await retrieval.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_execution("retrieval", execution_time)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 2.0, (
                f"Retrieval avg time {avg_time}s exceeds 2.0s threshold"
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_coding_agent_performance(self, agent_config, performance_monitor):
        """Benchmark coding agent performance."""
        with patch("src.agents.base.AsyncOpenAI") as mock_openai_class:
            # Setup mock for fast code generation
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
            mock_response.usage.total_tokens = 200
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            coding = CodingAgent(agent_config)

            # Benchmark coding performance
            execution_times = []
            for i in range(10):
                subtask = SubTask(
                    id=f"task-{i}",
                    type=TaskType.GEOMETRY,
                    description=f"Create cube {i}",
                )
                state = create_test_workflow_state(
                    prompt=f"Create cube {i}",
                    subtasks=[subtask],
                    documentation="Sample docs",
                )

                start_time = time.time()
                response = await coding.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_execution("coding", execution_time)
                performance_monitor.record_token_usage("coding", 200)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 1.5, f"Coding avg time {avg_time}s exceeds 1.5s threshold"


class TestWorkflowPerformance:
    """Test end-to-end workflow performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_full_workflow_performance(
        self,
        sample_workflow_state,  # noqa: ARG002
        performance_monitor,  # noqa: ARG002
    ):
        """Benchmark full workflow execution performance."""
        workflow = create_initial_workflow()

        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup all mocks for fast execution
            mock_openai_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.usage.total_tokens = 100

            # Quick responses
            planner_response = (
                '{"tasks":[{"id":"task-1","type":"geometry",'
                '"description":"Create cube","priority":1,"dependencies":[],'
                '"parameters":{"shape":"cube"}}],"reasoning":"Quick test"}'
            )
            coding_response = "import bpy\nbpy.ops.mesh.primitive_cube_add()"

            def side_effect(*args, **kwargs):  # noqa: ARG001
                # Determine response based on request content
                messages = args[0] if args else kwargs.get("messages", [])
                if messages:
                    content = str(messages).lower()
                    if "task" in content and "decompose" in content:
                        mock_response.choices[0].message.content = planner_response
                    else:
                        mock_response.choices[0].message.content = coding_response
                else:
                    # Default to planner response
                    mock_response.choices[0].message.content = planner_response
                return mock_response

            mock_openai_client.chat.completions.create = AsyncMock(
                side_effect=side_effect
            )
            mock_openai_class.return_value = mock_openai_client

            # Fast context7 mock
            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Fast documentation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            # Fast executor mock

            # BlenderExecutor is already mocked by the fixture

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            }

            # Run workflow performance test
            completion_times = []
            for i in range(5):  # Fewer iterations for full workflow
                test_state = create_test_workflow_state(
                    prompt=f"Create test object {i}",
                    original_prompt=f"Create test object {i}",
                )

                start_time = time.time()
                result = await workflow.ainvoke(
                    test_state, {"configurable": {"thread_id": f"perf_test_{i}"}}
                )
                completion_time = time.time() - start_time

                completion_times.append(completion_time)
                performance_monitor.record_workflow_completion(completion_time)

                # Verify successful completion
                assert result["asset_metadata"] is not None

            # Performance assertions
            avg_time = statistics.mean(completion_times)
            assert avg_time < 5.0, (
                f"Workflow avg time {avg_time}s exceeds 5.0s threshold"
            )
            assert max(completion_times) < 10.0, (
                "Max workflow time exceeds 10.0s threshold"
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_workflow_performance(self, performance_monitor):
        """Test performance under concurrent load."""
        workflow = create_initial_workflow()

        with (
            patch("src.agents.base.AsyncOpenAI") as mock_openai_class,
            patch(
                "src.agents.retrieval.Context7RetrievalService"
            ) as mock_context7_class,
            patch("src.workflow.graph._save_checkpoint", AsyncMock()),
            patch("src.utils.config.settings") as mock_settings,
        ):
            # Setup mocks (similar to above)
            mock_openai_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.usage.total_tokens = 100

            planner_response = (
                '{"tasks":[{"id":"task-1","type":"geometry",'
                '"description":"Create cube","priority":1,"dependencies":[],'
                '"parameters":{"shape":"cube"}}],"reasoning":"Concurrent test"}'
            )
            coding_response = "import bpy\nbpy.ops.mesh.primitive_cube_add()"

            def side_effect(*args, **kwargs):  # noqa: ARG001
                # Determine response based on request content
                messages = args[0] if args else kwargs.get("messages", [])
                if messages:
                    content = str(messages).lower()
                    if "task" in content and "decompose" in content:
                        mock_response.choices[0].message.content = planner_response
                    else:
                        mock_response.choices[0].message.content = coding_response
                else:
                    # Default to planner response
                    mock_response.choices[0].message.content = planner_response
                return mock_response

            mock_openai_client.chat.completions.create = AsyncMock(
                side_effect=side_effect
            )
            mock_openai_class.return_value = mock_openai_client

            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Concurrent documentation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            # BlenderExecutor is already mocked by the fixture

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            }

            # Run concurrent workflows
            async def run_single_workflow(
                workflow_id: int,
            ) -> tuple[dict[str, Any], float]:
                test_state = create_test_workflow_state(
                    prompt=f"Create concurrent object {workflow_id}",
                    original_prompt=f"Create concurrent object {workflow_id}",
                )

                start_time = time.time()
                result = await workflow.ainvoke(
                    test_state,
                    {"configurable": {"thread_id": f"concurrent_test_{workflow_id}"}},
                )
                completion_time = time.time() - start_time

                performance_monitor.record_workflow_completion(completion_time)
                return result, completion_time

            # Run 3 concurrent workflows
            start_time = time.time()
            tasks = [run_single_workflow(i) for i in range(3)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Verify all completed successfully
            for result, _completion_time in results:
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                assert result.get("asset_metadata") is not None

            # Performance assertions for concurrency
            completion_times = [completion_time for _, completion_time in results]
            avg_concurrent_time = statistics.mean(completion_times)

            # Concurrent execution should not be significantly slower per workflow
            assert avg_concurrent_time < 8.0, (
                f"Concurrent avg time {avg_concurrent_time}s exceeds threshold"
            )
            assert total_time < 15.0, (
                f"Total concurrent time {total_time}s exceeds threshold"
            )


class TestPerformanceReporting:
    """Test performance monitoring and reporting."""

    def test_performance_statistics_calculation(self, performance_monitor):
        """Test performance statistics calculation."""
        # Add sample data
        performance_monitor.record_execution("planner", 0.5)
        performance_monitor.record_execution("planner", 0.7)
        performance_monitor.record_execution("planner", 0.6)

        performance_monitor.record_workflow_completion(2.5)
        performance_monitor.record_workflow_completion(3.0)
        performance_monitor.record_workflow_completion(2.8)

        performance_monitor.record_token_usage("planner", 150)
        performance_monitor.record_token_usage("planner", 200)

        # Get statistics
        stats = performance_monitor.get_statistics()

        # Verify calculations
        assert stats["planner_avg_time"] == 0.6
        assert stats["planner_median_time"] == 0.6
        assert stats["workflow_avg_completion"] == 2.7666666666666666
        assert stats["planner_avg_tokens"] == 175.0
        assert stats["planner_total_tokens"] == 350

    def test_performance_thresholds(self):
        """Test performance threshold definitions."""
        thresholds = {
            "planner_max_time": 1.0,
            "retrieval_max_time": 2.0,
            "coding_max_time": 1.5,
            "workflow_max_time": 10.0,
            "concurrent_max_time": 8.0,
        }

        # Verify thresholds are reasonable
        assert all(threshold > 0 for threshold in thresholds.values())
        assert thresholds["workflow_max_time"] > sum(
            [
                thresholds["planner_max_time"],
                thresholds["retrieval_max_time"],
                thresholds["coding_max_time"],
            ]
        ), "Workflow threshold should account for all agent execution times"
