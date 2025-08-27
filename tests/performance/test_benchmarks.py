"""Performance benchmarks and monitoring for LL3M system."""

import asyncio
import statistics
import time
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coding import CodingAgent
from src.agents.planner import PlannerAgent
from src.agents.retrieval import RetrievalAgent
from src.utils.types import SubTask, TaskType, WorkflowState
from src.workflow.graph import create_initial_workflow


class PerformanceMonitor:
    """Monitor and collect performance metrics."""

    def __init__(self):
        self.metrics = {
            "agent_execution_times": {},
            "workflow_completion_times": [],
            "memory_usage": [],
            "token_usage": {},
            "error_rates": {},
        }

    def record_agent_execution(self, agent_name: str, execution_time: float):
        """Record agent execution time."""
        if agent_name not in self.metrics["agent_execution_times"]:
            self.metrics["agent_execution_times"][agent_name] = []
        self.metrics["agent_execution_times"][agent_name].append(execution_time)

    def record_workflow_completion(self, completion_time: float):
        """Record workflow completion time."""
        self.metrics["workflow_completion_times"].append(completion_time)

    def record_token_usage(self, agent_name: str, tokens: int):
        """Record token usage for an agent."""
        if agent_name not in self.metrics["token_usage"]:
            self.metrics["token_usage"][agent_name] = []
        self.metrics["token_usage"][agent_name].append(tokens)

    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = {}

        # Agent execution time statistics
        for agent_name, times in self.metrics["agent_execution_times"].items():
            if times:
                stats[f"{agent_name}_avg_time"] = statistics.mean(times)
                stats[f"{agent_name}_median_time"] = statistics.median(times)
                stats[f"{agent_name}_p95_time"] = sorted(times)[int(0.95 * len(times))]

        # Workflow completion statistics
        if self.metrics["workflow_completion_times"]:
            times = self.metrics["workflow_completion_times"]
            stats["workflow_avg_completion"] = statistics.mean(times)
            stats["workflow_median_completion"] = statistics.median(times)
            stats["workflow_p95_completion"] = sorted(times)[int(0.95 * len(times))]

        # Token usage statistics
        for agent_name, tokens in self.metrics["token_usage"].items():
            if tokens:
                stats[f"{agent_name}_avg_tokens"] = statistics.mean(tokens)
                stats[f"{agent_name}_total_tokens"] = sum(tokens)

        return stats


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
        with patch('src.agents.base.AsyncOpenAI') as mock_openai_class:
            # Setup fast mock response
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''
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
            '''
            mock_response.usage.total_tokens = 150
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            planner = PlannerAgent(agent_config)

            # Run multiple iterations for statistical significance
            execution_times = []
            for i in range(10):
                state = WorkflowState(prompt=f"Create object {i}")

                start_time = time.time()
                response = await planner.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_agent_execution("planner", execution_time)
                performance_monitor.record_token_usage("planner", 150)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 1.0, f"Planner avg time {avg_time}s exceeds 1.0s threshold"
            assert max(execution_times) < 2.0, "Max planner time exceeds 2.0s threshold"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_retrieval_agent_performance(self, agent_config, performance_monitor):
        """Benchmark retrieval agent performance."""
        with patch('src.agents.base.AsyncOpenAI') as mock_openai_class, \
             patch('src.agents.retrieval.Context7RetrievalService') as mock_context7_class:

            # Setup mocks
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_context7 = MagicMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Fast retrieval response"
            mock_context7.retrieve_documentation = AsyncMock(return_value=mock_retrieval_response)
            mock_context7_class.return_value = mock_context7

            retrieval = RetrievalAgent(agent_config)

            # Benchmark retrieval performance
            execution_times = []
            for i in range(10):
                subtask = SubTask(
                    id=f"task-{i}",
                    type=TaskType.GEOMETRY,
                    description=f"Create object {i}"
                )
                state = WorkflowState(
                    prompt=f"Create object {i}",
                    subtasks=[subtask]
                )

                start_time = time.time()
                response = await retrieval.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_agent_execution("retrieval", execution_time)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 2.0, f"Retrieval avg time {avg_time}s exceeds 2.0s threshold"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_coding_agent_performance(self, agent_config, performance_monitor):
        """Benchmark coding agent performance."""
        with patch('src.agents.base.AsyncOpenAI') as mock_openai_class:
            # Setup mock for fast code generation
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
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
                    description=f"Create cube {i}"
                )
                state = WorkflowState(
                    prompt=f"Create cube {i}",
                    subtasks=[subtask],
                    documentation="Sample docs"
                )

                start_time = time.time()
                response = await coding.process(state)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                performance_monitor.record_agent_execution("coding", execution_time)
                performance_monitor.record_token_usage("coding", 200)

                assert response.success is True

            # Performance assertions
            avg_time = statistics.mean(execution_times)
            assert avg_time < 1.5, f"Coding avg time {avg_time}s exceeds 1.5s threshold"


class TestWorkflowPerformance:
    """Test end-to-end workflow performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_full_workflow_performance(self, sample_workflow_state, performance_monitor):
        """Benchmark full workflow execution performance."""
        workflow = create_initial_workflow()

        with patch('src.agents.base.AsyncOpenAI') as mock_openai_class, \
             patch('src.agents.retrieval.Context7RetrievalService') as mock_context7_class, \
             patch('src.blender.executor.BlenderExecutor') as mock_executor_class, \
             patch('src.workflow.graph._save_checkpoint', AsyncMock()), \
             patch('src.utils.config.settings') as mock_settings:

            # Setup all mocks for fast execution
            mock_openai_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.usage.total_tokens = 100

            # Quick responses
            planner_response = '{"tasks":[{"id":"task-1","type":"geometry","description":"Create cube","priority":1,"dependencies":[],"parameters":{"shape":"cube"}}],"reasoning":"Quick test"}'
            coding_response = "import bpy\nbpy.ops.mesh.primitive_cube_add()"

            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    mock_response.choices[0].message.content = planner_response
                else:
                    mock_response.choices[0].message.content = coding_response
                return mock_response

            mock_openai_client.chat.completions.create = AsyncMock(side_effect=side_effect)
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
            from src.utils.types import ExecutionResult
            mock_executor = AsyncMock()
            mock_execution_result = ExecutionResult(
                success=True,
                errors=[],
                asset_path="/fast/asset.blend",
                screenshot_path="/fast/screenshot.png",
                execution_time=0.1
            )
            mock_executor.execute_code.return_value = mock_execution_result
            mock_executor_class.return_value = mock_executor

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Run workflow performance test
            completion_times = []
            for i in range(5):  # Fewer iterations for full workflow
                test_state = WorkflowState(
                    prompt=f"Create test object {i}",
                    original_prompt=f"Create test object {i}"
                )

                start_time = time.time()
                result = await workflow.ainvoke(test_state, config={"thread_id": f"perf_test_{i}"})
                completion_time = time.time() - start_time

                completion_times.append(completion_time)
                performance_monitor.record_workflow_completion(completion_time)

                # Verify successful completion
                assert result["asset_metadata"] is not None

            # Performance assertions
            avg_time = statistics.mean(completion_times)
            assert avg_time < 5.0, f"Workflow avg time {avg_time}s exceeds 5.0s threshold"
            assert max(completion_times) < 10.0, "Max workflow time exceeds 10.0s threshold"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_workflow_performance(self, performance_monitor):
        """Test performance under concurrent load."""
        workflow = create_initial_workflow()

        with patch('src.agents.base.AsyncOpenAI') as mock_openai_class, \
             patch('src.agents.retrieval.Context7RetrievalService') as mock_context7_class, \
             patch('src.blender.executor.BlenderExecutor') as mock_executor_class, \
             patch('src.workflow.graph._save_checkpoint', AsyncMock()), \
             patch('src.utils.config.settings') as mock_settings:

            # Setup mocks (similar to above)
            mock_openai_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.usage.total_tokens = 100

            planner_response = '{"tasks":[{"id":"task-1","type":"geometry","description":"Create cube","priority":1,"dependencies":[],"parameters":{"shape":"cube"}}],"reasoning":"Concurrent test"}'
            coding_response = "import bpy\nbpy.ops.mesh.primitive_cube_add()"

            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 1:
                    mock_response.choices[0].message.content = planner_response
                else:
                    mock_response.choices[0].message.content = coding_response
                return mock_response

            mock_openai_client.chat.completions.create = AsyncMock(side_effect=side_effect)
            mock_openai_class.return_value = mock_openai_client

            mock_context7 = MagicMock()
            mock_context7.retrieve_documentation = AsyncMock()
            mock_retrieval_response = MagicMock()
            mock_retrieval_response.success = True
            mock_retrieval_response.data = "Concurrent documentation"
            mock_context7.retrieve_documentation.return_value = mock_retrieval_response
            mock_context7_class.return_value = mock_context7

            from src.utils.types import ExecutionResult
            mock_executor = AsyncMock()
            mock_execution_result = ExecutionResult(
                success=True,
                errors=[],
                asset_path="/concurrent/asset.blend",
                screenshot_path="/concurrent/screenshot.png",
                execution_time=0.1
            )
            mock_executor.execute_code.return_value = mock_execution_result
            mock_executor_class.return_value = mock_executor

            mock_settings.get_agent_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Run concurrent workflows
            async def run_single_workflow(workflow_id: int):
                test_state = WorkflowState(
                    prompt=f"Create concurrent object {workflow_id}",
                    original_prompt=f"Create concurrent object {workflow_id}"
                )

                start_time = time.time()
                result = await workflow.ainvoke(
                    test_state,
                    config={"thread_id": f"concurrent_test_{workflow_id}"}
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
            for result, completion_time in results:
                assert result[0]["asset_metadata"] is not None

            # Performance assertions for concurrency
            completion_times = [completion_time for _, completion_time in results]
            avg_concurrent_time = statistics.mean(completion_times)

            # Concurrent execution should not be significantly slower per workflow
            assert avg_concurrent_time < 8.0, f"Concurrent avg time {avg_concurrent_time}s exceeds threshold"
            assert total_time < 15.0, f"Total concurrent time {total_time}s exceeds threshold"


class TestPerformanceReporting:
    """Test performance monitoring and reporting."""

    def test_performance_statistics_calculation(self, performance_monitor):
        """Test performance statistics calculation."""
        # Add sample data
        performance_monitor.record_agent_execution("planner", 0.5)
        performance_monitor.record_agent_execution("planner", 0.7)
        performance_monitor.record_agent_execution("planner", 0.6)

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
        assert stats["workflow_avg_completion"] == 2.766666666666667
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
        assert thresholds["workflow_max_time"] > sum([
            thresholds["planner_max_time"],
            thresholds["retrieval_max_time"],
            thresholds["coding_max_time"]
        ]), "Workflow threshold should account for all agent execution times"
