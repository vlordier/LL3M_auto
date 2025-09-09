"""Simple targeted tests to improve MCTS coverage."""

from unittest.mock import patch

import pytest

from src.agents.mcts import MCTS_CONFIG, MCTSNode, MCTSTaskDecomposer
from src.utils.types import SubTask, TaskType
from tests.utils.builders import WorkflowStateBuilder


class TestMCTSSimpleCoverage:
    """Simple tests targeting specific uncovered lines."""

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, mcts_config):
        """Test exception handling in main process method (lines 207-209)."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Mock _run_mcts to raise an exception
            with patch.object(agent, "_run_mcts", side_effect=Exception("Simulated MCTS failure")):
                state = WorkflowStateBuilder().with_prompt("Create a cube").build()
                response = await agent.process(state)

                # Should handle exception and return failure response
                assert not response.success
                assert "MCTS task decomposition failed" in response.message

    @pytest.mark.asyncio
    async def test_simulation_exception_handling(self, mcts_config):
        """Test exception handling in simulation method (lines 320-322)."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)
            node = MCTSNode(task_description="test task")
            state = WorkflowStateBuilder().with_prompt("Create a cube").build()

            # Force all evaluation methods to raise exceptions
            with (
                patch.object(
                    agent,
                    "_evaluate_feasibility",
                    side_effect=Exception("Feasibility error"),
                ),
                patch.object(
                    agent,
                    "_evaluate_complexity",
                    side_effect=Exception("Complexity error"),
                ),
                patch.object(
                    agent,
                    "_evaluate_completeness",
                    side_effect=Exception("Completeness error"),
                ),
            ):
                reward = await agent._simulate(node, state)
                # Should return low reward due to exception (line 322)
                assert reward == 0.1

    @pytest.mark.asyncio
    async def test_max_children_expansion_limit(self, mcts_config):
        """Test max children limit in expansion (line 275 break statement)."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create parent node with existing children at max limit
            parent = MCTSNode(task_description="parent")
            max_children = MCTS_CONFIG["max_children"]

            # Fill up to max_children
            for i in range(max_children):
                child = MCTSNode(task_description=f"child_{i}", parent=parent)
                parent.children.append(child)

            state = WorkflowStateBuilder().with_prompt("Create a cube").build()

            # Mock strategies that would exceed the limit
            extra_strategies = []
            for i in range(3):  # Try to add more than limit
                extra_strategies.append(
                    {
                        "task_description": f"extra_strategy_{i}",
                        "subtasks": [
                            SubTask(
                                id=f"extra-{i}",
                                type=TaskType.GEOMETRY,
                                description=f"Extra task {i}",
                                priority=3,
                                dependencies=[],
                            )
                        ],
                        "is_terminal": True,
                        "reasoning": f"Extra reasoning {i}",
                    }
                )

            with patch.object(
                agent,
                "_generate_decomposition_strategies",
                return_value=extra_strategies,
            ):
                result = await agent._expand_node(parent, state)
                # Should still have exactly max_children due to break statement
                assert len(result.children) == max_children

    @pytest.mark.asyncio
    async def test_completeness_with_lighting_and_scene_keywords(self, mcts_config):
        """Test completeness evaluation with lighting and scene keywords (lines 530, 532)."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Node with lighting tasks
            lighting_node = MCTSNode(
                task_description="lighting task",
                subtasks=[
                    SubTask(
                        id="light-1",
                        type=TaskType.LIGHTING,
                        description="Set up lighting",
                        priority=4,
                        dependencies=[],
                    ),
                    SubTask(
                        id="geo-1",
                        type=TaskType.GEOMETRY,
                        description="Create geometry",
                        priority=5,
                        dependencies=[],
                    ),
                ],
            )

            # Test lighting keywords (line 530)
            lighting_state = WorkflowStateBuilder().with_prompt("Create a scene with beautiful lighting").build()
            lighting_score = await agent._evaluate_completeness(lighting_node, lighting_state)
            assert lighting_score > 0.3  # Should get reasonable score

            # Node with scene setup tasks
            scene_node = MCTSNode(
                task_description="scene task",
                subtasks=[
                    SubTask(
                        id="scene-1",
                        type=TaskType.SCENE_SETUP,
                        description="Set up scene",
                        priority=5,
                        dependencies=[],
                    ),
                    SubTask(
                        id="geo-2",
                        type=TaskType.GEOMETRY,
                        description="Create geometry",
                        priority=4,
                        dependencies=[],
                    ),
                ],
            )

            # Test scene keywords (line 532)
            scene_state = WorkflowStateBuilder().with_prompt("Create a complex environment scene").build()
            scene_score = await agent._evaluate_completeness(scene_node, scene_state)
            assert scene_score > 0.3  # Should get reasonable score

    @pytest.mark.asyncio
    async def test_evaluation_exception_handlers(self, mcts_config):
        """Test exception handlers in evaluation methods (lines 488, 514, 551)."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)
            node = MCTSNode(
                task_description="test task",
                subtasks=[
                    SubTask(
                        id="test",
                        type=TaskType.GEOMETRY,
                        description="Test",
                        priority=3,
                        dependencies=[],
                    )
                ],
            )
            state = WorkflowStateBuilder().with_prompt("Create something").build()

            # Test feasibility evaluation exception (line 488)
            original_feasibility = agent._evaluate_feasibility

            async def failing_feasibility(*args, **kwargs):
                raise Exception("Feasibility calculation failed")

            agent._evaluate_feasibility = failing_feasibility
            score = await agent._simulate(node, state)
            assert score == 0.1  # Simulation should handle the exception

            # Restore and test complexity evaluation exception (line 514)
            agent._evaluate_feasibility = original_feasibility
            original_complexity = agent._evaluate_complexity

            async def failing_complexity(*args, **kwargs):
                raise Exception("Complexity calculation failed")

            agent._evaluate_complexity = failing_complexity
            score = await agent._simulate(node, state)
            assert score == 0.1  # Simulation should handle the exception

            # Restore and test completeness evaluation exception (line 551)
            agent._evaluate_complexity = original_complexity

            async def failing_completeness(*args, **kwargs):
                raise Exception("Completeness calculation failed")

            agent._evaluate_completeness = failing_completeness
            score = await agent._simulate(node, state)
            assert score == 0.1  # Simulation should handle the exception
