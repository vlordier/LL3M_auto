"""Tests for MCTS (LATS) task decomposition agent."""

import json
import math
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.mcts import (
    HIGH_QUALITY_THRESHOLD,
    MCTS_CONFIG,
    MIN_VISITS_FOR_EARLY_STOP,
    MCTSNode,
    MCTSState,
    MCTSTaskDecomposer,
)
from src.utils.types import AgentType, SubTask, TaskType, WorkflowState


@pytest.fixture
def mcts_config():
    """Basic MCTS agent configuration."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1500,
        "mcts_max_iterations": 10,
        "mcts_exploration_constant": 1.414,
        "mcts_max_depth": 3,
        "mcts_simulation_depth": 2,
    }


@pytest.fixture
def sample_mcts_node():
    """Sample MCTS node for testing."""
    return MCTSNode(
        task_description="Create a red cube",
        subtasks=[
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create cube geometry",
                priority=5,
                dependencies=[],
            ),
            SubTask(
                id="task-2",
                type=TaskType.MATERIAL,
                description="Apply red material",
                priority=4,
                dependencies=["task-1"],
            ),
        ],
        completion_strategy="sequential",
        visits=5,
        total_reward=3.5,
        feasibility_score=0.8,
        complexity_score=0.4,
        completeness_score=0.9,
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response for MCTS decomposition."""
    response_content = {
        "strategies": [
            {
                "strategy": "sequential_approach",
                "task_description": "Create red cube step by step",
                "subtasks": [
                    {
                        "task_type": "geometry",
                        "description": "Create basic cube mesh",
                        "dependencies": [],
                        "priority": 5,
                    },
                    {
                        "task_type": "material",
                        "description": "Apply red material to cube",
                        "dependencies": [],
                        "priority": 4,
                    },
                ],
                "is_terminal": True,
                "reasoning": "Sequential approach for simple cube creation",
            }
        ]
    }
    return json.dumps(response_content)


class TestMCTSNode:
    """Tests for MCTSNode class."""

    def test_node_initialization(self):
        """Test MCTS node can be initialized."""
        node = MCTSNode(task_description="Test task")

        assert node.task_description == "Test task"
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert node.is_terminal is False
        assert node.is_fully_expanded is False
        assert len(node.children) == 0
        assert node.parent is None
        assert node.id is not None  # UUID should be generated

    def test_average_reward_calculation(self, sample_mcts_node):
        """Test average reward (Q-value) calculation."""
        # Node with visits should calculate correctly
        expected_avg = sample_mcts_node.total_reward / sample_mcts_node.visits
        assert sample_mcts_node.average_reward == expected_avg

        # Node with no visits should return 0
        empty_node = MCTSNode()
        assert empty_node.average_reward == 0.0

    def test_ucb1_score_calculation(self):
        """Test UCB1 score calculation for node selection."""
        # Node with no visits should return infinity
        unvisited_node = MCTSNode()
        assert unvisited_node.ucb1_score == float("inf")

        # Node with no parent should return average reward
        root_node = MCTSNode(visits=5, total_reward=3.5)
        assert root_node.ucb1_score == root_node.average_reward

        # Node with parent should calculate UCB1 properly
        parent_node = MCTSNode(visits=10, total_reward=7.0)
        child_node = MCTSNode(visits=3, total_reward=2.1)
        parent_node.add_child(child_node)

        expected_exploration = MCTS_CONFIG["exploration_constant"] * math.sqrt(
            math.log(parent_node.visits) / child_node.visits
        )
        expected_ucb1 = child_node.average_reward + expected_exploration
        assert abs(child_node.ucb1_score - expected_ucb1) < 1e-10

    def test_node_hierarchy(self):
        """Test parent-child relationships."""
        parent = MCTSNode(task_description="Parent task")
        child1 = MCTSNode(task_description="Child 1")
        child2 = MCTSNode(task_description="Child 2")

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert child1.parent == parent
        assert child2.parent == parent
        assert not parent.is_leaf()
        assert child1.is_leaf()
        assert child2.is_leaf()

    def test_path_from_root(self):
        """Test path tracing from root to node."""
        root = MCTSNode(task_description="Root")
        child = MCTSNode(task_description="Child")
        grandchild = MCTSNode(task_description="Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        path = grandchild.get_path_from_root()
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child
        assert path[2] == grandchild


class TestMCTSState:
    """Tests for MCTSState class."""

    def test_state_initialization(self):
        """Test MCTS state initialization."""
        root = MCTSNode(task_description="Root task")
        state = MCTSState(root=root)

        assert state.root == root
        assert state.current_best is None
        assert state.iteration_count == 0
        assert state.total_nodes == 1
        assert state.max_depth_reached == 0

    def test_best_path_with_no_best(self):
        """Test getting best path when no best node is set."""
        root = MCTSNode(task_description="Root task")
        state = MCTSState(root=root)

        path = state.get_best_path()
        assert len(path) == 1
        assert path[0] == root

    def test_best_path_with_best_node(self):
        """Test getting best path when best node is set."""
        root = MCTSNode(task_description="Root task")
        child = MCTSNode(task_description="Child task")
        root.add_child(child)

        state = MCTSState(root=root, current_best=child)
        path = state.get_best_path()
        assert len(path) == 2
        assert path[0] == root
        assert path[1] == child


class TestMCTSTaskDecomposer:
    """Tests for MCTSTaskDecomposer agent."""

    def test_agent_initialization(self, mcts_config):
        """Test MCTS agent initialization."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            assert agent.agent_type == AgentType.MCTS_DECOMPOSER
            assert agent.name == "MCTS Task Decomposer"
            assert agent.max_iterations == mcts_config["mcts_max_iterations"]
            assert agent.exploration_constant == mcts_config["mcts_exploration_constant"]
            assert agent.max_depth == mcts_config["mcts_max_depth"]

    def test_agent_properties(self, mcts_config):
        """Test agent properties."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            assert agent.agent_type == AgentType.MCTS_DECOMPOSER
            assert agent.name == "MCTS Task Decomposer"

    @pytest.mark.asyncio
    async def test_validate_input_valid_state(self, mcts_config, sample_workflow_state):
        """Test input validation with valid workflow state."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            is_valid = await agent.validate_input(sample_workflow_state)
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_input_empty_prompt(self, mcts_config):
        """Test input validation with empty prompt."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            empty_state = WorkflowState(
                prompt="",
                original_prompt="",
                user_prompt="",
                user_feedback=None,
                documentation="",
                generated_code="",
                execution_result=None,
                refinement_request="",
                asset_metadata=None,
                error_message=None,
                previous_screenshot_path=None,
                critic_analysis=None,
                verification_result=None,
            )

            is_valid = await agent.validate_input(empty_state)
            assert is_valid is False

    def test_node_selection(self, mcts_config):
        """Test MCTS node selection phase."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create a tree structure
            root = MCTSNode(task_description="Root", visits=10, total_reward=5.0)
            child1 = MCTSNode(task_description="Child1", visits=3, total_reward=2.1)
            child2 = MCTSNode(task_description="Child2", visits=2, total_reward=1.8)

            root.add_child(child1)
            root.add_child(child2)

            # Selection should pick the child with higher UCB1 score
            selected = agent._select_node(root)

            # Should select one of the children (the one with higher UCB1)
            assert selected in [child1, child2]

    def test_backpropagation(self, mcts_config):
        """Test MCTS backpropagation phase."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create a path: root -> child -> grandchild
            root = MCTSNode(task_description="Root", visits=5, total_reward=2.5)
            child = MCTSNode(task_description="Child", visits=3, total_reward=1.5)
            grandchild = MCTSNode(task_description="Grandchild", visits=1, total_reward=0.5)

            root.add_child(child)
            child.add_child(grandchild)

            # Backpropagate a reward of 0.8
            reward = 0.8
            agent._backpropagate(grandchild, reward)

            # All nodes in the path should have increased visits and reward
            assert grandchild.visits == 2
            assert abs(grandchild.total_reward - (0.5 + reward)) < 1e-10

            assert child.visits == 4
            assert abs(child.total_reward - (1.5 + reward)) < 1e-10

            assert root.visits == 6
            assert abs(root.total_reward - (2.5 + reward)) < 1e-10

    @pytest.mark.asyncio
    async def test_decomposition_response_parsing(self, mcts_config, mock_openai_response):
        """Test parsing of LLM decomposition response."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            node = MCTSNode(task_description="Test task")
            strategies = agent._parse_decomposition_response(mock_openai_response, node)

            assert len(strategies) == 1
            strategy = strategies[0]

            assert strategy["strategy"] == "sequential_approach"
            assert strategy["task_description"] == "Create red cube step by step"
            assert len(strategy["subtasks"]) == 2
            assert strategy["is_terminal"] is True
            assert strategy["reasoning"] == "Sequential approach for simple cube creation"

            # Check subtask conversion
            subtask1 = strategy["subtasks"][0]
            assert subtask1.type == TaskType.GEOMETRY
            assert subtask1.description == "Create basic cube mesh"
            assert subtask1.priority == 5

    @pytest.mark.asyncio
    async def test_decomposition_response_parsing_invalid_json(self, mcts_config):
        """Test parsing of invalid JSON response."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            node = MCTSNode(task_description="Test task")
            strategies = agent._parse_decomposition_response("invalid json", node)

            # Should return fallback strategy
            assert len(strategies) == 1
            strategy = strategies[0]
            assert strategy["strategy"] == "fallback"
            assert strategy["task_description"] == "Test task"
            assert len(strategy["subtasks"]) == 0
            assert strategy["is_terminal"] is True

    @pytest.mark.asyncio
    async def test_feasibility_evaluation(self, mcts_config, sample_mcts_node, sample_workflow_state):
        """Test feasibility evaluation of task decomposition."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Test with subtasks
            feasibility = await agent._evaluate_feasibility(sample_mcts_node, sample_workflow_state)
            assert 0.0 <= feasibility <= 1.0

            # Test with empty node
            empty_node = MCTSNode()
            feasibility_empty = await agent._evaluate_feasibility(empty_node, sample_workflow_state)
            assert feasibility_empty == 0.5  # Neutral score for empty decompositions

    @pytest.mark.asyncio
    async def test_complexity_evaluation(self, mcts_config, sample_mcts_node, sample_workflow_state):
        """Test complexity evaluation of task decomposition."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Test with subtasks
            complexity = await agent._evaluate_complexity(sample_mcts_node, sample_workflow_state)
            assert 0.0 <= complexity <= 1.0

            # Test with empty node
            empty_node = MCTSNode()
            complexity_empty = await agent._evaluate_complexity(empty_node, sample_workflow_state)
            assert complexity_empty == 0.5

    @pytest.mark.asyncio
    async def test_completeness_evaluation(self, mcts_config, sample_mcts_node, sample_workflow_state):
        """Test completeness evaluation of task decomposition."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Test with subtasks
            completeness = await agent._evaluate_completeness(sample_mcts_node, sample_workflow_state)
            assert 0.0 <= completeness <= 1.0

            # Test with empty node
            empty_node = MCTSNode()
            completeness_empty = await agent._evaluate_completeness(empty_node, sample_workflow_state)
            assert completeness_empty == 0.2  # Low score for empty decomposition

    @pytest.mark.asyncio
    async def test_simulation_phase(self, mcts_config, sample_mcts_node, sample_workflow_state):
        """Test MCTS simulation phase."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            reward = await agent._simulate(sample_mcts_node, sample_workflow_state)

            # Reward should be between 0 and 1
            assert 0.0 <= reward <= 1.0

            # Node should have evaluation scores set
            assert hasattr(sample_mcts_node, "feasibility_score")
            assert hasattr(sample_mcts_node, "complexity_score")
            assert hasattr(sample_mcts_node, "completeness_score")
            assert hasattr(sample_mcts_node, "quality_score")

    def test_find_best_terminal_node(self, mcts_config):
        """Test finding the best terminal node in tree."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create tree with terminal nodes
            root = MCTSNode(task_description="Root")
            terminal1 = MCTSNode(
                task_description="Terminal1",
                is_terminal=True,
                visits=5,
                total_reward=3.0,
            )
            terminal2 = MCTSNode(
                task_description="Terminal2",
                is_terminal=True,
                visits=3,
                total_reward=2.7,  # Higher average reward (0.9 vs 0.6)
            )
            non_terminal = MCTSNode(
                task_description="NonTerminal",
                is_terminal=False,
                visits=10,
                total_reward=9.0,
            )

            root.add_child(terminal1)
            root.add_child(terminal2)
            root.add_child(non_terminal)

            best = agent._find_best_terminal_node(root)

            # Should find terminal2 (highest average reward among terminals)
            assert best == terminal2

    def test_extract_final_decomposition(self, mcts_config):
        """Test extracting final task decomposition from MCTS results."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create MCTS state with best path
            root = MCTSNode(task_description="Root")
            best_node = MCTSNode(
                task_description="Best",
                subtasks=[
                    SubTask(
                        id="t1",
                        type=TaskType.GEOMETRY,
                        description="Create geometry",
                        priority=5,
                        dependencies=[],
                    ),
                    SubTask(
                        id="t2",
                        type=TaskType.MATERIAL,
                        description="Apply material",
                        priority=4,
                        dependencies=["t1"],
                    ),
                ],
            )
            root.add_child(best_node)

            mcts_state = MCTSState(root=root, current_best=best_node)

            final_subtasks = agent._extract_final_decomposition(mcts_state)

            # Should extract subtasks and sort by priority
            assert len(final_subtasks) == 2
            assert final_subtasks[0].priority >= final_subtasks[1].priority  # Sorted by priority descending
            assert final_subtasks[0].type == TaskType.GEOMETRY
            assert final_subtasks[1].type == TaskType.MATERIAL

    def test_extract_final_decomposition_no_best(self, mcts_config):
        """Test extracting final decomposition when no best node exists."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            root = MCTSNode(task_description="Root")
            mcts_state = MCTSState(root=root)  # No current_best set

            final_subtasks = agent._extract_final_decomposition(mcts_state)

            # Should return empty list when no best solution
            assert len(final_subtasks) == 0

    @pytest.mark.asyncio
    async def test_process_user_termination(self, mcts_config):
        """Test process method with user termination."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Create workflow state with termination signal
            termination_state = WorkflowState(
                prompt="Create a cube",
                user_feedback="terminate",  # Termination signal
                original_prompt="Create a cube",
                user_prompt="Create a cube",
                documentation="",
                generated_code="",
                execution_result=None,
                refinement_request="",
                asset_metadata=None,
                error_message=None,
                previous_screenshot_path=None,
                critic_analysis=None,
                verification_result=None,
            )

            response = await agent.process(termination_state)

            assert response.success is False
            assert "terminated by user" in response.message.lower()
            assert response.data == []

    @pytest.mark.asyncio
    async def test_process_successful_decomposition(self, mcts_config, sample_workflow_state, mock_openai_response):
        """Test successful MCTS task decomposition process."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Mock the LLM request to return valid decomposition
            agent.make_openai_request = AsyncMock(return_value=mock_openai_response)

            response = await agent.process(sample_workflow_state)

            assert response.success is True
            assert "subtasks using MCTS" in response.message
            assert len(response.data) > 0  # Should have generated subtasks
            assert "mcts_iterations" in response.metadata
            assert "total_nodes" in response.metadata
            assert "exploration_strategy" in response.metadata

    @pytest.mark.asyncio
    async def test_early_stopping_mechanism(self, mcts_config, sample_workflow_state, mock_openai_response):
        """Test MCTS early stopping when high-quality solution is found."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            # Use very high iteration count to test early stopping
            config_with_high_iterations = mcts_config.copy()
            config_with_high_iterations["mcts_max_iterations"] = 100

            agent = MCTSTaskDecomposer(config_with_high_iterations)
            agent.make_openai_request = AsyncMock(return_value=mock_openai_response)

            # Mock the simulation to return high-quality scores quickly
            original_simulate = agent._simulate

            async def mock_simulate(node, workflow_state):
                _ = await original_simulate(node, workflow_state)
                # Force high reward to trigger early stopping
                return HIGH_QUALITY_THRESHOLD + 0.1

            agent._simulate = mock_simulate

            response = await agent.process(sample_workflow_state)

            assert response.success is True
            # Should have stopped early (less than max iterations)
            assert response.metadata["mcts_iterations"] < config_with_high_iterations["mcts_max_iterations"]

    def test_system_prompt_generation(self, mcts_config):
        """Test generation of system prompt for decomposition."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            system_prompt = agent._get_decomposition_system_prompt()

            assert "task decomposition specialist" in system_prompt.lower()
            assert "3d modeling" in system_prompt.lower()
            assert "blender" in system_prompt.lower()
            assert "geometry" in system_prompt.lower()
            assert "material" in system_prompt.lower()

    def test_decomposition_prompt_creation(self, mcts_config, sample_workflow_state):
        """Test creation of decomposition prompt for LLM."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            node = MCTSNode(task_description="Create a red cube")
            prompt = agent._create_decomposition_prompt(node, sample_workflow_state)

            assert sample_workflow_state.prompt in prompt
            assert node.task_description in prompt
            assert "JSON" in prompt
            assert "strategies" in prompt.lower()
            assert "subtasks" in prompt.lower()

    @pytest.mark.asyncio
    async def test_error_handling_in_process(self, mcts_config, sample_workflow_state):
        """Test error handling during MCTS process."""
        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(mcts_config)

            # Mock make_openai_request to raise an exception
            agent.make_openai_request = AsyncMock(side_effect=Exception("API Error"))

            response = await agent.process(sample_workflow_state)

            assert response.success is False
            assert "failed" in response.message.lower()
            assert response.data == []

    def test_config_validation(self):
        """Test MCTS configuration validation and defaults."""
        minimal_config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1500}

        with (
            patch("src.agents.base.get_settings"),
            patch("src.agents.base.LMStudioClient"),
            patch("src.agents.base.OpenAIClient"),
        ):
            agent = MCTSTaskDecomposer(minimal_config)

            # Should use defaults from MCTS_CONFIG for missing values
            assert agent.max_iterations == MCTS_CONFIG["max_iterations"]
            assert agent.exploration_constant == MCTS_CONFIG["exploration_constant"]
            assert agent.max_depth == MCTS_CONFIG["max_depth"]

    def test_constants_defined(self):
        """Test that important constants are properly defined."""
        assert HIGH_QUALITY_THRESHOLD > 0
        assert MIN_VISITS_FOR_EARLY_STOP > 0
        assert isinstance(MCTS_CONFIG, dict)
        assert "max_iterations" in MCTS_CONFIG
        assert "exploration_constant" in MCTS_CONFIG
