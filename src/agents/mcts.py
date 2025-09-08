"""MCTS (Monte Carlo Tree Search) agent for hierarchical task decomposition.

This agent implements a LATS-like approach using Monte Carlo Tree Search to:
1. Break down complex user requests into smaller, manageable tasks
2. Explore multiple decomposition strategies
3. Evaluate and select optimal task breakdowns
4. Provide tree-based reasoning with backpropagation
"""

import json
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import structlog

from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import check_user_termination, log_agent_exception

logger = structlog.get_logger(__name__)

# MCTS Configuration
MCTS_CONFIG = {
    "max_iterations": 50,
    "exploration_constant": 1.414,  # UCB1 exploration parameter (sqrt(2))
    "max_depth": 4,
    "simulation_depth": 2,
    "min_visits_for_expansion": 3,
    "temperature": 0.7,
    "max_children": 5,
}

# Quality thresholds for early stopping
HIGH_QUALITY_THRESHOLD = 0.8
MIN_VISITS_FOR_EARLY_STOP = 5


@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree for task decomposition."""

    # Node identification
    id: str = field(default_factory=lambda: str(uuid4()))
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)

    # Task information
    task_description: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    completion_strategy: str = ""

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    is_fully_expanded: bool = False

    # Evaluation scores
    feasibility_score: float = 0.0
    complexity_score: float = 0.0
    completeness_score: float = 0.0
    quality_score: float = 0.0

    @property
    def average_reward(self) -> float:
        """Calculate the average reward (Q-value) for this node."""
        return self.total_reward / max(self.visits, 1)

    @property
    def ucb1_score(self) -> float:
        """Calculate UCB1 score for node selection."""
        if self.visits == 0:
            return float("inf")

        if not self.parent or self.parent.visits == 0:
            return self.average_reward

        exploration_term = MCTS_CONFIG["exploration_constant"] * math.sqrt(math.log(self.parent.visits) / self.visits)
        return self.average_reward + exploration_term

    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node to this node."""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    def get_path_from_root(self) -> list["MCTSNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))


@dataclass
class MCTSState:
    """State information for MCTS tree search."""

    root: MCTSNode
    current_best: MCTSNode | None = None
    iteration_count: int = 0
    total_nodes: int = 1
    max_depth_reached: int = 0

    def get_best_path(self) -> list[MCTSNode]:
        """Get the path representing the best task decomposition."""
        if not self.current_best:
            return [self.root]
        return self.current_best.get_path_from_root()


class MCTSTaskDecomposer(EnhancedBaseAgent):
    """MCTS-based agent for hierarchical task decomposition using tree search."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize MCTS task decomposer."""
        super().__init__(config)
        self.error_handler = AgentErrorHandler()

        # MCTS-specific configuration
        self.max_iterations = config.get("mcts_max_iterations", MCTS_CONFIG["max_iterations"])
        self.exploration_constant = config.get("mcts_exploration_constant", MCTS_CONFIG["exploration_constant"])
        self.max_depth = config.get("mcts_max_depth", MCTS_CONFIG["max_depth"])
        self.simulation_depth = config.get("mcts_simulation_depth", MCTS_CONFIG["simulation_depth"])

        logger.info(
            "MCTS Task Decomposer initialized",
            max_iterations=self.max_iterations,
            exploration_constant=self.exploration_constant,
            max_depth=self.max_depth,
        )

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.MCTS_DECOMPOSER

    @property
    def name(self) -> str:
        """Return human-readable agent name."""
        return "MCTS Task Decomposer"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process the workflow state using MCTS task decomposition."""
        try:
            if check_user_termination(state):
                return AgentResponse(success=False, message="Task decomposition terminated by user", data=[])

            logger.info("Starting MCTS task decomposition", prompt=state.prompt)

            # Initialize MCTS tree with root node
            root_node = MCTSNode(task_description=state.prompt, subtasks=[], completion_strategy="root")

            mcts_state = MCTSState(root=root_node)

            # Run MCTS algorithm
            final_decomposition = await self._run_mcts(mcts_state, state)

            if not final_decomposition:
                return AgentResponse(success=False, message="Failed to generate task decomposition", data=[])

            logger.info(
                "MCTS task decomposition completed",
                total_subtasks=len(final_decomposition),
                iterations=mcts_state.iteration_count,
                total_nodes=mcts_state.total_nodes,
            )

            return AgentResponse(
                success=True,
                message=f"Generated {len(final_decomposition)} subtasks using MCTS",
                data=final_decomposition,
                metadata={
                    "mcts_iterations": mcts_state.iteration_count,
                    "total_nodes": mcts_state.total_nodes,
                    "max_depth_reached": mcts_state.max_depth_reached,
                    "exploration_strategy": "UCB1",
                },
            )

        except Exception as e:
            log_agent_exception(self.logger, "MCTS task decomposition", e)
            return AgentResponse(success=False, message=f"MCTS task decomposition failed: {str(e)}", data=[])

    async def _run_mcts(self, mcts_state: MCTSState, workflow_state: WorkflowState) -> list[SubTask]:
        """Run the main MCTS algorithm loop."""
        for iteration in range(self.max_iterations):
            mcts_state.iteration_count = iteration + 1

            # MCTS phases: Selection -> Expansion -> Simulation -> Backpropagation
            selected_node = self._select_node(mcts_state.root)
            expanded_node = await self._expand_node(selected_node, workflow_state)
            simulation_reward = await self._simulate(expanded_node, workflow_state)
            self._backpropagate(expanded_node, simulation_reward)

            # Update best path
            current_best = self._find_best_terminal_node(mcts_state.root)
            if current_best and (
                not mcts_state.current_best or current_best.average_reward > mcts_state.current_best.average_reward
            ):
                mcts_state.current_best = current_best

            # Early stopping if we have a high-quality solution
            if (
                mcts_state.current_best
                and mcts_state.current_best.average_reward > HIGH_QUALITY_THRESHOLD
                and mcts_state.current_best.visits > MIN_VISITS_FOR_EARLY_STOP
            ):
                logger.info(
                    "Early stopping: found high-quality solution",
                    reward=mcts_state.current_best.average_reward,
                    iteration=iteration,
                )
                break

        # Extract final task decomposition
        return self._extract_final_decomposition(mcts_state)

    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1 to find leaf node."""
        current = root
        path_depth = 0

        while not current.is_leaf() and path_depth < self.max_depth:
            # Select child with highest UCB1 score
            current = max(current.children, key=lambda child: child.ucb1_score)
            path_depth += 1

        return current

    async def _expand_node(self, node: MCTSNode, workflow_state: WorkflowState) -> MCTSNode:
        """Expansion phase: add new child nodes if conditions are met."""
        # Only expand if node has been visited enough times and isn't fully expanded
        if node.visits < MCTS_CONFIG["min_visits_for_expansion"] or node.is_fully_expanded or node.is_terminal:
            return node

        # Generate possible decomposition strategies for this node
        strategies = await self._generate_decomposition_strategies(node, workflow_state)

        # Add children for each strategy
        for strategy in strategies:
            if len(node.children) >= MCTS_CONFIG["max_children"]:
                break

            child = MCTSNode(
                task_description=strategy["task_description"],
                subtasks=strategy["subtasks"],
                completion_strategy=strategy["strategy"],
                is_terminal=strategy.get("is_terminal", False),
            )
            node.add_child(child)

        # Mark as fully expanded if we couldn't generate more strategies
        if len(strategies) == 0 or len(node.children) >= MCTS_CONFIG["max_children"]:
            node.is_fully_expanded = True

        # Return a random new child for simulation (using secrets for better randomness)
        if node.children:
            return secrets.choice(node.children)
        return node

    async def _simulate(self, node: MCTSNode, workflow_state: WorkflowState) -> float:
        """Simulation phase: evaluate the quality of this decomposition path."""
        try:
            # Evaluate different aspects of the task decomposition
            feasibility = await self._evaluate_feasibility(node, workflow_state)
            complexity = await self._evaluate_complexity(node, workflow_state)
            completeness = await self._evaluate_completeness(node, workflow_state)

            # Store individual scores
            node.feasibility_score = feasibility
            node.complexity_score = complexity
            node.completeness_score = completeness

            # Calculate overall quality score (weighted combination)
            quality_score = (
                0.4 * feasibility  # 40% weight on feasibility
                + 0.3 * (1.0 - complexity)  # 30% weight on simplicity (inverse complexity)
                + 0.3 * completeness  # 30% weight on completeness
            )

            node.quality_score = quality_score

            # Add small random noise to prevent deterministic behavior
            noise = (secrets.randbelow(1000) / 10000.0) - 0.05  # Range: -0.05 to 0.05
            return max(0.0, min(1.0, quality_score + noise))

        except Exception as e:
            log_agent_exception(self.logger, "MCTS simulation", e)
            return 0.1  # Return low score for failed simulations

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagation phase: update statistics up the tree."""
        current = node
        while current:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    async def _generate_decomposition_strategies(
        self, node: MCTSNode, workflow_state: WorkflowState
    ) -> list[dict[str, Any]]:
        """Generate possible task decomposition strategies for a node."""
        prompt = self._create_decomposition_prompt(node, workflow_state)

        try:
            response = await self.make_openai_request(
                messages=[
                    {"role": "system", "content": self._get_decomposition_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=MCTS_CONFIG["temperature"],
                max_tokens=1500,
            )

            # Parse the response to extract strategies
            return self._parse_decomposition_response(response, node)

        except Exception as e:
            log_agent_exception(self.logger, "strategy generation", e)
            return []

    def _create_decomposition_prompt(self, node: MCTSNode, workflow_state: WorkflowState) -> str:
        """Create a prompt for generating task decomposition strategies."""
        context = f"""
Original Request: {workflow_state.prompt}
Current Task: {node.task_description}
Current Depth: {len(node.get_path_from_root()) - 1}

Generate 2-3 different approaches to break down this task into smaller, executable subtasks.
Each approach should represent a different decomposition strategy.

Consider:
1. Sequential vs parallel execution
2. Different levels of granularity
3. Alternative implementation approaches
4. Dependencies between subtasks

Format your response as JSON with this structure:
{{
    "strategies": [
        {{
            "strategy": "strategy_name",
            "task_description": "refined_task_description",
            "subtasks": [
                {{
                    "task_type": "geometry|material|lighting|scene|export",
                    "description": "subtask_description",
                    "dependencies": [],
                    "priority": 1-5,
                    "estimated_complexity": 1-5
                }}
            ],
            "is_terminal": false,
            "reasoning": "why_this_approach"
        }}
    ]
}}
"""
        return context

    def _get_decomposition_system_prompt(self) -> str:
        """Get the system prompt for task decomposition."""
        return """You are an expert task decomposition specialist for 3D modeling and Blender workflows.

Your role is to break down complex 3D modeling tasks into smaller, manageable subtasks using different strategic
approaches.

Key principles:
1. Each subtask should be atomic and executable
2. Consider dependencies between tasks
3. Balance granularity - not too broad, not too narrow
4. Think about different execution strategies
5. Ensure completeness - all aspects of the original task should be covered

Task types:
- geometry: Creating 3D shapes, meshes, modifications
- material: Textures, shaders, material properties
- lighting: Lights, shadows, illumination setup
- scene: Camera, composition, environment
- export: File formats, optimization, output

Provide multiple strategic approaches to give the MCTS algorithm different paths to explore."""

    def _parse_decomposition_response(self, response: str, node: MCTSNode) -> list[dict[str, Any]]:
        """Parse LLM response to extract decomposition strategies."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            strategies = []

            for strategy_data in data.get("strategies", []):
                # Convert to our internal format
                subtasks = []
                for st in strategy_data.get("subtasks", []):
                    subtask = SubTask(
                        task_type=TaskType(st.get("task_type", "geometry")),
                        description=st.get("description", ""),
                        dependencies=st.get("dependencies", []),
                        priority=st.get("priority", 3),
                        estimated_complexity=st.get("estimated_complexity", 3),
                    )
                    subtasks.append(subtask)

                strategy = {
                    "strategy": strategy_data.get("strategy", f"strategy_{len(strategies)}"),
                    "task_description": strategy_data.get("task_description", node.task_description),
                    "subtasks": subtasks,
                    "is_terminal": strategy_data.get("is_terminal", len(subtasks) == 0),
                    "reasoning": strategy_data.get("reasoning", ""),
                }
                strategies.append(strategy)

        except (json.JSONDecodeError, KeyError) as e:
            log_agent_exception(self.logger, "response parsing", e)
            # Fallback: create a simple strategy
            strategies = [
                {
                    "strategy": "fallback",
                    "task_description": node.task_description,
                    "subtasks": [],
                    "is_terminal": True,
                    "reasoning": "Failed to parse LLM response",
                }
            ]

        return strategies

    async def _evaluate_feasibility(self, node: MCTSNode, _workflow_state: WorkflowState) -> float:
        """Evaluate how feasible this task decomposition is."""
        try:
            if not node.subtasks:
                return 0.5  # Neutral score for empty decompositions

            # Factors affecting feasibility:
            # 1. Reasonable number of subtasks (not too many, not too few)
            num_subtasks = len(node.subtasks)
            size_score = 1.0 - abs(num_subtasks - 5) / 10.0  # Optimal around 5 subtasks
            size_score = max(0.0, min(1.0, size_score))

            # 2. Balanced complexity distribution
            complexities = [st.estimated_complexity for st in node.subtasks]
            avg_complexity = sum(complexities) / len(complexities)
            complexity_score = 1.0 - abs(avg_complexity - 3) / 5.0  # Optimal around 3
            complexity_score = max(0.0, min(1.0, complexity_score))

            # 3. Clear task descriptions (length as proxy for clarity)
            description_lengths = [len(st.description.split()) for st in node.subtasks]
            avg_desc_length = sum(description_lengths) / len(description_lengths)
            clarity_score = min(1.0, avg_desc_length / 10.0)  # Good descriptions ~10 words

            return (size_score + complexity_score + clarity_score) / 3.0

        except Exception as e:
            log_agent_exception(self.logger, "feasibility evaluation", e)
            return 0.3

    async def _evaluate_complexity(self, node: MCTSNode, _workflow_state: WorkflowState) -> float:
        """Evaluate the complexity of this task decomposition."""
        try:
            if not node.subtasks:
                return 0.5

            # Complexity factors:
            # 1. Number of dependencies
            total_deps = sum(len(st.dependencies) for st in node.subtasks)
            dep_complexity = min(1.0, total_deps / (len(node.subtasks) * 2))

            # 2. Task complexity ratings
            complexity_ratings = [st.estimated_complexity for st in node.subtasks]
            avg_complexity = sum(complexity_ratings) / len(complexity_ratings)
            task_complexity = avg_complexity / 5.0  # Normalize to 0-1

            # 3. Number of different task types (more types = more complex)
            task_types = {st.task_type for st in node.subtasks}
            type_complexity = len(task_types) / 5.0  # Max 5 types

            return (dep_complexity + task_complexity + type_complexity) / 3.0

        except Exception as e:
            log_agent_exception(self.logger, "complexity evaluation", e)
            return 0.5

    async def _evaluate_completeness(self, node: MCTSNode, workflow_state: WorkflowState) -> float:
        """Evaluate how complete this task decomposition is."""
        try:
            if not node.subtasks:
                return 0.2  # Empty decomposition is incomplete

            # Completeness factors:
            # 1. Coverage of different task types
            task_types = {st.task_type for st in node.subtasks}
            expected_types = {TaskType.GEOMETRY, TaskType.MATERIAL}  # Minimum expected

            # Check for key task types based on the original prompt
            prompt_lower = workflow_state.prompt.lower()
            if any(word in prompt_lower for word in ["light", "illuminate", "shadow"]):
                expected_types.add(TaskType.LIGHTING)
            if any(word in prompt_lower for word in ["scene", "environment", "background"]):
                expected_types.add(TaskType.SCENE)

            coverage_score = len(task_types & expected_types) / len(expected_types)

            # 2. Task priority distribution (should have varied priorities)
            priorities = [st.priority for st in node.subtasks]
            priority_variance = len(set(priorities)) / 5.0  # More variety = better

            # 3. Logical flow (higher priority tasks should come first)
            is_logical_flow = all(
                i == 0 or node.subtasks[i - 1].priority >= node.subtasks[i].priority
                for i in range(1, len(node.subtasks))
            )
            flow_score = 1.0 if is_logical_flow else 0.5

            return (coverage_score + priority_variance + flow_score) / 3.0

        except Exception as e:
            log_agent_exception(self.logger, "completeness evaluation", e)
            return 0.4

    def _find_best_terminal_node(self, root: MCTSNode) -> MCTSNode | None:
        """Find the best terminal node in the tree."""
        best_node = None
        best_score = -1.0

        def _traverse(node: MCTSNode) -> None:
            nonlocal best_node, best_score

            if node.is_terminal and node.visits > 0 and node.average_reward > best_score:
                best_score = node.average_reward
                best_node = node

            for child in node.children:
                _traverse(child)

        _traverse(root)
        return best_node

    def _extract_final_decomposition(self, mcts_state: MCTSState) -> list[SubTask]:
        """Extract the final task decomposition from MCTS results."""
        if not mcts_state.current_best:
            logger.warning("No best solution found, using root node")
            return []

        best_path = mcts_state.get_best_path()

        # Combine subtasks from all nodes in the best path
        final_subtasks = []
        for node in best_path:
            if node.subtasks:
                final_subtasks.extend(node.subtasks)

        # Remove duplicates and sort by priority
        seen_descriptions = set()
        unique_subtasks = []

        for subtask in final_subtasks:
            if subtask.description not in seen_descriptions:
                seen_descriptions.add(subtask.description)
                unique_subtasks.append(subtask)

        # Sort by priority (highest first)
        unique_subtasks.sort(key=lambda x: x.priority, reverse=True)

        logger.info(
            "Final MCTS decomposition extracted",
            path_length=len(best_path),
            total_subtasks=len(unique_subtasks),
            best_reward=mcts_state.current_best.average_reward,
        )

        return unique_subtasks
