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

# Task complexity hierarchy (1 = simplest, 5 = most complex)
TASK_COMPLEXITY_LEVELS = {
    TaskType.GEOMETRY: 1,      # Basic shapes, primitives
    TaskType.SCENE_SETUP: 2,   # Camera, organization
    TaskType.MATERIAL: 3,      # Textures, shaders
    TaskType.LIGHTING: 4,      # Lighting setup
    TaskType.ANIMATION: 5,     # Motion, dynamics
}


def get_task_complexity_level(task_type: TaskType) -> int:
    """Get the complexity level for a task type."""
    return TASK_COMPLEXITY_LEVELS.get(task_type, 3)


def sort_tasks_by_complexity(subtasks: list[SubTask]) -> list[SubTask]:
    """Sort tasks by complexity level (simple to complex)."""
    return sorted(subtasks, key=lambda task: (
        get_task_complexity_level(task.type),  # Primary: complexity level
        -task.priority,                        # Secondary: priority (inverted for higher first)
        task.id                                # Tertiary: stable sort by id
    ))


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
        current: MCTSNode | None = self
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
        import time
        start_time = time.time()

        try:
            if check_user_termination(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    message="Task decomposition terminated by user",
                    data=[],
                    execution_time=time.time() - start_time
                )

            logger.info("Starting MCTS task decomposition", prompt=state.prompt)

            # Initialize MCTS tree with root node
            root_node = MCTSNode(task_description=state.prompt, subtasks=[], completion_strategy="root")

            mcts_state = MCTSState(root=root_node)

            # Run MCTS algorithm
            final_decomposition = await self._run_mcts(mcts_state, state)

            if not final_decomposition:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    message="Failed to generate task decomposition",
                    data=[],
                    execution_time=time.time() - start_time
                )

            logger.info(
                "MCTS task decomposition completed",
                total_subtasks=len(final_decomposition),
                iterations=mcts_state.iteration_count,
                total_nodes=mcts_state.total_nodes,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                message=f"Generated {len(final_decomposition)} subtasks using MCTS",
                data=final_decomposition,
                execution_time=time.time() - start_time,
                metadata={
                    "mcts_iterations": mcts_state.iteration_count,
                    "total_nodes": mcts_state.total_nodes,
                    "max_depth_reached": mcts_state.max_depth_reached,
                    "exploration_strategy": "UCB1",
                },
            )

        except Exception as e:
            log_agent_exception(self.logger, "MCTS task decomposition", e)
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                message=f"MCTS task decomposition failed: {str(e)}",
                data=[],
                execution_time=time.time() - start_time
            )

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
            # Note: complexity is inverted because lower complexity is better in our MCTS formulation
            quality_score = (
                0.35 * feasibility  # 35% weight on feasibility
                + 0.35 * (1.0 - complexity)  # 35% weight on simplicity (inverse complexity, includes hierarchy bonus)
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
        current: MCTSNode | None = node
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
                max_tokens=15000,  # Higher for GPT-5
            )

            # Parse the response to extract strategies
            return self._parse_decomposition_response(response, node)

        except Exception as e:
            log_agent_exception(self.logger, "strategy generation", e)
            return []

    def _create_decomposition_prompt(self, node: MCTSNode, workflow_state: WorkflowState) -> str:
        """Create a prompt for generating hierarchical task decomposition strategies."""
        depth = len(node.get_path_from_root()) - 1

        # Determine recommended complexity level based on depth
        recommended_levels = {
            0: "Focus primarily on GEOMETRY (level 1) - basic shapes and structures",
            1: "Balance GEOMETRY (level 1) and SCENE_SETUP (level 2) - organize the basic scene",
            2: "Include MATERIAL (level 3) tasks - add surface properties to existing geometry",
            3: "Add LIGHTING (level 4) and ANIMATION (level 5) - final details and dynamics"
        }
        depth_guidance = recommended_levels.get(depth, "Focus on appropriate complexity level for current context")

        context = f"""
Original Request: {workflow_state.prompt}
Current Task: {node.task_description}
Current Depth: {depth}

HIERARCHICAL GUIDANCE FOR DEPTH {depth}:
{depth_guidance}

Generate 2-3 different approaches to break down this task into smaller, executable subtasks.
Each approach should represent a different hierarchical decomposition strategy.

CRITICAL HIERARCHICAL PRINCIPLES:
1. HIERARCHY FIRST: Start with lowest complexity level tasks (geometry, basic shapes)
2. PROGRESSIVE BUILDING: Each level builds upon the previous with clear dependencies
3. EXPLICIT DEPENDENCIES: Complex tasks must explicitly depend on simpler foundation tasks
4. STEP-BY-STEP COMPOSITION: User sees scene building naturally from basic to detailed
5. LOGICAL FLOW: Geometry(1) → Scene Setup(2) → Materials(3) → Lighting(4) → Animation(5)

Format your response as JSON with this structure:
{{
    "strategies": [
        {{
            "strategy": "strategy_name",
            "task_description": "refined_task_description",
            "subtasks": [
                {{
                    "task_type": "geometry|scene_setup|material|lighting|animation",
                    "description": "subtask_description",
                    "dependencies": ["task_ids_this_depends_on"],
                    "priority": 1-5,
                    "complexity_level": "1=geometry, 2=scene_setup, 3=material, 4=lighting, 5=animation"
                }}
            ],
            "is_terminal": false,
            "reasoning": "why_this_hierarchical_approach_works"
        }}
    ]
}}

EXAMPLE HIERARCHICAL BREAKDOWN:
For "Create a wooden table with realistic lighting":
1. GEOMETRY tasks: Create table legs, create table top, assemble table structure
2. SCENE_SETUP tasks: Position table in scene, set camera angle
3. MATERIAL tasks: Apply wood texture to table, configure surface properties
4. LIGHTING tasks: Add key light, add ambient lighting, configure shadows
"""
        return context

    def _get_decomposition_system_prompt(self) -> str:
        """Get the system prompt for task decomposition."""
        return """# ROLE & SPECIALIZATION
You are an expert MCTS (Monte Carlo Tree Search) task decomposition specialist with deep expertise in:
- **3D Modeling Theory**: Topology, manifold surfaces, procedural generation
- **Blender Python API**: bpy, bmesh, mathutils, node systems
- **Strategic Planning**: MCTS algorithms, search space optimization
- **Workflow Architecture**: Task dependencies, parallel execution, resource management

# MCTS DECOMPOSITION METHODOLOGY
## Core Requirement: Generate 2-5 DISTINCT Strategic Approaches
Each strategy must offer a fundamentally different path to achieving the same goal, "
        "enabling the MCTS algorithm to explore diverse solution spaces.

## Mandatory Hierarchical Complexity Levels:
### LEVEL 1: GEOMETRY (Priority 1) - FOUNDATION LAYER
- **Basic Shapes**: Cubes, spheres, cylinders, planes
- **Primitive Operations**: Scale, rotate, translate, extrude
- **Core Structure**: Main components, structural framework
- **Topology Setup**: Edge loops, face organization

### LEVEL 2: SCENE_SETUP (Priority 2) - ORGANIZATION LAYER
- **Camera Positioning**: Framing, composition, angle selection
- **Scene Hierarchy**: Object organization, naming conventions
- **Basic Layout**: Spatial relationships, positioning
- **Reference Setup**: Coordinate systems, scale references

### LEVEL 3: MATERIAL (Priority 3) - SURFACE LAYER
- **Material Creation**: Principled BSDF node setups
- **Texture Application**: UV mapping, image textures
- **Surface Properties**: Roughness, metallic, normal maps
- **Shader Networks**: Node connections, material logic

### LEVEL 4: LIGHTING (Priority 4) - ILLUMINATION LAYER
- **Primary Lighting**: Key lights, sun, HDRI environments
- **Secondary Lighting**: Fill lights, rim lighting, bounce
- **Shadow Configuration**: Contact shadows, ray tracing
- **Atmospheric Effects**: Volumetrics, fog, ambient occlusion

### LEVEL 5: ANIMATION (Priority 5) - DYNAMICS LAYER
- **Keyframe Animation**: Transform animations, property changes
- **Physics Simulation**: Rigid body, soft body, fluid dynamics
- **Procedural Animation**: Drivers, constraints, expressions
- **Advanced Effects**: Particles, force fields, simulations

# CRITICAL DECOMPOSITION RULES
## Dependency Architecture:
- **Sequential Dependencies**: Level N depends on Level N-1 completion
- **Intra-level Dependencies**: Tasks within same level may depend on each other
- **Parallel Opportunities**: Independent components can execute simultaneously
- **Resource Dependencies**: Materials depend on existing geometry objects

## Strategic Variation Requirements:
### Strategy Type 1: LINEAR PROGRESSION
- Execute levels sequentially, complete each before next
- Optimal for complex, interdependent assets
- Lower risk, predictable outcomes

### Strategy Type 2: PARALLEL DEVELOPMENT
- Execute independent components simultaneously
- Merge components at integration points
- Faster execution, higher coordination complexity

### Strategy Type 3: ITERATIVE REFINEMENT
- Create basic versions across all levels
- Iteratively refine each level
- Balanced progress, continuous feedback

### Strategy Type 4: FEATURE-DRIVEN
- Focus on one major feature/component at a time
- Complete all levels for each feature before next
- Modular development, easier testing

### Strategy Type 5: RISK-MITIGATION
- Start with highest-risk/complexity components
- Address technical challenges early
- Front-load difficult decisions

# MANDATORY RESPONSE FORMAT
## JSON Structure (EXACT FORMAT REQUIRED):
```json
{
  "analysis": {
    "complexity_assessment": "low|medium|high",
    "key_challenges": ["challenge1", "challenge2"],
    "estimated_task_count": "number_range",
    "primary_components": ["component1", "component2"]
  },
  "strategies": [
    {
      "strategy": "strategy_name",
      "approach": "LINEAR|PARALLEL|ITERATIVE|FEATURE|RISK",
      "reasoning": "why_this_approach_is_effective",
      "task_description": "detailed_strategy_description",
      "subtasks": [
        {
          "task_id": "unique_identifier",
          "task_type": "geometry|material|lighting|scene_setup|animation",
          "description": "specific_task_description",
          "priority": 1-5,
          "dependencies": ["prerequisite_task_ids"],
          "complexity": "low|medium|high",
          "estimated_time": "time_estimate",
          "validation_criteria": ["success_metrics"]
        }
      ],
      "is_terminal": false,
      "expected_quality": "quality_assessment"
    }
  ]
}
```

# QUALITY ASSURANCE REQUIREMENTS
## Task Granularity Standards:
- **Atomic Operations**: Each task must be independently executable
- **Measurable Outcomes**: Clear success/failure criteria
- **Balanced Scope**: Neither too broad nor too narrow
- **Technical Precision**: Include specific Blender API operations

## Strategy Diversity Requirements:
- **Different Execution Orders**: Vary task sequencing between strategies
- **Resource Allocation Variance**: Different approaches to parallel vs sequential
- **Risk Distribution**: Spread technical challenges across strategies
- **Optimization Focus**: Speed vs quality vs reliability trade-offs

## Technical Validation:
- **API Correctness**: Use valid Blender Python operations
- **Dependency Logic**: Ensure prerequisite relationships are sound
- **Resource Management**: Account for memory and performance constraints
- **Error Recovery**: Consider failure modes and fallback options

CRITICAL: The MCTS algorithm's success depends on strategy diversity. "
        "Each approach must be genuinely different, not just cosmetic variations. "
        "Focus on fundamentally distinct paths to the same goal."""

    def _parse_decomposition_response(self, response: str, node: MCTSNode) -> list[dict[str, Any]]:
        """Parse LLM response to extract decomposition strategies."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            strategies: list[dict[str, Any]] = []

            for strategy_data in data.get("strategies", []):
                # Convert to our internal format
                subtasks: list[SubTask] = []
                for st in strategy_data.get("subtasks", []):
                    task_type = TaskType(st.get("task_type", "geometry"))

                    # Use complexity level as priority if provided, otherwise use given priority
                    # Priority aligns with complexity: geometry=1, scene_setup=2, material=3, lighting=4, animation=5
                    default_priority = get_task_complexity_level(task_type)

                    subtask = SubTask(
                        id=f"task-{len(subtasks)}-{secrets.token_hex(4)}",  # More unique IDs
                        type=task_type,
                        description=st.get("description", ""),
                        dependencies=st.get("dependencies", []),
                        priority=st.get("priority", default_priority),
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

            # 2. Balanced complexity distribution (using priority as complexity proxy)
            complexities = [st.priority for st in node.subtasks]
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
        """Evaluate the complexity of this task decomposition with hierarchical awareness."""
        try:
            if not node.subtasks:
                return 0.5

            # 1. Hierarchical progression bonus (reward proper simple-to-complex ordering)
            hierarchy_score = self._evaluate_hierarchical_progression(node.subtasks)

            # 2. Number of dependencies (more deps = higher complexity)
            total_deps = sum(len(st.dependencies) for st in node.subtasks)
            dep_complexity = min(1.0, total_deps / (len(node.subtasks) * 2))

            # 3. Task complexity based on complexity levels
            complexity_levels = [get_task_complexity_level(st.type) for st in node.subtasks]
            avg_complexity_level = sum(complexity_levels) / len(complexity_levels)
            level_complexity = avg_complexity_level / 5.0  # Normalize to 0-1

            # 4. Complexity spread (good to have range from simple to complex)
            if len(complexity_levels) > 1:
                complexity_spread = (max(complexity_levels) - min(complexity_levels)) / 4.0
            else:
                complexity_spread = 0.0

            # Weighted combination favoring hierarchical progression
            base_complexity = (dep_complexity + level_complexity + complexity_spread) / 3.0

            # Apply hierarchy bonus (lower complexity score = better for MCTS)
            return base_complexity * (1.0 - hierarchy_score * 0.3)  # Up to 30% reduction for good hierarchy

        except Exception as e:
            log_agent_exception(self.logger, "complexity evaluation", e)
            return 0.5

    def _evaluate_hierarchical_progression(self, subtasks: list[SubTask]) -> float:
        """Evaluate how well the subtasks follow hierarchical progression (0-1, higher is better)."""
        if len(subtasks) <= 1:
            return 1.0

        # Get complexity levels for all subtasks
        levels = [get_task_complexity_level(st.type) for st in subtasks]

        # Reward arrangements that generally progress from simple to complex
        progression_score = 0.0
        total_comparisons = 0

        for i in range(len(levels) - 1):
            for j in range(i + 1, len(levels)):
                # If later task is same or higher complexity, that's good progression
                if levels[j] >= levels[i]:
                    progression_score += 1.0
                # Penalize if later task is much simpler (regression)
                elif levels[i] - levels[j] > 2:  # Significant complexity regression
                    progression_score -= 0.5
                total_comparisons += 1

        if total_comparisons == 0:
            return 1.0

        # Normalize and ensure 0-1 range
        normalized_score = progression_score / total_comparisons
        return max(0.0, min(1.0, normalized_score))

    async def _evaluate_completeness(self, node: MCTSNode, workflow_state: WorkflowState) -> float:
        """Evaluate how complete this task decomposition is."""
        try:
            if not node.subtasks:
                return 0.2  # Empty decomposition is incomplete

            # Completeness factors:
            # 1. Coverage of different task types
            task_types = {st.type for st in node.subtasks}
            expected_types = {TaskType.GEOMETRY, TaskType.MATERIAL}  # Minimum expected

            # Check for key task types based on the original prompt
            prompt_lower = workflow_state.prompt.lower()
            if any(word in prompt_lower for word in ["light", "illuminate", "shadow"]):
                expected_types.add(TaskType.LIGHTING)
            if any(word in prompt_lower for word in ["scene", "environment", "background"]):
                expected_types.add(TaskType.SCENE_SETUP)

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

        # Sort the final subtasks hierarchically (simple to complex)
        hierarchical_subtasks = sort_tasks_by_complexity(unique_subtasks)

        self.logger.info(
            "Hierarchical task ordering completed",
            original_count=len(unique_subtasks),
            ordered_count=len(hierarchical_subtasks),
            complexity_sequence=[
                get_task_complexity_level(task.type)
                for task in hierarchical_subtasks[:5]
            ]  # First 5 for brevity
        )

        return hierarchical_subtasks
