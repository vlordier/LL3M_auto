"""Planner agent for decomposing natural language prompts into structured subtasks."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import check_user_termination, log_agent_exception

# Validation constants
MIN_PROMPT_LENGTH = 5
MAX_PROMPT_LENGTH = 2000


@dataclass
class TaskDecompositionPrompt:
    """Template for task decomposition prompts."""

    SYSTEM_PROMPT = (
        "# ROLE & EXPERTISE\n"
        "You are an expert 3D modeling task planner and workflow architect with deep knowledge of:\n"
        "- **3D Modeling Fundamentals**: Topology, geometry, manifold surfaces, edge flow\n"
        "- **Blender Python API**: bpy, bmesh, mathutils, node systems, operators\n"
        "- **Procedural Modeling**: Geometry nodes, modifiers, parametric design\n"
        "- **Material Science**: PBR workflows, node systems, texture mapping\n"
        "- **Lighting Theory**: Three-point lighting, HDRI, area lights, global illumination\n"
        "- **Scene Composition**: Camera work, rule of thirds, depth of field\n\n"

        "# TASK DECOMPOSITION METHODOLOGY\n"
        "## Hierarchical Planning Approach:\n"
        "### Phase 1: FOUNDATION (Priority 1-2)\n"
        "- **Geometry Creation**: Start with primitive shapes, build complexity gradually\n"
        "- **Basic Structure**: Establish main components, proper proportions\n"
        "- **Topology Optimization**: Ensure clean geometry, good edge flow\n\n"

        "### Phase 2: REFINEMENT (Priority 2-3)\n"
        "- **Detail Addition**: Add geometric details, surface features\n"
        "- **Modifier Application**: Subdivision, bevel, array operations\n"
        "- **Component Assembly**: Join parts, ensure structural integrity\n\n"

        "### Phase 3: MATERIALIZATION (Priority 3-4)\n"
        "- **Material Creation**: PBR-compliant materials, proper node setups\n"
        "- **Texture Application**: UV mapping, texture coordination\n"
        "- **Surface Properties**: Roughness, metallic, normal mapping\n\n"

        "### Phase 4: ILLUMINATION (Priority 4)\n"
        "- **Primary Lighting**: Key light establishment\n"
        "- **Secondary Lighting**: Fill lights, rim lighting\n"
        "- **Environmental Lighting**: HDRI, ambient lighting\n\n"

        "### Phase 5: PRESENTATION (Priority 5)\n"
        "- **Camera Setup**: Composition, focal length, depth of field\n"
        "- **Scene Organization**: Hierarchy, naming, cleanup\n"
        "- **Render Configuration**: Engine, samples, output settings\n\n"

        "# MANDATORY TASK SPECIFICATION FORMAT\n"
        "## Required Fields for Each Task:\n"
        "```json\n"
        "{\n"
        "  \"id\": \"unique_identifier\",\n"
        "  \"type\": \"geometry|material|lighting|scene_setup|animation\",\n"
        "  \"description\": \"Detailed description of what to accomplish\",\n"
        "  \"priority\": 1-5,\n"
        "  \"dependencies\": [\"prerequisite_task_ids\"],\n"
        "  \"parameters\": {\n"
        "    \"specific_blender_parameters\": \"with_exact_values\"\n"
        "  },\n"
        "  \"validation_criteria\": [\"specific_success_metrics\"],\n"
        "  \"estimated_complexity\": \"low|medium|high\"\n"
        "}\n"
        "```\n\n"

        "## Task Type Specifications:\n"
        "### geometry:\n"
        "- **Purpose**: Create, modify, or combine 3D meshes\n"
        "- **Parameters**: shape, dimensions, location, rotation, modifiers\n"
        "- **Examples**: Create cube, apply subdivision, boolean operations\n\n"

        "### material:\n"
        "- **Purpose**: Create and assign surface materials\n"
        "- **Parameters**: material_type, color, roughness, metallic, textures\n"
        "- **Examples**: PBR wood material, metal shader, glass material\n\n"

        "### lighting:\n"
        "- **Purpose**: Configure scene illumination\n"
        "- **Parameters**: light_type, intensity, color, position, angle\n"
        "- **Examples**: Sun light, HDRI environment, area light setup\n\n"

        "### scene_setup:\n"
        "- **Purpose**: Configure camera, render settings, organization\n"
        "- **Parameters**: camera_position, render_engine, resolution, samples\n"
        "- **Examples**: Camera framing, render optimization, scene cleanup\n\n"

        "### animation:\n"
        "- **Purpose**: Create motion, dynamics, or time-based effects\n"
        "- **Parameters**: keyframes, duration, easing, physics_properties\n"
        "- **Examples**: Rotation animation, physics simulation, particle systems\n\n"

        "# CRITICAL REQUIREMENTS\n"
        "## Dependency Management:\n"
        "- **Sequential Dependencies**: Geometry → Materials → Lighting → Scene\n"
        "- **Parallel Opportunities**: Independent geometric components\n"
        "- **Resource Dependencies**: Materials depend on existing objects\n\n"

        "## Parameter Precision:\n"
        "- **Exact Values**: Provide specific dimensions, coordinates, angles\n"
        "- **Blender Units**: Use Blender's default units (meters for scale)\n"
        "- **Realistic Proportions**: Ensure physically plausible dimensions\n\n"

        "## Quality Assurance:\n"
        "- **Validation Criteria**: Define success metrics for each task\n"
        "- **Complexity Assessment**: Realistic effort estimation\n"
        "- **Error Prevention**: Anticipate common failure points\n\n"

        "# OUTPUT FORMAT REQUIREMENTS\n"
        "## Mandatory Response Structure:\n"
        "```json\n"
        "{\n"
        "  \"asset_analysis\": {\n"
        "    \"complexity_level\": 1-5,\n"
        "    \"estimated_components\": \"number_of_major_parts\",\n"
        "    \"primary_challenges\": [\"challenge_list\"],\n"
        "    \"recommended_approach\": \"strategy_summary\"\n"
        "  },\n"
        "  \"tasks\": [\"array_of_task_objects\"]\n"
        "}\n"
        "```\n\n"

        "## Task Granularity Guidelines:\n"
        "- **Atomic Operations**: Each task should be independently executable\n"
        "- **Balanced Scope**: Not too broad (entire model) or narrow (single vertex)\n"
        "- **Clear Outcomes**: Each task produces a measurable result\n"
        "- **Logical Grouping**: Related operations in sequential tasks\n\n"

        "CRITICAL: Your task breakdown directly determines code generation success. "
        "Be thorough, precise, and technically accurate."
    )

    USER_TEMPLATE = """Analyze this prompt and create a detailed task breakdown:

Prompt: "{prompt}"

EXAMPLE (for "Create a 3D object of a chair"):
{{
  "tasks": [
    {{
      "id": "chair-seat",
      "type": "geometry",
      "description": "Create chair seat using cube primitive with specific dimensions",
      "priority": 1,
      "dependencies": [],
      "parameters": {{"shape": "cube", "dimensions": [1.8, 1.6, 0.1], "location": [0, 0, 0.45]}}
    }},
    {{
      "id": "chair-backrest",
      "type": "geometry",
      "description": "Create chair backrest attached to seat",
      "priority": 1,
      "dependencies": ["chair-seat"],
      "parameters": {{"shape": "cube", "dimensions": [1.8, 0.1, 1.2], "location": [0, -0.8, 1.0]}}
    }},
    {{
      "id": "chair-legs",
      "type": "geometry",
      "description": "Create four chair legs using cylinders",
      "priority": 2,
      "dependencies": ["chair-seat"],
      "parameters": {{"shape": "cylinder", "count": 4, "dimensions": [0.05, 0.05, 0.9], "positions": "corners"}}
    }}
  ],
  "reasoning": "Chair requires seat foundation first, then backrest attachment, finally legs for support"
}}

GUIDELINES:
- Break complex objects into logical components (seat, backrest, legs, etc.)
- Start with primary geometry, then secondary elements
- Include precise Blender parameters (dimensions, locations, shapes)
- Consider structural dependencies (foundation before details)
- Add materials and lighting after geometry is complete

Return your response as valid JSON:"""


class PlannerAgent(EnhancedBaseAgent):
    """Decomposes natural language prompts into structured 3D modeling subtasks."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize planner agent."""
        super().__init__(config)
        self.task_decomposer = TaskDecompositionPrompt()

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.PLANNER

    @property
    def name(self) -> str:
        """Return agent name."""
        return "Task Planner"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Decompose prompt into structured subtasks."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Check for user termination request
            if check_user_termination(state):
                self.logger.info("User requested workflow termination")
                # Set workflow to stop continuing
                state.should_continue = False
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    data=[],
                    message="Workflow terminated by user request",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            # Validate input
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message="Invalid input: prompt is required",
                    execution_time=0.0,
                )

            self.logger.info("Starting task decomposition", prompt=state.prompt[:100])

            # Create decomposition prompt
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self.task_decomposer.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.task_decomposer.USER_TEMPLATE.format(prompt=state.prompt),
                },
            ]

            # Get LLM response
            response_text = await self.make_openai_request(messages)

            # Parse JSON response - handle markdown code blocks
            try:
                # Clean markdown code blocks if present
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  # Remove ```
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                cleaned_response = cleaned_response.strip()

                response_data = json.loads(cleaned_response)
                tasks_data = response_data.get("tasks", [])
                reasoning = response_data.get("reasoning", "")
            except json.JSONDecodeError as e:
                log_agent_exception(self.logger, "parse JSON response", e)
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message=f"Failed to parse response: {e!s}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            # Convert to SubTask objects
            subtasks = []
            for i, task_data in enumerate(tasks_data):
                try:
                    subtask = SubTask(
                        id=task_data.get("id", f"task-{i + 1}"),
                        type=TaskType(task_data["type"]),
                        description=task_data["description"],
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        parameters=task_data.get("parameters", {}),
                    )
                    subtasks.append(subtask)
                except (KeyError, ValueError) as e:
                    self.logger.warning("Invalid task data", task_data=task_data, error=str(e))
                    continue

            if not subtasks:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message="No valid subtasks generated",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            # Sort by priority and dependencies
            ordered_subtasks = self._order_tasks_by_dependencies(subtasks)

            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.info(
                "Task decomposition completed",
                num_tasks=len(ordered_subtasks),
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=ordered_subtasks,
                message=f"Generated {len(ordered_subtasks)} subtasks: {reasoning}",
                execution_time=execution_time,
                metadata={"reasoning": reasoning, "original_prompt": state.prompt},
            )

        except Exception as decomposition_error:
            AgentErrorHandler.log_error(
                self.logger,
                decomposition_error,
                "task decomposition",
                prompt_preview=state.prompt[:100] if state.prompt else "No prompt",
            )
            return AgentErrorHandler.create_error_response(
                self.agent_type, decomposition_error, "task decomposition", start_time
            )

    def _order_tasks_by_dependencies(self, tasks: list[SubTask]) -> list[SubTask]:
        """Order tasks respecting dependencies and priorities."""
        # Implementation of topological sort with priority consideration
        ordered = []
        remaining = {task.id: task for task in tasks}

        while remaining:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = []
            for task in remaining.values():
                if all(dep_id not in remaining for dep_id in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or missing dependency - break it
                ready_tasks = [next(iter(remaining.values()))]
                self.logger.warning("Breaking circular or missing dependencies")

            # Sort ready tasks by priority (lower number = higher priority)
            ready_tasks.sort(key=lambda t: t.priority)

            # Add highest priority task
            next_task = ready_tasks[0]
            ordered.append(next_task)
            del remaining[next_task.id]

        return ordered

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for planner agent."""
        if not state.prompt:
            return False

        # Check prompt length (minimum and maximum)
        prompt_length = len(state.prompt.strip())
        return not (prompt_length < MIN_PROMPT_LENGTH or prompt_length > MAX_PROMPT_LENGTH)
