"""Planner agent for decomposing natural language prompts into structured subtasks."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import check_user_termination

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
        '  "id": "unique_identifier",\n'
        '  "type": "geometry|material|lighting|scene_setup|animation",\n'
        '  "description": "Detailed description of what to accomplish",\n'
        '  "priority": 1-5,\n'
        '  "dependencies": ["prerequisite_task_ids"],\n'
        '  "parameters": {\n'
        '    "specific_blender_parameters": "with_exact_values"\n'
        "  },\n"
        '  "validation_criteria": ["specific_success_metrics"],\n'
        '  "estimated_complexity": "low|medium|high"\n'
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
        '  "asset_analysis": {\n'
        '    "complexity_level": 1-5,\n'
        '    "estimated_components": "number_of_major_parts",\n'
        '    "primary_challenges": ["challenge_list"],\n'
        '    "recommended_approach": "strategy_summary"\n'
        "  },\n"
        '  "tasks": ["array_of_task_objects"]\n'
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

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        """Initialize planner agent."""
        super().__init__(config, client)
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
            # Robust JSON parsing with multiple recovery strategies
            response_data = None
            tasks_data = []
            reasoning = ""

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
                self.logger.warning(f"Initial JSON parsing failed at position {e.pos}: {e.msg}")

                # Recovery Strategy 1: Try to fix common JSON issues
                try:
                    fixed_response = self._attempt_json_repair(cleaned_response)
                    response_data = json.loads(fixed_response)
                    tasks_data = response_data.get("tasks", [])
                    reasoning = response_data.get("reasoning", "")
                    self.logger.info("JSON successfully repaired and parsed")

                except json.JSONDecodeError as e2:
                    self.logger.warning(f"JSON repair failed: {e2.msg}")

                    # Recovery Strategy 2: Extract partial JSON
                    try:
                        partial_data = self._extract_partial_json(response_text)
                        if partial_data:
                            tasks_data = partial_data.get("tasks", [])
                            reasoning = partial_data.get("reasoning", "")
                            self.logger.info("Extracted partial JSON data successfully")
                        else:
                            raise ValueError("No valid JSON found")

                    except Exception:
                        self.logger.exception("All JSON parsing strategies failed")

                        # Recovery Strategy 3: Generate fallback tasks
                        tasks_data = self._generate_fallback_tasks(state.prompt)
                        reasoning = f"Generated fallback tasks due to JSON parsing failure: {e.msg}"
                        self.logger.warning("Using fallback task generation")

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

    def _attempt_json_repair(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax errors."""
        import re

        # Fix 1: Add missing commas between object properties
        # Look for patterns like `"key": "value"\n    "nextkey"`
        fixed = re.sub(r'("[^"]*":\s*[^,\}\]]*)\s*\n\s*(")', r"\1,\n    \2", json_str)

        # Fix 2: Remove trailing commas before closing brackets
        fixed = re.sub(r",(\s*[\}\]])", r"\1", fixed)

        # Fix 3: Add missing closing brackets/braces (simple heuristic)
        open_braces = fixed.count("{")
        close_braces = fixed.count("}")
        if open_braces > close_braces:
            fixed += "}" * (open_braces - close_braces)

        open_brackets = fixed.count("[")
        close_brackets = fixed.count("]")
        if open_brackets > close_brackets:
            fixed += "]" * (open_brackets - close_brackets)

        # Fix 4: Handle unescaped quotes in strings (basic attempt)
        # This is more complex and risky, so we'll be conservative

        return fixed

    def _extract_partial_json(self, text: str) -> dict:
        """Try to extract valid JSON from partial/corrupted response."""
        import re

        # Strategy 1: Look for complete JSON objects
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 2: Try to find and extract tasks array specifically
        tasks_pattern = r'"tasks"\s*:\s*\[(.*?)\]'
        tasks_match = re.search(tasks_pattern, text, re.DOTALL)
        if tasks_match:
            try:
                tasks_json = f'{{"tasks": [{tasks_match.group(1)}]}}'
                return json.loads(tasks_json)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Look for any valid JSON fragments
        # Find strings that look like they might be JSON
        potential_jsons = re.findall(r"\{[^{}]*\}", text)
        for potential in potential_jsons:
            try:
                return json.loads(potential)
            except json.JSONDecodeError:
                continue

        return None

    def _generate_fallback_tasks(self, prompt: str) -> list[dict]:
        """Generate basic fallback tasks when JSON parsing completely fails."""
        # Analyze the prompt to determine appropriate tasks
        prompt_lower = prompt.lower()

        fallback_tasks = []

        # Basic task templates based on common prompt patterns
        if any(word in prompt_lower for word in ["room", "interior", "indoor"]):
            fallback_tasks.extend(
                [
                    {
                        "id": "setup-scene",
                        "type": "environment",
                        "description": "Set up basic scene environment and lighting",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"environment_type": "interior"},
                    },
                    {
                        "id": "create-room",
                        "type": "architecture",
                        "description": "Create basic room structure with walls and floor",
                        "priority": 2,
                        "dependencies": ["setup-scene"],
                        "parameters": {"room_type": "interior"},
                    },
                ]
            )

        if any(word in prompt_lower for word in ["castle", "medieval", "courtyard"]):
            fallback_tasks.extend(
                [
                    {
                        "id": "create-ground",
                        "type": "terrain",
                        "description": "Create ground plane and basic terrain",
                        "priority": 1,
                        "dependencies": [],
                        "parameters": {"terrain_type": "courtyard"},
                    },
                    {
                        "id": "create-walls",
                        "type": "architecture",
                        "description": "Create castle walls and structures",
                        "priority": 2,
                        "dependencies": ["create-ground"],
                        "parameters": {"style": "medieval"},
                    },
                ]
            )

        # Generic lighting task for most scenes
        fallback_tasks.append(
            {
                "id": "setup-lighting",
                "type": "lighting",
                "description": f"Set up lighting for: {prompt}",
                "priority": 3,
                "dependencies": [],
                "parameters": {"lighting_type": "atmospheric"},
            }
        )

        # If no specific tasks were generated, create a generic one
        if not fallback_tasks:
            fallback_tasks = [
                {
                    "id": "create-scene",
                    "type": "modeling",
                    "description": f"Create scene: {prompt}",
                    "priority": 1,
                    "dependencies": [],
                    "parameters": {"prompt": prompt},
                }
            ]

        return fallback_tasks

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for planner agent."""
        if not state.prompt:
            return False

        # Check prompt length (minimum and maximum)
        prompt_length = len(state.prompt.strip())
        return not (prompt_length < MIN_PROMPT_LENGTH or prompt_length > MAX_PROMPT_LENGTH)
