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
        "You are an expert 3D modeling task planner for Blender workflows.\n\n"
        "EXPERTISE: You understand 3D modeling fundamentals, Blender Python API, "
        "geometry creation, material systems, lighting principles, and scene composition.\n\n"
        "TASK: Decompose natural language prompts into structured, executable subtasks "
        "that create high-quality 3D assets through procedural generation.\n\n"
        "APPROACH:\n"
        "- Start with basic geometry (primitive shapes, modifiers, procedural modeling)\n"
        "- Add materials with proper node setups (Principled BSDF, textures, colors)\n"
        "- Include lighting for proper visualization (HDRI, area lights, sun)\n"
        "- Consider scale, positioning, and scene composition\n"
        "- Ensure tasks have clear dependencies and execution order\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Task type: geometry|material|lighting|scene_setup|animation\n"
        "- Priority: 1 (critical) to 5 (optional)\n"
        "- Dependencies: prerequisite task IDs\n"
        "- Parameters: specific values for Blender operations"
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
