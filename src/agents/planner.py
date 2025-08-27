"""Planner agent for decomposing natural language prompts into structured subtasks."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent


@dataclass
class TaskDecompositionPrompt:
    """Template for task decomposition prompts."""

    SYSTEM_PROMPT = """You are a 3D modeling task planner specializing in Blender workflows.

Your role is to analyze natural language prompts and decompose them into structured subtasks
for 3D asset creation. You understand geometry, materials, lighting, scene setup, and animation.

For each subtask, determine:
1. Task type (geometry, material, lighting, scene_setup, animation)
2. Priority (1-5, where 1 is highest priority)
3. Dependencies (which tasks must complete first)
4. Specific parameters needed for execution

Return tasks in optimal execution order with clear, actionable descriptions."""

    USER_TEMPLATE = """Analyze this prompt and create a detailed task breakdown:

Prompt: "{prompt}"

Return your response as a JSON array of tasks with this structure:
{{
  "tasks": [
    {{
      "id": "task-1",
      "type": "geometry|material|lighting|scene_setup|animation",
      "description": "Clear, specific description of what to create",
      "priority": 1-5,
      "dependencies": ["task-id-1", "task-id-2"],
      "parameters": {{"key": "value", ...}}
    }}
  ],
  "reasoning": "Brief explanation of the task breakdown approach"
}}

Guidelines:
- Create 1-8 tasks depending on complexity
- Be specific about shapes, colors, materials, positions
- Consider proper task ordering and dependencies
- Include realistic parameters for Blender operations"""


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
        start_time = time.monotonic()

        try:
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
            messages = [
                {"role": "system", "content": self.task_decomposer.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.task_decomposer.USER_TEMPLATE.format(
                        prompt=state.prompt
                    ),
                },
            ]

            # Get LLM response
            response_text = await self.make_openai_request(messages)

            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                tasks_data = response_data.get("tasks", [])
                reasoning = response_data.get("reasoning", "")
            except json.JSONDecodeError as e:
                self.logger.error("Failed to parse JSON response", error=str(e))
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message=f"Failed to parse response: {str(e)}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            # Convert to SubTask objects
            subtasks = []
            for i, task_data in enumerate(tasks_data):
                try:
                    subtask = SubTask(
                        id=task_data.get("id", f"task-{i+1}"),
                        type=TaskType(task_data["type"]),
                        description=task_data["description"],
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        parameters=task_data.get("parameters", {}),
                    )
                    subtasks.append(subtask)
                except (KeyError, ValueError) as e:
                    self.logger.warning(
                        "Invalid task data", task_data=task_data, error=str(e)
                    )
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

        except Exception as e:
            self.logger.error("Task decomposition failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data=[],
                message=f"Decomposition failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
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
                # Circular dependency or missing dependency - this indicates a planning failure.
                self.logger.error("Circular or missing dependency detected in task plan", remaining_tasks=list(remaining.keys()))
                raise ValueError("Invalid task plan: circular or missing dependency detected.")

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
        if prompt_length < 5 or prompt_length > 2000:
            return False

        return True