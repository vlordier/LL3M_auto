"""Selective state management for LangGraph workflows to handle context overflow."""

from typing import Any

from ..utils.types import SubTask, WorkflowState


class SelectiveStateManager:
    """Manages state by selectively retaining only essential information for each agent."""

    @staticmethod
    def prepare_for_agent(state: WorkflowState, agent_type: str) -> dict[str, Any]:
        """Prepare minimal state context for specific agent types."""
        base_context = {
            "prompt": state.prompt,
            "original_prompt": state.original_prompt,
            "refinement_request": state.refinement_request,
            "refinement_iterations": state.refinement_iterations,
        }

        if agent_type == "planner":
            return {
                **base_context,
                # Planner only needs the basic prompt and any refinement context
            }

        elif agent_type == "retrieval":
            return {
                **base_context,
                "subtasks": SelectiveStateManager._compress_subtasks(state.subtasks, max_tasks=5),
                # Retrieval needs subtasks to understand what to fetch
            }

        elif agent_type == "coding":
            return {
                **base_context,
                "subtasks": SelectiveStateManager._compress_subtasks(state.subtasks, max_tasks=3),
                "documentation": SelectiveStateManager._trim_documentation(state.documentation, max_chars=5000),
                # Coding needs subtasks and relevant docs, but not full history
            }

        elif agent_type == "critic":
            return {
                **base_context,
                "execution_result": state.execution_result,
                "previous_screenshot_path": state.previous_screenshot_path,
                # Critic only needs execution results and screenshots
            }

        elif agent_type == "verification":
            return {
                **base_context,
                "execution_result": state.execution_result,
                "generated_code": SelectiveStateManager._trim_code(state.generated_code),
                # Verification needs execution results and code summary
            }

        elif agent_type == "user_proxy":
            return {
                **base_context,
                "execution_result": state.execution_result,
                "critic_analysis": SelectiveStateManager._compress_analysis(state.critic_analysis),
                "verification_result": SelectiveStateManager._compress_analysis(state.verification_result),
                # User proxy needs results and summaries for decision making
            }

        # Default: return everything (fallback)
        return state.model_dump()

    @staticmethod
    def _compress_subtasks(subtasks: list[SubTask] | None, max_tasks: int = 3) -> list[dict[str, Any]]:
        """Compress subtasks to essential information."""
        if not subtasks:
            return []

        compressed = []
        for _i, task in enumerate(subtasks[:max_tasks]):  # Limit number of tasks
            compressed.append(
                {
                    "id": task.id,
                    "type": (task.type.value if hasattr(task.type, "value") else str(task.type)),
                    "description": (
                        task.description[:200] + "..." if len(task.description) > 200 else task.description
                    ),
                    "priority": task.priority,
                    "parameters": SelectiveStateManager._compress_dict(task.parameters, max_depth=2),
                }
            )

        return compressed

    @staticmethod
    def _trim_documentation(doc: str | None, max_chars: int = 3000) -> str:
        """Trim documentation to essential sections."""
        if not doc:
            return ""

        if len(doc) <= max_chars:
            return doc

        # Try to keep complete sections
        lines = doc.split("\n")
        trimmed_lines = []
        current_length = 0

        for line in lines:
            if current_length + len(line) > max_chars:
                break
            trimmed_lines.append(line)
            current_length += len(line)

        trimmed = "\n".join(trimmed_lines)
        return trimmed + "\n... [truncated for context management]"

    @staticmethod
    def _trim_code(code: str | None, max_lines: int = 30) -> str:
        """Trim code to main functions and key operations."""
        if not code:
            return ""

        lines = code.split("\n")
        if len(lines) <= max_lines:
            return code

        # Keep imports, function definitions, and key operations
        important_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(("import ", "from ", "def ", "class "))
                or "bpy.ops" in stripped
                or "bpy.context" in stripped
                or stripped.startswith("#")
            ):  # Keep comments for context
                important_lines.append(line)

        # If still too many, take first max_lines
        if len(important_lines) > max_lines:
            important_lines = important_lines[:max_lines]

        return "\n".join(important_lines) + "\n# ... [code truncated for context management]"

    @staticmethod
    def _compress_analysis(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
        """Compress analysis results to key metrics."""
        if not analysis:
            return None

        # Keep only essential fields
        essential = {}
        key_fields = [
            "overall_score",
            "quality_score",
            "needs_refinement",
            "critical_issues",
            "improvement_suggestions",
            "recommendations",
        ]

        for field in key_fields:
            if field in analysis:
                value = analysis[field]
                # Truncate lists to max 3 items
                if isinstance(value, list):
                    essential[field] = value[:3]
                # Truncate strings to max 200 chars
                elif isinstance(value, str):
                    essential[field] = value[:200] + "..." if len(value) > 200 else value
                else:
                    essential[field] = value

        return essential

    @staticmethod
    def _compress_dict(d: dict[str, Any] | None, max_depth: int = 2, current_depth: int = 0) -> dict[str, Any]:
        """Recursively compress dictionary to prevent deep nesting."""
        if not d or current_depth >= max_depth:
            return {} if not d else {"...": "truncated"}

        compressed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                compressed[key] = SelectiveStateManager._compress_dict(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                # Keep only first few items
                compressed[key] = value[:3] if len(value) > 3 else value
            elif isinstance(value, str):
                # Truncate long strings
                compressed[key] = value[:100] + "..." if len(value) > 100 else value
            else:
                compressed[key] = value

        return compressed


class ContextWindow:
    """Manages sliding window of context for conversation history."""

    def __init__(self, max_history: int = 3):
        """Initialize with maximum number of iterations to retain."""
        self.max_history = max_history
        self.history: list[dict[str, Any]] = []

    def add_iteration(self, state: WorkflowState, stage: str):
        """Add current state to history."""
        iteration = {
            "timestamp": state.model_dump().get("timestamp", "unknown"),
            "stage": stage,
            "refinement_iteration": state.refinement_iterations,
            "summary": {
                "prompt": (state.prompt[:100] + "..." if len(state.prompt or "") > 100 else state.prompt),
                "success": (getattr(state.execution_result, "success", None) if state.execution_result else None),
                "needs_refinement": state.needs_refinement,
                "quality_score": (
                    getattr(state.verification_result, "quality_score", None) if state.verification_result else None
                ),
            },
        }

        self.history.append(iteration)

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_context_summary(self) -> str:
        """Get summary of recent iterations for context."""
        if not self.history:
            return ""

        summary_parts = ["Recent workflow history:"]
        for i, iteration in enumerate(self.history[-3:], 1):  # Last 3 iterations
            summary_parts.append(
                f"{i}. {iteration['stage']} (iter {iteration['refinement_iteration']}): "
                f"Success={iteration['summary']['success']}, "
                f"Quality={iteration['summary'].get('quality_score', 'N/A')}"
            )

        return "\n".join(summary_parts)
