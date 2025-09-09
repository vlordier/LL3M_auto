"""Context adapters for agents to handle context length limitations."""

from typing import Any

from ..utils.types import WorkflowState
from ..workflow.selective_state_manager import SelectiveStateManager


class AgentContextAdapter:
    """Adapts workflow state for specific agents to prevent context overflow."""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.state_manager = SelectiveStateManager()

    def prepare_context(self, state: WorkflowState) -> dict[str, Any]:
        """Prepare optimized context for this agent type."""
        return self.state_manager.prepare_for_agent(state, self.agent_type)

    def estimate_tokens(self, context: dict[str, Any]) -> int:
        """Estimate token count for the prepared context."""
        import json

        context_str = json.dumps(context, default=str)
        return len(context_str) // 4  # Rough estimate: 1 token ≈ 4 chars


class CodingAgentContextAdapter(AgentContextAdapter):
    """Specialized adapter for coding agent with documentation management."""

    def __init__(self):
        super().__init__("coding")

    def prepare_context(self, state: WorkflowState) -> dict[str, Any]:
        """Prepare context with intelligent documentation selection."""
        base_context = super().prepare_context(state)

        # Enhance documentation selection based on subtasks
        if state.subtasks and state.documentation:
            base_context["documentation"] = self._select_relevant_docs(state.documentation, state.subtasks)

        return base_context

    def _select_relevant_docs(self, documentation: str, subtasks: list[Any]) -> str:
        """Select most relevant documentation sections based on subtasks."""
        doc_lines = documentation.split("\n")

        # Extract keywords from subtasks
        keywords = set()
        for task in subtasks[:3]:  # Focus on top 3 tasks
            desc_words = task.description.lower().split()
            keywords.update(word for word in desc_words if len(word) > 3)

            # Add parameter keywords
            for param_value in task.parameters.values():
                if isinstance(param_value, str):
                    keywords.add(param_value.lower())

        # Score documentation sections by keyword relevance
        relevant_sections = []
        current_section = []
        section_score = 0

        for line in doc_lines:
            line_lower = line.lower()

            # Check if this is a new section
            if line.startswith("#"):
                # Save previous section if relevant
                if current_section and section_score > 0:
                    relevant_sections.append(("\n".join(current_section), section_score))

                current_section = [line]
                section_score = 0
            else:
                current_section.append(line)

            # Score line based on keyword matches
            line_score = sum(1 for keyword in keywords if keyword in line_lower)
            section_score += line_score

        # Save last section
        if current_section and section_score > 0:
            relevant_sections.append(("\n".join(current_section), section_score))

        # Sort by relevance and take top sections
        relevant_sections.sort(key=lambda x: x[1], reverse=True)

        selected_docs = []
        total_chars = 0
        max_chars = 4000  # Conservative limit

        for section, _score in relevant_sections:
            if total_chars + len(section) <= max_chars:
                selected_docs.append(section)
                total_chars += len(section)
            else:
                break

        return "\n\n".join(selected_docs) if selected_docs else documentation[:max_chars]


class CriticAgentContextAdapter(AgentContextAdapter):
    """Specialized adapter for critic agent with screenshot focus."""

    def __init__(self):
        super().__init__("critic")

    def prepare_context(self, state: WorkflowState) -> dict[str, Any]:
        """Prepare minimal context focused on visual analysis."""
        context = {
            "prompt": state.original_prompt or state.prompt,
            "execution_result": {
                "success": (state.execution_result.success if state.execution_result else False),
                "screenshot_path": (state.execution_result.screenshot_path if state.execution_result else None),
            },
            "refinement_iteration": state.refinement_iterations,
        }

        # Include previous screenshot for comparison if available
        if state.previous_screenshot_path:
            context["previous_screenshot_path"] = state.previous_screenshot_path

        # Include minimal refinement history
        if state.refinement_request:
            context["previous_refinement"] = state.refinement_request[:200]

        return context


class VerificationAgentContextAdapter(AgentContextAdapter):
    """Specialized adapter for verification agent with code focus."""

    def __init__(self):
        super().__init__("verification")

    def prepare_context(self, state: WorkflowState) -> dict[str, Any]:
        """Prepare context focused on asset verification."""
        context = {
            "prompt": state.original_prompt or state.prompt,
            "execution_result": (state.execution_result.model_dump() if state.execution_result else None),
            "refinement_iteration": state.refinement_iterations,
        }

        # Include compressed code for verification
        if state.generated_code:
            context["code_summary"] = self._create_code_summary(state.generated_code)

        return context

    def _create_code_summary(self, code: str) -> dict[str, Any]:
        """Create a summary of the code for verification."""
        lines = code.split("\n")

        summary = {
            "total_lines": len(lines),
            "imports": [],
            "functions": [],
            "bpy_operations": [],
        }

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                summary["imports"].append(stripped[:50])
            elif stripped.startswith("def "):
                func_name = stripped.split("(")[0].replace("def ", "")
                summary["functions"].append(func_name)
            elif "bpy.ops." in stripped:
                summary["bpy_operations"].append(stripped[:100])

        # Limit list sizes
        for key in ["imports", "functions", "bpy_operations"]:
            summary[key] = summary[key][:5]  # Max 5 items each

        return summary


# Factory for creating appropriate context adapters
def create_context_adapter(agent_type: str) -> AgentContextAdapter:
    """Create appropriate context adapter for agent type."""
    adapters = {
        "coding": CodingAgentContextAdapter,
        "critic": CriticAgentContextAdapter,
        "verification": VerificationAgentContextAdapter,
    }

    adapter_class = adapters.get(agent_type, AgentContextAdapter)
    return adapter_class() if agent_type in adapters else AgentContextAdapter(agent_type)
