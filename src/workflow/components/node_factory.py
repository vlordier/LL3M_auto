"""Streamlined node factory for workflow components."""

from collections.abc import Callable
from typing import Any

import structlog

from ...utils.types import WorkflowState

logger = structlog.get_logger(__name__)


class NodeFactory:
    """Factory for creating standardized workflow nodes with DRY principles."""

    def __init__(self, config):  # noqa: D107
        self.config = config

    def create_agent_node(self, agent_type: str, field_mapping: dict[str, str]) -> Callable:
        """Create a standardized agent node with consistent error handling.

        Args:
            agent_type: Type of agent to create (planner, retrieval, coding, etc.)
            field_mapping: Maps response.data to state fields {'data_field': 'state_field'}
        """
        from ...agents.factory import get_agent_factory

        async def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute agent with standardized error handling."""
            try:
                # Create agent
                factory = get_agent_factory()
                agent = getattr(factory, f"create_{agent_type}")()

                # Process
                response = await agent.process(state)

                # Handle response
                if response.success:
                    # Map response data to state fields
                    for data_field, state_field in field_mapping.items():
                        if hasattr(response, "data") and response.data:
                            setattr(state, state_field, response.data)
                        else:
                            setattr(state, state_field, getattr(response, data_field, None))
                else:
                    state.error_message = response.message or f"{agent_type.title()} execution failed"

            except Exception as e:
                logger.exception("%s node failed", agent_type.title())
                state.error_message = f"{agent_type.title()} error: {e}"

            return state

        agent_node.__name__ = f"{agent_type}_node"
        return agent_node

    def create_assessment_node(self, agent_type: str) -> Callable:
        """Create a standardized assessment node.

        Args:
            agent_type: Type of assessment agent (critic, verification)
        """
        from ...agents.factory import get_agent_factory

        async def assessment_node(state: WorkflowState) -> dict[str, Any]:
            """Execute assessment with standardized response format."""
            try:
                # Create agent
                factory = get_agent_factory()
                agent = getattr(factory, f"create_{agent_type}")()

                # Process
                response = await agent.process(state)

                # Standardized response format
                if response.success:
                    return {
                        "score": 3.0,
                        "issues": [],
                        "feedback": response.data or f"{agent_type.title()} assessment completed",
                    }
                else:
                    return {
                        "score": 1.0,
                        "issues": (
                            [response.message] if response.message else [f"{agent_type.title()} assessment failed"]
                        ),
                        "feedback": response.message or f"{agent_type.title()} assessment failed",
                    }

            except Exception as e:
                logger.exception("%s assessment failed", agent_type.title())
                return {
                    "score": 1.0,
                    "issues": [str(e)],
                    "feedback": f"{agent_type.title()} assessment error: {e}",
                }

        assessment_node.__name__ = f"{agent_type}_assessment"
        return assessment_node

    def create_execution_node(self) -> Callable:
        """Create the Blender execution node."""
        from ...blender.enhanced_executor import EnhancedBlenderExecutor

        async def execution_node(state: WorkflowState) -> WorkflowState:
            """Execute Blender code with streamlined result handling."""
            try:
                executor = EnhancedBlenderExecutor()
                result = await executor.execute_code(
                    code=state.generated_code,
                    prompt=state.prompt,
                    output_dir=state.output_directory or "outputs",
                )

                # Update state with results
                state.execution_result = result
                state.blend_file_path = result.asset_path
                state.screenshot_path = result.screenshot_path

                if not result.success:
                    state.error_message = "; ".join(result.errors) if result.errors else "Execution failed"

            except Exception as e:
                logger.exception("Execution failed")
                state.error_message = f"Execution error: {e}"

            return state

        return execution_node


class ConditionFactory:
    """Factory for creating standardized workflow conditions."""

    def __init__(self, config):  # noqa: D107
        self.config = config

    def create_continue_condition(self) -> Callable:
        """Create standardized continue/end condition."""

        def should_continue(state: WorkflowState) -> str:
            """Determine if workflow should continue based on errors and limits."""
            if state.error_message:
                return "end"
            if state.refinement_iterations >= state.max_refinements:
                return "end"
            return "continue"

        return should_continue

    def create_refinement_condition(self) -> Callable:
        """Create standardized refinement condition."""

        def should_refine(state: WorkflowState) -> str:
            """Determine if refinement is needed."""
            # Hard stops
            if state.error_message and "max_iterations" in state.error_message.lower():
                return "end"
            if state.refinement_iterations >= self.config.max_refinements:
                return "end"

            # Success conditions
            if (
                state.execution_result
                and state.execution_result.success
                and not state.execution_result.errors
                and state.refinement_iterations >= 1
            ):
                return "end"

            # Quality conditions
            if (
                hasattr(state, "quality_score")
                and state.quality_score is not None
                and state.quality_score >= self.config.quality_threshold
            ):
                return "end"

            # Refine conditions
            if state.error_message or (
                hasattr(state, "quality_score")
                and state.quality_score is not None
                and state.quality_score < self.config.quality_threshold
            ):
                return "refine"

            return "end"

        return should_refine

    def create_quality_condition(self) -> Callable:
        """Create standardized quality assessment condition."""

        def should_assess_quality(state: WorkflowState) -> str:
            """Determine if quality assessment should be performed."""
            if not state.execution_result or not state.execution_result.success:
                return "end"

            # Skip if already refined successfully
            if (
                state.execution_result.success
                and not state.execution_result.errors
                and state.refinement_iterations >= 1
            ):
                return "end"

            return "assess"

        return should_assess_quality
