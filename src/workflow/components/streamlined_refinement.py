"""Streamlined refinement components with DRY principles."""

from collections.abc import Callable

import structlog

from ...utils.types import WorkflowState
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class RefinementEngine:
    """Simplified refinement engine with standardized logic."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def create_simple_refinement(self) -> Callable:
        """Create simple refinement that updates prompts."""

        async def simple_refinement(state: WorkflowState) -> WorkflowState:
            """Apply simple refinement by updating the prompt."""
            logger.info("Running simple refinement")

            # Check if refinement should be attempted
            if not self._should_attempt_refinement(state):
                logger.info("Skipping refinement - conditions not met")
                return state

            try:
                # Apply refinement
                state.refinement_iterations = getattr(state, "refinement_iterations", 0) + 1

                # Update prompt with error context
                if state.error_message:
                    error_context = (
                        f"\n\nPrevious attempt failed with error: {state.error_message}\nPlease fix this issue."
                    )
                    state.prompt = f"{state.original_prompt or state.prompt}{error_context}"
                    state.error_message = ""  # Clear error to allow retry

                logger.info(f"Simple refinement applied (iteration {state.refinement_iterations})")
                return state

            except Exception as e:
                logger.error(f"Simple refinement failed: {e}")
                state.error_message = f"Refinement error: {e}"
                return state

        return simple_refinement

    def create_adaptive_refinement(self) -> Callable:
        """Create adaptive refinement based on issue types."""

        async def adaptive_refinement(state: WorkflowState) -> WorkflowState:
            """Apply adaptive refinement based on detected issues."""
            logger.info("Running adaptive refinement")

            if not self._should_attempt_refinement(state):
                logger.info("Skipping refinement - conditions not met")
                return state

            try:
                # Determine refinement strategy based on issues
                strategy = self._select_refinement_strategy(state)
                logger.info(f"Selected refinement strategy: {strategy}")

                # Apply strategy-specific refinement
                if strategy == "error_focused":
                    self._apply_error_focused_refinement(state)
                elif strategy == "quality_focused":
                    self._apply_quality_focused_refinement(state)
                elif strategy == "output_focused":
                    self._apply_output_focused_refinement(state)
                else:
                    self._apply_general_refinement(state)

                # Update refinement counter
                state.refinement_iterations = getattr(state, "refinement_iterations", 0) + 1

                logger.info(f"Adaptive refinement applied (iteration {state.refinement_iterations})")
                return state

            except Exception as e:
                logger.error(f"Adaptive refinement failed: {e}")
                state.error_message = f"Adaptive refinement error: {e}"
                return state

        return adaptive_refinement

    def _should_attempt_refinement(self, state: WorkflowState) -> bool:
        """Check if refinement should be attempted."""
        # Check refinement limits
        refinement_count = getattr(state, "refinement_iterations", 0)
        if refinement_count >= self.config.max_refinements:
            logger.info("Max refinements reached, skipping refinement")
            return False

        # Check if there's something to refine
        has_errors = bool(state.error_message)
        has_quality_issues = (
            hasattr(state, "quality_score")
            and state.quality_score is not None
            and state.quality_score < self.config.quality_threshold
        )
        has_critical_issues = (
            hasattr(state, "quality_issues") and len(state.quality_issues) > self.config.critical_issues_threshold
        )

        should_refine = has_errors or has_quality_issues or has_critical_issues

        if not should_refine:
            logger.info("No refinement needed - quality sufficient")

        return should_refine

    def _select_refinement_strategy(self, state: WorkflowState) -> str:
        """Select appropriate refinement strategy based on state."""
        # Error-focused if we have execution errors
        if state.error_message or (state.execution_result and not state.execution_result.success):
            return "error_focused"

        # Quality-focused if quality score is low
        if (
            hasattr(state, "quality_score")
            and state.quality_score is not None
            and state.quality_score < self.config.quality_threshold
        ):
            return "quality_focused"

        # Output-focused if missing required outputs
        if not (state.blend_file_path and state.screenshot_path):
            return "output_focused"

        # General fallback
        return "general"

    def _apply_error_focused_refinement(self, state: WorkflowState):
        """Apply error-focused refinement."""
        if state.error_message:
            error_context = f"\n\nFocus on fixing these errors: {state.error_message}"
            if state.execution_result and state.execution_result.errors:
                error_context += f"\nExecution errors: {'; '.join(state.execution_result.errors)}"
            state.prompt = f"{state.original_prompt or state.prompt}{error_context}"
            state.error_message = ""

    def _apply_quality_focused_refinement(self, state: WorkflowState):
        """Apply quality-focused refinement."""
        quality_context = f"\n\nCurrent quality score: {getattr(state, 'quality_score', 0):.1f}"
        if hasattr(state, "quality_issues") and state.quality_issues:
            quality_context += f"\nAddress these quality issues: {'; '.join(state.quality_issues)}"
        if hasattr(state, "quality_feedback") and state.quality_feedback:
            quality_context += f"\nQuality feedback: {state.quality_feedback}"
        state.prompt = f"{state.original_prompt or state.prompt}{quality_context}"

    def _apply_output_focused_refinement(self, state: WorkflowState):
        """Apply output-focused refinement."""
        missing_outputs = []
        if not state.blend_file_path:
            missing_outputs.append("blend file")
        if not state.screenshot_path:
            missing_outputs.append("screenshot")

        if missing_outputs:
            output_context = f"\n\nEnsure these outputs are generated: {', '.join(missing_outputs)}"
            state.prompt = f"{state.original_prompt or state.prompt}{output_context}"

    def _apply_general_refinement(self, state: WorkflowState):
        """Apply general refinement."""
        general_context = "\n\nPlease improve the solution based on any issues you can identify."
        state.prompt = f"{state.original_prompt or state.prompt}{general_context}"


# Simplified condition functions
def should_refine_enhanced(state: WorkflowState) -> str:
    """Enhanced refinement decision with comprehensive stop conditions."""
    logger.debug("Evaluating refinement decision")

    # Hard stop conditions
    if state.error_message and "max_iterations" in state.error_message.lower():
        logger.info("Stopping: Max iterations reached")
        return "end"

    refinement_count = getattr(state, "refinement_iterations", 0)
    if refinement_count >= 2:  # Hard limit
        logger.info("Stopping: Max refinements reached")
        return "end"

    # Success conditions
    if (
        state.execution_result
        and state.execution_result.success
        and not state.execution_result.errors
        and refinement_count >= 1
    ):
        logger.info("Stopping: Successful execution after refinement")
        return "end"

    # No refinement needed conditions
    if not state.error_message and refinement_count > 0:
        logger.info("Stopping: No errors after refinement")
        return "end"

    # Quality-based refinement decision
    if hasattr(state, "quality_score") and state.quality_score is not None and state.quality_score >= 2.0:
        logger.info("Stopping: Quality threshold met")
        return "end"

    # Refine if we have errors or quality issues
    if state.error_message or (
        hasattr(state, "quality_score") and state.quality_score is not None and state.quality_score < 2.0
    ):
        logger.info("Continuing: Refinement needed")
        return "refine"

    logger.info("Stopping: Default - no refinement needed")
    return "end"


def create_simple_refinement_check(config: WorkflowConfig) -> Callable:
    """Create simple refinement check for fast workflows."""

    def simple_refinement_check(state: WorkflowState) -> str:
        """Simple check - refine if there are errors and we haven't hit limits."""
        if not state.error_message:
            return "end"

        refinement_count = getattr(state, "refinement_iterations", 0)
        if refinement_count >= config.max_refinements:
            return "end"

        return "refine"

    return simple_refinement_check


def create_simple_refinement_node(config: WorkflowConfig) -> Callable:
    """Create simple refinement node for fast workflows."""
    engine = RefinementEngine(config)
    return engine.create_simple_refinement()


# Factory functions for backward compatibility
def create_adaptive_refinement_node(config: WorkflowConfig) -> Callable:
    """Factory function for adaptive refinement."""
    engine = RefinementEngine(config)
    return engine.create_adaptive_refinement()
