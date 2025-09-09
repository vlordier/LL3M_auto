"""Error recovery middleware for LangGraph workflows to handle context overflow."""

import logging
from collections.abc import Callable

from ..utils.types import WorkflowState
from .context_manager import ContextCompactor
from .selective_state_manager import SelectiveStateManager

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """Raised when context length exceeds model limits."""

    pass


class ErrorRecoveryMiddleware:
    """Middleware to handle context overflow and other recoverable errors."""

    def __init__(self):
        """Initialize error recovery with context compaction and state management."""
        self.compactor = ContextCompactor(
            max_tokens=100000, compression_threshold=0.8
        )  # Adjusted for GPT-5's 128k context
        self.state_manager = SelectiveStateManager()
        self.max_retries = 3

    def is_context_error(self, error: Exception) -> bool:
        """Check if error is related to context length."""
        error_indicators = [
            "context length",
            "maximum context length",
            "token limit",
            "context_length_exceeded",
            "requested too many tokens",
        ]

        error_str = str(error).lower()
        return any(indicator in error_str for indicator in error_indicators)

    async def recover_from_context_error(
        self,
        state: WorkflowState,
        agent_function: Callable,
        agent_type: str,
        error: Exception,
    ) -> WorkflowState:
        """Attempt to recover from context overflow error."""
        print(f"🔄 Context error recovery for {agent_type}: {error}")

        # Recovery strategies in order of preference
        recovery_strategies = [
            self._strategy_selective_context,
            self._strategy_aggressive_compression,
            self._strategy_minimal_context,
            self._strategy_reset_with_summary,
        ]

        for i, strategy in enumerate(recovery_strategies, 1):
            try:
                print(f"🔧 Attempting recovery strategy {i}/{len(recovery_strategies)}: {strategy.__name__}")

                recovered_state = await strategy(state, agent_function, agent_type)

                print(f"✅ Recovery successful with strategy {i}")
                return recovered_state

            except Exception as recovery_error:
                if self.is_context_error(recovery_error):
                    print(f"❌ Strategy {i} failed: context still too large")
                    continue
                print(f"❌ Strategy {i} failed with different error: {recovery_error}")
                # If it's not a context error, the strategy worked but there's another issue
                raise

        # If all strategies failed
        print(f"💥 All recovery strategies failed for {agent_type}")
        state.error_message = f"Context overflow unrecoverable for {agent_type}: {error}"
        state.should_continue = False
        return state

    async def _strategy_selective_context(
        self, state: WorkflowState, agent_function: Callable, agent_type: str
    ) -> WorkflowState:
        """Strategy 1: Use selective context based on agent type."""
        selective_context = self.state_manager.prepare_for_agent(state, agent_type)

        # Create minimal state with only essential info
        minimal_state = WorkflowState(
            prompt=selective_context.get("prompt", state.prompt),
            original_prompt=selective_context.get("original_prompt", state.original_prompt),
            refinement_request=selective_context.get("refinement_request", state.refinement_request),
            refinement_iterations=state.refinement_iterations,
        )

        # Add agent-specific fields
        for key, value in selective_context.items():
            if hasattr(minimal_state, key) and key not in [
                "prompt",
                "original_prompt",
                "refinement_request",
            ]:
                setattr(minimal_state, key, value)

        return await agent_function(minimal_state)

    async def _strategy_aggressive_compression(
        self, state: WorkflowState, agent_function: Callable, agent_type: str
    ) -> WorkflowState:
        """Strategy 2: Apply aggressive compression to all state."""
        # Use more aggressive compression settings
        aggressive_compactor = ContextCompactor(
            max_tokens=80000, compression_threshold=0.6
        )  # Aggressive compression for GPT-5
        compressed_state = aggressive_compactor.compress_state(state)

        return await agent_function(compressed_state)

    async def _strategy_minimal_context(
        self, state: WorkflowState, agent_function: Callable, agent_type: str
    ) -> WorkflowState:
        """Strategy 3: Use only the absolute minimum context."""
        minimal_state = WorkflowState(
            prompt=state.prompt[:500] if state.prompt else "",  # Truncate prompt
            original_prompt=(state.original_prompt or state.prompt or "")[:200],
            refinement_iterations=state.refinement_iterations,
            should_continue=state.should_continue,
        )

        # Add only essential execution result
        if state.execution_result and agent_type in ["critic", "verification"]:
            minimal_state.execution_result = state.execution_result

        return await agent_function(minimal_state)

    async def _strategy_reset_with_summary(
        self, state: WorkflowState, agent_function: Callable, agent_type: str
    ) -> WorkflowState:
        """Strategy 4: Reset state with only a summary of progress."""
        # Create a summary of what's been done
        summary_parts = []

        if state.subtasks:
            summary_parts.append(f"Planned {len(state.subtasks)} subtasks")

        if state.execution_result and state.execution_result.success:
            summary_parts.append("Asset generation successful")

        if state.critic_analysis:
            score = state.critic_analysis.get("overall_score", "N/A")
            summary_parts.append(f"Quality score: {score}")

        if state.refinement_iterations > 0:
            summary_parts.append(f"Refinement iteration {state.refinement_iterations}")

        context_summary = ". ".join(summary_parts)

        # Create fresh state with summary
        fresh_state = WorkflowState(
            prompt=f"{state.original_prompt or state.prompt} [Context: {context_summary}]",
            original_prompt=state.original_prompt,
            refinement_iterations=state.refinement_iterations,
            should_continue=True,
        )

        # Add minimal essential data based on agent type
        if agent_type == "critic" and state.execution_result:
            fresh_state.execution_result = state.execution_result

        return await agent_function(fresh_state)


def with_error_recovery(agent_type: str):
    """Decorator to add error recovery to agent functions."""

    def decorator(agent_func: Callable) -> Callable:
        async def wrapper(state: WorkflowState) -> WorkflowState:
            recovery = ErrorRecoveryMiddleware()

            try:
                return await agent_func(state)
            except Exception as e:
                if recovery.is_context_error(e):
                    return await recovery.recover_from_context_error(state, agent_func, agent_type, e)
                else:
                    # Re-raise non-context errors
                    raise

        return wrapper

    return decorator


# Utility function for manual error handling
async def handle_context_error(
    state: WorkflowState, agent_function: Callable, agent_type: str, error: Exception
) -> WorkflowState:
    """Manually handle context error using recovery middleware."""
    recovery = ErrorRecoveryMiddleware()
    return await recovery.recover_from_context_error(state, agent_function, agent_type, error)
