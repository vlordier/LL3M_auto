"""Retry and error handling utilities for the LL3M workflow."""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import structlog

from ..utils.types import WorkflowState

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic with exponential backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):  # +1 to include initial attempt
                try:
                    if attempt > 0:
                        logger.info(f"Retrying {func.__name__} attempt {attempt + 1}/{max_retries + 1}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                    # If this is the last attempt, don't sleep
                    if attempt == max_retries:
                        break

            # If we get here, all retries failed
            logger.error(
                f"All retry attempts failed for {func.__name__}",
                final_error=str(last_exception),
            )
            raise last_exception

        return wrapper

    return decorator


def inject_execution_errors(state: WorkflowState, errors: list[str]) -> None:
    """Inject errors into the execution result for testing retry mechanisms."""
    if state.execution_result:
        # Add errors to existing execution result
        existing_errors = state.execution_result.errors or []
        state.execution_result.errors = existing_errors + errors
        logger.info(f"Injected errors into execution result: {errors}")
    else:
        logger.warning("Cannot inject errors: no execution result present")


def should_retry_on_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    # Don't retry on certain critical errors
    non_retryable_errors = [
        "GraphRecursionError",
        "KeyboardInterrupt",
        "SystemExit",
        "MemoryError",
    ]

    error_name = type(error).__name__
    if error_name in non_retryable_errors:
        logger.info(f"Error {error_name} is not retryable")
        return False

    # Retry on most other errors
    return True


class WorkflowRetryHandler:
    """Handles retry logic for workflow operations."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts: dict[str, int] = {}

    async def execute_with_retry(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """Execute an operation with retry logic."""
        retry_count = self.retry_counts.get(operation_name, 0)

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Executing {operation_name}, attempt {attempt + 1}")
                result = await operation_func(*args, **kwargs)

                # Reset retry count on success
                if operation_name in self.retry_counts:
                    del self.retry_counts[operation_name]

                return result

            except Exception as e:
                retry_count += 1
                self.retry_counts[operation_name] = retry_count

                if not should_retry_on_error(e):
                    logger.exception("Non-retryable error in %s", operation_name)
                    raise

                if attempt == self.max_retries - 1:
                    logger.exception("All retries exhausted for %s", operation_name)
                    raise

                delay = self.base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1} failed for {operation_name}, retrying in {delay}s",
                    error=str(e),
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"Unexpected error: retries exhausted for {operation_name}")


# Global retry handler instance
workflow_retry_handler = WorkflowRetryHandler()


def log_workflow_state(state: WorkflowState, stage: str) -> None:
    """Log current workflow state for debugging."""
    logger.info(
        f"Workflow state at {stage}",
        refinement_iterations=state.refinement_iterations,
        needs_refinement=state.needs_refinement,
        should_continue=state.should_continue,
        has_errors=bool(state.error_message),
        execution_success=(state.execution_result.success if state.execution_result else None),
        execution_errors_count=(
            len(state.execution_result.errors) if state.execution_result and state.execution_result.errors else 0
        ),
    )
