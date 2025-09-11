"""Common utilities for agent implementations to reduce duplication."""

from typing import Any

import structlog

from ..utils.types import WorkflowState


def check_user_termination(state: WorkflowState) -> bool:
    """Check if user requested workflow termination.

    Args:
        state: Current workflow state

    Returns:
        True if user requested termination, False otherwise
    """
    return bool(state.user_feedback and state.user_feedback.upper().strip() == "TERMINATE")


def get_asset_path_safely(state: WorkflowState) -> str | None:
    """Safely extract asset path from execution result.

    Args:
        state: Current workflow state

    Returns:
        Asset path if available, None otherwise
    """
    return state.execution_result.asset_path if state.execution_result else None


def log_agent_exception(logger: structlog.BoundLogger, operation: str, error: Exception, **context: Any) -> None:
    """Standardized exception logging for agents.

    Args:
        logger: Agent's logger instance
        operation: Description of the operation that failed
        error: The exception that occurred
        **context: Additional context to include in the log
    """
    logger.exception(
        f"Agent operation failed: {operation}",
        error=str(error),
        error_type=type(error).__name__,
        **context,
    )


def create_error_response(message: str, error: Exception | None = None, **metadata: Any) -> dict[str, Any]:
    """Create standardized error response for agents.

    Args:
        message: Human-readable error message
        error: Optional exception for additional context
        **metadata: Additional metadata to include

    Returns:
        Standardized error response dictionary
    """
    response = {"success": False, "message": message, "data": None, **metadata}

    if error:
        response["error_type"] = type(error).__name__
        response["error_details"] = str(error)

    return response


def validate_workflow_state(state: WorkflowState, required_fields: list[str]) -> tuple[bool, str]:
    """Validate that workflow state has required fields.

    Args:
        state: Workflow state to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_message)
    """
    for field_name in required_fields:
        if not hasattr(state, field_name):
            return False, f"Missing required field: {field_name}"

        field_value = getattr(state, field_name)
        if field_value is None or (isinstance(field_value, str | list) and not field_value):
            return False, f"Required field is empty: {field_name}"

    return True, ""


def extract_agent_metrics(
    operation_name: str, execution_time: float, success: bool, **additional_metrics: Any
) -> dict[str, Any]:
    """Extract standardized metrics for agent operations.

    Args:
        operation_name: Name of the operation
        execution_time: Time taken in seconds
        success: Whether operation was successful
        **additional_metrics: Additional metrics to include

    Returns:
        Standardized metrics dictionary
    """
    return {
        "operation": operation_name,
        "execution_time": execution_time,
        "success": success,
        "timestamp": None,  # Will be set by monitoring system
        **additional_metrics,
    }
