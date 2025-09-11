"""Standardized error handling utilities for agents."""

import asyncio
from typing import Any

from ..utils.types import AgentResponse, AgentType

# Type alias for common empty data structures
EmptyData = dict[str, Any] | list[Any] | str


class AgentErrorHandler:
    """Standardized error handling for agents."""

    @staticmethod
    def create_error_response(
        agent_type: AgentType,
        error: Exception,
        operation_name: str,
        start_time: float,
        empty_data: EmptyData | None = None,
    ) -> AgentResponse:
        """Create a standardized error response.

        Args:
            agent_type: The type of agent that failed
            error: The exception that occurred
            operation_name: Human-readable name of the operation (e.g., "task decomposition")
            start_time: When the operation started (for timing)
            empty_data: Default empty data value for this agent type
        """
        # Determine appropriate empty data based on agent type
        if empty_data is None:
            empty_data = AgentErrorHandler._get_default_empty_data(agent_type)

        execution_time = asyncio.get_event_loop().time() - start_time

        return AgentResponse(
            agent_type=agent_type,
            success=False,
            data=empty_data,
            message=f"{operation_name.capitalize()} failed: {error!s}",
            execution_time=execution_time,
            metadata={
                "error_type": type(error).__name__,
                "operation": operation_name,
            },
        )

    @staticmethod
    def _get_default_empty_data(agent_type: AgentType) -> EmptyData:
        """Get appropriate empty data structure for agent type."""
        # Map agent types to their expected data structures
        empty_data_map: dict[AgentType, EmptyData] = {
            AgentType.PLANNER: [],  # Planner returns list of SubTasks
            AgentType.CODING: "",  # Coding returns generated code string
            AgentType.CRITIC: {},  # Critic returns analysis dict
            AgentType.VERIFICATION: {},  # Verification returns verification dict
            AgentType.USER_PROXY: {},  # User proxy returns feedback dict
            AgentType.RETRIEVAL: "",  # Retrieval returns documentation string
        }
        return empty_data_map.get(agent_type, {})

    @staticmethod
    def log_error(logger: Any, error: Exception, operation_name: str, **context: Any) -> None:
        """Log error with standardized format.

        Args:
            logger: The logger instance
            error: The exception that occurred
            operation_name: Human-readable operation name
            **context: Additional context for structured logging
        """
        logger.exception(
            f"{operation_name.capitalize()} failed",
            error=str(error),
            error_type=type(error).__name__,
            operation=operation_name,
            **context,
        )
