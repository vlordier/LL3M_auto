"""Base classes and interfaces for LL3M agents."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import structlog
from openai import AsyncOpenAI

from ..utils.config import settings
from ..utils.types import AgentResponse, AgentType, WorkflowState

logger = structlog.get_logger(__name__)


@runtime_checkable
class LL3MAgent(Protocol):
    """Enhanced protocol for all LL3M agents with logging and metrics."""

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable agent name."""
        ...

    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        ...

    @abstractmethod
    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input state before processing."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return agent performance metrics."""
        ...


class EnhancedBaseAgent:
    """Enhanced base class with OpenAI integration, metrics, and error handling."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize enhanced base agent."""
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.max_retries = config.get("max_retries", 3)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)

        # Metrics tracking
        self.metrics = {
            "execution_times": [],
            "total_tokens": 0,
            "requests_count": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
        }

        self.logger = structlog.get_logger(self.__class__.__name__)

    async def make_openai_request(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        """Make OpenAI API request with retry logic and error handling."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )

                # Update metrics
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, response.usage.total_tokens)

                self.logger.info(
                    "OpenAI request successful",
                    attempt=attempt + 1,
                    execution_time=execution_time,
                    tokens_used=response.usage.total_tokens,
                )

                return response.choices[0].message.content

            except Exception as e:
                self.logger.warning(
                    "OpenAI request failed",
                    attempt=attempt + 1,
                    error=str(e),
                    max_retries=self.max_retries,
                )

                if attempt == self.max_retries - 1:
                    self.metrics["failed_requests"] += 1
                    raise

                # Exponential backoff
                await asyncio.sleep(2**attempt)

        self.metrics["failed_requests"] += 1
        raise RuntimeError(
            f"Failed to complete OpenAI request after {self.max_retries} attempts"
        )

    def _update_metrics(self, execution_time: float, tokens_used: int) -> None:
        """Update agent performance metrics."""
        self.metrics["successful_requests"] += 1
        self.metrics["total_tokens_used"] += tokens_used
        
        # New metrics for test compatibility
        self.metrics["execution_times"].append(execution_time)
        self.metrics["total_tokens"] += tokens_used
        self.metrics["requests_count"] += 1

        # Update rolling average response time
        total_successful = self.metrics["successful_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            current_avg * (total_successful - 1) + execution_time
        ) / total_successful

    def get_metrics(self) -> dict[str, Any]:
        """Return current agent metrics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0.0
            ),
            "agent_type": self.agent_type.value,
            "model": self.model,
        }

    async def validate_input(self, state: WorkflowState) -> bool:
        """Default input validation - can be overridden by subclasses."""
        return state.prompt is not None and len(state.prompt.strip()) > 0

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable agent name."""
        pass

    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        pass


# Legacy alias for backward compatibility
class BaseAgent(EnhancedBaseAgent, ABC):
    """Legacy base agent class - use EnhancedBaseAgent for new implementations."""

    pass


# Protocol alias for backward compatibility
Agent = LL3MAgent
