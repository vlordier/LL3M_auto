"""Base classes and interfaces for LL3M agents."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import structlog
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..utils.config import get_settings
from ..utils.llm_client import LMStudioClient, OpenAIClient
from ..utils.types import AgentResponse, AgentType, WorkflowState
from .utils import log_agent_exception

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
        self.model = config.get("model", "gpt-5-chat-latest")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 15000)  # GPT-5 max completion tokens
        self.max_retries = config.get("max_retries", 3)

        # Initialize LLM client based on configuration
        settings = get_settings()
        self.client: LMStudioClient | OpenAIClient
        if settings.app.use_local_llm:
            self.client = LMStudioClient()
        else:
            self.client = OpenAIClient()

        # Metrics tracking
        self.metrics: dict[str, Any] = {
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
        self,
        messages: Sequence[ChatCompletionMessageParam],
        model: str = "gpt-5-chat-latest",  # noqa: ARG002
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> str:
        """Make OpenAI API request with retry logic and error handling."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        for attempt in range(self.max_retries):
            try:
                response_data = await self.client.chat_completion(
                    messages=list(messages),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )

                # Update metrics
                execution_time = time.time() - start_time
                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)
                self._update_metrics(execution_time, tokens_used)

                self.logger.info(
                    "LLM request successful",
                    attempt=attempt + 1,
                    execution_time=execution_time,
                    tokens_used=tokens_used,
                )

                content: str | None = response_data.get("choices", [{}])[0].get("message", {}).get("content")
                if content is None:
                    self._raise_empty_response_error()
                else:
                    return content

            except Exception as e:
                log_agent_exception(self.logger, "OpenAI request", e, attempt=attempt + 1, max_retries=self.max_retries)

                if attempt == self.max_retries - 1:
                    self.metrics["failed_requests"] += 1
                    raise

                # Exponential backoff
                await asyncio.sleep(2**attempt)

        # This should never be reached due to the raise in the loop
        raise RuntimeError("All retry attempts exhausted")

    async def make_structured_request(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_format: type[BaseModel],
        model: str = "gpt-5-chat-latest",  # noqa: ARG002
        temperature: float = 0.7,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> BaseModel:
        """Make structured API request with fallback for all client types."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        # Use OpenAI client's structured completion if available
        if isinstance(self.client, OpenAIClient):
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.structured_completion(
                        messages=list(messages),
                        response_format=response_format,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **kwargs,
                    )

                    # Update metrics
                    execution_time = time.time() - start_time
                    self._update_metrics(execution_time, 0)

                    self.logger.info(
                        "Structured LLM request successful",
                        attempt=attempt + 1,
                        execution_time=execution_time,
                        response_type=response_format.__name__,
                    )

                    return response

                except Exception as e:
                    log_agent_exception(self.logger, "Structured OpenAI request", e, attempt=attempt + 1, max_retries=self.max_retries)

                    if attempt == self.max_retries - 1:
                        self.metrics["failed_requests"] += 1
                        raise

                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        else:
            # Fallback to regular completion with JSON schema prompt
            import json

            schema = response_format.model_json_schema()

            # Add schema instruction to the last message
            enhanced_messages = list(messages)
            enhanced_messages[-1] = {
                **enhanced_messages[-1],
                "content": f"{enhanced_messages[-1]['content']}\n\nIMPORTANT: Respond with valid JSON matching this exact schema:\n{json.dumps(schema, indent=2)}\n\nRequirements:\n- Response must contain ONLY the JSON object\n- No additional text, explanations, or markdown"
            }

            for attempt in range(self.max_retries):
                try:
                    response_text = await self.make_openai_request(
                        enhanced_messages,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **kwargs,
                    )

                    # Parse JSON from response
                    response_text = response_text.strip()

                    # Remove markdown code blocks if present
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]

                    # Extract JSON object using brace counting
                    brace_count = 0
                    start_idx = -1

                    for i, char in enumerate(response_text):
                        if char == '{':
                            if start_idx == -1:
                                start_idx = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_idx != -1:
                                json_str = response_text[start_idx:i+1]
                                break
                    else:
                        # If no complete JSON found, try parsing the whole response
                        json_str = response_text

                    # Parse and validate with Pydantic
                    data = json.loads(json_str)
                    response = response_format(**data)

                    # Update metrics
                    execution_time = time.time() - start_time
                    self._update_metrics(execution_time, 0)

                    self.logger.info(
                        "Fallback structured request successful",
                        attempt=attempt + 1,
                        execution_time=execution_time,
                        response_type=response_format.__name__,
                        client_type=type(self.client).__name__,
                    )

                    return response

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    self.logger.warning(
                        "JSON parsing failed in structured request",
                        attempt=attempt + 1,
                        error=str(e),
                        response_preview=response_text[:200] if 'response_text' in locals() else "No response",
                    )

                    if attempt == self.max_retries - 1:
                        self.metrics["failed_requests"] += 1
                        raise ValueError(f"Failed to parse structured response after {self.max_retries} attempts: {e}")

                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        # This should never be reached due to the raise in the loop
        raise RuntimeError("All retry attempts exhausted")

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

    def _raise_empty_response_error(self) -> None:
        """Raise an error for empty OpenAI response content."""
        raise ValueError("OpenAI returned empty response content")

    def get_metrics(self) -> dict[str, int | float | list[float] | str]:
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

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable agent name."""

    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""


# Legacy alias for backward compatibility
class BaseAgent(EnhancedBaseAgent, ABC):
    """Legacy base agent class - use EnhancedBaseAgent for new implementations."""


# Protocol alias for backward compatibility
Agent = LL3MAgent
