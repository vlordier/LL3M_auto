"""Base classes and interfaces for LL3M agents."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from ..utils.types import AgentResponse, AgentType, WorkflowState


@runtime_checkable
class Agent(Protocol):
    """Protocol for all LL3M agents."""

    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        ...

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        ...


class BaseAgent(ABC):
    """Abstract base class for LL3M agents."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize base agent with configuration."""
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)

    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        pass
