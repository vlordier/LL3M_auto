"""Test agent base classes."""

import pytest

from src.agents.base import BaseAgent
from src.utils.types import AgentResponse, AgentType, WorkflowState


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.PLANNER

    async def process(self, state: WorkflowState) -> AgentResponse:  # noqa: ARG002
        """Process workflow state."""
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data="mock result",
            message="Mock processing complete",
            execution_time=1.0,
        )


class TestBaseAgent:
    """Test base agent functionality."""

    def test_agent_initialization(self) -> None:
        """Test agent initialization with config."""
        config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1500,
        }

        agent = MockAgent(config)

        assert agent.config == config
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 1500

    def test_agent_defaults(self) -> None:
        """Test agent initialization with default values."""
        config: dict[str, str] = {}
        agent = MockAgent(config)

        assert agent.model == "gpt-4"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000

    @pytest.mark.asyncio
    async def test_agent_process(self) -> None:
        """Test agent processing."""
        config = {"model": "gpt-3.5-turbo"}
        agent = MockAgent(config)

        state = WorkflowState(
            prompt="Test prompt",
            user_feedback=None,
            documentation="",
            generated_code="",
            execution_result=None,
            asset_metadata=None,
            error_message=None,
        )
        result = await agent.process(state)

        assert isinstance(result, AgentResponse)
        assert result.success is True
        assert result.agent_type == AgentType.PLANNER
        assert result.data == "mock result"
