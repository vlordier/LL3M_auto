"""Test harness for agent testing with mocking and assertion utilities."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents import AgentFactory, get_agent_factory
from src.config import BaseAgentConfig
from src.utils.types import AgentResponse, AgentType, WorkflowState
from .builders import AgentResponseBuilder, WorkflowStateBuilder


class AgentTestHarness:
    """Test harness for comprehensive agent testing."""
    
    def __init__(self, agent_type: AgentType) -> None:
        """Initialize test harness for specific agent type.
        
        Args:
            agent_type: The type of agent to test
        """
        self.agent_type = agent_type
        self.mock_responses: List[AgentResponse] = []
        self.call_count = 0
        self.captured_states: List[WorkflowState] = []
        self.captured_configs: List[BaseAgentConfig] = []
        
        # Mock objects
        self.mock_openai_client = AsyncMock()
        self.mock_settings = MagicMock()
        self.mock_config = MagicMock()
        
    def mock_openai_responses(self, responses: List[str]) -> "AgentTestHarness":
        """Mock OpenAI API responses.
        
        Args:
            responses: List of response texts to return in sequence
        """
        async def mock_create(*args, **kwargs):
            if self.call_count >= len(responses):
                raise IndexError(f"No more mock responses available (call {self.call_count})")
            
            response_text = responses[self.call_count]
            self.call_count += 1
            
            # Mock OpenAI response structure
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = response_text
            
            return mock_response
        
        self.mock_openai_client.create = mock_create
        return self
    
    def mock_agent_responses(self, responses: List[AgentResponse]) -> "AgentTestHarness":
        """Mock agent process() method responses.
        
        Args:
            responses: List of AgentResponse objects to return
        """
        self.mock_responses = responses
        return self
    
    def simulate_agent_failure(self, error_message: str = "Simulated failure") -> "AgentTestHarness":
        """Configure the harness to simulate agent failure.
        
        Args:
            error_message: Error message for the failure
        """
        failed_response = AgentResponseBuilder(self.agent_type).with_failure(error_message).build()
        self.mock_responses = [failed_response]
        return self
    
    def simulate_timeout(self, timeout_seconds: float = 30.0) -> "AgentTestHarness":
        """Configure the harness to simulate timeout.
        
        Args:
            timeout_seconds: Timeout duration
        """
        async def mock_process_timeout(*args, **kwargs):
            await asyncio.sleep(timeout_seconds)
            return AgentResponseBuilder(self.agent_type).with_failure("Timeout").build()
        
        self.mock_process = mock_process_timeout
        return self
    
    def capture_metrics(self) -> "AgentTestHarness":
        """Enable metrics capture during agent execution."""
        self.metrics_enabled = True
        return self
    
    async def create_agent_with_mocks(self, config_overrides: Dict[str, Any] = None) -> Any:
        """Create an agent instance with all mocks applied.
        
        Args:
            config_overrides: Optional configuration overrides
            
        Returns:
            Agent instance with mocks applied
        """
        factory = get_agent_factory()
        
        # Apply configuration overrides
        overrides = config_overrides or {}
        overrides.setdefault("timeout_seconds", 5)  # Short timeout for tests
        
        agent = factory.create_agent(self.agent_type.value, overrides)
        
        # Apply mocks based on agent type
        if hasattr(agent, 'openai_client'):
            agent.openai_client = self.mock_openai_client
        
        # Mock process method if responses are configured
        if self.mock_responses:
            async def mock_process(state: WorkflowState) -> AgentResponse:
                self.captured_states.append(state)
                if self.call_count >= len(self.mock_responses):
                    raise IndexError("No more mock responses available")
                
                response = self.mock_responses[self.call_count]
                self.call_count += 1
                return response
            
            agent.process = mock_process
        
        return agent
    
    def assert_agent_called(self, times: int = 1) -> None:
        """Assert that agent process() was called specific number of times.
        
        Args:
            times: Expected number of calls
        """
        assert self.call_count == times, f"Expected {times} calls, got {self.call_count}"
    
    def assert_agent_received_state(self, expected_prompt: str = None, **state_attrs) -> None:
        """Assert that agent received expected state.
        
        Args:
            expected_prompt: Expected prompt in the state
            **state_attrs: Additional state attributes to check
        """
        assert self.captured_states, "No states were captured"
        
        last_state = self.captured_states[-1]
        
        if expected_prompt:
            assert last_state.prompt == expected_prompt
        
        for attr, expected_value in state_attrs.items():
            actual_value = getattr(last_state, attr)
            assert actual_value == expected_value, f"Expected {attr}={expected_value}, got {actual_value}"
    
    def assert_execution_time_under(self, max_seconds: float) -> None:
        """Assert that all responses completed within time limit.
        
        Args:
            max_seconds: Maximum allowed execution time
        """
        if not self.mock_responses:
            return
        
        for response in self.mock_responses[:self.call_count]:
            assert response.execution_time <= max_seconds, (
                f"Execution time {response.execution_time}s exceeds limit {max_seconds}s"
            )
    
    def get_captured_states(self) -> List[WorkflowState]:
        """Get all captured workflow states."""
        return self.captured_states.copy()
    
    def reset(self) -> "AgentTestHarness":
        """Reset the harness for reuse."""
        self.call_count = 0
        self.captured_states.clear()
        self.captured_configs.clear()
        return self


class MockAgentFactory:
    """Mock agent factory for testing."""
    
    def __init__(self) -> None:
        """Initialize mock factory."""
        self.created_agents: Dict[str, List[Any]] = {}
        self.call_counts: Dict[str, int] = {}
    
    def create_agent(self, agent_type: str, config_overrides: Dict[str, Any] = None) -> MagicMock:
        """Create a mock agent.
        
        Args:
            agent_type: Type of agent to create
            config_overrides: Configuration overrides
            
        Returns:
            Mock agent instance
        """
        mock_agent = MagicMock()
        mock_agent.agent_type = agent_type
        mock_agent.config = config_overrides or {}
        
        # Track creation
        if agent_type not in self.created_agents:
            self.created_agents[agent_type] = []
            self.call_counts[agent_type] = 0
        
        self.created_agents[agent_type].append(mock_agent)
        self.call_counts[agent_type] += 1
        
        return mock_agent
    
    def create_planner(self, **config_overrides: Any) -> MagicMock:
        """Create mock planner agent."""
        return self.create_agent("planner", config_overrides)
    
    def create_retrieval(self, **config_overrides: Any) -> MagicMock:
        """Create mock retrieval agent."""
        return self.create_agent("retrieval", config_overrides)
    
    def create_coding(self, **config_overrides: Any) -> MagicMock:
        """Create mock coding agent."""
        return self.create_agent("coding", config_overrides)
    
    def create_critic(self, **config_overrides: Any) -> MagicMock:
        """Create mock critic agent."""
        return self.create_agent("critic", config_overrides)
    
    def create_verification(self, **config_overrides: Any) -> MagicMock:
        """Create mock verification agent."""
        return self.create_agent("verification", config_overrides)
    
    def create_user_proxy(self, **config_overrides: Any) -> MagicMock:
        """Create mock user proxy agent."""
        return self.create_agent("user_proxy", config_overrides)
    
    def assert_agent_created(self, agent_type: str, times: int = 1) -> None:
        """Assert that specific agent type was created."""
        actual_count = self.call_counts.get(agent_type, 0)
        assert actual_count == times, f"Expected {agent_type} created {times} times, got {actual_count}"
    
    def get_created_agents(self, agent_type: str) -> List[MagicMock]:
        """Get all created agents of specific type."""
        return self.created_agents.get(agent_type, [])


@pytest.fixture
def agent_test_harness():
    """Pytest fixture for agent test harness."""
    def _create_harness(agent_type: AgentType) -> AgentTestHarness:
        return AgentTestHarness(agent_type)
    return _create_harness


@pytest.fixture
def mock_agent_factory():
    """Pytest fixture for mock agent factory."""
    return MockAgentFactory()


# Context managers for common test scenarios
class MockWorkflowExecution:
    """Context manager for mocking complete workflow execution."""
    
    def __init__(self, success_responses: Dict[AgentType, Any] = None):
        """Initialize with success responses for each agent type."""
        self.success_responses = success_responses or {}
        self.patches = []
    
    async def __aenter__(self):
        """Enter context with mocked workflow."""
        # Mock all agent types with successful responses
        for agent_type in AgentType:
            response_data = self.success_responses.get(agent_type, {})
            mock_response = AgentResponseBuilder(agent_type).with_success(response_data).build()
            
            # Mock the agent's process method
            patch_target = f"src.workflow.enhanced_graph.{agent_type.value.title()}Agent"
            patcher = patch(patch_target)
            mock_agent_class = patcher.start()
            mock_agent_class.return_value.process.return_value = mock_response
            
            self.patches.append(patcher)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup patches."""
        for patcher in self.patches:
            patcher.stop()


# Example usage functions for documentation
def create_test_scenario_successful_workflow() -> WorkflowState:
    """Example: Create a test scenario for successful workflow execution."""
    return (
        WorkflowStateBuilder()
        .with_prompt("Create a blue sphere")
        .with_subtasks(count=2)
        .with_successful_execution("/tmp/blue_sphere.blend")
        .build()
    )


def create_test_scenario_failed_execution() -> WorkflowState:
    """Example: Create a test scenario for failed execution."""
    return (
        WorkflowStateBuilder()
        .with_prompt("Invalid Blender code")
        .with_failed_execution("Syntax error in generated code")
        .with_detected_issues(count=1)
        .build()
    )