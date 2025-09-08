"""Typed configuration objects for agents.

Replaces dictionary-based configuration with strongly typed configuration classes
for better type safety, validation, and IDE support.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseAgentConfig(BaseModel):
    """Base configuration for all agents."""

    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(ge=0.0, le=2.0, description="Temperature for LLM responses")
    max_tokens: int = Field(gt=0, description="Maximum tokens for LLM responses")
    timeout_seconds: int = Field(default=60, gt=0, description="Timeout for agent operations")

    # Retry configuration
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.0, description="Delay between retries in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level for this agent")

    model_config = ConfigDict(extra="forbid")  # Prevent accidental configuration


class PlannerAgentConfig(BaseAgentConfig):
    """Configuration for PlannerAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.7
    max_tokens: int = 5000

    # Planner-specific settings
    max_subtasks: int = Field(default=10, gt=0, description="Maximum number of subtasks to generate")
    enable_dependency_analysis: bool = Field(default=True, description="Enable subtask dependency analysis")


class RetrievalAgentConfig(BaseAgentConfig):
    """Configuration for RetrievalAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.3
    max_tokens: int = 10000

    # Retrieval-specific settings
    max_search_results: int = Field(default=5, gt=0, description="Maximum search results to retrieve")
    enable_fallback_docs: bool = Field(default=True, description="Enable fallback documentation")
    context7_enabled: bool = Field(default=True, description="Enable Context7 integration")


class CodingAgentConfig(BaseAgentConfig):
    """Configuration for CodingAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.3
    max_tokens: int = 15000

    # Coding-specific settings
    enable_code_validation: bool = Field(default=True, description="Enable Python code validation")
    enable_template_enhancement: bool = Field(default=True, description="Enable template-based code enhancement")
    safety_checks: bool = Field(default=True, description="Enable safety checks for generated code")


class CriticAgentConfig(BaseAgentConfig):
    """Configuration for CriticAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.5
    max_tokens: int = 12000

    # Critic-specific settings
    enable_visual_analysis: bool = Field(default=True, description="Enable visual analysis of screenshots")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum quality threshold")
    enable_geometry_check: bool = Field(default=True, description="Enable geometry validation")
    enable_material_check: bool = Field(default=True, description="Enable material validation")


class VerificationAgentConfig(BaseAgentConfig):
    """Configuration for VerificationAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.3
    max_tokens: int = 8000

    # Verification-specific settings
    enable_asset_validation: bool = Field(default=True, description="Enable asset file validation")
    enable_metrics_collection: bool = Field(default=True, description="Enable metrics collection")
    polygon_count_threshold: int = Field(default=100, gt=0, description="Minimum polygon count threshold")


class UserProxyAgentConfig(BaseAgentConfig):
    """Configuration for UserProxyAgent."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.6
    max_tokens: int = 10000

    # User proxy-specific settings
    enable_interactive_mode: bool = Field(default=True, description="Enable interactive user prompts")
    auto_accept_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Quality threshold for auto-acceptance"
    )
    max_user_iterations: int = Field(default=5, gt=0, description="Maximum user interaction iterations")


class MCTSAgentConfig(BaseAgentConfig):
    """Configuration for MCTSTaskDecomposer (MCTS-based task decomposition agent)."""

    model: str = "gpt-5-chat-latest"
    temperature: float = 0.7
    max_tokens: int = 15000

    # MCTS-specific settings
    max_iterations: int = Field(default=20, gt=0, description="Maximum MCTS iterations")
    max_depth: int = Field(default=3, gt=0, description="Maximum tree depth for task decomposition")
    exploration_weight: float = Field(
        default=1.414, gt=0.0, description="UCB1 exploration weight (sqrt(2) recommended)"
    )
    min_subtasks: int = Field(default=2, ge=1, description="Minimum number of subtasks to generate")
    max_subtasks: int = Field(default=8, gt=1, description="Maximum number of subtasks to generate")

    # Evaluation thresholds
    feasibility_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum feasibility score")
    complexity_penalty: float = Field(default=0.3, ge=0.0, le=1.0, description="Penalty factor for high complexity")
    completeness_bonus: float = Field(default=0.2, ge=0.0, le=1.0, description="Bonus factor for completeness")


class AgentConfigRegistry:
    """Registry for agent configurations with type safety."""

    def __init__(self) -> None:
        """Initialize the agent configuration registry."""
        self._configs: dict[str, type[BaseAgentConfig]] = {
            "planner": PlannerAgentConfig,
            "retrieval": RetrievalAgentConfig,
            "coding": CodingAgentConfig,
            "critic": CriticAgentConfig,
            "verification": VerificationAgentConfig,
            "user_proxy": UserProxyAgentConfig,
            "mcts": MCTSAgentConfig,
        }

    def get_config_class(self, agent_type: str) -> type[BaseAgentConfig]:
        """Get configuration class for an agent type."""
        config_class = self._configs.get(agent_type)
        if config_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return config_class

    def create_config(self, agent_type: str, **overrides: Any) -> BaseAgentConfig:
        """Create a typed configuration instance for an agent."""
        config_class = self.get_config_class(agent_type)
        return config_class(**overrides)

    def get_supported_agent_types(self) -> list[str]:
        """Get list of supported agent types."""
        return list(self._configs.keys())

    def register_config(self, agent_type: str, config_class: type[BaseAgentConfig]) -> None:
        """Register a new agent configuration type."""
        if not issubclass(config_class, BaseAgentConfig):
            raise TypeError("Config class must inherit from BaseAgentConfig")
        self._configs[agent_type] = config_class


# Global registry instance
agent_config_registry = AgentConfigRegistry()
