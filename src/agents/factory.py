"""Agent factory for centralized agent creation and dependency injection."""

from typing import Any, Protocol

import structlog

from ..config import BaseAgentConfig, agent_config_registry
from ..utils.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class AgentProtocol(Protocol):
    """Protocol that all agents must implement."""
    
    def __init__(self, config: dict[str, Any] | BaseAgentConfig) -> None:
        """Initialize agent with configuration."""
        ...
    
    async def process(self, state: Any) -> Any:
        """Process workflow state and return agent response."""
        ...


class AgentFactory:
    """Factory for creating agents with proper dependency injection."""
    
    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize agent factory.
        
        Args:
            settings: Application settings. If None, will use get_settings()
        """
        self.settings = settings or get_settings()
        self._agent_classes: dict[str, type[AgentProtocol]] = {}
        
        # Register default agent types
        self._register_default_agents()
    
    def _register_default_agents(self) -> None:
        """Register default agent implementations."""
        # Import here to avoid circular dependencies
        from .coding import CodingAgent
        from .critic import CriticAgent
        from .planner import PlannerAgent
        from .retrieval import RetrievalAgent
        from .user_proxy import UserProxyAgent
        from .verification import VerificationAgent
        
        self._agent_classes.update({
            "planner": PlannerAgent,
            "retrieval": RetrievalAgent, 
            "coding": CodingAgent,
            "critic": CriticAgent,
            "verification": VerificationAgent,
            "user_proxy": UserProxyAgent,
        })
    
    def register_agent(self, agent_type: str, agent_class: type[AgentProtocol]) -> None:
        """Register a new agent implementation.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class implementation
        """
        if not hasattr(agent_class, 'process'):
            raise TypeError(f"Agent class {agent_class.__name__} must have a 'process' method")
        
        self._agent_classes[agent_type] = agent_class
        logger.info("Registered agent type", agent_type=agent_type, class_name=agent_class.__name__)
    
    def create_agent(self, agent_type: str, config_overrides: dict[str, Any] | None = None) -> AgentProtocol:
        """Create an agent instance with proper configuration.
        
        Args:
            agent_type: Type of agent to create (planner, coding, etc.)
            config_overrides: Optional configuration overrides
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is not registered
            TypeError: If agent class doesn't implement required interface
        """
        if agent_type not in self._agent_classes:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Registered types: {list(self._agent_classes.keys())}"
            )
        
        agent_class = self._agent_classes[agent_type]
        
        # Get base configuration
        try:
            # Try to get typed configuration first
            config = self.settings.get_typed_agent_config(agent_type)
            
            # Apply overrides if provided
            if config_overrides:
                # Convert to dict, apply overrides, then back to typed config
                config_dict = config.model_dump()
                config_dict.update(config_overrides)
                config = agent_config_registry.create_config(agent_type, **config_dict)
                
        except (ValueError, AttributeError):
            # Fall back to legacy dict-based config
            config = self.settings.get_agent_config(agent_type)
            if config_overrides:
                config.update(config_overrides)
        
        try:
            agent_instance = agent_class(config)
            logger.debug("Created agent instance", agent_type=agent_type, class_name=agent_class.__name__)
            return agent_instance
            
        except Exception as e:
            logger.exception(
                "Failed to create agent instance",
                agent_type=agent_type,
                class_name=agent_class.__name__,
                error=str(e)
            )
            raise RuntimeError(f"Failed to create {agent_type} agent: {e}") from e
    
    def create_planner(self, **config_overrides: Any) -> AgentProtocol:
        """Create a planner agent with optional configuration overrides."""
        return self.create_agent("planner", config_overrides)
    
    def create_retrieval(self, **config_overrides: Any) -> AgentProtocol:
        """Create a retrieval agent with optional configuration overrides."""
        return self.create_agent("retrieval", config_overrides)
    
    def create_coding(self, **config_overrides: Any) -> AgentProtocol:
        """Create a coding agent with optional configuration overrides."""
        return self.create_agent("coding", config_overrides)
    
    def create_critic(self, **config_overrides: Any) -> AgentProtocol:
        """Create a critic agent with optional configuration overrides."""
        return self.create_agent("critic", config_overrides)
    
    def create_verification(self, **config_overrides: Any) -> AgentProtocol:
        """Create a verification agent with optional configuration overrides."""
        return self.create_agent("verification", config_overrides)
    
    def create_user_proxy(self, **config_overrides: Any) -> AgentProtocol:
        """Create a user proxy agent with optional configuration overrides."""
        return self.create_agent("user_proxy", config_overrides)
    
    def get_supported_agent_types(self) -> list[str]:
        """Get list of supported agent types."""
        return list(self._agent_classes.keys())
    
    def clone(self) -> "AgentFactory":
        """Create a copy of this factory for testing or isolation."""
        new_factory = AgentFactory(self.settings)
        new_factory._agent_classes = self._agent_classes.copy()
        return new_factory


# Global factory instance
_default_factory: AgentFactory | None = None


def get_agent_factory(settings: Settings | None = None) -> AgentFactory:
    """Get the default agent factory instance.
    
    Args:
        settings: Optional settings override. If provided, creates a new factory.
        
    Returns:
        AgentFactory instance
    """
    global _default_factory
    
    if settings is not None:
        # Create new factory with custom settings
        return AgentFactory(settings)
    
    if _default_factory is None:
        _default_factory = AgentFactory()
    
    return _default_factory


def set_agent_factory(factory: AgentFactory) -> None:
    """Set the default agent factory (primarily for testing).
    
    Args:
        factory: Agent factory instance to set as default
    """
    global _default_factory
    _default_factory = factory