"""Configuration for context management in LangGraph workflows."""

from pydantic import BaseModel, Field


class ContextManagementConfig(BaseModel):
    """Configuration for context management and compression."""

    # Token limits for different models
    gpt_4_max_tokens: int = Field(8192, description="GPT-4 context window size")
    gpt_4_turbo_max_tokens: int = Field(128000, description="GPT-4 Turbo context window size")
    gpt_4o_max_tokens: int = Field(128000, description="GPT-4o context window size")
    gpt_5_max_tokens: int = Field(128000, description="GPT-5 context window size")

    # Compression thresholds
    compression_threshold: float = Field(0.75, ge=0.1, le=0.95,
                                       description="Fraction of max tokens that triggers compression")
    aggressive_compression_threshold: float = Field(0.5, ge=0.1, le=0.8,
                                                   description="Threshold for aggressive compression")

    # Context monitoring
    enable_monitoring: bool = Field(True, description="Enable context size monitoring")
    warning_threshold: float = Field(0.8, ge=0.5, le=0.95,
                                   description="Warn when context exceeds this fraction")

    # Agent-specific limits (adjusted for GPT-5's 128k context)
    coding_agent_max_tokens: int = Field(25000, description="Max tokens for coding agent")
    critic_agent_max_tokens: int = Field(20000, description="Max tokens for critic agent")
    verification_agent_max_tokens: int = Field(15000, description="Max tokens for verification agent")

    # Documentation compression (adjusted for GPT-5's 128k context)
    max_documentation_chars: int = Field(30000, description="Max chars for documentation")
    max_code_lines: int = Field(200, description="Max lines for code summaries")
    max_subtasks: int = Field(20, description="Max subtasks to retain")

    # Error recovery
    enable_error_recovery: bool = Field(True, description="Enable automatic error recovery")
    max_recovery_attempts: int = Field(3, description="Max recovery attempts per error")

    # History management
    max_conversation_history: int = Field(3, description="Max iterations to keep in history")

    def get_model_limits(self, model_name: str) -> int:
        """Get token limits for specific model."""
        model_limits = {
            "gpt-4": self.gpt_4_max_tokens,
            "gpt-4-turbo": self.gpt_4_turbo_max_tokens,
            "gpt-4o": self.gpt_4o_max_tokens,
            "gpt-4-vision-preview": self.gpt_4_max_tokens,  # Legacy
            "gpt-5": self.gpt_5_max_tokens,
            "gpt-5-mini": self.gpt_5_max_tokens,
            "gpt-5-nano": self.gpt_5_max_tokens,
        }

        return model_limits.get(model_name, self.gpt_5_max_tokens)  # Default to GPT-5

    def should_compress(self, current_tokens: int, model_name: str) -> bool:
        """Check if compression should be triggered."""
        max_tokens = self.get_model_limits(model_name)
        threshold_tokens = int(max_tokens * self.compression_threshold)
        return current_tokens > threshold_tokens

    def should_warn(self, current_tokens: int, model_name: str) -> bool:
        """Check if warning should be issued."""
        max_tokens = self.get_model_limits(model_name)
        warning_tokens = int(max_tokens * self.warning_threshold)
        return current_tokens > warning_tokens


# Default configuration instance
DEFAULT_CONTEXT_CONFIG = ContextManagementConfig()


def get_context_config() -> ContextManagementConfig:
    """Get context management configuration."""
    return DEFAULT_CONTEXT_CONFIG
