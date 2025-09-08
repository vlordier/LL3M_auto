"""Configuration management for LL3M."""

import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_openai_api_key() -> str:
    """Get default OpenAI API key with environment awareness."""
    # Check if we're in test environment
    if os.getenv("ENVIRONMENT") == "test":
        return "sk-test-mock-key-for-testing"

    # For non-test environments, require explicit API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable. "
            "This provides clearer error messages than runtime failures."
        )
    return api_key


class LMStudioConfig(BaseSettings):
    """LM Studio local LLM configuration."""

    base_url: str = Field(default="http://localhost:1234/v1", description="LM Studio server URL")
    api_key: str = Field(default="lm-studio", description="API key for LM Studio (can be any string)")
    model: str = Field(
        default="local-model",
        description="Model name (will be auto-detected from LM Studio)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=15000, gt=0)  # GPT-5 max completion tokens
    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="LMSTUDIO_", env_file=".env", extra="ignore")


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=_get_default_openai_api_key)

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, api_key: str) -> str:
        """Validate API key format and provide helpful error messages."""
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty")

        if not (api_key.startswith(("sk-", "sk-test-"))):
            import warnings

            warnings.warn(
                f"OpenAI API key format appears invalid (got: {api_key[:10]}...). "
                "Ensure OPENAI_API_KEY environment variable is set correctly.",
                UserWarning,
                stacklevel=3,
            )

        return api_key

    model: str = Field(default="gpt-5-chat-latest", description="Default model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=15000, gt=0)  # GPT-5 max completion tokens

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")


class Context7Config(BaseSettings):
    """Context7 MCP configuration."""

    mcp_server: str = Field(default="http://localhost:8080", description="Context7 MCP server URL")
    api_key: str | None = Field(default=None, description="Context7 API key (optional for some endpoints)")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if self.api_key is None:
            import os
            import warnings

            # Only warn in production-like environments
            if os.getenv("ENVIRONMENT", "development") != "test":
                warnings.warn(
                    "Context7 API key is not configured. "
                    "Some endpoints may not be available. "
                    "Set CONTEXT7_API_KEY environment variable if needed.",
                    UserWarning,
                    stacklevel=2,
                )

    model_config = SettingsConfigDict(env_prefix="CONTEXT7_", extra="ignore")


class BlenderConfig(BaseSettings):
    """Blender configuration."""

    executable: str = Field(
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        description="Path to Blender executable",
    )
    output_dir: str = Field(
        default=str(Path.cwd() / "outputs"),
        description="Default output directory for generated assets",
    )
    timeout: int = Field(default=120, gt=0, description="Blender script execution timeout")

    model_config = SettingsConfigDict(env_prefix="BLENDER_", env_file=".env", extra="ignore")
    """Blender execution configuration."""

    path: str = Field(
        default="/Applications/Blender.app/Contents/MacOS/Blender",
        description="Path to Blender executable",
    )
    headless: bool = Field(default=True, description="Run Blender in headless mode")
    timeout: int = Field(default=300, gt=0, description="Execution timeout in seconds")
    screenshot_resolution: tuple[int, int] = Field(default=(800, 600))
    mcp_server_port: int = Field(default=3001, description="Blender MCP server port")
    mcp_server_url: str = Field(default="http://localhost:3001", description="Blender MCP server URL")

    model_config = SettingsConfigDict(env_prefix="BLENDER_", extra="ignore")


class AppConfig(BaseSettings):
    """Main application configuration."""

    env: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development"),
        description="Application environment (development, test, production)",
    )
    log_level: str = Field(default="INFO", description="Logging level")
    output_directory: Path = Field(default=Path("./outputs"), description="Output directory")
    max_refinement_iterations: int = Field(default=3, ge=1, description="Max refinements")
    min_verification_score: float = Field(
        default=7.0, ge=1.0, le=10.0, description="Minimum verification score to pass quality check"
    )
    enable_async: bool = Field(default=True, description="Enable async processing")
    development: bool = Field(default=False, description="Development mode")
    debug: bool = Field(default=False, description="Debug mode")
    use_local_llm: bool = Field(
        default_factory=lambda: os.getenv("USE_LOCAL_LLM", "false").lower() in ("true", "1", "yes"),
        description="Use local LLM (LM Studio) instead of OpenAI"
    )

    # Security settings
    allowed_origins: list[str] = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else [],
        description="Allowed CORS origins",
    )
    allowed_hosts: list[str] = Field(
        default_factory=lambda: os.getenv("ALLOWED_HOSTS", "").split(",") if os.getenv("ALLOWED_HOSTS") else [],
        description="Trusted hosts for production",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")


class Settings:
    """Global settings manager."""

    def __init__(self) -> None:
        """Initialize settings."""
        # Load from default .env or environment variables
        self.app = AppConfig()
        self.lmstudio = LMStudioConfig()
        self.context7 = Context7Config()
        self.blender = BlenderConfig()

        # Only initialize OpenAI config if not using local LLM
        if not self.app.use_local_llm:
            self.openai = OpenAIConfig()
        else:
            # Create a dummy OpenAI config with test key for compatibility
            self.openai = OpenAIConfig(api_key="sk-test-local-llm-mode")

        # Ensure output directory exists
        self.app.output_directory.mkdir(parents=True, exist_ok=True)

    def get_agent_config(self, agent_type: str) -> dict[str, Any]:
        """Get configuration for specific agent type.

        DEPRECATED: Use get_typed_agent_config() for new code.
        This method maintained for backward compatibility.
        """
        # Delegate to new typed config system and convert back to dict

        try:
            typed_config = self.get_typed_agent_config(agent_type)
            return typed_config.model_dump()
        except ValueError:
            # Fallback for unknown agent types
            return {
                "model": self.openai.model,
                "temperature": self.openai.temperature,
                "max_tokens": self.openai.max_tokens,
            }

    def get_typed_agent_config(self, agent_type: str):
        """Get strongly typed configuration for specific agent type."""
        from ..config import agent_config_registry

        # Create configuration with OpenAI settings as base
        base_overrides = {
            "model": self.openai.model,
        }

        # Add agent-specific overrides
        agent_overrides = {
            "planner": {"temperature": 0.7, "max_tokens": 5000},
            "retrieval": {"temperature": 0.3, "max_tokens": 10000},
            "coding": {"temperature": 0.3, "max_tokens": 15000},
            "critic": {"model": "gpt-5-chat-latest", "temperature": 0.5, "max_tokens": 12000},
            "verification": {"temperature": 0.3, "max_tokens": 8000},
            "user_proxy": {"model": "gpt-5-chat-latest", "temperature": 0.6, "max_tokens": 10000},
        }

        # Merge base and agent-specific overrides
        final_overrides = {**base_overrides}
        if agent_type in agent_overrides:
            final_overrides.update(agent_overrides[agent_type])

        return agent_config_registry.create_config(agent_type, **final_overrides)


settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings
