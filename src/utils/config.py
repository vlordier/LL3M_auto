"""Configuration management for LL3M."""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(
        default="", description="OpenAI API key (required for production)"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if not self.api_key or not self.api_key.startswith("sk-"):
            import os
            import warnings

            # Only warn in production-like environments
            if os.getenv("ENVIRONMENT", "development") != "test":
                warnings.warn(
                    "OpenAI API key is empty or invalid. "
                    "Set OPENAI_API_KEY environment variable for production use.",
                    UserWarning,
                    stacklevel=2,
                )

    model: str = Field(default="gpt-4", description="Default model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)

    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class Context7Config(BaseSettings):
    """Context7 MCP configuration."""

    mcp_server: str = Field(
        default="http://localhost:8080", description="Context7 MCP server URL"
    )
    api_key: str | None = Field(
        default=None, description="Context7 API key (optional for some endpoints)"
    )
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

    model_config = SettingsConfigDict(env_prefix="CONTEXT7_")


class BlenderConfig(BaseSettings):
    """Blender execution configuration."""

    path: str = Field(default="blender", description="Path to Blender executable")
    headless: bool = Field(default=True, description="Run Blender in headless mode")
    timeout: int = Field(default=300, gt=0, description="Execution timeout in seconds")
    screenshot_resolution: tuple[int, int] = Field(default=(800, 600))

    model_config = SettingsConfigDict(env_prefix="BLENDER_")


class AppConfig(BaseSettings):
    """Main application configuration."""

    log_level: str = Field(default="INFO", description="Logging level")
    output_directory: Path = Field(
        default=Path("./outputs"), description="Output directory"
    )
    max_refinement_iterations: int = Field(
        default=3, ge=1, description="Max refinements"
    )
    enable_async: bool = Field(default=True, description="Enable async processing")
    development: bool = Field(default=False, description="Development mode")
    debug: bool = Field(default=False, description="Debug mode")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


class Settings:
    """Global settings manager."""

    def __init__(self) -> None:
        """Initialize settings."""
        # Load from default .env or environment variables
        self.app = AppConfig()

        self.openai = OpenAIConfig()
        self.context7 = Context7Config()
        self.blender = BlenderConfig()

        # Ensure output directory exists
        self.app.output_directory.mkdir(parents=True, exist_ok=True)

    def get_agent_config(self, agent_type: str) -> dict[str, Any]:
        """Get configuration for specific agent type."""
        # This will be loaded from YAML files later
        default_configs = {
            "planner": {
                "model": self.openai.model,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "retrieval": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 1500,
            },
            "coding": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            "critic": {
                "model": "gpt-4-vision-preview",
                "temperature": 0.5,
                "max_tokens": 1500,
            },
            "verification": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 1000,
            },
        }

        return default_configs.get(
            agent_type,
            {
                "model": self.openai.model,
                "temperature": self.openai.temperature,
                "max_tokens": self.openai.max_tokens,
            },
        )


# Global settings instance
settings = Settings()
