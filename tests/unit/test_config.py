"""Test configuration management."""

from pathlib import Path
from unittest.mock import patch

from src.utils.config import AppConfig, OpenAIConfig, Settings


class TestSettings:
    """Test settings management."""

    def test_app_config_defaults(self) -> None:
        """Test default app configuration."""
        config = AppConfig()

        assert config.log_level == "INFO"
        assert config.output_directory == Path("./outputs")
        assert config.max_refinement_iterations == 3
        assert config.enable_async is True
        assert config.development is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4"})
    def test_openai_config_from_env(self) -> None:
        """Test OpenAI config from environment variables."""
        config = OpenAIConfig()

        assert config.api_key == "test-key"
        assert config.model == "gpt-4"

    def test_settings_initialization(self) -> None:
        """Test settings initialization."""
        settings = Settings()

        assert settings.app is not None
        assert settings.openai is not None
        assert settings.context7 is not None
        assert settings.blender is not None

    def test_get_agent_config(self) -> None:
        """Test agent configuration retrieval."""
        settings = Settings()

        planner_config = settings.get_agent_config("planner")
        assert "model" in planner_config
        assert "temperature" in planner_config
        assert "max_tokens" in planner_config

        # Test unknown agent type
        unknown_config = settings.get_agent_config("unknown")
        assert "model" in unknown_config


class TestTypes:
    """Test type definitions."""

    def test_subtask_creation(self) -> None:
        """Test SubTask model creation."""
        from src.utils.types import SubTask, TaskType

        subtask = SubTask(
            id="test-1", type=TaskType.GEOMETRY, description="Create a cube"
        )

        assert subtask.id == "test-1"
        assert subtask.type == TaskType.GEOMETRY
        assert subtask.description == "Create a cube"
        assert subtask.priority == 1
        assert subtask.dependencies == []

    def test_execution_result_creation(self) -> None:
        """Test ExecutionResult model creation."""
        from src.utils.types import ExecutionResult

        result = ExecutionResult(
            success=True,
            asset_path="/path/to/asset.blend",
            screenshot_path=None,
            execution_time=2.5,
        )

        assert result.success is True
        assert result.asset_path == "/path/to/asset.blend"
        assert result.execution_time == 2.5
        assert result.logs == []
        assert result.errors == []
