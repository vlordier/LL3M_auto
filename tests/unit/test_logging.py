"""Test logging configuration."""

from typing import Any
from unittest.mock import MagicMock, patch

from src.utils.logging import get_logger, setup_logging


class TestLogging:
    """Test logging setup and configuration."""

    def test_setup_logging(self) -> None:
        """Test logging setup completes without error."""
        # This just ensures the setup function runs
        setup_logging()
        assert True  # Just verify it doesn't crash

    def test_get_logger(self) -> None:
        """Test logger creation."""
        logger = get_logger(__name__)
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_get_logger_with_different_names(self) -> None:
        """Test logger creation with different names."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")

        assert logger1 is not None
        assert logger2 is not None
        # They should be different instances but both functional
        assert hasattr(logger1, "info")
        assert hasattr(logger2, "error")

    @patch("src.utils.config.settings")
    def test_setup_logging_development_mode(self, mock_settings: Any) -> None:
        """Test logging setup in development mode."""
        mock_settings.app.development = True
        mock_settings.app.log_level = "DEBUG"

        setup_logging()
        assert True  # Just verify it doesn't crash

    @patch("src.utils.config.settings")
    def test_setup_logging_production_mode(self, mock_settings: Any) -> None:
        """Test logging setup in production mode."""
        mock_settings.app.development = False
        mock_settings.app.log_level = "INFO"

        setup_logging()
        assert True  # Just verify it doesn't crash

    @patch("src.utils.config.settings")
    @patch("pathlib.Path.mkdir")
    def test_setup_logging_creates_logs_directory(
        self, mock_mkdir: MagicMock, mock_settings: Any
    ) -> None:
        """Test that setup_logging creates logs directory."""
        mock_settings.app.development = False
        mock_settings.app.log_level = "INFO"

        setup_logging()

        # Verify that mkdir was called to create logs directory
        mock_mkdir.assert_called_once_with(exist_ok=True)
