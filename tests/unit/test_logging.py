"""Test logging configuration."""

from unittest.mock import patch

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

    @patch("src.utils.config.settings")
    def test_setup_logging_development_mode(self, mock_settings) -> None:
        """Test logging setup in development mode."""
        mock_settings.app.development = True
        mock_settings.app.log_level = "DEBUG"

        setup_logging()
        assert True  # Just verify it doesn't crash
