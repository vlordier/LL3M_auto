"""Logging configuration for LL3M."""

import logging
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging() -> None:
    """Set up structured logging with Rich formatting."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.app.development
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.app.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up rich console for pretty output
    console = Console()

    # Configure rich handler for development
    if settings.app.development:
        rich_handler = RichHandler(
            console=console, show_time=True, show_path=True, markup=True
        )

        # Add rich handler to root logger for pretty development output
        logging.basicConfig(
            level=getattr(logging, settings.app.log_level.upper()),
            format="%(message)s",
            handlers=[rich_handler],
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
