"""Abstract interface for Blender services."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..blender.export_manager import ExportFormat
from ..utils.types import ExecutionResult


class BlenderService(ABC):
    """Abstract base class for Blender services."""

    @abstractmethod
    async def execute_script(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a Python script in Blender.

        Args:
            code: The Python code to execute.
            context: A dictionary of context variables to pass to the script.
            timeout: Execution timeout in seconds.

        Returns:
            An ExecutionResult containing the outcome.
        """
        pass

    @abstractmethod
    async def export_asset(
        self,
        output_path: Path,
        export_format: ExportFormat,
        settings: dict[str, Any] | None = None,
    ) -> Path:
        """Export an asset from Blender.

        Args:
            output_path: The path to the output file.
            export_format: The format to export to.
            settings: Custom export settings.

        Returns:
            The path to the exported file.
        """
        pass
