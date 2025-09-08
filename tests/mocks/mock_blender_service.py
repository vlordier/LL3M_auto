"""Mock implementation of the Blender service for testing."""

from pathlib import Path
from typing import Any

from src.services.blender_service import BlenderService, ExportFormat
from src.utils.types import ExecutionResult


class MockBlenderService(BlenderService):
    """Mock implementation of the Blender service for testing."""

    def __init__(self) -> None:
        """Initialize the mock Blender service."""
        self.should_succeed = True
        self.asset_path = "/tmp/mock_asset.blend"
        self.screenshot_path = "/tmp/mock_screenshot.png"

    async def execute_script(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Simulate executing a Python script in Blender."""
        if self.should_succeed:
            return ExecutionResult(
                success=True,
                asset_path=self.asset_path,
                screenshot_path=self.screenshot_path,
                logs=["Script executed successfully."],
                errors=[],
                execution_time=0.1,
            )
        else:
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=["Script execution failed."],
                execution_time=0.1,
            )

    async def export_asset(
        self,
        output_path: Path,
        export_format: ExportFormat,
        settings: dict[str, Any] | None = None,
    ) -> Path:
        """Simulate exporting an asset from Blender."""
        if self.should_succeed:
            # Create a dummy file to simulate the export
            output_path.touch()
            return output_path
        else:
            raise RuntimeError("Failed to export asset.")
