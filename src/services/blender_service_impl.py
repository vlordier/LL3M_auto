"""Concrete implementation of the Blender service."""

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

from ..blender.enhanced_executor import BlenderProcessManager
from ..blender.export_manager import ExportFormat, ExportFormatManager
from ..blender.script_wrapper import BlenderScriptWrapper
from ..utils.config import get_settings
from ..utils.types import ExecutionResult
from .blender_service import BlenderService

logger = structlog.get_logger(__name__)


class BlenderServiceImpl(BlenderService):
    """Concrete implementation of the Blender service."""

    def __init__(self) -> None:
        """Initialize the Blender service implementation."""
        self.settings = get_settings()
        self.output_dir = Path(self.settings.blender.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.process_manager = BlenderProcessManager()
        self.script_wrapper = BlenderScriptWrapper(self.output_dir)
        self.export_manager = ExportFormatManager()

    async def execute_script(
        self,
        code: str,
        context: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a Python script in Blender."""
        wrapped_code, script_path = self.script_wrapper.wrap_script(code, context)

        cmd = [
            self.settings.blender.executable,
            "--background",
            "--python",
            str(script_path),
        ]

        start_time = asyncio.get_event_loop().time()
        try:
            process_id, _ = await self.process_manager.start_process(cmd, timeout)
            stdout, stderr, returncode = await self.process_manager.wait_for_process(
                process_id, timeout
            )
            execution_time = asyncio.get_event_loop().time() - start_time

            if returncode != 0:
                return ExecutionResult(
                    success=False,
                    errors=[stderr.decode("utf-8", "ignore")],
                    execution_time=execution_time,
                    asset_path=None,
                    screenshot_path=None,
                    logs=[],
                )

            # Process stdout for structured output
            stdout_str = stdout.decode("utf-8", "ignore")
            try:
                result_data = json.loads(stdout_str)
                return ExecutionResult(
                    success=result_data.get("success", False),
                    asset_path=result_data.get("asset_path"),
                    screenshot_path=result_data.get("screenshot_path"),
                    logs=result_data.get("logs", []),
                    errors=result_data.get("errors", []),
                    execution_time=execution_time,
                )
            except json.JSONDecodeError:
                return ExecutionResult(
                    success=True,
                    logs=[stdout_str],
                    errors=[],
                    execution_time=execution_time,
                    asset_path=None,
                    screenshot_path=None,
                )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.exception("Blender script execution failed", error=str(e))
            return ExecutionResult(
                success=False,
                errors=[str(e)],
                execution_time=execution_time,
                asset_path=None,
                screenshot_path=None,
                logs=[],
            )

    async def export_asset(
        self,
        output_path: Path,
        export_format: ExportFormat,
        settings: dict[str, Any] | None = None,
    ) -> Path:
        """Export an asset from Blender."""
        export_settings = self.export_manager.get_export_settings(export_format, settings)

        code = f"""
import bpy
bpy.ops.export_scene.{export_format.value}(filepath="{output_path}", **{export_settings})
"""
        result = await self.execute_script(code)

        if not result.success:
            raise RuntimeError(f"Failed to export asset: {result.stderr}")

        return output_path
