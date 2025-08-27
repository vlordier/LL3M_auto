"""Blender execution engine for running generated Python code."""

import asyncio
import json
import tempfile
from pathlib import Path

import structlog

from ..utils.config import settings
from ..utils.types import ExecutionResult

logger = structlog.get_logger(__name__)


class BlenderExecutor:
    """Executes Python code in Blender environment."""

    def __init__(self) -> None:
        """Initialize the Blender executor."""
        self.blender_path = settings.blender.path
        self.headless = settings.blender.headless
        self.timeout = settings.blender.timeout
        self.output_dir = settings.app.output_directory

        # Ensure Blender is available
        self._validate_blender_installation()

    def _validate_blender_installation(self) -> None:
        """Validate that Blender is installed and accessible."""
        if not Path(self.blender_path).exists():
            raise RuntimeError(f"Blender not found at: {self.blender_path}")

        logger.info("Blender installation validated", path=self.blender_path)

    async def execute_code(
        self, code: str, asset_name: str = "asset", export_format: str = "blend"
    ) -> ExecutionResult:
        """Execute Blender Python code and return result."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_content = self._wrap_code_for_execution(
                    code, asset_name, export_format
                )
                script_file.write(script_content)
                script_path = script_file.name

            # Execute Blender with script
            result = await self._run_blender_script(script_path)

            # Clean up temporary file
            Path(script_path).unlink(missing_ok=True)

            execution_time = asyncio.get_event_loop().time() - start_time

            if result.success:
                logger.info(
                    "Blender execution successful",
                    asset_name=asset_name,
                    execution_time=execution_time,
                )
            else:
                logger.error(
                    "Blender execution failed",
                    asset_name=asset_name,
                    errors=result.errors,
                )

            return ExecutionResult(
                success=result.success,
                asset_path=result.asset_path,
                screenshot_path=result.screenshot_path,
                logs=result.logs,
                errors=result.errors,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error("Blender execution exception", error=str(e))

            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[str(e)],
                execution_time=execution_time,
            )

    def _wrap_code_for_execution(
        self, code: str, asset_name: str, export_format: str
    ) -> str:
        """Wrap user code with necessary Blender setup and export logic."""
        output_dir = self.output_dir

        wrapped_code = f'''
import bpy
import bmesh
import sys
import traceback
import json
from pathlib import Path

# Setup
output_dir = Path("{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

asset_path = output_dir / "{asset_name}.{export_format}"
screenshot_path = output_dir / "{asset_name}_screenshot.png"

logs = []
errors = []
success = False

try:
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Execute user code
    logs.append("Starting user code execution")

{self._indent_code(code, "    ")}

    logs.append("User code executed successfully")

    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath=str(asset_path))
    logs.append(f"Asset saved to: {{asset_path}}")

    # Take screenshot
    bpy.context.scene.render.filepath = str(screenshot_path)
    bpy.context.scene.render.resolution_x = {settings.blender.screenshot_resolution[0]}
    bpy.context.scene.render.resolution_y = {settings.blender.screenshot_resolution[1]}
    bpy.ops.render.render(write_still=True)
    logs.append(f"Screenshot saved to: {{screenshot_path}}")

    success = True
except Exception as e:
    error_msg = f"Execution error: {{str(e)}}"
    errors.append(error_msg)
    errors.append(traceback.format_exc())
    logs.append(f"Error occurred: {{error_msg}}")

# Output result as JSON for parsing
result = {{
    "success": success,
    "asset_path": str(asset_path) if success else None,
    "screenshot_path": str(screenshot_path) if success else None,
    "logs": logs,
    "errors": errors
}}

print("EXECUTION_RESULT_JSON:", json.dumps(result))
'''
        return wrapped_code

    def _indent_code(self, code: str, indent: str) -> str:
        """Indent code lines."""
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)

    async def _run_blender_script(self, script_path: str) -> ExecutionResult:
        """Run Blender with the given script."""
        cmd = [self.blender_path, "--background", "--python", script_path]

        if self.headless:
            cmd.insert(1, "--no-window")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )

            stdout_text = stdout.decode("utf-8")
            stderr_text = stderr.decode("utf-8")

            # Parse result from stdout
            result = self._parse_execution_result(stdout_text, stderr_text)

            return result

        except asyncio.TimeoutError:
            logger.error("Blender execution timed out", timeout=self.timeout)
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Execution timed out after {self.timeout} seconds"],
                execution_time=float(self.timeout),
            )
        except Exception as e:
            logger.error("Failed to run Blender", error=str(e))
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Failed to run Blender: {str(e)}"],
                execution_time=0.0,
            )

    def _parse_execution_result(self, stdout: str, stderr: str) -> ExecutionResult:
        """Parse execution result from Blender output."""
        try:
            # Look for our JSON result marker
            for line in stdout.split("\n"):
                if line.startswith("EXECUTION_RESULT_JSON:"):
                    json_str = line.replace("EXECUTION_RESULT_JSON:", "").strip()
                    result_data = json.loads(json_str)

                    return ExecutionResult(
                        success=result_data["success"],
                        asset_path=result_data.get("asset_path"),
                        screenshot_path=result_data.get("screenshot_path"),
                        logs=result_data.get("logs", []),
                        errors=result_data.get("errors", []),
                        execution_time=0.0,  # Will be set by caller
                    )
        except Exception as e:
            logger.error("Failed to parse execution result", error=str(e))

        # Fallback: parse from stderr/stdout
        return ExecutionResult(
            success="Error" not in stderr,
            asset_path=None,
            screenshot_path=None,
            logs=stdout.split("\n") if stdout else [],
            errors=stderr.split("\n") if stderr else [],
            execution_time=0.0,
        )

    async def take_screenshot(self, blend_file_path: str, output_path: str) -> bool:
        """Take a screenshot of a Blender file."""
        try:
            script_content = f'''
import bpy

# Open the blend file
bpy.ops.wm.open_mainfile(filepath="{blend_file_path}")

# Set up rendering
bpy.context.scene.render.filepath = "{output_path}"
bpy.context.scene.render.resolution_x = {settings.blender.screenshot_resolution[0]}
bpy.context.scene.render.resolution_y = {settings.blender.screenshot_resolution[1]}

# Render screenshot
bpy.ops.render.render(write_still=True)
'''

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_file.write(script_content)
                script_path = script_file.name

            result = await self._run_blender_script(script_path)
            Path(script_path).unlink(missing_ok=True)

            return result.success

        except Exception as e:
            logger.error("Failed to take screenshot", error=str(e))
            return False
