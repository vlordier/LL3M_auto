"""Enhanced Blender execution engine with production features."""

import ast
import asyncio
import json
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog

from ..utils.config import get_settings
from ..utils.types import ExecutionResult

logger = structlog.get_logger(__name__)


class BlenderProcessManager:
    """Manages Blender process lifecycle with robust process management."""

    def __init__(self) -> None:
        """Initialize the enhanced Blender executor."""
        self.active_processes: dict[str, asyncio.subprocess.Process] = {}
        self.process_counter = 0

    async def start_process(
        self,
        cmd: list[str],
        _timeout: float = 30.0,
        process_id: str | None = None,
    ) -> tuple[str, asyncio.subprocess.Process]:
        """Start a Blender process with tracking."""
        if not process_id:
            self.process_counter += 1
            process_id = f"blender_process_{self.process_counter}"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            )

            self.active_processes[process_id] = process
            logger.debug(
                "Blender process started", process_id=process_id, pid=process.pid
            )

            return process_id, process

        except Exception as e:
            logger.error("Failed to start Blender process", error=str(e))
            raise

    async def wait_for_process(
        self, process_id: str, timeout: float = 30.0
    ) -> tuple[bytes, bytes, int]:
        """Wait for process completion with timeout and cleanup."""
        if process_id not in self.active_processes:
            raise ValueError(f"Process {process_id} not found")

        process = self.active_processes[process_id]

        try:
            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            returncode = process.returncode

            logger.debug(
                "Blender process completed",
                process_id=process_id,
                returncode=returncode,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )

            return stdout, stderr, returncode

        except TimeoutError:
            logger.warning(
                "Blender process timed out, terminating", process_id=process_id
            )
            await self._terminate_process(process_id)
            raise

        finally:
            # Cleanup process from tracking
            if process_id in self.active_processes:
                del self.active_processes[process_id]

    async def terminate_process(self, process_id: str) -> bool:
        """Gracefully terminate a process."""
        return await self._terminate_process(process_id)

    async def _terminate_process(self, process_id: str) -> bool:
        """Internal process termination with escalation."""
        if process_id not in self.active_processes:
            return False

        process = self.active_processes[process_id]

        try:
            # First, try graceful termination
            process.terminate()

            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.debug("Process terminated gracefully", process_id=process_id)
                return True
            except TimeoutError:
                # Force kill if graceful termination fails
                process.kill()
                await process.wait()
                logger.warning("Process force killed", process_id=process_id)
                return True

        except Exception as e:
            logger.error(
                "Failed to terminate process", process_id=process_id, error=str(e)
            )
            return False

        finally:
            if process_id in self.active_processes:
                del self.active_processes[process_id]

    async def cleanup_all_processes(self) -> None:
        """Cleanup all active processes."""
        if not self.active_processes:
            return

        logger.info(
            "Cleaning up active Blender processes", count=len(self.active_processes)
        )

        for process_id in list(self.active_processes.keys()):
            await self._terminate_process(process_id)


class PythonCodeValidator:
    """Validates Python code for safety and correctness."""

    ALLOWED_MODULES = {
        "bpy",
        "bmesh",
        "mathutils",
        "math",
        "random",
        "time",
        "json",
        "sys",
        "traceback",
        "pathlib",
        "os.path",
    }

    FORBIDDEN_FUNCTIONS = {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "raw_input",
        "file",
    }

    FORBIDDEN_ATTRIBUTES = {"__builtins__", "__globals__", "__locals__"}

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate Python code for safety and syntax."""
        issues = []

        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, issues

        # AST-based security analysis
        security_issues = self._analyze_ast_security(tree)
        issues.extend(security_issues)

        # Check for Blender API usage
        if not self._has_blender_api_usage(tree):
            issues.append("Code does not appear to use Blender API")

        return len(issues) == 0, issues

    def _analyze_ast_security(self, tree: ast.AST) -> list[str]:
        """Analyze AST for security issues."""
        issues = []

        for node in ast.walk(tree):
            issues.extend(self._check_forbidden_calls(node))
            issues.extend(self._check_forbidden_attributes(node))
            issues.extend(self._check_unsafe_imports(node))

        return issues

    def _check_forbidden_calls(self, node: ast.AST) -> list[str]:
        issues = []
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.FORBIDDEN_FUNCTIONS:
                    issues.append(f"Forbidden function call: {func_name}")
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in self.FORBIDDEN_FUNCTIONS:
                    issues.append(f"Forbidden method call: {attr_name}")
        return issues

    def _check_forbidden_attributes(self, node: ast.AST) -> list[str]:
        issues = []
        if isinstance(node, ast.Attribute):
            if node.attr in self.FORBIDDEN_ATTRIBUTES:
                issues.append(f"Forbidden attribute access: {node.attr}")
        return issues

    def _check_unsafe_imports(self, node: ast.AST) -> list[str]:
        issues = []
        if isinstance(node, ast.Import | ast.ImportFrom):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if module and not any(
                    module.startswith(allowed) for allowed in self.ALLOWED_MODULES
                ):
                    issues.append(f"Potentially unsafe import: {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not any(
                        alias.name.startswith(allowed)
                        for allowed in self.ALLOWED_MODULES
                    ):
                        issues.append(f"Potentially unsafe import: {alias.name}")
        return issues

    def _has_blender_api_usage(self, tree: ast.AST) -> bool:
        """Check if code uses Blender API."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check for bpy.* usage
                if isinstance(node.value, ast.Name) and node.value.id == "bpy":
                    return True
            elif isinstance(node, ast.Call):
                # Check for bmesh operations
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "bmesh"
                    ):
                        return True

        return False


class EnhancedBlenderExecutor:
    """Enhanced Blender executor with production features."""

    def __init__(self) -> None:
        """Initialize the enhanced Blender executor."""
        self.blender_path = get_settings().blender.path
        self.headless = get_settings().blender.headless
        self.timeout = get_settings().blender.timeout
        self.output_dir = get_settings().app.output_directory

        self.process_manager = BlenderProcessManager()
        self.code_validator = PythonCodeValidator()
        self.execution_history: list[dict[str, Any]] = []

        # Supported export formats
        self.export_formats = {
            "blend": {"extension": "blend", "requires_addon": False},
            "obj": {"extension": "obj", "requires_addon": False},
            "fbx": {"extension": "fbx", "requires_addon": True},
            "gltf": {"extension": "gltf", "requires_addon": True},
            "collada": {"extension": "dae", "requires_addon": True},
            "stl": {"extension": "stl", "requires_addon": False},
            "ply": {"extension": "ply", "requires_addon": False},
        }

        # Ensure Blender is available
        self._validate_blender_installation()

    def _validate_blender_installation(self) -> None:
        """Validate that Blender is installed and accessible."""
        if not Path(self.blender_path).exists():
            raise RuntimeError(f"Blender not found at: {self.blender_path}")

        logger.info("Enhanced Blender installation validated", path=self.blender_path)

    async def execute_code(
        self,
        code: str,
        asset_name: str = "asset",
        export_formats: list[str] | None = None,
        validate_code: bool = True,
        quality_settings: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute Blender Python code with enhanced features."""
        start_time = time.time()

        if export_formats is None:
            export_formats = ["blend"]

        script_path = None
        try:
            # Validate code if requested
            if validate_code:
                is_valid, validation_issues = self.code_validator.validate_code(code)
                if not is_valid:
                    return ExecutionResult(
                        success=False,
                        asset_path=None,
                        screenshot_path=None,
                        logs=["Code validation failed"],
                        errors=validation_issues,
                        execution_time=time.time() - start_time,
                    )

            # Create temporary script file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_content = self._wrap_code_for_execution(
                    code, asset_name, export_formats, quality_settings
                )
                script_file.write(script_content)
                script_path = script_file.name

            # Execute Blender with script
            result = await self._run_blender_script_enhanced(script_path)

            execution_time = time.time() - start_time

            # Update execution history
            self._update_execution_history(
                asset_name, result, execution_time, export_formats
            )

            if result.success:
                logger.info(
                    "Enhanced Blender execution successful",
                    asset_name=asset_name,
                    execution_time=execution_time,
                    export_formats=export_formats,
                )
            else:
                logger.error(
                    "Enhanced Blender execution failed",
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
            execution_time = time.time() - start_time
            logger.error("Enhanced Blender execution exception", error=str(e))

            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[str(e)],
                execution_time=execution_time,
            )
        finally:
            # Clean up temporary file
            if script_path:
                Path(script_path).unlink(missing_ok=True)

    def _wrap_code_for_execution(
        self,
        code: str,
        asset_name: str,
        export_formats: list[str],
        quality_settings: dict[str, Any] | None = None,
    ) -> str:
        """Wrap user code with enhanced setup and export logic."""
        output_dir = self.output_dir

        if quality_settings is None:
            quality_settings = {}

        # Build export operations
        export_operations = []
        for fmt in export_formats:
            if fmt in self.export_formats:
                format_info = self.export_formats[fmt]
                extension = format_info["extension"]

                if fmt == "blend":
                    export_operations.append(
                        f'    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                        f'    exported_files.append(str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                    )
                elif fmt == "obj":
                    export_operations.append(
                        f'    bpy.ops.export_scene.obj(filepath=str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                        f'    exported_files.append(str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                    )
                elif fmt == "fbx":
                    export_operations.append(
                        f'    bpy.ops.export_scene.fbx(filepath=str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                        f'    exported_files.append(str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                    )
                elif fmt == "gltf":
                    export_operations.append(
                        f'    bpy.ops.export_scene.gltf(filepath=str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                        f'    exported_files.append(str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                    )
                elif fmt == "stl":
                    export_operations.append(
                        f'    bpy.ops.export_mesh.stl(filepath=str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                        f'    exported_files.append(str(output_dir / "'
                        f'{asset_name}.{extension}"))\n'
                    )

        export_code = "".join(export_operations)

        # Quality settings
        screenshot_resolution = quality_settings.get(
            "screenshot_resolution", get_settings().blender.screenshot_resolution
        )
        render_engine = quality_settings.get("render_engine", "EEVEE")
        render_samples = quality_settings.get("render_samples", 64)

        wrapped_code = f'''
import bpy
import bmesh
import sys
import traceback
import json
import time
from pathlib import Path

# Enhanced setup
output_dir = Path("{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

screenshot_path = output_dir / "{asset_name}_screenshot.png"
exported_files = []

logs = []
errors = []
success = False
execution_start_time = time.time()

try:
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Configure render settings for quality
    scene = bpy.context.scene
    scene.render.engine = "{render_engine}"
    scene.render.resolution_x = {screenshot_resolution[0]}
    scene.render.resolution_y = {screenshot_resolution[1]}

    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = {render_samples}
    elif scene.render.engine == 'EEVEE':
        scene.eevee.taa_render_samples = {render_samples}

    logs.append(f"Render settings configured: {{scene.render.engine}}, "
                f"{{scene.render.resolution_x}}x{{scene.render.resolution_y}}")

    # Execute user code
    logs.append("Starting user code execution")

    code_start_time = time.time()
{self._indent_code(code, "    ")}
    logs.append(f"User code executed in {{time.time() - code_start_time:.2f}}s")

    # Export logic
    logs.append("Starting export operations")
    export_start_time = time.time()
{export_code}
    logs.append(
        f"Export operations completed in {{time.time() - export_start_time:.2f}}s"
    )

    # Take screenshot
    logs.append("Taking screenshot")
    bpy.context.scene.render.filepath = str(screenshot_path)
    bpy.ops.render.render(write_still=True)
    logs.append(f"Screenshot saved to {{screenshot_path}}")

    success = True

except Exception:
    error_trace = traceback.format_exc()
    errors.append(error_trace)
    logs.append("An error occurred during execution")
    success = False

finally:
    execution_time = time.time() - execution_start_time
    result = {{
        "success": success,
        "asset_path": exported_files[0] if exported_files else None,
        "screenshot_path": str(screenshot_path) if success else None,
        "logs": logs,
        "errors": errors,
        "execution_time": execution_time,
    }}
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
'''
        return wrapped_code

    async def _run_blender_script_enhanced(self, script_path: str) -> ExecutionResult:
        """Run Blender script with enhanced process management and result parsing."""
        cmd = [self.blender_path]
        if self.headless:
            cmd.append("-b")
        cmd.extend(["--python", script_path])

        process_id, _ = await self.process_manager.start_process(cmd, self.timeout)

        try:
            stdout, stderr, returncode = await self.process_manager.wait_for_process(
                process_id, self.timeout
            )

            output_dir = Path(self.output_dir)
            result_path = output_dir / "result.json"

            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                return ExecutionResult(**result_data)

            # If result.json is not found, create a failure result
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[stdout.decode(errors="ignore")],
                errors=[stderr.decode(errors="ignore")],
                execution_time=0,
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=["Blender execution timed out"],
                execution_time=self.timeout,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"An unexpected error occurred: {e}"],
                execution_time=0,
            )

    def _update_execution_history(
        self,
        asset_name: str,
        result: ExecutionResult,
        execution_time: float,
        export_formats: list[str],
    ) -> None:
        """Update the execution history with the latest result."""
        self.execution_history.append(
            {
                "timestamp": time.time(),
                "asset_name": asset_name,
                "success": result.success,
                "execution_time": execution_time,
                "export_formats": export_formats,
                "asset_path": result.asset_path,
                "screenshot_path": result.screenshot_path,
            }
        )

    def get_execution_history(self) -> list[dict[str, Any]]:
        """Return the execution history."""
        return self.execution_history

    def _indent_code(self, code: str, prefix: str) -> str:
        """Indent a block of code."""
        return "".join(prefix + line for line in code.splitlines(True))
