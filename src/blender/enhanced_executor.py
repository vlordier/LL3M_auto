"""Enhanced Blender execution engine with production features."""

import ast
import asyncio
import json
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import structlog

from ..utils.config import settings
from ..utils.types import ExecutionResult

logger = structlog.get_logger(__name__)


class BlenderProcessManager:
    """Manages Blender process lifecycle with robust process management."""

    def __init__(self) -> None:
        """Initialize BlenderProcessManager."""
        self.active_processes: dict[str, asyncio.subprocess.Process] = {}
        self.process_counter = 0

    async def start_process(
        self, cmd: list[str], _timeout: float = 30.0, process_id: Optional[str] = None
    ) -> tuple[str, asyncio.subprocess.Process]:
        """Start a Blender process with tracking.

        Args:
            cmd: Command to execute
            _timeout: Process timeout (reserved for future use)
            process_id: Optional process identifier
        """
        # Note: _timeout parameter reserved for future implementation
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
    ) -> tuple[bytes, bytes, Optional[int]]:
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

        except asyncio.TimeoutError:
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
            except asyncio.TimeoutError:
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
            issues.extend(self._check_node_security(node))
        return issues

    def _check_node_security(self, node: ast.AST) -> list[str]:
        """Check security issues for a single AST node."""
        if isinstance(node, ast.Call):
            return self._check_function_call_security(node)
        elif isinstance(node, ast.Attribute):
            return self._check_attribute_security(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            return self._check_import_security(node)
        return []

    def _check_function_call_security(self, node: ast.Call) -> list[str]:
        """Check security for function calls."""
        issues = []
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.FORBIDDEN_FUNCTIONS:
                issues.append(f"Forbidden function call: {func_name}")
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in self.FORBIDDEN_FUNCTIONS:
                issues.append(f"Forbidden method call: {attr_name}")
        return issues

    def _check_attribute_security(self, node: ast.Attribute) -> list[str]:
        """Check security for attribute access."""
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            return [f"Forbidden attribute access: {node.attr}"]
        return []

    def _check_import_security(self, node: ast.AST) -> list[str]:
        """Check security for imports."""
        issues = []
        if isinstance(node, ast.ImportFrom):
            module = node.module
            if module and not any(
                module.startswith(allowed) for allowed in self.ALLOWED_MODULES
            ):
                issues.append(f"Potentially unsafe import: {module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not any(
                    alias.name.startswith(allowed) for allowed in self.ALLOWED_MODULES
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
        self.blender_path = settings.blender.path
        self.headless = settings.blender.headless
        self.timeout = settings.blender.timeout
        self.output_dir = settings.app.output_directory

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
        export_formats: Optional[list[str]] = None,
        validate_code: bool = True,
        quality_settings: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute Blender Python code with enhanced features."""
        start_time = time.time()

        if export_formats is None:
            export_formats = ["blend"]

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
            result = await self._run_blender_script_enhanced(script_path, asset_name)

            # Clean up temporary file
            Path(script_path).unlink(missing_ok=True)

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

    def _wrap_code_for_execution(
        self,
        code: str,
        asset_name: str,
        export_formats: list[str],
        quality_settings: Optional[dict[str, Any]] = None,
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
                    filepath = f'str(output_dir / "{asset_name}.{extension}")'
                    export_operations.append(
                        f"    bpy.ops.wm.save_as_mainfile(filepath={filepath})\n"
                        f"    exported_files.append({filepath})\n"
                    )
                elif fmt == "obj":
                    filepath = f'str(output_dir / "{asset_name}.{extension}")'
                    export_operations.append(
                        f"    bpy.ops.export_scene.obj(filepath={filepath})\n"
                        f"    exported_files.append({filepath})\n"
                    )
                elif fmt == "fbx":
                    filepath = f'str(output_dir / "{asset_name}.{extension}")'
                    export_operations.append(
                        f"    bpy.ops.export_scene.fbx(filepath={filepath})\n"
                        f"    exported_files.append({filepath})\n"
                    )
                elif fmt == "gltf":
                    filepath = f'str(output_dir / "{asset_name}.{extension}")'
                    export_operations.append(
                        f"    bpy.ops.export_scene.gltf(filepath={filepath})\n"
                        f"    exported_files.append({filepath})\n"
                    )
                elif fmt == "stl":
                    filepath = f'str(output_dir / "{asset_name}.{extension}")'
                    export_operations.append(
                        f"    bpy.ops.export_mesh.stl(filepath={filepath})\n"
                        f"    exported_files.append({filepath})\n"
                    )

        export_code = "".join(export_operations)

        # Quality settings
        screenshot_resolution = quality_settings.get(
            "screenshot_resolution", settings.blender.screenshot_resolution
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

    logs.append(
        f"Render settings configured: {{scene.render.engine}}, "
        f"{{scene.render.resolution_x}}x{{scene.render.resolution_y}}"
    )

    # Execute user code
    logs.append("Starting user code execution")

    code_start_time = time.time()
{self._indent_code(code, "    ")}

    code_execution_time = time.time() - code_start_time
    logs.append(
        f"User code executed successfully in {{code_execution_time:.2f}} seconds"
    )

    # Export files in requested formats
    export_start_time = time.time()
{export_code}

    export_time = time.time() - export_start_time
    logs.append(
        f"Assets exported in {{export_time:.2f}} seconds: {{len(exported_files)}} files"
    )

    # Take high-quality screenshot
    screenshot_start_time = time.time()
    scene.render.filepath = str(screenshot_path)

    # Ensure proper camera and lighting for screenshot
    if not bpy.data.cameras:
        bpy.ops.object.camera_add(location=(7, -7, 5))
        bpy.context.scene.camera = bpy.context.active_object

    if not bpy.data.lights:
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

    bpy.ops.render.render(write_still=True)
    screenshot_time = time.time() - screenshot_start_time
    logs.append(
        f"Screenshot saved to: {{screenshot_path}} in {{screenshot_time:.2f}} seconds"
    )

    success = True

    # Asset statistics
    mesh_count = len([obj for obj in bpy.data.objects if obj.type == 'MESH'])
    material_count = len(bpy.data.materials)
    total_vertices = sum(len(obj.data.vertices) for obj in bpy.data.objects
                        if obj.type == 'MESH' and obj.data)
    total_faces = sum(len(obj.data.polygons) for obj in bpy.data.objects
                     if obj.type == 'MESH' and obj.data)

    logs.append(f"Asset stats: {{mesh_count}} meshes, {{material_count}} materials, "
               f"{{total_vertices}} vertices, {{total_faces}} faces")

except Exception as e:
    error_msg = f"Execution error: {{str(e)}}"
    errors.append(error_msg)
    errors.append(traceback.format_exc())
    logs.append(f"Error occurred: {{error_msg}}")

# Calculate total execution time
total_execution_time = time.time() - execution_start_time

# Output enhanced result as JSON
result = {{
    "success": success,
    "asset_path": exported_files[0] if exported_files else None,
    "exported_files": exported_files,
    "screenshot_path": str(screenshot_path) if success else None,
    "logs": logs,
    "errors": errors,
    "execution_time": total_execution_time,
    "export_formats": {json.dumps(export_formats)},
    "quality_settings": {json.dumps(quality_settings)}
}}

print("ENHANCED_EXECUTION_RESULT_JSON:", json.dumps(result))
'''
        return wrapped_code

    def _indent_code(self, code: str, indent: str) -> str:
        """Indent code lines."""
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)

    async def _run_blender_script_enhanced(
        self, script_path: str, asset_name: str
    ) -> ExecutionResult:
        """Run Blender with enhanced process management."""
        cmd = [self.blender_path, "--background", "--python", script_path]

        if self.headless:
            cmd.insert(1, "--no-window")

        try:
            # Start process with tracking
            process_id, process = await self.process_manager.start_process(
                cmd, _timeout=self.timeout, process_id=f"execution_{asset_name}"
            )

            # Wait for completion
            stdout, stderr, returncode = await self.process_manager.wait_for_process(
                process_id, timeout=self.timeout
            )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            # Parse enhanced result
            result = self._parse_enhanced_execution_result(stdout_text, stderr_text)

            logger.debug(
                "Blender process completed",
                process_id=process_id,
                returncode=returncode,
                success=result.success,
            )

            return result

        except asyncio.TimeoutError:
            logger.error("Enhanced Blender execution timed out", timeout=self.timeout)
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Execution timed out after {self.timeout} seconds"],
                execution_time=float(self.timeout),
            )

        except Exception as e:
            logger.error("Failed to run enhanced Blender", error=str(e))
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Failed to run Blender: {str(e)}"],
                execution_time=0.0,
            )

    def _parse_enhanced_execution_result(
        self, stdout: str, stderr: str
    ) -> ExecutionResult:
        """Parse enhanced execution result from Blender output."""
        try:
            # Look for enhanced JSON result marker
            for line in stdout.split("\n"):
                if line.startswith("ENHANCED_EXECUTION_RESULT_JSON:"):
                    json_str = line.replace(
                        "ENHANCED_EXECUTION_RESULT_JSON:", ""
                    ).strip()
                    result_data = json.loads(json_str)

                    return ExecutionResult(
                        success=result_data["success"],
                        asset_path=result_data.get("asset_path"),
                        screenshot_path=result_data.get("screenshot_path"),
                        logs=result_data.get("logs", []),
                        errors=result_data.get("errors", []),
                        execution_time=result_data.get("execution_time", 0.0),
                        metadata={
                            "exported_files": result_data.get("exported_files", []),
                            "export_formats": result_data.get("export_formats", []),
                            "quality_settings": result_data.get("quality_settings", {}),
                        },
                    )

        except Exception as e:
            logger.error("Failed to parse enhanced execution result", error=str(e))

        # Fallback: parse from stderr/stdout
        return ExecutionResult(
            success="Error" not in stderr and "Exception" not in stderr,
            asset_path=None,
            screenshot_path=None,
            logs=stdout.split("\n") if stdout else [],
            errors=stderr.split("\n") if stderr else [],
            execution_time=0.0,
        )

    def _update_execution_history(
        self,
        asset_name: str,
        result: ExecutionResult,
        execution_time: float,
        export_formats: list[str],
    ) -> None:
        """Update execution history for monitoring."""
        self.execution_history.append(
            {
                "timestamp": time.time(),
                "asset_name": asset_name,
                "success": result.success,
                "execution_time": execution_time,
                "export_formats": export_formats,
                "errors": result.errors,
                "logs_count": len(result.logs),
            }
        )

        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

    async def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}

        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h["success"])
        avg_execution_time = (
            sum(h["execution_time"] for h in self.execution_history) / total
        )

        recent_executions = (
            self.execution_history[-10:]
            if len(self.execution_history) >= 10
            else self.execution_history
        )
        recent_success_rate = sum(1 for h in recent_executions if h["success"]) / len(
            recent_executions
        )

        return {
            "total_executions": total,
            "success_rate": successful / total,
            "recent_success_rate": recent_success_rate,
            "avg_execution_time": avg_execution_time,
            "active_processes": len(self.process_manager.active_processes),
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.process_manager.cleanup_all_processes()
        logger.info("Enhanced Blender executor cleaned up")
