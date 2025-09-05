"""Blender script wrapping and code generation utilities."""

import tempfile
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BlenderScriptWrapper:
    """Handles wrapping user code with Blender-specific execution context."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the script wrapper.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir

    def wrap_code_for_execution(
        self,
        user_code: str,
        asset_name: str = "asset",
        export_formats: list[str] | None = None,
        quality_settings: dict[str, Any] | None = None,
    ) -> str:
        """Wrap user code with execution context and export logic.

        Args:
            user_code: The user's Blender Python code
            asset_name: Name for the generated asset
            export_formats: List of export formats to generate
            quality_settings: Rendering and quality settings

        Returns:
            Complete Blender Python script ready for execution
        """
        if export_formats is None:
            export_formats = ["blend"]

        if quality_settings is None:
            quality_settings = {}

        # Generate timestamp for unique filenames
        timestamp = int(time.time())

        # Build export code for each format
        export_code_blocks = []
        for fmt in export_formats:
            export_block = self._generate_export_code(fmt, asset_name, timestamp)
            if export_block:
                export_code_blocks.append(export_block)

        export_code = "\\n".join(export_code_blocks)

        # Generate screenshot code
        screenshot_code = self._generate_screenshot_code(asset_name, timestamp)

        # Apply quality settings
        quality_code = self._generate_quality_settings_code(quality_settings)

        # Combine all parts
        wrapped_script = f'''
import bpy
import os
from pathlib import Path

# Ensure output directory exists
output_dir = Path("{self.output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

# Clear existing scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

try:
    # Apply quality settings
    {quality_code}

    # Execute user code
    {user_code}

    # Export in requested formats
    {export_code}

    # Take screenshot
    {screenshot_code}

    print("EXECUTION_SUCCESS")

except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''

        return wrapped_script

    def _generate_export_code(self, format_name: str, asset_name: str, timestamp: int) -> str:
        """Generate export code for a specific format.

        Args:
            format_name: Export format (blend, obj, fbx, etc.)
            asset_name: Name for the asset
            timestamp: Timestamp for unique filenames

        Returns:
            Python code for exporting in the specified format
        """
        format_configs = {
            "blend": {
                "extension": "blend",
                "code": """
# Save Blender file
blend_path = output_dir / f"{asset_name}_{timestamp}.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
print(f"Saved Blender file: {{blend_path}}")
                """,
            },
            "obj": {
                "extension": "obj",
                "code": """
# Export OBJ
obj_path = output_dir / f"{asset_name}_{timestamp}.obj"
bpy.ops.export_scene.obj(filepath=str(obj_path), use_selection=False)
print(f"Exported OBJ: {{obj_path}}")
                """,
            },
            "fbx": {
                "extension": "fbx",
                "code": """
# Export FBX
fbx_path = output_dir / f"{asset_name}_{timestamp}.fbx"
bpy.ops.export_scene.fbx(filepath=str(fbx_path), use_selection=False)
print(f"Exported FBX: {{fbx_path}}")
                """,
            },
            "gltf": {
                "extension": "gltf",
                "code": """
# Export glTF
gltf_path = output_dir / f"{asset_name}_{timestamp}.gltf"
bpy.ops.export_scene.gltf(filepath=str(gltf_path), export_format='GLTF_SEPARATE')
print(f"Exported glTF: {{gltf_path}}")
                """,
            },
            "stl": {
                "extension": "stl",
                "code": """
# Export STL
stl_path = output_dir / f"{asset_name}_{timestamp}.stl"
bpy.ops.export_mesh.stl(filepath=str(stl_path), use_selection=False)
print(f"Exported STL: {{stl_path}}")
                """,
            },
        }

        config = format_configs.get(format_name.lower())
        if not config:
            logger.warning("Unknown export format", format=format_name)
            return ""

        return config["code"].format(asset_name=asset_name, timestamp=timestamp)

    def _generate_screenshot_code(self, asset_name: str, timestamp: int) -> str:
        """Generate code to take a screenshot of the rendered scene.

        Args:
            asset_name: Name for the asset
            timestamp: Timestamp for unique filenames

        Returns:
            Python code for taking a screenshot
        """
        return f'''
# Take screenshot
screenshot_path = output_dir / f"{asset_name}_{timestamp}_screenshot.png"

# Set render settings
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(screenshot_path)

# Render current view
bpy.ops.render.render(write_still=True)
print(f"Screenshot saved: {{screenshot_path}}")
'''

    def _generate_quality_settings_code(self, quality_settings: dict[str, Any]) -> str:
        """Generate code to apply quality and render settings.

        Args:
            quality_settings: Dictionary of quality settings

        Returns:
            Python code to apply quality settings
        """
        if not quality_settings:
            return "# No quality settings specified"

        code_lines = ["# Apply quality settings"]

        # Render resolution
        if "resolution_x" in quality_settings:
            code_lines.append(f"bpy.context.scene.render.resolution_x = {quality_settings['resolution_x']}")
        if "resolution_y" in quality_settings:
            code_lines.append(f"bpy.context.scene.render.resolution_y = {quality_settings['resolution_y']}")

        # Render engine
        if "engine" in quality_settings:
            engine = quality_settings["engine"].upper()
            if engine in ["BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"]:
                code_lines.append(f"bpy.context.scene.render.engine = '{engine}'")

        # Sample count for Cycles
        has_samples = "samples" in quality_settings
        has_engine = "engine" in quality_settings
        is_cycles = has_engine and quality_settings["engine"].upper() == "CYCLES"
        if has_samples and is_cycles:
            code_lines.append(f"bpy.context.scene.cycles.samples = {quality_settings['samples']}")

        # Anti-aliasing
        if "anti_aliasing" in quality_settings:
            aa = quality_settings["anti_aliasing"]
            if isinstance(aa, bool) and aa:
                code_lines.append("bpy.context.scene.render.use_antialiasing = True")

        return "\\n".join(code_lines)

    def create_temp_script(self, script_content: str) -> Path:
        """Create a temporary script file with the given content.

        Args:
            script_content: The Python script content

        Returns:
            Path to the temporary script file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="blender_script_") as temp_file:
            temp_file.write(script_content)
            temp_path = Path(temp_file.name)

        logger.debug("Created temporary script", path=str(temp_path))
        return temp_path

    def cleanup_temp_script(self, script_path: Path) -> None:
        """Clean up a temporary script file.

        Args:
            script_path: Path to the script file to remove
        """
        try:
            script_path.unlink()
            logger.debug("Cleaned up temporary script", path=str(script_path))
        except Exception as e:
            logger.warning("Failed to cleanup temp script", path=str(script_path), error=str(e))
