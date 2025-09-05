"""Export format management for Blender assets."""

from enum import Enum
from typing import Any, NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""
    
    BLEND = "blend"
    OBJ = "obj"
    FBX = "fbx"
    GLTF = "gltf"
    COLLADA = "collada"
    STL = "stl"
    PLY = "ply"


class ExportConfig(NamedTuple):
    """Configuration for an export format."""
    
    extension: str
    requires_addon: bool
    description: str
    default_settings: dict[str, Any]


class ExportFormatManager:
    """Manages export format configurations and validation."""
    
    def __init__(self) -> None:
        """Initialize export format manager with default configurations."""
        self._format_configs = {
            ExportFormat.BLEND: ExportConfig(
                extension="blend",
                requires_addon=False,
                description="Native Blender file format",
                default_settings={},
            ),
            ExportFormat.OBJ: ExportConfig(
                extension="obj",
                requires_addon=False,
                description="Wavefront OBJ format",
                default_settings={
                    "use_selection": False,
                    "use_animation": False,
                    "use_mesh_modifiers": True,
                    "use_smooth_groups": True,
                },
            ),
            ExportFormat.FBX: ExportConfig(
                extension="fbx",
                requires_addon=True,
                description="Autodesk FBX format",
                default_settings={
                    "use_selection": False,
                    "global_scale": 1.0,
                    "apply_unit_scale": True,
                    "bake_space_transform": False,
                },
            ),
            ExportFormat.GLTF: ExportConfig(
                extension="gltf",
                requires_addon=True,
                description="GL Transmission Format",
                default_settings={
                    "export_format": "GLTF_SEPARATE",
                    "export_texcoords": True,
                    "export_normals": True,
                    "export_materials": "EXPORT",
                },
            ),
            ExportFormat.COLLADA: ExportConfig(
                extension="dae",
                requires_addon=True,
                description="COLLADA Digital Asset Exchange",
                default_settings={
                    "apply_modifiers": True,
                    "export_mesh_type": 0,
                    "selected": False,
                },
            ),
            ExportFormat.STL: ExportConfig(
                extension="stl",
                requires_addon=False,
                description="STereoLithography format",
                default_settings={
                    "use_selection": False,
                    "global_scale": 1.0,
                    "use_mesh_modifiers": True,
                },
            ),
            ExportFormat.PLY: ExportConfig(
                extension="ply",
                requires_addon=False,
                description="Polygon File Format",
                default_settings={
                    "use_selection": False,
                    "use_mesh_modifiers": True,
                    "use_normals": True,
                },
            ),
        }
    
    def get_supported_formats(self) -> list[ExportFormat]:
        """Get list of all supported export formats.
        
        Returns:
            List of supported ExportFormat enums
        """
        return list(self._format_configs.keys())
    
    def get_format_config(self, format_name: str | ExportFormat) -> ExportConfig | None:
        """Get configuration for a specific export format.
        
        Args:
            format_name: Name or enum of the export format
            
        Returns:
            ExportConfig if format is supported, None otherwise
        """
        if isinstance(format_name, str):
            try:
                format_enum = ExportFormat(format_name.lower())
            except ValueError:
                logger.warning("Unknown export format", format=format_name)
                return None
        else:
            format_enum = format_name
        
        return self._format_configs.get(format_enum)
    
    def validate_formats(self, format_names: list[str]) -> list[ExportFormat]:
        """Validate and convert format names to ExportFormat enums.
        
        Args:
            format_names: List of format names to validate
            
        Returns:
            List of valid ExportFormat enums
        """
        valid_formats = []
        
        for format_name in format_names:
            try:
                format_enum = ExportFormat(format_name.lower())
                valid_formats.append(format_enum)
            except ValueError:
                logger.warning("Skipping unknown export format", format=format_name)
                continue
        
        if not valid_formats:
            logger.info("No valid formats provided, defaulting to BLEND")
            valid_formats = [ExportFormat.BLEND]
        
        return valid_formats
    
    def get_required_addons(self, formats: list[ExportFormat]) -> list[str]:
        """Get list of Blender addons required for the given formats.
        
        Args:
            formats: List of export formats
            
        Returns:
            List of addon names that need to be enabled
        """
        required_addons = set()
        
        addon_mapping = {
            ExportFormat.FBX: "io_scene_fbx",
            ExportFormat.GLTF: "io_scene_gltf2",
            ExportFormat.COLLADA: "io_scene_collada",
        }
        
        for format_enum in formats:
            config = self._format_configs.get(format_enum)
            if config and config.requires_addon:
                addon_name = addon_mapping.get(format_enum)
                if addon_name:
                    required_addons.add(addon_name)
        
        return list(required_addons)
    
    def generate_addon_enable_code(self, formats: list[ExportFormat]) -> str:
        """Generate Python code to enable required addons.
        
        Args:
            formats: List of export formats that will be used
            
        Returns:
            Python code to enable necessary addons
        """
        required_addons = self.get_required_addons(formats)
        
        if not required_addons:
            return "# No addons required for selected formats"
        
        code_lines = ["# Enable required addons"]
        code_lines.append("import addon_utils")
        
        for addon in required_addons:
            code_lines.extend([
                f"if not addon_utils.check('{addon}')[1]:",
                f"    addon_utils.enable('{addon}')",
                f"    print(f'Enabled addon: {addon}')",
            ])
        
        return "\\n".join(code_lines)
    
    def get_format_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all supported formats.
        
        Returns:
            Dictionary with format information for display/documentation
        """
        format_info = {}
        
        for format_enum, config in self._format_configs.items():
            format_info[format_enum.value] = {
                "extension": config.extension,
                "requires_addon": config.requires_addon,
                "description": config.description,
                "default_settings": config.default_settings,
            }
        
        return format_info