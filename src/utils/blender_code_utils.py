"""Blender code generation and validation utilities."""

import ast
import re
from typing import Any


class BlenderCodeValidator:
    """Validates Blender Python code for safety and correctness."""

    SAFE_MODULES = {
        "bpy",
        "bmesh",
        "mathutils",
        "bpy_extras",
        "bl_ui",
        "math",
        "random",
        "time",
        "datetime",
        "json",
        "os.path",
    }

    DANGEROUS_FUNCTIONS = {
        "exec",
        "eval",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "hasattr",
    }

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate Blender Python code for safety and syntax."""
        errors = []

        try:
            # Parse the code to check syntax
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors

        # Check for dangerous patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.DANGEROUS_FUNCTIONS:
                        errors.append(f"Dangerous function call: {node.func.id}")

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_safe_module(alias.name):
                        errors.append(f"Unsafe module import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_safe_module(node.module):
                    errors.append(f"Unsafe module import: {node.module}")

        return len(errors) == 0, errors

    def _is_safe_module(self, module_name: str) -> bool:
        """Check if a module is safe to import."""
        return any(module_name.startswith(safe) for safe in self.SAFE_MODULES)


class BlenderCodeGenerator:
    """Generates Blender Python code from high-level descriptions."""

    TEMPLATES = {
        "cube": """
import bpy

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create cube
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "{name}"

# Apply material if specified
{material_code}
""",
        "sphere": """
import bpy

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
sphere = bpy.context.active_object
sphere.name = "{name}"

# Apply material if specified
{material_code}
""",
        "material": """
# Create material
material = bpy.data.materials.new(name="{material_name}")
material.use_nodes = True

# Get the principled BSDF node
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = ({r}, {g}, {b}, 1.0)  # Base Color

# Assign material to active object
if bpy.context.active_object:
    if bpy.context.active_object.data.materials:
        bpy.context.active_object.data.materials[0] = material
    else:
        bpy.context.active_object.data.materials.append(material)
""",
    }

    def generate_object_code(
        self,
        shape: str,
        name: str = "GeneratedObject",
        color: tuple[float, float, float] | None = None,
    ) -> str:
        """Generate code to create a basic 3D object."""
        if shape.lower() not in self.TEMPLATES:
            raise ValueError(f"Unsupported shape: {shape}")

        template = self.TEMPLATES[shape.lower()]

        # Generate material code if color is specified
        material_code = ""
        if color:
            material_code = self.TEMPLATES["material"].format(
                material_name=f"{name}_Material", r=color[0], g=color[1], b=color[2]
            )

        return template.format(name=name, material_code=material_code)

    def generate_complex_code(self, description: str, **kwargs) -> str:
        """Generate complex Blender code from natural language description."""
        # This is a simplified version - in production would use LLM

        description_lower = description.lower()

        # Extract shape
        if "cube" in description_lower or "box" in description_lower:
            shape = "cube"
        elif "sphere" in description_lower or "ball" in description_lower:
            shape = "sphere"
        else:
            shape = "cube"  # Default

        # Extract color
        color = self._extract_color_from_description(description)

        # Extract name
        name = kwargs.get("name", "GeneratedObject")

        return self.generate_object_code(shape, name, color)

    def _extract_color_from_description(
        self, description: str
    ) -> tuple[float, float, float] | None:
        """Extract color information from text description."""
        color_map = {
            "red": (0.8, 0.2, 0.2),
            "blue": (0.2, 0.2, 0.8),
            "green": (0.2, 0.8, 0.2),
            "yellow": (0.8, 0.8, 0.2),
            "purple": (0.8, 0.2, 0.8),
            "orange": (0.8, 0.5, 0.2),
            "pink": (0.8, 0.2, 0.6),
            "cyan": (0.2, 0.8, 0.8),
            "white": (0.9, 0.9, 0.9),
            "black": (0.1, 0.1, 0.1),
            "gray": (0.5, 0.5, 0.5),
            "grey": (0.5, 0.5, 0.5),
        }

        description_lower = description.lower()
        for color_name, rgb in color_map.items():
            if color_name in description_lower:
                return rgb

        return None


def extract_blender_objects(code: str) -> list[dict[str, Any]]:
    """Extract information about Blender objects created in code."""
    objects = []

    # Simple pattern matching for object creation
    patterns = [
        r"bpy\.ops\.mesh\.primitive_(\w+)_add\([^)]*location=\(([^)]+)\)",
        r"(\w+)\s*=\s*bpy\.context\.active_object",
        r"(\w+)\.name\s*=\s*[\"']([^\"']+)[\"']",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, code)
        for match in matches:
            if "primitive" in pattern:
                objects.append({"type": match.group(1), "location": match.group(2)})

    return objects


def format_blender_code(code: str) -> str:
    """Format Blender Python code with proper indentation and structure."""
    try:
        ast.parse(code)
        # In a real implementation, would use a code formatter
        return code  # Simplified for now
    except SyntaxError:
        return code  # Return original if parsing fails


def optimize_blender_code(code: str) -> str:
    """Optimize Blender Python code for performance."""
    # Remove redundant operations
    optimizations = [
        # Remove duplicate select_all calls
        (
            r"bpy\.ops\.object\.select_all\(action='SELECT'\)\s*\n\s*bpy\.ops\.object\.select_all\(action='SELECT'\)",
            r"bpy.ops.object.select_all(action='SELECT')",
        ),
        # Combine material assignments
        (
            r"(material = bpy\.data\.materials\.new\([^)]+\))\s*\n\s*(material\.use_nodes = True)",
            r"\1\n\2",
        ),
    ]

    optimized_code = code
    for pattern, replacement in optimizations:
        optimized_code = re.sub(
            pattern, replacement, optimized_code, flags=re.MULTILINE
        )

    return optimized_code
