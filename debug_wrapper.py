#!/usr/bin/env python3
"""Debug the script wrapper to see the actual output."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.blender.script_wrapper import BlenderScriptWrapper

# Test the script wrapper
wrapper = BlenderScriptWrapper("/tmp/debug_output")

# Test with problematic user code
test_code = """# Broken user code that might have been generated
try:
    import bpy
    bpy.ops.mesh.primitive_cube_add()
    # Missing except/finally block
"""

# Wrap the code
wrapped_script = wrapper.wrap_code_for_execution(
    test_code, "debug_asset", ["blend"], {"render_engine": "BLENDER_EEVEE_NEXT"}
)

print("Full wrapped script:")
print("=" * 60)
for i, line in enumerate(wrapped_script.split("\n"), 1):
    print(f"{i:2d}: {line}")
print("=" * 60)
