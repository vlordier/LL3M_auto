"""Blender verification scripts for asset analysis and benchmarking."""

import json
from typing import Any


def create_asset_analysis_script() -> str:
    """Create Blender Python script for comprehensive asset analysis."""
    return '''
import bpy
import bmesh
import json
import sys

def analyze_asset():
    """Analyze the loaded asset."""
    results = {
        "geometry_valid": True,
        "material_count": 0,
        "polygon_count": 0,
        "vertex_count": 0,
        "has_textures": False,
        "has_lighting": False,
        "scene_bounds": [0.0, 0.0, 0.0]
    }

    try:
        # Count materials
        results["material_count"] = len(bpy.data.materials)

        # Check for textures
        results["has_textures"] = len(bpy.data.images) > 0

        # Check for lights
        results["has_lighting"] = len(bpy.data.lights) > 0

        # Analyze mesh objects
        total_polygons = 0
        total_vertices = 0
        has_invalid_geometry = False
        max_bounds = [0.0, 0.0, 0.0]

        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                # Update bounds
                if obj.dimensions:
                    for i in range(3):
                        max_bounds[i] = max(max_bounds[i], obj.dimensions[i])

                # Count polygons and vertices
                if obj.data:
                    total_polygons += len(obj.data.polygons)
                    total_vertices += len(obj.data.vertices)

                    # Check geometry validity
                    try:
                        bm = bmesh.new()
                        bm.from_mesh(obj.data)

                        # Check for non-manifold geometry
                        bmesh.ops.dissolve_degenerate(
                            bm, dist=0.0001, edges=bm.edges[:]
                        )
                        non_manifold = [v for v in bm.verts if not v.is_manifold]
                        if non_manifold:
                            has_invalid_geometry = True

                        bm.free()
                    except Exception:
                        has_invalid_geometry = True

        results["polygon_count"] = total_polygons
        results["vertex_count"] = total_vertices
        results["geometry_valid"] = not has_invalid_geometry
        results["scene_bounds"] = max_bounds

    except Exception as e:
        print(f"Analysis error: {e}", file=sys.stderr)
        results["error"] = str(e)

    # Print results as JSON
    print("ANALYSIS_RESULTS:", json.dumps(results))

if __name__ == "__main__":
    analyze_asset()
'''


def create_benchmark_script(asset_path: str) -> str:
    """Create Blender Python script for performance benchmarking."""
    return f'''
import bpy
import json
import os
import sys
import tempfile
import time

def benchmark_asset():
    """Benchmark asset performance."""
    results = {{}}

    try:
        # Load the asset to benchmark
        bpy.ops.wm.open_mainfile(filepath="{asset_path}")

        # Render benchmark
        start_time = time.time()

        # Set up basic render settings
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        temp_dir = tempfile.gettempdir()
        scene.render.filepath = os.path.join(temp_dir, 'benchmark_render.png')

        # Render
        bpy.ops.render.render(write_still=True)

        render_time = time.time() - start_time
        results["render_time"] = render_time

        # Memory usage approximation
        results["estimated_memory_mb"] = len(bpy.data.meshes) * 0.1  # Rough estimate

        # Scene complexity metrics
        results["object_count"] = len(bpy.data.objects)
        results["mesh_count"] = len(bpy.data.meshes)
        results["material_count"] = len(bpy.data.materials)

    except Exception as e:
        print(f"Benchmark error: {{e}}", file=sys.stderr)
        results["error"] = str(e)

    # Print results as JSON
    print("BENCHMARK_RESULTS:", json.dumps(results))

if __name__ == "__main__":
    benchmark_asset()
'''


def parse_analysis_results(logs: list[str]) -> dict[str, Any] | None:
    """Parse Blender analysis output from logs."""
    try:
        for line in logs:
            if line.startswith("ANALYSIS_RESULTS:"):
                json_str = line.replace("ANALYSIS_RESULTS:", "", 1).strip()
                parsed_result = json.loads(json_str)
                return parsed_result if isinstance(parsed_result, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def parse_benchmark_results(logs: list[str]) -> dict[str, Any]:
    """Parse Blender benchmark output from logs."""
    try:
        for line in logs:
            if line.startswith("BENCHMARK_RESULTS:"):
                json_str = line.replace("BENCHMARK_RESULTS:", "", 1).strip()
                parsed_result = json.loads(json_str)
                return parsed_result if isinstance(parsed_result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}
