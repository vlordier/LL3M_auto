#!/usr/bin/env python3
"""Generate 3D assets using hierarchical prompts from YAML test data."""

import argparse
import asyncio
import json

# Add the project root to the Python path
import sys
import time
from pathlib import Path
from typing import Any

import structlog
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.orchestrator import LL3MOrchestrator

logger = structlog.get_logger(__name__)
console = Console()


class HierarchicalAssetGenerator:
    """Generate 3D assets using hierarchical prompts."""

    def __init__(self, output_dir: Path | None = None, level_filter: int | None = None):
        """Initialize the generator."""
        self.project_root = project_root
        self.output_dir = output_dir or (project_root / "outputs" / "hierarchical_tests")
        self.level_filter = level_filter

        # Load test prompts
        yaml_path = project_root / "tests" / "data" / "hierarchical_prompts.yaml"
        with open(yaml_path) as f:
            self.test_data = yaml.safe_load(f)

        # Initialize orchestrator
        self.orchestrator = LL3MOrchestrator()

        # Create output directory structure
        self.setup_output_directories()

    def setup_output_directories(self):
        """Create organized output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each level
        for level in range(1, 6):
            level_dir = self.output_dir / f"level_{level}"
            level_dir.mkdir(exist_ok=True)

            # Create asset type subdirectories
            (level_dir / "code").mkdir(exist_ok=True)
            (level_dir / "renders").mkdir(exist_ok=True)
            (level_dir / "blends").mkdir(exist_ok=True)
            (level_dir / "exports").mkdir(exist_ok=True)
            (level_dir / "logs").mkdir(exist_ok=True)

    def get_prompts_for_level(self, level: int) -> list[dict[str, Any]]:
        """Get prompts for a specific complexity level."""
        level_map = {
            1: "level_1_basic_geometry",
            2: "level_2_simple_objects",
            3: "level_3_textured_objects",
            4: "level_4_lit_scenes",
            5: "level_5_animated_complex"
        }

        level_name = level_map.get(level)
        if not level_name:
            return []

        return self.test_data["test_prompts"].get(level_name, [])

    async def generate_asset(self, prompt_data: dict[str, Any], level: int, prompt_index: int) -> dict[str, Any]:
        """Generate a single 3D asset from prompt data."""
        prompt = prompt_data["prompt"]
        description = prompt_data.get("description", "No description")

        console.print(f"\n🎯 Generating Level {level} Asset #{prompt_index + 1}")
        console.print(f"   Prompt: {prompt}")
        console.print(f"   Description: {description}")

        # Create asset-specific output directory
        asset_name = f"level_{level}_asset_{prompt_index + 1:02d}"
        asset_dir = self.output_dir / f"level_{level}" / asset_name
        asset_dir.mkdir(exist_ok=True)

        start_time = time.time()

        try:
            # Generate the asset using the orchestrator
            result = await self.orchestrator.generate_asset(
                prompt=prompt,
                export_formats=['blend', 'obj', 'gltf'],
                asset_name=asset_name,
                output_directory=str(asset_dir)
            )

            generation_time = time.time() - start_time

            # Process and organize results
            asset_result = await self._process_generation_result(
                result, prompt_data, asset_dir, level, prompt_index, generation_time
            )

            console.print(f"   ✅ Generated in {generation_time:.2f}s")
            return asset_result

        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)

            console.print(f"   ❌ Failed after {generation_time:.2f}s: {error_msg}")

            # Log error details
            error_log = asset_dir / "error.json"
            error_data = {
                "prompt": prompt,
                "error": error_msg,
                "generation_time": generation_time,
                "level": level,
                "prompt_index": prompt_index
            }

            with open(error_log, "w") as f:
                json.dump(error_data, f, indent=2)

            return {
                "success": False,
                "error": error_msg,
                "generation_time": generation_time,
                "prompt": prompt,
                "level": level
            }

    async def _process_generation_result(
        self,
        result: dict[str, Any],
        prompt_data: dict[str, Any],
        asset_dir: Path,
        level: int,
        prompt_index: int,
        generation_time: float
    ) -> dict[str, Any]:
        """Process and organize generation results."""
        # Create metadata file
        metadata = {
            "prompt": prompt_data["prompt"],
            "description": prompt_data.get("description", ""),
            "expected_complexity": prompt_data.get("expected_complexity", level),
            "expected_tasks": prompt_data.get("expected_tasks", []),
            "level": level,
            "prompt_index": prompt_index,
            "generation_time": generation_time,
            "timestamp": time.time(),
            "result": result
        }

        # Save metadata (convert Pydantic models to dicts for JSON serialization)
        metadata_file = asset_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=self._serialize_pydantic)

        # Organize generated files
        files_organized = await self._organize_generated_files(result, asset_dir, level)

        # Generate additional export formats if .blend file exists
        blend_file = asset_dir / "asset.blend"
        if blend_file.exists():
            await self._export_additional_formats(blend_file, asset_dir)

        return {
            "success": result.get("success", False),
            "prompt": prompt_data["prompt"],
            "level": level,
            "generation_time": generation_time,
            "asset_dir": str(asset_dir),
            "files_organized": files_organized,
            "metadata_file": str(metadata_file)
        }

    async def _organize_generated_files(self, result: dict[str, Any], asset_dir: Path, level: int) -> dict[str, list[str]]:
        """Organize generated files into appropriate subdirectories."""
        organized_files = {
            "code": [],
            "renders": [],
            "blends": [],
            "exports": [],
            "other": []
        }

        # Move files based on extension
        for file_path in asset_dir.glob("*"):
            if file_path.is_file() and file_path.name not in ["metadata.json", "error.json"]:
                suffix = file_path.suffix.lower()

                if suffix == ".py":
                    dest_dir = asset_dir / "code"
                    organized_files["code"].append(file_path.name)
                elif suffix in [".png", ".jpg", ".jpeg", ".exr", ".hdr"]:
                    dest_dir = asset_dir / "renders"
                    organized_files["renders"].append(file_path.name)
                elif suffix == ".blend":
                    dest_dir = asset_dir / "blends"
                    organized_files["blends"].append(file_path.name)
                elif suffix in [".obj", ".fbx", ".gltf", ".glb", ".ply", ".stl", ".dae"]:
                    dest_dir = asset_dir / "exports"
                    organized_files["exports"].append(file_path.name)
                else:
                    dest_dir = asset_dir / "other"
                    organized_files["other"].append(file_path.name)

                dest_dir.mkdir(exist_ok=True)
                dest_file = dest_dir / file_path.name
                if not dest_file.exists():
                    file_path.rename(dest_file)

        return organized_files

    async def _export_additional_formats(self, blend_file: Path, asset_dir: Path):
        """Export the .blend file to additional 3D formats."""
        export_dir = asset_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        # Define export formats
        export_formats = [
            ("obj", "Wavefront OBJ"),
            ("fbx", "Autodesk FBX"),
            ("gltf", "glTF 2.0"),
            ("ply", "Stanford PLY"),
            ("stl", "STL")
        ]

        # Generate Blender export script
        export_script = self._generate_export_script(blend_file, export_dir, export_formats)
        script_file = asset_dir / "export_script.py"

        with open(script_file, "w") as f:
            f.write(export_script)

        # Try to run Blender export (if Blender is available)
        try:
            import subprocess

            # Detect Blender path
            blender_path = self._find_blender_executable()
            if blender_path:
                cmd = [
                    blender_path,
                    str(blend_file),
                    "--background",
                    "--python", str(script_file)
                ]

                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                console.print("   📦 Exported to additional formats")
            else:
                console.print("   ⚠️  Blender not found - skipping format exports")

        except Exception as e:
            console.print(f"   ⚠️  Export failed: {str(e)}")

    def _find_blender_executable(self) -> str | None:
        """Find Blender executable on the system."""
        import platform
        import shutil

        system = platform.system()

        if system == "Darwin":  # macOS
            blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
            if Path(blender_path).exists():
                return blender_path
        elif system == "Linux":
            blender_path = shutil.which("blender")
            if blender_path:
                return blender_path
        elif system == "Windows":
            # Common Windows install paths
            possible_paths = [
                "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
                "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    return path

        return None

    def _generate_export_script(self, blend_file: Path, export_dir: Path, formats: list[tuple]) -> str:
        """Generate Blender Python script for exporting to multiple formats."""
        script = '''
import bpy
import bmesh
from pathlib import Path

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Load the blend file (already loaded by command line)
export_dir = Path(r"{export_dir}")
blend_name = "{blend_name}"

# Select all mesh objects
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)

if bpy.context.selected_objects:
    # Export to different formats
    formats_to_export = {formats}

    for ext, description in formats_to_export:
        try:
            output_file = export_dir / f"{blend_name}.{ext}"

            if ext == "obj":
                bpy.ops.export_scene.obj(
                    filepath=str(output_file),
                    use_selection=True,
                    use_materials=True,
                    use_uvs=True
                )
            elif ext == "fbx":
                bpy.ops.export_scene.fbx(
                    filepath=str(output_file),
                    use_selection=True,
                    embed_textures=True
                )
            elif ext == "gltf":
                bpy.ops.export_scene.gltf(
                    filepath=str(output_file),
                    use_selection=True,
                    export_materials='EXPORT'
                )
            elif ext == "ply":
                bpy.ops.export_mesh.ply(
                    filepath=str(output_file),
                    use_selection=True
                )
            elif ext == "stl":
                bpy.ops.export_mesh.stl(
                    filepath=str(output_file),
                    use_selection=True
                )

            print(f"Exported {description} to {output_file}")

        except Exception as e:
            print(f"Failed to export {description}: {e}")
else:
    print("No mesh objects found to export")
'''.format(
            export_dir=export_dir,
            blend_name=blend_file.stem,
            formats=formats
        )

        return script

    async def run_generation(self, start_level: int = 1, end_level: int = 5, max_per_level: int = None) -> dict[str, Any]:
        """Run asset generation for specified levels."""
        console.print(Panel("🚀 Starting Hierarchical 3D Asset Generation", border_style="blue"))
        console.print(f"Output directory: {self.output_dir}")

        results = {
            "total_generated": 0,
            "total_failed": 0,
            "total_time": 0,
            "levels": {}
        }

        overall_start = time.time()

        for level in range(start_level, end_level + 1):
            if self.level_filter and level != self.level_filter:
                continue

            console.print(f"\n{'='*60}")
            console.print(f"🎯 LEVEL {level} GENERATION")
            console.print(f"{'='*60}")

            prompts = self.get_prompts_for_level(level)
            if max_per_level:
                prompts = prompts[:max_per_level]

            if not prompts:
                console.print(f"No prompts found for level {level}")
                continue

            level_results = {
                "generated": 0,
                "failed": 0,
                "time": 0,
                "assets": []
            }

            level_start = time.time()

            for i, prompt_data in enumerate(prompts):
                asset_result = await self.generate_asset(prompt_data, level, i)

                level_results["assets"].append(asset_result)
                level_results["time"] += asset_result.get("generation_time", 0)

                if asset_result.get("success", False):
                    level_results["generated"] += 1
                    results["total_generated"] += 1
                else:
                    level_results["failed"] += 1
                    results["total_failed"] += 1

            level_end = time.time()
            level_total_time = level_end - level_start
            level_results["total_time"] = level_total_time

            results["levels"][level] = level_results
            results["total_time"] += level_total_time

            # Level summary
            console.print(f"\n📊 LEVEL {level} SUMMARY:")
            console.print(f"   Generated: {level_results['generated']}")
            console.print(f"   Failed: {level_results['failed']}")
            console.print(f"   Total Time: {level_total_time:.2f}s")
            console.print(f"   Avg Time: {level_results['time'] / len(prompts):.2f}s per asset")

        overall_end = time.time()
        results["total_time"] = overall_end - overall_start

        # Save overall results
        results_file = self.output_dir / "generation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=self._serialize_pydantic)

        self._display_final_summary(results)

        return results

    def _serialize_pydantic(self, obj):
        """Custom JSON serializer for Pydantic models."""
        from datetime import datetime

        from pydantic import BaseModel

        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _display_final_summary(self, results: dict[str, Any]):
        """Display final generation summary."""
        console.print(f"\n{'='*60}")
        console.print("🏁 FINAL GENERATION SUMMARY")
        console.print(f"{'='*60}")

        table = Table()
        table.add_column("Level", justify="center")
        table.add_column("Generated", justify="center")
        table.add_column("Failed", justify="center")
        table.add_column("Success Rate", justify="center")
        table.add_column("Avg Time", justify="center")

        for level, level_data in results["levels"].items():
            total_assets = level_data["generated"] + level_data["failed"]
            success_rate = (level_data["generated"] / total_assets * 100) if total_assets > 0 else 0
            avg_time = level_data["time"] / total_assets if total_assets > 0 else 0

            table.add_row(
                str(level),
                str(level_data["generated"]),
                str(level_data["failed"]),
                f"{success_rate:.1f}%",
                f"{avg_time:.1f}s"
            )

        console.print(table)

        console.print("\n📈 OVERALL STATISTICS:")
        console.print(f"   Total Generated: {results['total_generated']}")
        console.print(f"   Total Failed: {results['total_failed']}")
        console.print(f"   Overall Success Rate: {(results['total_generated'] / (results['total_generated'] + results['total_failed']) * 100):.1f}%")
        console.print(f"   Total Time: {results['total_time']:.2f}s")
        console.print(f"   Output Directory: {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate 3D assets using hierarchical prompts")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5],
                       help="Generate assets for a specific complexity level only")
    parser.add_argument("--start-level", type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help="Start generation from this level (default: 1)")
    parser.add_argument("--end-level", type=int, default=5, choices=[1, 2, 3, 4, 5],
                       help="End generation at this level (default: 5)")
    parser.add_argument("--max-per-level", type=int,
                       help="Maximum number of assets to generate per level")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for generated assets")

    args = parser.parse_args()

    try:
        generator = HierarchicalAssetGenerator(
            output_dir=args.output_dir,
            level_filter=args.level
        )

        results = asyncio.run(generator.run_generation(
            start_level=args.start_level,
            end_level=args.end_level,
            max_per_level=args.max_per_level
        ))

        console.print("\n🎉 Generation complete!")

    except KeyboardInterrupt:
        console.print("\n👋 Generation interrupted by user.")
    except Exception as e:
        console.print(f"\n❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
