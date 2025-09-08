#!/usr/bin/env python3
"""Demo version of hierarchical generation that creates mock assets without requiring LLM."""

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


logger = structlog.get_logger(__name__)
console = Console()


class MockHierarchicalAssetGenerator:
    """Mock generator for demonstration purposes - no LLM required."""

    def __init__(self, output_dir: Path | None = None, level_filter: int | None = None):
        """Initialize the mock generator."""
        self.project_root = project_root
        self.output_dir = output_dir or (project_root / "outputs" / "hierarchical_demo")
        self.level_filter = level_filter

        # Load test prompts
        yaml_path = project_root / "tests" / "data" / "hierarchical_prompts.yaml"
        with open(yaml_path) as f:
            self.test_data = yaml.safe_load(f)

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

    async def generate_mock_asset(self, prompt_data: dict[str, Any], level: int, prompt_index: int) -> dict[str, Any]:
        """Generate a mock 3D asset (no real generation)."""
        prompt = prompt_data["prompt"]
        description = prompt_data.get("description", "No description")
        expected_tasks = prompt_data.get("expected_tasks", [])

        console.print(f"\n🎯 Generating Level {level} Mock Asset #{prompt_index + 1}")
        console.print(f"   Prompt: {prompt}")
        console.print(f"   Description: {description}")
        console.print(f"   Expected Tasks: {' → '.join(expected_tasks)}")

        # Create asset-specific output directory
        asset_name = f"level_{level}_asset_{prompt_index + 1:02d}"
        asset_dir = self.output_dir / f"level_{level}" / asset_name
        asset_dir.mkdir(exist_ok=True)

        start_time = time.time()

        # Mock generation delay (simulate processing time)
        mock_delay = level * 0.5 + 1.0  # Increasing delay with complexity
        await asyncio.sleep(mock_delay)

        try:
            # Create mock files
            self._create_mock_files(asset_dir, prompt_data, level, expected_tasks)

            generation_time = time.time() - start_time

            # Create metadata
            metadata = {
                "prompt": prompt,
                "description": description,
                "expected_complexity": prompt_data.get("expected_complexity", level),
                "expected_tasks": expected_tasks,
                "level": level,
                "prompt_index": prompt_index,
                "generation_time": generation_time,
                "timestamp": time.time(),
                "mock_mode": True,
                "success": True
            }

            # Save metadata
            metadata_file = asset_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            console.print(f"   ✅ Mock generated in {generation_time:.2f}s")

            return {
                "success": True,
                "prompt": prompt,
                "level": level,
                "generation_time": generation_time,
                "asset_dir": str(asset_dir),
                "metadata_file": str(metadata_file),
                "mock_mode": True
            }

        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)

            console.print(f"   ❌ Mock failed after {generation_time:.2f}s: {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "generation_time": generation_time,
                "prompt": prompt,
                "level": level,
                "mock_mode": True
            }

    def _create_mock_files(self, asset_dir: Path, prompt_data: dict[str, Any], level: int, expected_tasks: list[str]):
        """Create mock files to simulate real generation."""
        prompt = prompt_data["prompt"]

        # Create mock Python code
        code_dir = asset_dir / "code"
        code_dir.mkdir(exist_ok=True)
        code_file = code_dir / "generated_script.py"
        mock_code = f'''import bpy

# Generated mock code for: {prompt}
# Level {level} complexity
# Expected tasks: {", ".join(expected_tasks)}

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Mock {prompt.lower()}
'''

        if level == 1:  # Basic geometry
            if "ovoid" in prompt.lower():
                mock_code += '''
# Create ovoid (ellipsoid)
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
bpy.context.active_object.scale = (1, 1, 1.5)  # Make it ovoid
bpy.context.active_object.name = "Ovoid"
'''
            elif "cube" in prompt.lower():
                mock_code += '''
# Create cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.context.active_object.name = "Cube"
'''
            elif "sphere" in prompt.lower():
                mock_code += '''
# Create sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
bpy.context.active_object.name = "Sphere"
'''

        elif level >= 2:  # More complex
            mock_code += '''
# More complex operations for higher levels would go here
# This is mock code for demonstration purposes
'''

        mock_code += '''
# Mock rendering setup
bpy.context.scene.render.filepath = "/tmp/render.png"
bpy.context.scene.render.image_settings.file_format = 'PNG'

print(f"Mock generation complete for: {prompt}")
'''

        with open(code_file, "w") as f:
            f.write(mock_code)

        # Create mock render image (placeholder)
        renders_dir = asset_dir / "renders"
        renders_dir.mkdir(exist_ok=True)
        render_file = renders_dir / "render.png"
        with open(render_file, "w") as f:
            f.write("# Mock PNG image placeholder\n")

        # Create mock .blend file (placeholder)
        blends_dir = asset_dir / "blends"
        blends_dir.mkdir(exist_ok=True)
        blend_file = blends_dir / "asset.blend"
        with open(blend_file, "w") as f:
            f.write("# Mock .blend file placeholder\n")

        # Create mock export files
        exports_dir = asset_dir / "exports"
        exports_dir.mkdir(exist_ok=True)
        export_formats = ["obj", "fbx", "gltf", "ply", "stl"]
        for fmt in export_formats:
            export_file = exports_dir / f"asset.{fmt}"
            with open(export_file, "w") as f:
                f.write(f"# Mock {fmt.upper()} export placeholder\n")

        # Create mock log
        logs_dir = asset_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "generation.log"
        with open(log_file, "w") as f:
            f.write(f"Mock generation log for: {prompt}\n")
            f.write(f"Level: {level}\n")
            f.write(f"Expected tasks: {', '.join(expected_tasks)}\n")
            f.write("Generation completed successfully (mock mode)\n")

    async def run_generation(self, start_level: int = 1, end_level: int = 5, max_per_level: int = None) -> dict[str, Any]:
        """Run mock asset generation for specified levels."""
        console.print(Panel("🎭 Starting Mock Hierarchical 3D Asset Generation", border_style="blue"))
        console.print(f"Output directory: {self.output_dir}")
        console.print("Note: This is a demo mode with mock files (no real LLM/Blender required)")

        results = {
            "total_generated": 0,
            "total_failed": 0,
            "total_time": 0,
            "levels": {},
            "mock_mode": True
        }

        overall_start = time.time()

        for level in range(start_level, end_level + 1):
            if self.level_filter and level != self.level_filter:
                continue

            console.print(f"\n{'='*60}")
            console.print(f"🎯 LEVEL {level} MOCK GENERATION")
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
                asset_result = await self.generate_mock_asset(prompt_data, level, i)

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
            json.dump(results, f, indent=2)

        self._display_final_summary(results)

        return results

    def _display_final_summary(self, results: dict[str, Any]):
        """Display final generation summary."""
        console.print(f"\n{'='*60}")
        console.print("🏁 FINAL MOCK GENERATION SUMMARY")
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
        total_attempts = results['total_generated'] + results['total_failed']
        if total_attempts > 0:
            console.print(f"   Overall Success Rate: {(results['total_generated'] / total_attempts * 100):.1f}%")
        console.print(f"   Total Time: {results['total_time']:.2f}s")
        console.print(f"   Output Directory: {self.output_dir}")
        console.print("   🎭 Mode: Mock/Demo (no real LLM required)")


def main():
    """Main entry point for mock generation."""
    parser = argparse.ArgumentParser(description="Generate mock 3D assets using hierarchical prompts")
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
        generator = MockHierarchicalAssetGenerator(
            output_dir=args.output_dir,
            level_filter=args.level
        )

        results = asyncio.run(generator.run_generation(
            start_level=args.start_level,
            end_level=args.end_level,
            max_per_level=args.max_per_level
        ))

        console.print("\n🎉 Mock generation complete!")
        console.print("💡 This demo shows the hierarchical structure without requiring LLM/Blender.")
        console.print("   To run real generation, configure OpenAI API key and use the main script.")

    except KeyboardInterrupt:
        console.print("\n👋 Mock generation interrupted by user.")
    except Exception as e:
        console.print(f"\n❌ Mock generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
