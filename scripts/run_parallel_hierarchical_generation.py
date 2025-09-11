#!/usr/bin/env python3
"""Generate 3D assets using hierarchical prompts in parallel for maximum speed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.workflow.orchestrator import LL3MOrchestrator

console = Console()


@dataclass
class GenerationTask:
    """A single generation task."""

    prompt_data: dict[str, Any]
    level: int
    prompt_index: int
    output_dir: Path


class ParallelHierarchicalAssetGenerator:
    """Parallel hierarchical 3D asset generator."""

    def __init__(
        self,
        output_dir: str = "outputs/hierarchical_tests",
        level_filter: int = None,
        max_concurrent: int = 4,
    ):
        """Initialize the generator.

        Args:
            output_dir: Base output directory
            level_filter: Only generate for specific level (1-5)
            max_concurrent: Maximum number of concurrent generation tasks
        """
        self.output_dir = Path(output_dir)
        self.level_filter = level_filter
        self.max_concurrent = max_concurrent
        self.orchestrator = LL3MOrchestrator()

        # Load test prompts
        prompts_file = Path("tests/data/hierarchical_prompts.yaml")
        with open(prompts_file) as f:
            self.test_data = yaml.safe_load(f)

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create level directories
        for level in range(1, 6):
            level_dir = self.output_dir / f"level_{level}"
            level_dir.mkdir(exist_ok=True)

            for subdir in ["blends", "code", "exports", "renders", "logs"]:
                (level_dir / subdir).mkdir(exist_ok=True)

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

    async def generate_single_asset(self, task: GenerationTask) -> dict[str, Any]:
        """Generate a single asset from a task."""
        prompt_data = task.prompt_data
        level = task.level
        prompt_index = task.prompt_index

        prompt = prompt_data["prompt"]
        asset_name = f"level_{level}_asset_{prompt_index + 1:02d}"

        # Create asset directory
        asset_dir = task.output_dir / f"level_{level}" / asset_name
        asset_dir.mkdir(exist_ok=True)

        console.print(f"🎯 [Level {level}] Generating Asset #{prompt_index + 1}")
        console.print(f"   Prompt: {prompt}")
        console.print(f"   Description: {prompt_data.get('description', 'N/A')}")

        start_time = time.time()

        try:
            # Generate the asset using the orchestrator
            result = await self.orchestrator.generate_asset(
                prompt=prompt,
                export_formats=["blend", "obj", "gltf"],
                asset_name=asset_name,
            )

            generation_time = time.time() - start_time

            # Process and organize results
            asset_result = await self._process_generation_result(
                result, prompt_data, asset_dir, level, prompt_index, generation_time
            )

            console.print(f"   ✅ [Level {level}] Asset #{prompt_index + 1} generated in {generation_time:.2f}s")
            return asset_result

        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = str(e)

            console.print(
                f"   ❌ [Level {level}] Asset #{prompt_index + 1} failed after {generation_time:.2f}s: {error_msg}"
            )

            # Log error details
            error_log = asset_dir / "error.json"
            error_data = {
                "prompt": prompt,
                "error": error_msg,
                "generation_time": generation_time,
                "level": level,
                "prompt_index": prompt_index,
            }

            with open(error_log, "w") as f:
                json.dump(error_data, f, indent=2)

            return {
                "success": False,
                "error": error_msg,
                "generation_time": generation_time,
                "prompt": prompt,
                "level": level,
            }

    async def _process_generation_result(
        self,
        result,
        prompt_data: dict[str, Any],
        asset_dir: Path,
        level: int,
        prompt_index: int,
        generation_time: float,
    ) -> dict[str, Any]:
        """Process and organize generation results."""
        metadata = {
            **prompt_data,
            "level": level,
            "prompt_index": prompt_index,
            "generation_time": generation_time,
            "timestamp": time.time(),
            "result": result,
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
            "success": True,
            "generation_time": generation_time,
            "files_organized": files_organized,
            "prompt": prompt_data["prompt"],
            "level": level,
        }

    async def _organize_generated_files(self, result, asset_dir: Path, level: int) -> dict[str, list[str]]:
        """Organize generated files into appropriate directories."""
        organized_files = {
            "code": [],
            "blends": [],
            "exports": [],
            "renders": [],
            "other": [],
        }

        # Move files based on extension
        for file_path in asset_dir.glob("*"):
            if file_path.is_file() and file_path.name not in [
                "metadata.json",
                "error.json",
            ]:
                suffix = file_path.suffix.lower()

                if suffix == ".py":
                    dest_dir = asset_dir / "code"
                    organized_files["code"].append(file_path.name)
                elif suffix == ".blend":
                    dest_dir = asset_dir / "blends"
                    organized_files["blends"].append(file_path.name)
                elif suffix in [
                    ".obj",
                    ".mtl",
                    ".gltf",
                    ".glb",
                    ".fbx",
                    ".stl",
                    ".ply",
                ]:
                    dest_dir = asset_dir / "exports"
                    organized_files["exports"].append(file_path.name)
                elif suffix in [".png", ".jpg", ".jpeg", ".exr", ".hdr"]:
                    dest_dir = asset_dir / "renders"
                    organized_files["renders"].append(file_path.name)
                else:
                    dest_dir = asset_dir / "other"
                    organized_files["other"].append(file_path.name)

                dest_dir.mkdir(exist_ok=True)

                # Move file to organized directory
                dest_path = dest_dir / file_path.name
                if not dest_path.exists():
                    file_path.rename(dest_path)

        return organized_files

    async def _export_additional_formats(self, blend_file: Path, asset_dir: Path):
        """Export .blend file to additional formats."""
        try:
            # This would use Blender's command line export capabilities
            # For now, we'll just log that we would do this
            console.print(f"   📦 Would export {blend_file.name} to additional formats")
        except Exception as e:
            console.print(f"   ⚠️ Export failed: {e}")

    def _get_prompts_for_level(self, level: int) -> list[dict[str, Any]]:
        """Get prompts for a specific level."""
        level_key = (
            f"level_{level}_"
            + [
                "basic_geometry",
                "simple_objects",
                "textured_objects",
                "lit_scenes",
                "animated_complex",
            ][level - 1]
        )

        return self.test_data.get("test_prompts", {}).get(level_key, [])

    def _create_generation_tasks(
        self, start_level: int, end_level: int, max_per_level: int = None
    ) -> list[GenerationTask]:
        """Create a list of generation tasks for parallel execution."""
        tasks = []

        for level in range(start_level, end_level + 1):
            if self.level_filter and level != self.level_filter:
                continue

            prompts = self._get_prompts_for_level(level)
            if not prompts:
                continue

            # Limit prompts per level if specified
            if max_per_level:
                prompts = prompts[:max_per_level]

            for i, prompt_data in enumerate(prompts):
                task = GenerationTask(
                    prompt_data=prompt_data,
                    level=level,
                    prompt_index=i,
                    output_dir=self.output_dir,
                )
                tasks.append(task)

        return tasks

    async def run_parallel_generation(
        self, start_level: int = 1, end_level: int = 5, max_per_level: int = None
    ) -> dict[str, Any]:
        """Run asset generation in parallel across levels and prompts."""
        console.print(
            Panel(
                "🚀 Starting Parallel Hierarchical 3D Asset Generation",
                border_style="blue",
            )
        )
        console.print(f"Output directory: {self.output_dir}")
        console.print(f"Max concurrent tasks: {self.max_concurrent}")

        # Create all generation tasks
        tasks = self._create_generation_tasks(start_level, end_level, max_per_level)
        console.print(f"Total tasks to process: {len(tasks)}")

        results = {
            "total_generated": 0,
            "total_failed": 0,
            "total_time": 0,
            "levels": {},
            "tasks_completed": 0,
            "tasks_total": len(tasks),
        }

        overall_start = time.time()

        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(task: GenerationTask):
            async with semaphore:
                return await self.generate_single_asset(task)

        # Run all tasks in parallel with concurrency limit
        console.print(f"\n🔥 Running {len(tasks)} tasks in parallel (max {self.max_concurrent} concurrent)...")

        # Execute tasks in parallel
        task_results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks], return_exceptions=True)

        # Process results
        for task, result in zip(tasks, task_results, strict=False):
            level = task.level

            if level not in results["levels"]:
                results["levels"][level] = {"generated": 0, "failed": 0, "assets": []}

            if isinstance(result, Exception):
                # Handle exceptions
                error_result = {
                    "success": False,
                    "error": str(result),
                    "generation_time": 0,
                    "prompt": task.prompt_data["prompt"],
                    "level": level,
                }
                results["levels"][level]["assets"].append(error_result)
                results["levels"][level]["failed"] += 1
                results["total_failed"] += 1
            elif result.get("success", False):
                results["levels"][level]["assets"].append(result)
                results["levels"][level]["generated"] += 1
                results["total_generated"] += 1
            else:
                results["levels"][level]["assets"].append(result)
                results["levels"][level]["failed"] += 1
                results["total_failed"] += 1

        overall_end = time.time()
        results["total_time"] = overall_end - overall_start

        # Save overall results
        results_file = self.output_dir / "parallel_generation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=self._serialize_pydantic)

        self._display_final_summary(results)

        return results

    def _display_final_summary(self, results: dict[str, Any]):
        """Display final generation summary."""
        console.print(f"\n{'=' * 60}")
        console.print("🏁 PARALLEL GENERATION SUMMARY")
        console.print(f"{'=' * 60}")

        table = Table()
        table.add_column("Level", justify="center")
        table.add_column("Generated", justify="center")
        table.add_column("Failed", justify="center")
        table.add_column("Success Rate", justify="center")

        for level, level_data in results["levels"].items():
            generated = level_data["generated"]
            failed = level_data["failed"]
            total = generated + failed
            success_rate = (generated / total * 100) if total > 0 else 0

            table.add_row(str(level), str(generated), str(failed), f"{success_rate:.1f}%")

        console.print(table)

        console.print("\n📈 OVERALL STATISTICS:")
        console.print(f"   Total Generated: {results['total_generated']}")
        console.print(f"   Total Failed: {results['total_failed']}")
        total_tasks = results["total_generated"] + results["total_failed"]
        overall_success = (results["total_generated"] / total_tasks * 100) if total_tasks > 0 else 0
        console.print(f"   Overall Success Rate: {overall_success:.1f}%")
        console.print(f"   Total Time: {results['total_time']:.2f}s")
        console.print(f"   Output Directory: {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate hierarchical 3D assets in parallel for maximum speed")

    parser.add_argument("--start-level", type=int, default=1, help="Starting complexity level (1-5)")
    parser.add_argument("--end-level", type=int, default=5, help="Ending complexity level (1-5)")
    parser.add_argument("--level", type=int, help="Generate only for specific level (1-5)")
    parser.add_argument("--max-per-level", type=int, help="Maximum assets to generate per level")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent generation tasks (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hierarchical_tests",
        help="Output directory for generated assets",
    )

    args = parser.parse_args()

    try:
        generator = ParallelHierarchicalAssetGenerator(
            output_dir=args.output_dir,
            level_filter=args.level,
            max_concurrent=args.max_concurrent,
        )

        asyncio.run(
            generator.run_parallel_generation(
                start_level=args.start_level,
                end_level=args.end_level,
                max_per_level=args.max_per_level,
            )
        )

        console.print("\n🎉 Parallel generation complete!")

    except KeyboardInterrupt:
        console.print("\n👋 Generation interrupted by user.")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
