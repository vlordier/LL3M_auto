#!/usr/bin/env python3
"""Enhanced hierarchical generation using the new output management system."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.output.asset_browser import AssetBrowser
from src.output.workflow_integration import WorkflowOutputIntegration

logger = structlog.get_logger(__name__)
console = Console()


class EnhancedHierarchicalGenerator:
    """Enhanced hierarchical asset generator with organized output management."""

    def __init__(self, base_output_dir: str = "outputs/enhanced"):
        """Initialize the enhanced generator.

        Args:
            base_output_dir: Base output directory
        """
        self.base_dir = Path(base_output_dir)
        self.integration = WorkflowOutputIntegration(self.base_dir)
        self.directory_manager = self.integration.directory_manager

        # Load hierarchical prompts
        yaml_path = project_root / "tests" / "data" / "hierarchical_prompts.yaml"
        with open(yaml_path) as f:
            self.test_data = yaml.safe_load(f)

    async def generate_hierarchical_assets(
        self,
        start_level: int = 1,
        end_level: int = 5,
        max_per_level: int = None,
        enable_interactive: bool = True,
        create_project_per_level: bool = False,
    ) -> dict[str, Any]:
        """Generate hierarchical assets with enhanced output management.

        Args:
            start_level: Starting complexity level
            end_level: Ending complexity level
            max_per_level: Maximum assets per level
            enable_interactive: Enable interactive refinement
            create_project_per_level: Create separate project for each level

        Returns:
            Generation results summary
        """
        console.print(
            Panel(
                "🚀 Enhanced Hierarchical 3D Asset Generation",
                subtitle=f"Levels {start_level}-{end_level} with Interactive Refinement",
                border_style="blue",
            )
        )

        results = {
            "total_generated": 0,
            "total_failed": 0,
            "levels": {},
            "projects_created": [],
            "interactive_sessions": [],
        }

        # Create main project if not creating per-level projects
        if not create_project_per_level:
            main_project = self.directory_manager.create_project(
                "Hierarchical Generation Suite",
                f"Complete hierarchical generation from level {start_level} to {end_level}",
            )
            results["main_project"] = str(main_project)

        # Generate assets level by level
        with Progress() as progress:
            main_task = progress.add_task(
                f"[cyan]Generating levels {start_level}-{end_level}...",
                total=end_level - start_level + 1,
            )

            for level in range(start_level, end_level + 1):
                level_results = await self._generate_level(
                    level,
                    max_per_level,
                    enable_interactive,
                    create_project_per_level,
                    progress,
                )

                results["levels"][level] = level_results
                results["total_generated"] += level_results.get("successful", 0)
                results["total_failed"] += level_results.get("failed", 0)

                if level_results.get("project_path"):
                    results["projects_created"].append(level_results["project_path"])

                if level_results.get("interactive_sessions"):
                    results["interactive_sessions"].extend(level_results["interactive_sessions"])

                progress.update(main_task, advance=1)

        # Display final results
        self._display_final_results(results)

        # Optionally launch asset browser
        if results["total_generated"] > 0:
            launch_browser = input("\n🔍 Launch asset browser to explore results? (y/n): ").lower() == "y"
            if launch_browser:
                await self._launch_asset_browser()

        return results

    async def _generate_level(
        self,
        level: int,
        max_per_level: int,
        enable_interactive: bool,
        create_project_per_level: bool,
        progress: Progress,
    ) -> dict[str, Any]:
        """Generate assets for a specific complexity level."""
        level_name = (
            f"level_{level}_"
            + [
                "basic_geometry",
                "simple_objects",
                "textured_objects",
                "lit_scenes",
                "animated_complex",
            ][level - 1]
        )

        prompts = self.test_data["test_prompts"].get(level_name, [])
        if max_per_level:
            prompts = prompts[:max_per_level]

        if not prompts:
            return {"successful": 0, "failed": 0, "assets": []}

        console.print(f"\n{'=' * 60}")
        console.print(f"🎯 LEVEL {level} - {level_name.replace('_', ' ').title()}")
        console.print(f"{'=' * 60}")

        # Create project for this level if requested
        project_path = None
        if create_project_per_level:
            project_name = f"Level {level} - {level_name.replace('_', ' ').title()}"
            project_description = f"Complexity level {level} assets with {len(prompts)} variations"
            project_dir = self.directory_manager.create_project(project_name, project_description)
            project_path = str(project_dir)

        # Generate assets for this level
        level_task = progress.add_task(f"[yellow]Level {level} assets...", total=len(prompts))

        level_results = {
            "successful": 0,
            "failed": 0,
            "assets": [],
            "project_path": project_path,
            "interactive_sessions": [],
        }

        for i, prompt_data in enumerate(prompts):
            asset_result = await self._generate_single_asset(prompt_data, level, i, enable_interactive, project_path)

            level_results["assets"].append(asset_result)

            if asset_result.get("success"):
                level_results["successful"] += 1

                # Add interactive session info if available
                if asset_result.get("refinement_info"):
                    level_results["interactive_sessions"].append(
                        {
                            "asset_name": asset_result.get("asset_name"),
                            "session_id": asset_result["refinement_info"]["session_id"],
                            "asset_path": asset_result.get("asset_dir"),
                        }
                    )
            else:
                level_results["failed"] += 1

            progress.update(level_task, advance=1)

        return level_results

    async def _generate_single_asset(
        self,
        prompt_data: dict[str, Any],
        level: int,
        index: int,
        enable_interactive: bool,
        project_path: str = None,
    ) -> dict[str, Any]:
        """Generate a single asset with enhanced output management."""
        prompt = prompt_data["prompt"]
        description = prompt_data.get("description", "")
        expected_tasks = prompt_data.get("expected_tasks", [])

        console.print(f"\n🎨 Generating Asset #{index + 1}: {description}")
        console.print(f"   Prompt: {prompt}")
        console.print(f"   Expected: {' → '.join(expected_tasks)}")

        try:
            # Generate using enhanced integration
            result = await self.integration.generate_with_enhanced_output(
                prompt=prompt,
                project_name=f"Level {level} Generation" if not project_path else None,
                complexity_level=level,
                enable_interactive=enable_interactive,
                export_formats=["blend", "obj", "gltf", "stl"],
            )

            console.print("   ✅ Generated successfully!")

            # Display key information
            if result.get("quality_score"):
                console.print(f"   📊 Quality Score: {result['quality_score']:.1f}/10")

            if result.get("refinement_info"):
                session_id = result["refinement_info"]["session_id"]
                console.print(f"   🛠️  Interactive session: {session_id}")

            return {
                "success": True,
                "asset_name": self.integration._generate_asset_name(prompt),
                "asset_dir": result.get("asset_dir"),
                "prompt": prompt,
                "level": level,
                "quality_score": result.get("quality_score"),
                "refinement_info": result.get("refinement_info"),
                "organized_files": result.get("organized_files", {}),
            }

        except Exception as e:
            console.print(f"   ❌ Failed: {e}")
            logger.exception("Asset generation failed")

            return {
                "success": False,
                "asset_name": self.integration._generate_asset_name(prompt),
                "error": str(e),
                "prompt": prompt,
                "level": level,
            }

    def _display_final_results(self, results: dict[str, Any]):
        """Display final generation results."""
        console.print(f"\n{'=' * 60}")
        console.print("🏁 ENHANCED GENERATION COMPLETE")
        console.print(f"{'=' * 60}")

        # Summary table
        summary_table = Table(title="📊 Generation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Generated", str(results["total_generated"]))
        summary_table.add_row("Total Failed", str(results["total_failed"]))
        summary_table.add_row("Projects Created", str(len(results["projects_created"])))
        summary_table.add_row("Interactive Sessions", str(len(results["interactive_sessions"])))

        total_attempts = results["total_generated"] + results["total_failed"]
        if total_attempts > 0:
            success_rate = (results["total_generated"] / total_attempts) * 100
            summary_table.add_row("Success Rate", f"{success_rate:.1f}%")

        console.print(summary_table)

        # Level breakdown
        if results["levels"]:
            levels_table = Table(title="📈 Level Breakdown")
            levels_table.add_column("Level", style="cyan")
            levels_table.add_column("Generated", style="green")
            levels_table.add_column("Failed", style="red")
            levels_table.add_column("Success Rate", style="yellow")
            levels_table.add_column("Interactive", style="magenta")

            for level, level_data in results["levels"].items():
                successful = level_data.get("successful", 0)
                failed = level_data.get("failed", 0)
                total = successful + failed
                success_rate = (successful / total * 100) if total > 0 else 0
                interactive_count = len(level_data.get("interactive_sessions", []))

                levels_table.add_row(
                    str(level),
                    str(successful),
                    str(failed),
                    f"{success_rate:.1f}%",
                    str(interactive_count),
                )

            console.print(levels_table)

        # Interactive sessions summary
        if results["interactive_sessions"]:
            console.print("\n[bold cyan]🛠️  Interactive Refinement Sessions Available:[/bold cyan]")
            for session in results["interactive_sessions"][:5]:  # Show first 5
                console.print(f"   • {session['asset_name']} - Session: {session['session_id']}")
                console.print(f"     📁 {session['asset_path']}")

        # Directory information
        console.print(f"\n[bold green]📁 Enhanced Output Directory:[/bold green] {self.base_dir}")
        console.print("[bold yellow]💡 Use the interactive asset manager to explore and refine:[/bold yellow]")
        console.print("   python scripts/interactive_asset_manager.py browse")

    async def _launch_asset_browser(self):
        """Launch the interactive asset browser."""
        console.print("\n🚀 Launching Interactive Asset Browser...")

        browser = AssetBrowser(self.directory_manager)
        try:
            browser.show_dashboard()
        except KeyboardInterrupt:
            console.print("\n👋 Asset browser closed")

    async def demonstrate_interactive_refinement(self, asset_path: str):
        """Demonstrate interactive refinement on a specific asset."""
        asset_path = Path(asset_path)

        if not asset_path.exists():
            console.print(f"[red]❌ Asset not found: {asset_path}[/red]")
            return

        console.print(
            Panel(
                f"🛠️ Interactive Refinement Demo: {asset_path.name}",
                border_style="green",
            )
        )

        # Load asset metadata
        metadata_file = asset_path / "asset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                asset_meta = json.load(f)
        else:
            asset_meta = {"asset_name": asset_path.name}

        console.print(f"Asset: {asset_meta.get('asset_name', 'Unknown')}")
        console.print(f"Quality: {asset_meta.get('quality_score', 'N/A')}")

        # Get refinement suggestions
        suggestions = self.integration.get_interactive_suggestions(asset_path)

        console.print("\n[bold cyan]💡 AI Suggestions:[/bold cyan]")
        for i, suggestion in enumerate(suggestions["suggestions"][:5], 1):
            console.print(f"{i}. {suggestion}")

        # Demonstrate batch refinement
        sample_instructions = [
            "Make it more colorful",
            "Add more geometric detail",
            "Improve the lighting",
        ]

        console.print("\n[yellow]Demonstrating batch refinement with sample instructions...[/yellow]")

        result = await self.integration.refine_existing_asset(asset_path, sample_instructions, create_variants=True)

        console.print("[green]✅ Batch refinement completed![/green]")
        console.print(f"   Session: {result['session_id']}")
        console.print(f"   Success: {result['successful_refinements']}/{result['total_instructions']}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Hierarchical 3D Asset Generation")

    parser.add_argument(
        "--start-level",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Start generation from this level (default: 1)",
    )
    parser.add_argument(
        "--end-level",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="End generation at this level (default: 5)",
    )
    parser.add_argument(
        "--max-per-level",
        type=int,
        help="Maximum number of assets to generate per level",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/enhanced",
        help="Base output directory (default: outputs/enhanced)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive refinement setup",
    )
    parser.add_argument(
        "--project-per-level",
        action="store_true",
        help="Create separate project for each complexity level",
    )
    parser.add_argument(
        "--demo-refinement",
        help="Demonstrate interactive refinement on existing asset path",
    )
    parser.add_argument(
        "--browser-only",
        action="store_true",
        help="Launch asset browser without generation",
    )

    args = parser.parse_args()

    try:
        generator = EnhancedHierarchicalGenerator(args.output_dir)

        if args.browser_only:
            # Just launch browser
            console.print("🔍 Launching Asset Browser...")
            await generator._launch_asset_browser()

        elif args.demo_refinement:
            # Demonstrate refinement
            await generator.demonstrate_interactive_refinement(args.demo_refinement)

        else:
            # Full generation
            results = await generator.generate_hierarchical_assets(
                start_level=args.start_level,
                end_level=args.end_level,
                max_per_level=args.max_per_level,
                enable_interactive=not args.no_interactive,
                create_project_per_level=args.project_per_level,
            )

            # Show final statistics
            total_generated = results.get("total_generated", 0)
            if total_generated > 0:
                console.print(f"\n🎉 Successfully generated {total_generated} enhanced 3D assets!")
                console.print(f"📁 Organized in: {args.output_dir}")
                console.print(
                    f"🛠️  {len(results.get('interactive_sessions', []))} assets ready for interactive refinement"
                )
            else:
                console.print("[yellow]No assets were generated.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n👋 Generation interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
