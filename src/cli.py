"""Command-line interface for LL3M."""

import asyncio
from pathlib import Path
from typing import Any

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Import settings with fallback for both mypy and entry point usage
try:
    # Try relative import first (for mypy and internal usage)
    from .utils.config import get_settings
except ImportError:
    # Fall back to absolute import (for entry point usage)
    from utils.config import get_settings  # type: ignore[no-redef]

from .workflow.orchestrator import LL3MOrchestrator

logger = structlog.get_logger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="LL3M")
def cli() -> None:
    """Large Language 3D Modelers - Multi-agent system for generating 3D assets."""
    pass


@cli.command()
@click.argument("prompt", required=True)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for generated assets",
    type=click.Path(path_type=Path),
)
@click.option(
    "--format",
    "-f",
    default="blend",
    help="Export format (blend, gltf, obj, fbx)",
    type=click.Choice(["blend", "gltf", "obj", "fbx"]),
)
@click.option(
    "--no-refine",
    is_flag=True,
    help="Skip automatic refinement phase",
)
def generate(
    prompt: str,
    output: Path | None,
    format: str,  # noqa: ARG001
    no_refine: bool,  # noqa: ARG001
) -> None:
    """Generate a 3D asset from a text prompt."""
    settings = get_settings()
    if output:
        settings.app.output_directory = output

    console.print("[bold blue]LL3M[/bold blue] Generating 3D asset from prompt:")
    console.print(f"[italic]{prompt}[/italic]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing workflow...", total=None)

        async def run_generation() -> Any:
            try:
                orchestrator = LL3MOrchestrator()
                progress.update(task, description="Planning asset generation...")
                result = await orchestrator.generate_asset(
                    prompt=prompt,
                    export_format=format,
                    skip_refinement=no_refine,
                )
                progress.update(task, description="Generation complete!")
                return result

            except Exception as e:
                progress.update(task, description=f"Error: {str(e)}")
                logger.error("Generation failed", error=str(e))
                raise

        try:
            result = asyncio.run(run_generation())

            if result.success:
                console.print(
                    "\n[bold green]✓[/bold green] Asset generated successfully!"
                )
                console.print(f"Asset path: [cyan]{result.asset_path}[/cyan]")
                if result.screenshot_path:
                    console.print(f"Screenshot: [cyan]{result.screenshot_path}[/cyan]")
                console.print(
                    f"Execution time: [yellow]{result.execution_time:.2f}s[/yellow]"
                )
            else:
                console.print("\n[bold red]✗[/bold red] Asset generation failed!")
                for error in result.errors:
                    console.print(f"[red]Error:[/red] {error}")

        except Exception as e:
            console.print(f"\n[bold red]✗[/bold red] Generation failed: {str(e)}")
            raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("asset_id", required=True)
@click.argument("feedback", required=True)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for refined assets",
    type=click.Path(path_type=Path),
)
def refine(asset_id: str, feedback: str, output: Path | None) -> None:
    """Refine an existing asset with user feedback."""
    settings = get_settings()
    if output:
        settings.app.output_directory = output

    console.print(
        f"[bold blue]LL3M[/bold blue] Refining asset: [cyan]{asset_id}[/cyan]"
    )
    console.print(f"Feedback: [italic]{feedback}[/italic]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing refinement...", total=None)

        async def run_refinement() -> Any:
            try:
                orchestrator = LL3MOrchestrator()
                progress.update(task, description="Processing feedback...")
                result = await orchestrator.refine_asset(
                    asset_id=asset_id,
                    user_feedback=feedback,
                )
                progress.update(task, description="Refinement complete!")
                return result

            except Exception as e:
                progress.update(task, description=f"Error: {str(e)}")
                logger.error("Refinement failed", error=str(e))
                raise

        try:
            result = asyncio.run(run_refinement())

            if result.success:
                console.print(
                    "\n[bold green]✓[/bold green] Asset refined successfully!"
                )
                console.print(f"Asset path: [cyan]{result.asset_path}[/cyan]")
                if result.screenshot_path:
                    console.print(f"Screenshot: [cyan]{result.screenshot_path}[/cyan]")
            else:
                console.print("\n[bold red]✗[/bold red] Asset refinement failed!")
                for error in result.errors:
                    console.print(f"[red]Error:[/red] {error}")

        except Exception as e:
            console.print(f"\n[bold red]✗[/bold red] Refinement failed: {str(e)}")
            raise click.ClickException(str(e)) from e


@cli.command()
@click.option(
    "--format",
    "-f",
    default="table",
    help="Output format (table, json)",
    type=click.Choice(["table", "json"]),
)
@click.option(
    "--limit",
    "-l",
    default=10,
    help="Maximum number of assets to display",
    type=int,
)
@click.option(
    "--min-quality",
    "-q",
    default=None,
    help="Minimum quality score filter",
    type=float,
)
def list_assets(_format: str, limit: int, min_quality: float | None) -> None:
    """List all generated assets."""
    try:
        orchestrator = LL3MOrchestrator()
        assets = orchestrator.list_assets(
            min_quality_score=min_quality,
            limit=limit,
        )

        if not assets:
            console.print("[yellow]No assets found[/yellow]")
            return

        if _format == "json":
            import json

            console.print(
                json.dumps([asset.model_dump() for asset in assets], indent=2)
            )
        else:
            from rich.table import Table

            table = Table(title="Generated Assets")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="magenta")
            table.add_column("Quality", justify="right", style="green")
            table.add_column("Created", style="blue")
            table.add_column("Versions", justify="right", style="yellow")

            for asset in assets:
                latest_version = asset.latest_version
                quality_score = (
                    f"{latest_version.quality_score:.1f}/10"
                    if latest_version and latest_version.quality_score
                    else "N/A"
                )

                from datetime import datetime

                created_date = datetime.fromtimestamp(asset.created_at).strftime(
                    "%Y-%m-%d"
                )

                table.add_row(
                    asset.id[:8] + "...",  # Shortened ID
                    asset.name,
                    quality_score,
                    created_date,
                    str(len(asset.versions)),
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing assets: {str(e)}[/red]")
        logger.error("Asset listing failed", error=str(e))


@cli.command()
@click.option(
    "--check-blender",
    is_flag=True,
    help="Check Blender installation",
)
@click.option(
    "--check-openai",
    is_flag=True,
    help="Check OpenAI API connection",
)
def status(check_blender: bool, check_openai: bool) -> None:
    """Check LL3M system status and configuration."""
    settings = get_settings()
    console.print("[bold blue]LL3M System Status[/bold blue]")
    console.print()

    # Basic configuration
    console.print(f"Output directory: [cyan]{settings.app.output_directory}[/cyan]")
    console.print(f"Log level: [cyan]{settings.app.log_level}[/cyan]")
    console.print()

    if check_blender:
        console.print("Checking Blender installation...")
        blender_path = Path(settings.blender.path)
        if blender_path.exists():
            console.print(f"[green]✓[/green] Blender found at: {blender_path}")
        else:
            console.print(f"[red]✗[/red] Blender not found at: {blender_path}")

    if check_openai:
        console.print("Checking OpenAI API connection...")
        # This would test the API connection
        if settings.openai.api_key:
            console.print("[green]✓[/green] OpenAI API key configured")
        else:
            console.print("[red]✗[/red] OpenAI API key not configured")


def main() -> None:
    """Entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        logger.error("CLI error", error=str(e))


if __name__ == "__main__":
    main()
