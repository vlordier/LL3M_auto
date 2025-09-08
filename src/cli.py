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
    import utils.config

    get_settings = utils.config.get_settings

# Use absolute import consistently
from .workflow.orchestrator import LL3MOrchestrator

logger = structlog.get_logger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="LL3M")
def cli() -> None:
    """Large Language 3D Modelers - Multi-agent system for generating 3D assets."""


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
@click.option(
    "--session-id",
    type=str,
    help="Optional session ID for continuing a previous session",
)
def generate(
    prompt: str,
    output: Path | None,
    format: str,
    no_refine: bool,  # noqa: ARG001 - deprecated parameter kept for CLI compatibility
    session_id: str | None,
) -> None:
    """Generate a 3D asset from a text prompt."""
    settings = get_settings()
    if output:
        settings.app.output_directory = output

    # Convert format options to list
    export_formats = [format] if format else ["blend"]
    # Note: no_refine option is deprecated since we now only have the comprehensive workflow

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
                from .output.workflow_integration import WorkflowOutputIntegration

                integration = WorkflowOutputIntegration(output)
                progress.update(task, description="Planning asset generation...")

                result = await integration.generate_with_enhanced_output(
                    prompt=prompt,
                    export_formats=export_formats,
                    session_id=session_id,
                )
                progress.update(task, description="Generation complete!")
            except Exception as generation_error:
                progress.update(task, description=f"Error: {generation_error!s}")
                logger.exception("Generation failed", error=str(generation_error))
                return {"success": False, "error": str(generation_error)}
            else:
                return result

        try:
            result = asyncio.run(run_generation())

            if result.get("success"):
                console.print("\n[bold green]✓[/bold green] Asset generated successfully!")

                # Display comprehensive artifact information
                console.print(f"Session ID: [bold cyan]{result.get('session_id')}[/bold cyan]")
                console.print(f"Session Directory: [cyan]{result.get('session_directory')}[/cyan]")

                # Display artifact summary
                artifact_summary = result.get("artifact_summary", "No artifacts")
                console.print("\n[bold]Generated Artifacts:[/bold]")
                console.print(artifact_summary)

                # Display quality score if available
                quality_score = result.get("quality_score")
                if quality_score is not None:
                    console.print(f"\nQuality Score: [bold yellow]{quality_score:.1f}/10[/bold yellow]")

                # Display resume information
                if result.get("session_id"):
                    console.print("\n[bold]To Resume This Session:[/bold]")
                    console.print(f"ll3m resume {result['session_id']}")

            else:
                console.print("\n[bold red]✗[/bold red] Asset generation failed!")
                error = result.get("error", "Unknown error")
                console.print(f"[red]Error:[/red] {error}")

        except Exception as workflow_error:
            console.print(f"\n[bold red]✗[/bold red] Generation failed: {workflow_error!s}")
            raise click.ClickException(str(workflow_error)) from workflow_error


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

    console.print(f"[bold blue]LL3M[/bold blue] Refining asset: [cyan]{asset_id}[/cyan]")
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
                orchestrator = LL3MOrchestrator()  # Always use comprehensive workflow
                progress.update(task, description="Processing feedback...")
                result = await orchestrator.refine_asset(
                    original_prompt=asset_id,  # Use asset_id as prompt for now
                    feedback=feedback,
                    asset_name="refined_asset",
                )
                progress.update(task, description="Refinement complete!")
            except Exception as e:
                progress.update(task, description=f"Error: {e!s}")
                logger.exception("Refinement failed", error=str(e))
                return {"success": False, "error": str(e)}
            else:
                return result

        try:
            result = asyncio.run(run_refinement())

            if result.success:
                console.print("\n[bold green]✓[/bold green] Asset refined successfully!")
                console.print(f"Asset path: [cyan]{result.asset_path}[/cyan]")
                if result.screenshot_path:
                    console.print(f"Screenshot: [cyan]{result.screenshot_path}[/cyan]")
            else:
                console.print("\n[bold red]✗[/bold red] Asset refinement failed!")
                for error in result.errors:
                    console.print(f"[red]Error:[/red] {error}")

        except Exception as e:
            console.print(f"\n[bold red]✗[/bold red] Refinement failed: {e!s}")
            raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("session_id", required=True)
@click.option(
    "--feedback",
    "-f",
    type=str,
    help="Additional user feedback for resumption",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for resumed session",
    type=click.Path(path_type=Path),
)
def resume(session_id: str, feedback: str | None, output: Path | None) -> None:
    """Resume a previous generation session."""
    settings = get_settings()
    if output:
        settings.app.output_directory = output

    console.print(f"[bold blue]LL3M[/bold blue] Resuming session: [cyan]{session_id}[/cyan]")
    if feedback:
        console.print(f"Additional feedback: [italic]{feedback}[/italic]")
    console.print()

    async def _resume() -> None:
        """Async wrapper for resumption."""
        from .output.workflow_integration import WorkflowOutputIntegration

        integration = WorkflowOutputIntegration(output)

        try:
            result = await integration.resume_session(session_id, feedback)

            if result["success"]:
                console.print("[bold green]✓[/bold green] Session resumed successfully!")

                console.print(f"Session Directory: [bold]{result['session_directory']}[/bold]")

                # Display artifact summary
                if result.get("artifact_summary"):
                    console.print("\n[bold]Current Artifacts:[/bold]")
                    console.print(result["artifact_summary"])

                # Display new resume prompt
                if result.get("new_resume_prompt"):
                    console.print("\n[bold]Updated Resume Prompt:[/bold]")
                    console.print(result["new_resume_prompt"])

            else:
                console.print("[bold red]✗[/bold red] Failed to resume session")
                if "error" in result:
                    console.print(f"Error: {result['error']}")

        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Resumption failed: {e}")
            raise click.ClickException(str(e)) from e

    try:
        asyncio.run(_resume())
    except Exception as workflow_error:
        console.print(f"\n[bold red]✗[/bold red] Resumption failed: {workflow_error!s}")
        raise click.ClickException(str(workflow_error)) from workflow_error


@cli.command()
@click.option(
    "--limit",
    "-l",
    default=10,
    help="Maximum number of sessions to list",
    type=int,
)
@click.option(
    "--format",
    "-f",
    default="table",
    help="Output format (table, json)",
    type=click.Choice(["table", "json"]),
)
def list_sessions(limit: int, format: str) -> None:
    """List resumable generation sessions."""
    from .output.workflow_integration import WorkflowOutputIntegration

    try:
        integration = WorkflowOutputIntegration()
        sessions = integration.get_resumable_sessions(limit)

        if not sessions:
            console.print("[yellow]No resumable sessions found[/yellow]")
            return

        if format == "json":
            import json
            console.print(json.dumps(sessions, indent=2))
        else:
            # Table format
            from rich.table import Table

            table = Table(title="Resumable Sessions")
            table.add_column("Session ID", style="cyan")
            table.add_column("Prompt", style="white")
            table.add_column("Iterations", style="yellow")
            table.add_column("Last Updated", style="green")
            table.add_column("Artifacts", style="blue")

            for session in sessions:
                table.add_row(
                    session["session_id"],
                    session["prompt"][:50] + "..." if len(session["prompt"]) > 50 else session["prompt"],
                    str(session["refinement_iterations"]),
                    session["last_updated"],
                    str(session["artifacts_count"]),
                )

            console.print(table)
            console.print("\n[dim]Use 'll3m resume <session_id>' to continue a session[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to list sessions: {e}")
        raise click.ClickException(str(e)) from e


@cli.command()
@click.argument("session_id", required=True)
def show_resume_prompt(session_id: str) -> None:
    """Show detailed resumption information for a session."""
    from .output.workflow_integration import WorkflowOutputIntegration

    try:
        integration = WorkflowOutputIntegration()
        resume_prompt = integration.create_session_resume_prompt(session_id)

        console.print(f"[bold blue]Resume Information for Session: {session_id}[/bold blue]")
        console.print()
        console.print(resume_prompt)

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Failed to show resume prompt: {e}")
        raise click.ClickException(str(e)) from e


@cli.command()
@click.option(
    "--format",
    "-f",
    default="table",
    help="Output format (table, json)",
    type=click.Choice(["table", "json"]),
)
def list_assets(_format: str) -> None:
    """List all generated assets."""
    # This would typically query a database or asset management system
    console.print("[yellow]Asset listing not yet implemented[/yellow]")


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
    except Exception as unexpected_error:
        console.print(f"\n[red]Unexpected error: {unexpected_error!s}[/red]")
        logger.exception("CLI error", error=str(unexpected_error))


if __name__ == "__main__":
    main()
