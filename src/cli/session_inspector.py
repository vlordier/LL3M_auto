"""Session Inspector CLI

Provides comprehensive tools for inspecting and analyzing stateful sessions:
- List and filter sessions
- View detailed execution traces
- Compare code versions
- Analyze workflow performance
- Resume interrupted sessions
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.session_integration import resume_session_tracker
from src.services.stateful_session_manager import StatefulSessionManager

console = Console()


def format_duration(seconds: float | None) -> str:
    """Format duration in human-readable format."""
    if seconds is None:
        return "N/A"

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def list_sessions_command(manager: StatefulSessionManager, limit: int = 20,
                         filter_phase: str | None = None, filter_success: bool | None = None):
    """List all sessions with filtering options."""
    sessions = manager.list_sessions(limit=limit)

    # Apply filters
    if filter_phase:
        sessions = [s for s in sessions if s['current_phase'] == filter_phase]

    if filter_success is not None:
        sessions = [s for s in sessions if s['success'] == filter_success]

    if not sessions:
        console.print("[yellow]No sessions found matching criteria[/yellow]")
        return

    # Create table
    table = Table(title=f"Sessions ({len(sessions)} found)")
    table.add_column("Session ID", style="cyan")
    table.add_column("Prompt", style="white", width=40)
    table.add_column("Asset Name", style="green")
    table.add_column("Phase", style="magenta")
    table.add_column("Success", style="bold")
    table.add_column("Created", style="dim")
    table.add_column("Completion", style="blue")

    for session in sessions:
        success_text = "[green]✓[/green]" if session.get('success') else "[red]✗[/red]"
        phase_text = session.get('current_phase', 'unknown')
        completion = session.get('completion_percentage', 0)

        table.add_row(
            session['session_id'][-12:],  # Show last 12 chars
            session['prompt'][:40] + ("..." if len(session['prompt']) > 40 else ""),
            session['asset_name'],
            phase_text,
            success_text,
            format_timestamp(datetime.fromisoformat(session['created_at'])),
            f"{completion:.0f}%"
        )

    console.print(table)


def inspect_session_command(manager: StatefulSessionManager, session_id: str,
                           show_code: bool = False, show_traces: bool = False):
    """Inspect a specific session in detail."""
    session_state = manager.load_session(session_id)

    if not session_state:
        console.print(f"[red]Session {session_id} not found[/red]")
        return

    # Session overview
    console.print(Panel(
        f"[bold cyan]Session: {session_state.session_id}[/bold cyan]\n"
        f"[white]Prompt:[/white] {session_state.prompt}\n"
        f"[white]Asset:[/white] {session_state.asset_name}\n"
        f"[white]Phase:[/white] {session_state.current_phase.value}\n"
        f"[white]Success:[/white] {'✓' if session_state.success else '✗'}\n"
        f"[white]Completion:[/white] {session_state.completion_percentage:.1f}%\n"
        f"[white]Duration:[/white] {format_duration(session_state.total_duration)}\n"
        f"[white]Created:[/white] {format_timestamp(session_state.created_at)}\n"
        f"[white]Updated:[/white] {format_timestamp(session_state.updated_at)}",
        title="Session Overview"
    ))

    # Artifacts summary
    if session_state.artifacts:
        artifact_table = Table(title="Artifacts")
        artifact_table.add_column("Type", style="cyan")
        artifact_table.add_column("File", style="white")
        artifact_table.add_column("Phase", style="magenta")
        artifact_table.add_column("Version", style="green")
        artifact_table.add_column("Size", style="dim")

        for artifact in session_state.artifacts[-10:]:  # Show last 10
            size_text = f"{artifact.size_bytes:,} bytes" if artifact.size_bytes else "N/A"
            artifact_table.add_row(
                artifact.artifact_type.value,
                artifact.relative_path,
                artifact.phase.value,
                str(artifact.version),
                size_text
            )

        console.print(artifact_table)

    # Code versions summary
    if session_state.code_versions and show_code:
        console.print("\n[bold yellow]Code Evolution:[/bold yellow]")

        for version in session_state.code_versions:
            status_color = {
                'valid': 'green',
                'invalid': 'red',
                'healing': 'yellow',
                'final': 'blue'
            }.get(version.validation_status, 'white')

            console.print(Panel(
                f"[white]Phase:[/white] {version.source_phase.value}\n"
                f"[white]Status:[/white] [{status_color}]{version.validation_status}[/{status_color}]\n"
                f"[white]Issues:[/white] {len(version.ruff_issues)} ruff, {len(version.execution_errors)} execution\n"
                f"[white]Healing:[/white] {'Yes' if version.healing_applied else 'No'}\n"
                f"[white]Quality:[/white] {version.quality_score or 'N/A'}",
                title=f"Code Version {version.version}"
            ))

            if len(version.code_content) < 2000:  # Show code for small versions
                console.print(Syntax(version.code_content, "python", theme="monokai", line_numbers=True))
            else:
                console.print(f"[dim]Code too long ({len(version.code_content)} chars), use --show-code-file[/dim]")

    # Execution traces
    if session_state.traces and show_traces:
        console.print("\n[bold yellow]Execution Traces:[/bold yellow]")

        for trace in session_state.traces:
            status_color = {
                'completed': 'green',
                'running': 'yellow',
                'failed': 'red',
                'skipped': 'dim'
            }.get(trace.status, 'white')

            llm_calls = len(trace.llm_calls)
            agent_actions = len(trace.agent_actions)
            errors = len(trace.errors)

            console.print(Panel(
                f"[white]Duration:[/white] {format_duration(trace.duration_seconds)}\n"
                f"[white]Status:[/white] [{status_color}]{trace.status}[/{status_color}]\n"
                f"[white]LLM Calls:[/white] {llm_calls}\n"
                f"[white]Agent Actions:[/white] {agent_actions}\n"
                f"[white]Errors:[/white] {errors}\n"
                f"[white]Logs:[/white] {len(trace.logs)} entries",
                title=f"Phase: {trace.phase.value}"
            ))


def compare_code_versions_command(manager: StatefulSessionManager, session_id: str,
                                 version1: int, version2: int):
    """Compare two code versions from a session."""
    session_state = manager.load_session(session_id)

    if not session_state:
        console.print(f"[red]Session {session_id} not found[/red]")
        return

    versions = session_state.code_versions

    if version1 > len(versions) or version2 > len(versions):
        console.print(f"[red]Version not found. Available versions: 1-{len(versions)}[/red]")
        return

    v1 = versions[version1 - 1]
    v2 = versions[version2 - 1]

    # Show version comparison
    console.print(f"[bold]Comparing Version {version1} vs Version {version2}[/bold]")

    # Create side-by-side comparison
    table = Table()
    table.add_column(f"Version {version1}", style="cyan")
    table.add_column(f"Version {version2}", style="magenta")

    table.add_row(
        f"Phase: {v1.source_phase.value}",
        f"Phase: {v2.source_phase.value}"
    )
    table.add_row(
        f"Status: {v1.validation_status}",
        f"Status: {v2.validation_status}"
    )
    table.add_row(
        f"Ruff Issues: {len(v1.ruff_issues)}",
        f"Ruff Issues: {len(v2.ruff_issues)}"
    )
    table.add_row(
        f"Execution Errors: {len(v1.execution_errors)}",
        f"Execution Errors: {len(v2.execution_errors)}"
    )
    table.add_row(
        f"Healing Applied: {v1.healing_applied}",
        f"Healing Applied: {v2.healing_applied}"
    )
    table.add_row(
        f"Quality Score: {v1.quality_score or 'N/A'}",
        f"Quality Score: {v2.quality_score or 'N/A'}"
    )

    console.print(table)

    # Show code differences (simplified)
    console.print("\n[bold yellow]Code Differences:[/bold yellow]")
    console.print(f"Version {version1}: {len(v1.code_content)} characters")
    console.print(f"Version {version2}: {len(v2.code_content)} characters")


def resume_session_command(session_id: str):
    """Resume an interrupted session."""
    tracker = resume_session_tracker(session_id)

    if not tracker:
        console.print(f"[red]Cannot resume session {session_id}[/red]")
        return

    console.print(f"[green]Successfully loaded session {session_id}[/green]")
    console.print(f"Current phase: {tracker.session_state.current_phase.value}")
    console.print(f"Completion: {tracker.session_state.completion_percentage:.1f}%")

    # TODO: Integrate with orchestrator to actually resume execution
    console.print("[yellow]Note: Actual resumption requires orchestrator integration[/yellow]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LL3M Session Inspector")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List sessions
    list_parser = subparsers.add_parser("list", help="List sessions")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum sessions to show")
    list_parser.add_argument("--phase", help="Filter by phase")
    list_parser.add_argument("--success", type=bool, help="Filter by success status")

    # Inspect session
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a session")
    inspect_parser.add_argument("session_id", help="Session ID to inspect")
    inspect_parser.add_argument("--code", action="store_true", help="Show code versions")
    inspect_parser.add_argument("--traces", action="store_true", help="Show execution traces")

    # Compare code versions
    compare_parser = subparsers.add_parser("compare", help="Compare code versions")
    compare_parser.add_argument("session_id", help="Session ID")
    compare_parser.add_argument("version1", type=int, help="First version number")
    compare_parser.add_argument("version2", type=int, help="Second version number")

    # Resume session
    resume_parser = subparsers.add_parser("resume", help="Resume a session")
    resume_parser.add_argument("session_id", help="Session ID to resume")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = StatefulSessionManager()

    if args.command == "list":
        list_sessions_command(manager, args.limit, args.phase, args.success)
    elif args.command == "inspect":
        inspect_session_command(manager, args.session_id, args.code, args.traces)
    elif args.command == "compare":
        compare_code_versions_command(manager, args.session_id, args.version1, args.version2)
    elif args.command == "resume":
        resume_session_command(args.session_id)


if __name__ == "__main__":
    main()
