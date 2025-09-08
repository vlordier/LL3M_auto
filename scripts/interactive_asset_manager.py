#!/usr/bin/env python3
"""Interactive Asset Manager - CLI for browsing and refining 3D assets."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.output.directory_manager import OutputDirectoryManager
from src.output.asset_browser import AssetBrowser
from src.output.interactive_refiner import InteractiveRefiner
from rich.console import Console
from rich.panel import Panel

console = Console()


async def create_sample_project():
    """Create a sample project for demonstration."""
    directory_manager = OutputDirectoryManager()
    
    project_dir = directory_manager.create_project(
        name="Demo Project", 
        description="Sample project to demonstrate the interactive system"
    )
    
    # Create a sample session
    session_dir = directory_manager.create_session(project_dir)
    
    # Create some sample asset directories
    sample_assets = [
        ("Red Cube", 1, "A simple red cube for testing"),
        ("Wooden Table", 2, "A wooden table with realistic materials"),
        ("Living Room Scene", 4, "A complete living room with furniture and lighting"),
    ]
    
    for asset_name, complexity, description in sample_assets:
        asset_dir = directory_manager.create_asset_directory(
            session_dir, asset_name, complexity, stage="raw"
        )
        
        # Add some sample files
        (asset_dir / "source" / "generated_script.py").touch()
        (asset_dir / "outputs" / "render.png").touch()
        (asset_dir / "exports" / "asset.obj").touch()
        
    console.print(f"[green]✅ Created sample project at: {project_dir}[/green]")
    return project_dir


async def quick_refine(asset_path: str, instruction: str):
    """Quick refinement without interactive browser."""
    asset_path = Path(asset_path)
    
    if not asset_path.exists():
        console.print(f"[red]❌ Asset path not found: {asset_path}[/red]")
        return
        
    directory_manager = OutputDirectoryManager()
    refiner = InteractiveRefiner(directory_manager)
    
    console.print(f"[cyan]🛠️ Refining asset: {asset_path.name}[/cyan]")
    console.print(f"[yellow]📝 Instruction: {instruction}[/yellow]")
    
    # Start refinement session
    session = await refiner.start_refinement_session(
        asset_path, 
        f"Original asset: {asset_path.name}"
    )
    
    # Parse and apply instruction
    parsed_instruction = refiner.parse_instruction(instruction)
    console.print(f"[green]✅ Parsed as: {parsed_instruction.intent} (confidence: {parsed_instruction.confidence:.2f})[/green]")
    
    # Apply refinement
    console.print("[yellow]🔄 Applying refinement...[/yellow]")
    result = await refiner.apply_instruction(session, parsed_instruction)
    
    if result.get("success"):
        console.print("[green]✅ Refinement completed successfully![/green]")
        console.print(f"[cyan]📁 Results saved in: {asset_path / 'refinements'}[/cyan]")
    else:
        console.print(f"[red]❌ Refinement failed: {result.get('error', 'Unknown error')}[/red]")


def list_recent_assets():
    """List recent assets for quick access."""
    directory_manager = OutputDirectoryManager()
    assets = directory_manager.find_assets()
    
    if not assets:
        console.print("[yellow]No assets found.[/yellow]")
        return
        
    console.print(Panel("📋 Recent Assets", border_style="blue"))
    
    # Show top 10 most recent
    for i, asset in enumerate(assets[:10]):
        quality = asset.get("quality_score", 0)
        quality_str = f" (Q: {quality:.1f})" if quality else ""
        console.print(f"{i:2d}. {asset.get('asset_name', 'Unnamed')}{quality_str}")
        console.print(f"     📁 {asset['path']}")
        console.print(f"     📅 {asset.get('created_at', 'Unknown')[:19]}")
        console.print()


def show_statistics():
    """Show system statistics."""
    directory_manager = OutputDirectoryManager()
    stats = directory_manager.get_directory_stats()
    
    console.print(Panel("📊 System Statistics", border_style="magenta"))
    
    console.print(f"📁 Projects: {stats['total_projects']}")
    console.print(f"💾 Storage: {stats['total_size_mb']:.1f} MB")
    console.print(f"🌟 Gallery: {sum(stats['gallery_counts'].values())} assets")
    
    console.print("\n📈 Complexity Levels:")
    for level, count in stats["top_complexity_levels"].items():
        console.print(f"   Level {level}: {count} assets")
        
    console.print("\n🏛️ Gallery Categories:")
    for category, count in stats["gallery_counts"].items():
        console.print(f"   {category.title()}: {count} assets")


async def batch_refine():
    """Interactive batch refinement of multiple assets."""
    directory_manager = OutputDirectoryManager()
    assets = directory_manager.find_assets()
    
    if not assets:
        console.print("[yellow]No assets found for batch refinement.[/yellow]")
        return
        
    console.print(Panel("🔄 Batch Refinement", border_style="green"))
    console.print(f"Found {len(assets)} assets")
    
    # Filter selection
    console.print("\nFilter options:")
    min_quality = input("Minimum quality score to refine (or Enter for all): ").strip()
    if min_quality:
        try:
            min_quality = float(min_quality)
            assets = [a for a in assets if (a.get("quality_score") or 0) >= min_quality]
        except ValueError:
            pass
            
    complexity = input("Complexity level to refine (1-5, or Enter for all): ").strip()
    if complexity and complexity.isdigit():
        complexity = int(complexity)
        assets = [a for a in assets if a.get("complexity_level") == complexity]
        
    console.print(f"Selected {len(assets)} assets for batch refinement")
    
    if not assets:
        console.print("[yellow]No assets match the filters.[/yellow]")
        return
        
    # Get batch instruction
    instruction = input("\nEnter refinement instruction to apply to all assets: ").strip()
    if not instruction:
        console.print("[red]Instruction is required for batch refinement.[/red]")
        return
        
    # Confirm
    confirm = input(f"Apply '{instruction}' to {len(assets)} assets? (y/n): ").lower()
    if confirm != 'y':
        console.print("Batch refinement cancelled.")
        return
        
    # Apply to each asset
    refiner = InteractiveRefiner(directory_manager)
    success_count = 0
    
    for i, asset in enumerate(assets):
        console.print(f"\n[cyan]Processing {i+1}/{len(assets)}: {asset.get('asset_name', 'Unnamed')}[/cyan]")
        
        try:
            asset_path = Path(asset["path"])
            session = await refiner.start_refinement_session(
                asset_path,
                f"Batch refinement of {asset.get('asset_name', 'unnamed')}"
            )
            
            parsed_instruction = refiner.parse_instruction(instruction)
            result = await refiner.apply_instruction(session, parsed_instruction)
            
            if result.get("success"):
                success_count += 1
                console.print("[green]✅ Success[/green]")
            else:
                console.print(f"[red]❌ Failed: {result.get('error', 'Unknown')}[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            
    console.print(f"\n[bold green]Batch refinement completed: {success_count}/{len(assets)} successful[/bold green]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Interactive 3D Asset Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Browse command
    browse_parser = subparsers.add_parser("browse", help="Launch interactive asset browser")
    
    # List command  
    list_parser = subparsers.add_parser("list", help="List recent assets")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    
    # Refine command
    refine_parser = subparsers.add_parser("refine", help="Refine a specific asset")
    refine_parser.add_argument("asset_path", help="Path to asset directory")
    refine_parser.add_argument("instruction", help="Refinement instruction")
    
    # Batch refine command
    batch_parser = subparsers.add_parser("batch", help="Batch refine multiple assets")
    
    # Create sample command
    sample_parser = subparsers.add_parser("sample", help="Create sample project for testing")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup directory structure")
    setup_parser.add_argument("--base-dir", default="outputs", help="Base output directory")
    
    args = parser.parse_args()
    
    if args.command == "browse":
        console.print(Panel(
            "🚀 Starting Interactive Asset Browser",
            subtitle="Navigate with menu options",
            border_style="blue"
        ))
        asyncio.run(browse_mode())
        
    elif args.command == "list":
        list_recent_assets()
        
    elif args.command == "stats":
        show_statistics()
        
    elif args.command == "refine":
        asyncio.run(quick_refine(args.asset_path, args.instruction))
        
    elif args.command == "batch":
        asyncio.run(batch_refine())
        
    elif args.command == "sample":
        asyncio.run(create_sample_project())
        
    elif args.command == "setup":
        directory_manager = OutputDirectoryManager(Path(args.base_dir))
        console.print(f"[green]✅ Setup completed in: {args.base_dir}[/green]")
        
    else:
        # Default: show help and quick start options
        console.print(Panel(
            "🎨 Interactive 3D Asset Manager",
            subtitle="Organize, browse, and refine your 3D assets with natural language",
            border_style="cyan"
        ))
        
        console.print("\n[bold cyan]Quick Start:[/bold cyan]")
        console.print("  🔍 python interactive_asset_manager.py browse   - Launch interactive browser")
        console.print("  📋 python interactive_asset_manager.py list     - List recent assets")
        console.print("  📊 python interactive_asset_manager.py stats    - Show statistics")
        console.print("  🛠️  python interactive_asset_manager.py refine   - Quick refinement")
        console.print("  🔄 python interactive_asset_manager.py batch    - Batch refinement")
        console.print("  📦 python interactive_asset_manager.py sample   - Create sample project")
        
        console.print("\n[bold yellow]Examples:[/bold yellow]")
        console.print("  # Quick refinement")
        console.print('  python interactive_asset_manager.py refine ./outputs/projects/my_project_*/assets/*/raw/* "make it blue"')
        console.print("\n  # Interactive browser")
        console.print("  python interactive_asset_manager.py browse")
        
        # Show current system status
        try:
            directory_manager = OutputDirectoryManager()
            stats = directory_manager.get_directory_stats()
            console.print(f"\n[dim]Current system: {stats['total_projects']} projects, {stats['total_size_mb']:.1f}MB[/dim]")
        except Exception:
            console.print("\n[dim]Run 'python interactive_asset_manager.py setup' to initialize[/dim]")


async def browse_mode():
    """Launch the interactive browser."""
    directory_manager = OutputDirectoryManager()
    browser = AssetBrowser(directory_manager)
    
    try:
        browser.show_dashboard()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Exiting interactive browser...[/yellow]")


if __name__ == "__main__":
    main()