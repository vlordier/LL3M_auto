"""Asset browser and viewer for easy navigation and interaction with generated 3D assets."""

import json
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .directory_manager import OutputDirectoryManager
from .interactive_refiner import InteractiveRefiner

logger = structlog.get_logger(__name__)


class AssetBrowser:
    """Interactive browser for navigating and managing 3D assets."""

    def __init__(self, directory_manager: OutputDirectoryManager):
        """Initialize the asset browser.

        Args:
            directory_manager: Directory manager instance
        """
        self.directory_manager = directory_manager
        self.refiner = InteractiveRefiner(directory_manager)
        self.console = Console()

    def show_dashboard(self):
        """Display the main dashboard with overview."""
        self.console.clear()

        # Header
        self.console.print(Panel(
            "🎨 LL3M Asset Browser & Interactive Refinement System",
            title="Dashboard",
            border_style="blue"
        ))

        # Statistics
        stats = self.directory_manager.get_directory_stats()

        stats_table = Table(title="📊 System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        stats_table.add_row("Total Projects", str(stats["total_projects"]))
        stats_table.add_row("Storage Used", f"{stats['total_size_mb']:.1f} MB")
        stats_table.add_row("Gallery Assets", str(sum(stats["gallery_counts"].values())))

        complexity_stats = stats["top_complexity_levels"]
        for level, count in complexity_stats.items():
            stats_table.add_row(f"Level {level} Assets", str(count))

        self.console.print(stats_table)
        self.console.print()

        # Recent projects
        recent_projects = self.directory_manager.get_recent_projects(5)
        if recent_projects:
            projects_table = Table(title="📁 Recent Projects")
            projects_table.add_column("Project", style="green")
            projects_table.add_column("Description", style="white")
            projects_table.add_column("Created", style="yellow")
            projects_table.add_column("Status", style="cyan")

            for project in recent_projects:
                created = project.get("created_at", "")[:10]  # Just date part
                projects_table.add_row(
                    project.get("name", "Unnamed"),
                    project.get("description", "No description")[:50],
                    created,
                    project.get("status", "unknown")
                )

            self.console.print(projects_table)

        self.console.print()
        self._show_main_menu()

    def _show_main_menu(self):
        """Show the main navigation menu."""
        menu_options = """
[bold cyan]Main Menu:[/bold cyan]

1. 🔍 Browse Projects
2. 🎨 Browse Gallery
3. 🛠️  Interactive Refinement
4. 📊 Asset Search
5. 🗂️  Create New Project
6. 🧹 Cleanup & Maintenance
7. 📈 Detailed Statistics
8. ❌ Exit

Choose an option (1-8): """

        self.console.print(Panel(menu_options, border_style="green"))

        try:
            choice = input().strip()
            self._handle_main_menu_choice(choice)
        except KeyboardInterrupt:
            self.console.print("\n👋 Goodbye!")

    def _handle_main_menu_choice(self, choice: str):
        """Handle main menu selection."""
        if choice == "1":
            self.browse_projects()
        elif choice == "2":
            self.browse_gallery()
        elif choice == "3":
            self.interactive_refinement_menu()
        elif choice == "4":
            self.asset_search()
        elif choice == "5":
            self.create_new_project()
        elif choice == "6":
            self.cleanup_menu()
        elif choice == "7":
            self.detailed_statistics()
        elif choice == "8":
            self.console.print("👋 Goodbye!")
            return
        else:
            self.console.print("[red]Invalid option. Please try again.[/red]")
            input("Press Enter to continue...")
            self.show_dashboard()

    def browse_projects(self):
        """Browse all projects."""
        self.console.clear()
        projects = self.directory_manager.get_recent_projects(20)

        if not projects:
            self.console.print("[yellow]No projects found.[/yellow]")
            input("Press Enter to return to main menu...")
            self.show_dashboard()
            return

        self.console.print(Panel("🔍 Project Browser", border_style="blue"))

        # Create project tree
        tree = Tree("📁 Projects")

        for i, project in enumerate(projects):
            project_node = tree.add(f"[green]{project.get('name', 'Unnamed')}[/green]")
            project_node.add(f"📅 Created: {project.get('created_at', '')[:10]}")
            project_node.add(f"📊 Status: {project.get('status', 'unknown')}")
            project_node.add(f"📝 Description: {project.get('description', 'No description')[:60]}")

            # Add assets count
            project_path = Path(project["path"])
            assets = self.directory_manager.find_assets(project_path)
            project_node.add(f"🎨 Assets: {len(assets)}")

        self.console.print(tree)

        # Project selection menu
        self.console.print("\n[bold cyan]Options:[/bold cyan]")
        self.console.print("Enter project number (0-based) to view details, or 'back' to return: ", end="")

        try:
            user_input = input().strip().lower()
            if user_input == "back":
                self.show_dashboard()
                return

            project_idx = int(user_input)
            if 0 <= project_idx < len(projects):
                self.view_project_details(Path(projects[project_idx]["path"]))
            else:
                self.console.print("[red]Invalid project number.[/red]")
                input("Press Enter to continue...")
                self.browse_projects()
        except (ValueError, IndexError):
            self.console.print("[red]Invalid input.[/red]")
            input("Press Enter to continue...")
            self.browse_projects()

    def view_project_details(self, project_path: Path):
        """View detailed information about a specific project."""
        self.console.clear()

        # Load project metadata
        metadata_file = project_path / "metadata" / "project.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                project_meta = json.load(f)
        else:
            project_meta = {"name": project_path.name}

        self.console.print(Panel(
            f"📁 Project: {project_meta.get('name', 'Unnamed')}",
            subtitle=f"Path: {project_path}",
            border_style="green"
        ))

        # Project information
        info_table = Table(title="Project Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Name", project_meta.get("name", "Unnamed"))
        info_table.add_row("Description", project_meta.get("description", "No description"))
        info_table.add_row("Created", project_meta.get("created_at", "Unknown"))
        info_table.add_row("Status", project_meta.get("status", "Unknown"))
        info_table.add_row("Total Assets", str(project_meta.get("total_assets", 0)))

        self.console.print(info_table)

        # Find and display assets
        assets = self.directory_manager.find_assets(project_path)

        if assets:
            assets_table = Table(title="🎨 Project Assets")
            assets_table.add_column("Asset Name", style="green")
            assets_table.add_column("Level", style="cyan")
            assets_table.add_column("Stage", style="yellow")
            assets_table.add_column("Quality", style="magenta")
            assets_table.add_column("Created", style="white")

            for asset in assets[:10]:  # Show top 10
                quality = asset.get("quality_score", 0)
                quality_str = f"{quality:.1f}" if quality else "N/A"
                created = asset.get("created_at", "")[:16]  # Date + time

                assets_table.add_row(
                    asset.get("asset_name", "Unnamed"),
                    str(asset.get("complexity_level", "?")),
                    asset.get("stage", "unknown"),
                    quality_str,
                    created
                )

            self.console.print(assets_table)

            # Asset selection options
            self.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.console.print("1. 🔍 View Asset Details")
            self.console.print("2. 🛠️  Refine Asset")
            self.console.print("3. 🌟 Promote to Gallery")
            self.console.print("4. 🗂️  Browse Other Projects")
            self.console.print("5. 🏠 Main Menu")

            choice = input("\nChoose option: ").strip()

            if choice == "1":
                asset_idx = int(input("Enter asset index (0-based): "))
                if 0 <= asset_idx < len(assets):
                    self.view_asset_details(Path(assets[asset_idx]["path"]))
            elif choice == "2":
                asset_idx = int(input("Enter asset index to refine (0-based): "))
                if 0 <= asset_idx < len(assets):
                    self.refine_asset_interactive(Path(assets[asset_idx]["path"]))
            elif choice == "3":
                asset_idx = int(input("Enter asset index to promote (0-based): "))
                if 0 <= asset_idx < len(assets):
                    category = input("Gallery category (featured/recent/favorites): ").strip()
                    if category in ["featured", "recent", "favorites"]:
                        self.directory_manager.promote_to_gallery(Path(assets[asset_idx]["path"]), category)
                        self.console.print(f"[green]✅ Promoted to {category} gallery![/green]")
            elif choice == "4":
                self.browse_projects()
                return
            elif choice == "5":
                self.show_dashboard()
                return
        else:
            self.console.print("[yellow]No assets found in this project.[/yellow]")

        input("Press Enter to continue...")
        self.view_project_details(project_path)

    def view_asset_details(self, asset_path: Path):
        """View detailed information about a specific asset."""
        self.console.clear()

        # Load asset metadata
        metadata_file = asset_path / "asset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                asset_meta = json.load(f)
        else:
            asset_meta = {"asset_name": asset_path.name}

        self.console.print(Panel(
            f"🎨 Asset: {asset_meta.get('asset_name', 'Unnamed')}",
            subtitle=f"Path: {asset_path}",
            border_style="cyan"
        ))

        # Asset details table
        details_table = Table(title="Asset Details")
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="white")

        details_table.add_row("Name", asset_meta.get("asset_name", "Unnamed"))
        details_table.add_row("Complexity Level", str(asset_meta.get("complexity_level", "?")))
        details_table.add_row("Stage", asset_meta.get("stage", "unknown"))
        details_table.add_row("Status", asset_meta.get("status", "unknown"))
        details_table.add_row("Created", asset_meta.get("created_at", "Unknown"))

        quality = asset_meta.get("quality_score")
        if quality:
            details_table.add_row("Quality Score", f"{quality:.1f}/10")

        user_rating = asset_meta.get("user_rating")
        if user_rating:
            details_table.add_row("User Rating", f"{user_rating}/10")

        self.console.print(details_table)

        # File listing
        files_table = Table(title="Generated Files")
        files_table.add_column("Directory", style="yellow")
        files_table.add_column("Files", style="green")

        for subdir in asset_path.iterdir():
            if subdir.is_dir():
                files = [f.name for f in subdir.iterdir() if f.is_file()]
                files_str = ", ".join(files[:5])  # Show first 5 files
                if len(files) > 5:
                    files_str += f" ... (+{len(files)-5} more)"
                files_table.add_row(subdir.name, files_str)

        self.console.print(files_table)

        # Options
        self.console.print("\n[bold cyan]Options:[/bold cyan]")
        self.console.print("1. 📁 Open Asset Directory")
        self.console.print("2. 🖼️  View Screenshots (if available)")
        self.console.print("3. 🛠️  Start Refinement")
        self.console.print("4. 🌟 Rate Asset")
        self.console.print("5. ⬅️  Back")

        choice = input("\nChoose option: ").strip()

        if choice == "1":
            self._open_directory(asset_path)
        elif choice == "2":
            self._view_screenshots(asset_path)
        elif choice == "3":
            self.refine_asset_interactive(asset_path)
            return
        elif choice == "4":
            rating = int(input("Rate this asset (1-10): "))
            if 1 <= rating <= 10:
                asset_meta["user_rating"] = rating
                with open(metadata_file, 'w') as f:
                    json.dump(asset_meta, f, indent=2)
                self.console.print(f"[green]✅ Rated {rating}/10![/green]")
        elif choice == "5":
            return

        input("Press Enter to continue...")
        self.view_asset_details(asset_path)

    def refine_asset_interactive(self, asset_path: Path):
        """Start interactive refinement for an asset."""
        self.console.clear()

        # Load asset metadata to get original prompt
        metadata_file = asset_path / "asset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                asset_meta = json.load(f)
        else:
            asset_meta = {}

        asset_name = asset_meta.get("asset_name", asset_path.name)

        self.console.print(Panel(
            f"🛠️ Interactive Refinement: {asset_name}",
            subtitle="Give natural language instructions to improve the asset",
            border_style="green"
        ))

        # Check for existing refinement session
        refinement_dir = asset_path / "refinements"
        session_files = list(refinement_dir.glob("refine_*.json")) if refinement_dir.exists() else []

        if session_files:
            self.console.print(f"[yellow]Found {len(session_files)} existing refinement session(s).[/yellow]")
            use_existing = input("Continue existing session? (y/n): ").strip().lower()

            if use_existing == 'y':
                # Load most recent session
                latest_session_file = max(session_files, key=lambda p: p.stat().st_mtime)
                session = self.refiner.load_session(latest_session_file)
                self.console.print(f"[green]Loaded session with {len(session.instructions)} instructions.[/green]")
            else:
                # Create new session
                original_prompt = asset_meta.get("original_prompt", "3D asset")
                session = await self.refiner.start_refinement_session(asset_path, original_prompt)
        else:
            # Create new session
            original_prompt = input("Enter original prompt (or press Enter for default): ").strip()
            if not original_prompt:
                original_prompt = asset_meta.get("asset_name", "3D asset")
            session = await self.refiner.start_refinement_session(asset_path, original_prompt)

        self.console.print(f"\n[bold cyan]Original Prompt:[/bold cyan] {session.original_prompt}")

        # Show existing instructions if any
        if session.instructions:
            self.console.print("\n[bold yellow]Previous Instructions:[/bold yellow]")
            for i, inst in enumerate(session.instructions, 1):
                self.console.print(f"{i}. {inst.instruction_text}")

        # Interactive refinement loop
        while True:
            self.console.print("\n" + "="*60)
            self.console.print("[bold green]Enter refinement instruction (or 'quit' to exit):[/bold green]")

            # Show suggestions
            suggestions = self.refiner.get_refinement_suggestions(asset_path)
            self.console.print("\n[dim]💡 Suggestions:[/dim]")
            for i, suggestion in enumerate(suggestions[:3], 1):
                self.console.print(f"[dim]{i}. {suggestion}[/dim]")

            user_input = input("\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'done']:
                break

            if not user_input:
                self.console.print("[yellow]Please enter an instruction or 'quit' to exit.[/yellow]")
                continue

            # Parse and apply instruction
            self.console.print(f"[cyan]Processing instruction: {user_input}[/cyan]")

            try:
                instruction = self.refiner.parse_instruction(user_input)
                self.console.print(f"[green]✅ Parsed as: {instruction.intent} (confidence: {instruction.confidence:.2f})[/green]")

                # Apply the instruction
                self.console.print("[yellow]🔄 Generating refined asset...[/yellow]")
                result = await self.refiner.apply_instruction(session, instruction)

                if result.get("success"):
                    self.console.print(f"[green]✅ Refinement iteration {session.current_iteration} completed![/green]")

                    # Show result summary
                    if "quality_score" in str(result):
                        self.console.print("[cyan]Quality assessment will be available after generation completes.[/cyan]")
                else:
                    self.console.print(f"[red]❌ Refinement failed: {result.get('error', 'Unknown error')}[/red]")

            except Exception as e:
                self.console.print(f"[red]❌ Error processing instruction: {e}[/red]")
                logger.error(f"Refinement error: {e}")

        # Session summary
        self.console.print("\n[bold green]Refinement session completed![/bold green]")
        self.console.print(f"Total instructions applied: {len(session.instructions)}")
        self.console.print(f"Current iteration: {session.current_iteration}")

        # Ask for user satisfaction rating
        try:
            satisfaction = int(input("\nRate your satisfaction with the refinements (1-10): "))
            if 1 <= satisfaction <= 10:
                session.user_satisfaction = satisfaction
                # Save updated session
                refinement_dir = asset_path / "refinements"
                session_file = refinement_dir / f"{session.session_id}.json"
                with open(session_file, 'w') as f:
                    json.dump(session.model_dump(), f, indent=2, default=str)
        except ValueError:
            pass

        input("\nPress Enter to return to main menu...")
        self.show_dashboard()

    def browse_gallery(self):
        """Browse the gallery of promoted assets."""
        self.console.clear()
        self.console.print(Panel("🌟 Asset Gallery", border_style="magenta"))

        categories = ["featured", "recent", "favorites", "archived"]

        for category in categories:
            category_dir = self.directory_manager.gallery_dir / category
            if not category_dir.exists():
                continue

            assets = list(category_dir.iterdir())
            self.console.print(f"\n[bold {self._get_category_color(category)}]{category.title()} ({len(assets)} assets)[/bold {self._get_category_color(category)}]")

            if assets:
                for asset_link in assets[:5]:  # Show first 5
                    if asset_link.is_symlink():
                        asset_path = asset_link.resolve()
                        metadata_file = asset_path / "asset_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                meta = json.load(f)
                            name = meta.get("asset_name", asset_link.name)
                            quality = meta.get("quality_score", 0)
                            quality_str = f" (Q: {quality:.1f})" if quality else ""
                            self.console.print(f"  • {name}{quality_str}")

        input("\nPress Enter to return to main menu...")
        self.show_dashboard()

    def _get_category_color(self, category: str) -> str:
        """Get color for gallery category."""
        colors = {
            "featured": "gold3",
            "recent": "green",
            "favorites": "red",
            "archived": "dim"
        }
        return colors.get(category, "white")

    def _open_directory(self, path: Path):
        """Open directory in system file manager."""
        import platform
        import subprocess

        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(path)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(path)])
            self.console.print(f"[green]📁 Opened directory: {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Could not open directory: {e}[/red]")

    def _view_screenshots(self, asset_path: Path):
        """View screenshots if available."""
        screenshot_files = []

        # Look for screenshots in various directories
        for subdir in ["outputs", "renders"]:
            subdir_path = asset_path / subdir
            if subdir_path.exists():
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    screenshot_files.extend(subdir_path.glob(ext))

        if screenshot_files:
            self.console.print(f"[green]Found {len(screenshot_files)} screenshot(s)[/green]")
            for screenshot in screenshot_files:
                self.console.print(f"  📸 {screenshot.name}")

            # Try to open first screenshot
            if input("Open first screenshot? (y/n): ").lower() == 'y':
                try:
                    import platform
                    import subprocess
                    if platform.system() == "Darwin":
                        subprocess.run(["open", str(screenshot_files[0])])
                    elif platform.system() == "Windows":
                        subprocess.run(["start", str(screenshot_files[0])], shell=True)
                    else:
                        subprocess.run(["xdg-open", str(screenshot_files[0])])
                except Exception as e:
                    self.console.print(f"[red]Could not open screenshot: {e}[/red]")
        else:
            self.console.print("[yellow]No screenshots found.[/yellow]")

    def create_new_project(self):
        """Create a new project interactively."""
        self.console.clear()
        self.console.print(Panel("🗂️ Create New Project", border_style="green"))

        name = input("Project name: ").strip()
        if not name:
            self.console.print("[red]Project name is required.[/red]")
            input("Press Enter to continue...")
            self.show_dashboard()
            return

        description = input("Project description (optional): ").strip()

        try:
            project_dir = self.directory_manager.create_project(name, description)
            self.console.print(f"[green]✅ Created project: {project_dir}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Failed to create project: {e}[/red]")

        input("Press Enter to continue...")
        self.show_dashboard()

    async def interactive_refinement_menu(self):
        """Show interactive refinement options."""
        self.console.clear()
        self.console.print(Panel("🛠️ Interactive Refinement", border_style="cyan"))

        # Find assets that can be refined
        all_assets = self.directory_manager.find_assets()
        refinable_assets = [a for a in all_assets if a.get("stage") in ["raw", "refined"]]

        if not refinable_assets:
            self.console.print("[yellow]No assets available for refinement.[/yellow]")
            input("Press Enter to return...")
            self.show_dashboard()
            return

        self.console.print(f"Found {len(refinable_assets)} assets available for refinement:")

        # Show refinable assets
        assets_table = Table()
        assets_table.add_column("Index", style="cyan")
        assets_table.add_column("Asset", style="green")
        assets_table.add_column("Level", style="yellow")
        assets_table.add_column("Quality", style="magenta")

        for i, asset in enumerate(refinable_assets[:20]):  # Show top 20
            quality = asset.get("quality_score", 0)
            quality_str = f"{quality:.1f}" if quality else "N/A"
            assets_table.add_row(
                str(i),
                asset.get("asset_name", "Unnamed"),
                str(asset.get("complexity_level", "?")),
                quality_str
            )

        self.console.print(assets_table)

        try:
            asset_idx = int(input("\nEnter asset index to refine: "))
            if 0 <= asset_idx < len(refinable_assets):
                await self.refine_asset_interactive(Path(refinable_assets[asset_idx]["path"]))
            else:
                self.console.print("[red]Invalid asset index.[/red]")
        except ValueError:
            self.console.print("[red]Invalid input.[/red]")

        input("Press Enter to continue...")
        self.show_dashboard()

    def asset_search(self):
        """Search for assets with filters."""
        self.console.clear()
        self.console.print(Panel("🔍 Asset Search", border_style="blue"))

        # Search filters
        self.console.print("Search filters (press Enter to skip):")

        complexity_input = input("Complexity level (1-5): ").strip()
        complexity = int(complexity_input) if complexity_input.isdigit() and 1 <= int(complexity_input) <= 5 else None

        stage = input("Stage (raw/refined/final/variants/experiments): ").strip()
        stage = stage if stage in ["raw", "refined", "final", "variants", "experiments"] else None

        quality_input = input("Minimum quality score (1-10): ").strip()
        min_quality = float(quality_input) if quality_input.replace('.', '').isdigit() else None

        tags_input = input("Tags (comma-separated): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else None

        # Perform search
        results = self.directory_manager.find_assets(
            complexity_level=complexity,
            stage=stage,
            min_quality=min_quality,
            tags=tags
        )

        self.console.print(f"\n[green]Found {len(results)} matching assets:[/green]")

        if results:
            results_table = Table()
            results_table.add_column("Asset", style="green")
            results_table.add_column("Level", style="cyan")
            results_table.add_column("Stage", style="yellow")
            results_table.add_column("Quality", style="magenta")
            results_table.add_column("Created", style="white")

            for result in results[:15]:  # Show top 15
                quality = result.get("quality_score", 0)
                quality_str = f"{quality:.1f}" if quality else "N/A"
                created = result.get("created_at", "")[:10]

                results_table.add_row(
                    result.get("asset_name", "Unnamed"),
                    str(result.get("complexity_level", "?")),
                    result.get("stage", "unknown"),
                    quality_str,
                    created
                )

            self.console.print(results_table)

        input("Press Enter to return...")
        self.show_dashboard()

    def cleanup_menu(self):
        """Cleanup and maintenance menu."""
        self.console.clear()
        self.console.print(Panel("🧹 Cleanup & Maintenance", border_style="red"))

        self.console.print("Cleanup options:")
        self.console.print("1. Remove old temporary files (30+ days)")
        self.console.print("2. Archive completed projects")
        self.console.print("3. Organize gallery")
        self.console.print("4. Back to main menu")

        choice = input("\nChoose option: ").strip()

        if choice == "1":
            days = int(input("Delete files older than how many days? (default 30): ") or "30")
            self.directory_manager.cleanup_old_files(days)
            self.console.print("[green]✅ Cleanup completed![/green]")
        elif choice == "2":
            # Find projects to archive
            projects = self.directory_manager.get_recent_projects()
            self.console.print("Select projects to archive (implement as needed)")
        elif choice == "3":
            self.console.print("Gallery organization (implement as needed)")
        elif choice == "4":
            self.show_dashboard()
            return

        input("Press Enter to continue...")
        self.cleanup_menu()

    def detailed_statistics(self):
        """Show detailed system statistics."""
        self.console.clear()
        stats = self.directory_manager.get_directory_stats()

        self.console.print(Panel("📈 Detailed Statistics", border_style="magenta"))

        # Overview stats
        overview_table = Table(title="System Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="white")

        overview_table.add_row("Total Projects", str(stats["total_projects"]))
        overview_table.add_row("Storage Used", f"{stats['total_size_mb']:.1f} MB")
        overview_table.add_row("Gallery Assets", str(sum(stats["gallery_counts"].values())))

        self.console.print(overview_table)

        # Complexity level breakdown
        complexity_table = Table(title="Assets by Complexity Level")
        complexity_table.add_column("Level", style="cyan")
        complexity_table.add_column("Count", style="green")
        complexity_table.add_column("Percentage", style="yellow")

        total_assets = sum(stats["top_complexity_levels"].values())
        for level, count in stats["top_complexity_levels"].items():
            percentage = (count / total_assets * 100) if total_assets > 0 else 0
            complexity_table.add_row(str(level), str(count), f"{percentage:.1f}%")

        self.console.print(complexity_table)

        # Gallery breakdown
        gallery_table = Table(title="Gallery Categories")
        gallery_table.add_column("Category", style="magenta")
        gallery_table.add_column("Assets", style="green")

        for category, count in stats["gallery_counts"].items():
            gallery_table.add_row(category.title(), str(count))

        self.console.print(gallery_table)

        input("Press Enter to return...")
        self.show_dashboard()


async def main():
    """Main entry point for the asset browser."""
    # Initialize directory manager
    directory_manager = OutputDirectoryManager()

    # Create and run browser
    browser = AssetBrowser(directory_manager)
    browser.show_dashboard()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
