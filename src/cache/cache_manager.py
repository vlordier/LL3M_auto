"""Cache management utilities and CLI for RAG system."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from .retrieval_service import get_retrieval_service
from .vector_db import get_vector_db

logger = structlog.get_logger(__name__)
console = Console()


class CacheManager:
    """Utility class for managing the RAG cache system."""

    def __init__(self):
        """Initialize the cache manager."""
        self.retrieval_service = get_retrieval_service()
        self.vector_db = get_vector_db()

    async def import_from_json(self, json_file: Path) -> dict[str, Any]:
        """Import cache entries from a JSON file.

        Args:
            json_file: Path to JSON file with cache entries

        Returns:
            Import statistics
        """
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with open(json_file) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of cache entries")

        stats = {
            "total": len(data),
            "imported": 0,
            "skipped": 0,
            "errors": 0
        }

        console.print(f"[cyan]Importing {stats['total']} cache entries...[/cyan]")

        with Progress() as progress:
            task = progress.add_task("Importing...", total=stats['total'])

            for entry_data in data:
                try:
                    # Validate required fields
                    if not all(key in entry_data for key in ['prompt', 'code']):
                        stats['errors'] += 1
                        continue

                    # Check if entry already exists
                    existing = await self.vector_db.get_by_prompt(
                        entry_data['prompt'],
                        entry_data.get('complexity_level', 1)
                    )

                    if existing:
                        stats['skipped'] += 1
                    else:
                        await self.vector_db.add_entry(
                            prompt=entry_data['prompt'],
                            code=entry_data['code'],
                            complexity_level=entry_data.get('complexity_level', 1),
                            quality_score=entry_data.get('quality_score'),
                            metadata=entry_data.get('metadata', {})
                        )
                        stats['imported'] += 1

                except Exception as e:
                    logger.error(f"Error importing entry: {e}")
                    stats['errors'] += 1

                progress.advance(task)

        console.print(
            f"[green]Import completed:[/green] {stats['imported']} imported, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        return stats

    async def export_to_json(self, json_file: Path, include_metadata: bool = True) -> int:
        """Export cache entries to a JSON file.

        Args:
            json_file: Output JSON file path
            include_metadata: Include metadata in export

        Returns:
            Number of entries exported
        """
        console.print("[cyan]Exporting cache entries...[/cyan]")

        # Get all entries from database
        import sqlite3
        entries = []

        with sqlite3.connect(self.vector_db.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, prompt, code, embedding, metadata, complexity_level,
                       quality_score, success_rate, usage_count, created_at, last_used
                FROM cache_entries
                ORDER BY created_at DESC
            """)

            for row in cursor:
                entry_data = {
                    "id": row[0],
                    "prompt": row[1],
                    "code": row[2],
                    "complexity_level": row[5],
                    "quality_score": row[6],
                    "success_rate": row[7],
                    "usage_count": row[8],
                    "created_at": row[9]
                }

                if include_metadata:
                    entry_data["metadata"] = json.loads(row[4] or '{}')
                    entry_data["last_used"] = row[10]

                entries.append(entry_data)

        # Write to JSON file
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(entries, f, indent=2, default=str)

        console.print(f"[green]Exported {len(entries)} entries to {json_file}[/green]")
        return len(entries)

    async def cleanup_cache(
        self,
        min_success_rate: float = 0.3,
        min_usage_count: int = 3,
        dry_run: bool = False
    ) -> dict[str, Any]:
        """Clean up low-quality cache entries.

        Args:
            min_success_rate: Minimum success rate to keep
            min_usage_count: Minimum usage count required
            dry_run: If True, only report what would be cleaned

        Returns:
            Cleanup statistics
        """
        console.print(Panel(
            f"Cache Cleanup {'(DRY RUN)' if dry_run else ''}",
            subtitle=f"Removing entries with success rate < {min_success_rate} and usage >= {min_usage_count}",
            border_style="yellow"
        ))

        if dry_run:
            import sqlite3
            with sqlite3.connect(self.vector_db.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM cache_entries
                    WHERE usage_count >= ? AND success_rate < ?
                """, (min_usage_count, min_success_rate))
                count = cursor.fetchone()[0]

            console.print(f"[yellow]Would remove {count} entries[/yellow]")
            return {"would_remove": count, "dry_run": True}
        else:
            removed_count = await self.vector_db.cleanup_low_quality(
                min_success_rate, min_usage_count
            )
            console.print(f"[green]Removed {removed_count} low-quality entries[/green]")
            return {"removed": removed_count, "dry_run": False}

    def display_statistics(self, stats: dict[str, Any]):
        """Display cache statistics in a formatted table."""
        console.print(Panel("📊 RAG Cache Statistics", border_style="blue"))

        # Main stats table
        main_table = Table(title="Cache Overview")
        main_table.add_column("Metric", style="cyan")
        main_table.add_column("Value", style="green")

        main_table.add_row("Total Entries", str(stats.get("total_entries", 0)))
        main_table.add_row("Database Size", f"{stats.get('database_size', 0) / 1024:.1f} KB")

        console.print(main_table)

        # Complexity breakdown
        if "complexity_breakdown" in stats:
            complexity_table = Table(title="By Complexity Level")
            complexity_table.add_column("Level", style="cyan")
            complexity_table.add_column("Count", style="green")
            complexity_table.add_column("Avg Quality", style="yellow")
            complexity_table.add_column("Success Rate", style="magenta")

            for level, level_stats in stats["complexity_breakdown"].items():
                complexity_table.add_row(
                    str(level),
                    str(level_stats["count"]),
                    f"{level_stats['avg_quality']:.1f}",
                    f"{level_stats['avg_success_rate']:.2%}"
                )

            console.print(complexity_table)

        # Top performing entries
        if "top_performing" in stats:
            console.print("\n[bold cyan]🏆 Top Performing Entries:[/bold cyan]")
            for entry in stats["top_performing"][:5]:
                console.print(
                    f"  • {entry['prompt']} "
                    f"(L{entry['complexity']}, Q:{entry['quality']:.1f}, "
                    f"S:{entry['success_rate']:.2%}, Used:{entry['usage_count']}x)"
                )

    async def test_cache_retrieval(self, test_prompts: list[str] = None) -> dict[str, Any]:
        """Test cache retrieval with sample prompts.

        Args:
            test_prompts: List of prompts to test (uses defaults if None)

        Returns:
            Test results
        """
        if not test_prompts:
            test_prompts = [
                "Create a red cube",
                "Generate a simple chair",
                "Make a textured wooden table",
                "Build a lit interior scene",
                "Create an animated character dancing"
            ]

        console.print(Panel("🧪 Testing Cache Retrieval", border_style="green"))

        results = {
            "total_tests": len(test_prompts),
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_retrieval_time": 0.0,
            "test_details": []
        }

        total_time = 0.0

        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"\n[cyan]Test {i}/{len(test_prompts)}:[/cyan] {prompt}")

            try:
                cache_result = await self.retrieval_service.retrieve_cached_code(
                    prompt=prompt,
                    complexity_level=2,  # Default test level
                    include_alternatives=True
                )

                total_time += cache_result.retrieval_time_ms

                test_detail = {
                    "prompt": prompt,
                    "cache_hit": cache_result.cache_hit,
                    "retrieval_time_ms": cache_result.retrieval_time_ms,
                    "similarity_score": cache_result.similarity_score,
                    "alternatives_count": len(cache_result.alternatives)
                }

                if cache_result.cache_hit:
                    results["cache_hits"] += 1
                    console.print(f"  ✅ Cache HIT - Similarity: {cache_result.similarity_score:.3f}")
                    console.print(f"     Entry: {cache_result.entry.id}")
                else:
                    results["cache_misses"] += 1
                    console.print(f"  ❌ Cache MISS - {len(cache_result.alternatives)} alternatives found")

                console.print(f"  ⏱️  Retrieval time: {cache_result.retrieval_time_ms:.1f}ms")

                results["test_details"].append(test_detail)

            except Exception as e:
                console.print(f"  ❌ Test failed: {e}")
                results["test_details"].append({
                    "prompt": prompt,
                    "error": str(e)
                })

        results["avg_retrieval_time"] = total_time / len(test_prompts)

        # Summary
        console.print("\n[bold green]Test Summary:[/bold green]")
        console.print(f"Cache Hits: {results['cache_hits']}/{results['total_tests']}")
        console.print(f"Average Retrieval Time: {results['avg_retrieval_time']:.1f}ms")

        return results

    async def rebuild_embeddings(self) -> dict[str, Any]:
        """Rebuild all embeddings in the cache (e.g., if switching embedding models).

        Returns:
            Rebuild statistics
        """
        console.print(Panel("🔄 Rebuilding Cache Embeddings", border_style="red"))

        import sqlite3

        # Get all entries
        entries_to_rebuild = []
        with sqlite3.connect(self.vector_db.db_path) as conn:
            cursor = conn.execute("SELECT id, prompt FROM cache_entries")
            entries_to_rebuild = cursor.fetchall()

        if not entries_to_rebuild:
            console.print("[yellow]No entries found to rebuild[/yellow]")
            return {"total": 0, "rebuilt": 0}

        stats = {"total": len(entries_to_rebuild), "rebuilt": 0, "errors": 0}

        with Progress() as progress:
            task = progress.add_task("Rebuilding embeddings...", total=stats["total"])

            for entry_id, prompt in entries_to_rebuild:
                try:
                    # Generate new embedding
                    embedding_result = await self.vector_db.embedder.embed_text(
                        prompt, context_type="prompt"
                    )

                    # Update database
                    with sqlite3.connect(self.vector_db.db_path) as conn:
                        conn.execute("""
                            UPDATE cache_entries
                            SET embedding = ?
                            WHERE id = ?
                        """, (json.dumps(embedding_result.embedding), entry_id))
                        conn.commit()

                    stats["rebuilt"] += 1

                except Exception as e:
                    logger.error(f"Error rebuilding embedding for {entry_id}: {e}")
                    stats["errors"] += 1

                progress.advance(task)

        console.print(f"[green]Rebuilt {stats['rebuilt']} embeddings[/green]")
        if stats["errors"]:
            console.print(f"[red]{stats['errors']} errors encountered[/red]")

        return stats


async def main():
    """CLI entry point for cache management."""
    parser = argparse.ArgumentParser(description="RAG Cache Management Utilities")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Stats command
    subparsers.add_parser("stats", help="Display cache statistics")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import cache entries from JSON")
    import_parser.add_argument("json_file", help="JSON file to import from")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export cache entries to JSON")
    export_parser.add_argument("json_file", help="JSON file to export to")
    export_parser.add_argument("--no-metadata", action="store_true", help="Exclude metadata")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up low-quality entries")
    cleanup_parser.add_argument("--min-success-rate", type=float, default=0.3,
                               help="Minimum success rate (default: 0.3)")
    cleanup_parser.add_argument("--min-usage-count", type=int, default=3,
                               help="Minimum usage count (default: 3)")
    cleanup_parser.add_argument("--dry-run", action="store_true",
                               help="Show what would be cleaned without deleting")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test cache retrieval")
    test_parser.add_argument("--prompts", nargs="+", help="Custom test prompts")

    # Rebuild command
    subparsers.add_parser("rebuild", help="Rebuild all embeddings")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cache entries")
    clear_parser.add_argument("--confirm", action="store_true",
                             help="Confirm deletion (required)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        manager = CacheManager()

        if args.command == "stats":
            stats = await manager.retrieval_service.get_service_statistics()
            manager.display_statistics(stats["database_stats"])

        elif args.command == "import":
            await manager.import_from_json(Path(args.json_file))

        elif args.command == "export":
            await manager.export_to_json(
                Path(args.json_file),
                include_metadata=not args.no_metadata
            )

        elif args.command == "cleanup":
            await manager.cleanup_cache(
                min_success_rate=args.min_success_rate,
                min_usage_count=args.min_usage_count,
                dry_run=args.dry_run
            )

        elif args.command == "test":
            await manager.test_cache_retrieval(args.prompts)

        elif args.command == "rebuild":
            await manager.rebuild_embeddings()

        elif args.command == "clear":
            if not args.confirm:
                console.print("[red]ERROR: --confirm required to clear cache[/red]")
                return

            await manager.vector_db.clear_cache()
            console.print("[green]Cache cleared successfully[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())
