#!/usr/bin/env python3
"""Check configuration for hierarchical generation."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table

console = Console()


def check_llm_configuration():
    """Check LLM configuration for generation."""
    console.print("🔧 Checking LLM Configuration for Hierarchical Generation\n")

    issues = []

    # Check environment variable
    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() in ("true", "1", "yes")

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Status", style="green")

    # Check LLM mode
    if use_local_llm:
        table.add_row("LLM Mode", "Local (LMStudio)", "❌ Requires loaded model")
        issues.append("Local LLM mode requires LMStudio with a loaded model")
    else:
        table.add_row("LLM Mode", "OpenAI API", "✅ Preferred for generation")

    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        masked_key = f"{openai_key[:8]}...{openai_key[-4:]}" if len(openai_key) > 12 else "***"
        table.add_row("OpenAI API Key", masked_key, "✅ Configured")
    else:
        table.add_row("OpenAI API Key", "Not set", "❌ Required for OpenAI mode")
        if not use_local_llm:
            issues.append("OpenAI API key not found - set OPENAI_API_KEY in .env")

    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        table.add_row(".env file", str(env_file), "✅ Found")
    else:
        table.add_row(".env file", "Not found", "⚠️  Optional")
        issues.append("Consider creating .env file for configuration")

    console.print(table)
    console.print()

    # Show recommendations
    if issues:
        console.print("⚠️  Configuration Issues Found:")
        for i, issue in enumerate(issues, 1):
            console.print(f"   {i}. {issue}")

        console.print("\n💡 Recommended Solutions:")

        if not use_local_llm and not openai_key:
            console.print("   • Set OpenAI API key:")
            console.print("     export OPENAI_API_KEY='your-api-key-here'")
            console.print("     # or add to .env file: OPENAI_API_KEY=your-api-key-here")

        if use_local_llm:
            console.print("   • Use OpenAI instead (recommended):")
            console.print("     export USE_LOCAL_LLM=false")
            console.print("   • Or load a model in LMStudio:")
            console.print("     lms load <model-name>")

        return False
    else:
        console.print("✅ Configuration looks good for hierarchical generation!")
        return True


def show_generation_help():
    """Show help for generation commands."""
    console.print("\n🚀 Generation Commands:")

    help_table = Table()
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    help_table.add_column("Requirements", style="yellow")

    help_table.add_row("make generate-basic", "Generate Level 1 basic geometry", "OpenAI API key")
    help_table.add_row("make generate-hierarchical", "Generate all levels (1-5)", "OpenAI API key")
    help_table.add_row("make generate-level-N", "Generate specific level (1-5)", "OpenAI API key")
    help_table.add_row("make generate-gangnam", "Ultimate test: Gangnam Style", "OpenAI API key")

    console.print(help_table)


def main():
    """Main configuration checker."""
    config_ok = check_llm_configuration()
    show_generation_help()

    if config_ok:
        console.print("\n🎯 Ready to run: make generate-basic")
    else:
        console.print("\n❌ Please fix configuration issues before running generation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
