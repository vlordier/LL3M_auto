#!/usr/bin/env python3
"""Demo script to showcase hierarchical prompt decomposition capabilities."""

# Add the project root to the Python path
import sys
from pathlib import Path

import structlog
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


logger = structlog.get_logger(__name__)
console = Console()


class HierarchicalPromptDemo:
    """Demo class for showcasing hierarchical prompt capabilities."""

    def __init__(self):
        """Initialize the demo with test data."""
        yaml_path = project_root / "tests" / "data" / "hierarchical_prompts.yaml"
        with open(yaml_path) as f:
            self.test_data = yaml.safe_load(f)

    def display_welcome(self):
        """Display welcome message and overview."""
        welcome_text = """
🎯 Hierarchical 3D Prompt Decomposition Demo

This demo showcases how the MCTS planner decomposes 3D generation prompts
from simple basic geometry to complex animated scenes using hierarchical
task decomposition.

Complexity Levels:
• Level 1: Basic Geometry (ovoid, cube, sphere)
• Level 2: Simple Objects (table, chair, house)
• Level 3: Textured Objects (wooden table, brick wall)
• Level 4: Lit Scenes (cozy room, castle courtyard)
• Level 5: Complex Animations (CharMorph Vitruvian characters)
        """

        console.print(
            Panel(
                welcome_text.strip(),
                title="🚀 Hierarchical Decomposition Demo",
                border_style="blue",
            )
        )

    def display_complexity_levels(self):
        """Display the task complexity level mapping."""
        table = Table(title="Task Complexity Hierarchy")
        table.add_column("Level", justify="center", style="cyan")
        table.add_column("Task Type", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Examples", style="yellow")

        level_info = [
            (1, "GEOMETRY", "Basic shapes, primitives", "Create cube, make sphere"),
            (
                2,
                "SCENE_SETUP",
                "Camera, positioning",
                "Position camera, organize scene",
            ),
            (
                3,
                "MATERIAL",
                "Textures, shaders",
                "Apply wood texture, metallic surface",
            ),
            (4, "LIGHTING", "Illumination, shadows", "Add lighting, configure shadows"),
            (5, "ANIMATION", "Motion, dynamics", "Dance animation, character movement"),
        ]

        for level, task_type, desc, examples in level_info:
            table.add_row(str(level), task_type, desc, examples)

        console.print(table)
        console.print()

    def display_prompt_examples(self):
        """Display example prompts for each complexity level."""
        for level in range(1, 6):
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

            prompts = self.test_data["test_prompts"][level_name]

            # Create a table for this level
            table = Table(title=f"Level {level} Examples")
            table.add_column("Prompt", style="green", width=50)
            table.add_column("Expected Tasks", style="cyan")
            table.add_column("Description", style="yellow", width=30)

            for prompt_data in prompts[:3]:  # Show first 3 examples
                task_str = " → ".join(prompt_data["expected_tasks"])
                table.add_row(prompt_data["prompt"], task_str, prompt_data["description"])

            console.print(table)
            console.print()

    def display_charmorph_specifications(self):
        """Display CharMorph Vitruvian specifications."""
        specs = self.test_data["charmorph_vitruvian_specs"]

        charmorph_text = f"""
CharMorph Vitruvian System: {specs["base_system"]}
Model Type: {specs["model_type"]}

Animation Capabilities:
• Dance Moves: {", ".join(specs["animation_capabilities"]["dance_moves"])}
• Action Sequences: {", ".join(specs["animation_capabilities"]["action_sequences"])}
• Basic Movements: {", ".join(specs["animation_capabilities"]["basic_movements"])}

Customization Options:
• Hair Colors: {", ".join(specs["customization_options"]["hair_colors"])}
• Clothing: {", ".join(specs["customization_options"]["clothing"])}
• Gender: {", ".join(specs["customization_options"]["gender"])}
        """

        console.print(
            Panel(
                charmorph_text.strip(),
                title="🎭 CharMorph Vitruvian Specifications",
                border_style="magenta",
            )
        )

    def display_gangnam_style_example(self):
        """Highlight the most complex example - Gangnam Style prompt."""
        level_5_prompts = self.test_data["test_prompts"]["level_5_animated_complex"]

        gangnam_prompt = None
        for prompt_data in level_5_prompts:
            if "Gangnam Style" in prompt_data["prompt"]:
                gangnam_prompt = prompt_data
                break

        if gangnam_prompt:
            gangnam_text = f"""
🕺 Most Complex Test Case:
Prompt: "{gangnam_prompt["prompt"]}"

Expected Task Decomposition:
{" → ".join(gangnam_prompt["expected_tasks"])}

CharMorph Specifications:
• Base Model: {gangnam_prompt["charmorph_spec"]["base_model"]}
• Gender: {gangnam_prompt["charmorph_spec"]["gender"]}
• Hair: {gangnam_prompt["charmorph_spec"]["hair_color"]}
• Clothing: {gangnam_prompt["charmorph_spec"]["clothing"]}
• Animation: {gangnam_prompt["charmorph_spec"]["animation"]}

This represents the pinnacle of hierarchical decomposition complexity!
            """

            console.print(
                Panel(
                    gangnam_text.strip(),
                    title="💃 Gangnam Style - Ultimate Complexity",
                    border_style="red",
                )
            )

    def display_performance_expectations(self):
        """Display performance benchmarks for each level."""
        benchmarks = self.test_data["performance_benchmarks"]

        table = Table(title="Performance Expectations")
        table.add_column("Complexity Level", justify="center", style="cyan")
        table.add_column("Generation Time", style="yellow")
        table.add_column("Task Count Range", style="green")
        table.add_column("Example", style="white")

        examples = [
            "Simple ovoid",
            "Wooden table",
            "Textured brick wall",
            "Lit living room",
            "Gangnam Style dance",
        ]

        for i in range(1, 6):
            gen_time = benchmarks[f"level_{i}_generation_time"]
            task_range = benchmarks["complexity_scaling"][f"task_count_level_{i}"]
            example = examples[i - 1]

            table.add_row(f"Level {i}", gen_time, str(task_range), example)

        console.print(table)

    def run_demo(self):
        """Run the complete demonstration."""
        self.display_welcome()
        console.input("\n🎯 Press Enter to see the complexity level mapping...")

        self.display_complexity_levels()
        console.input("🎯 Press Enter to see prompt examples for each level...")

        self.display_prompt_examples()
        console.input("🎯 Press Enter to see CharMorph Vitruvian specifications...")

        self.display_charmorph_specifications()
        console.input("🎯 Press Enter to see the most complex example...")

        self.display_gangnam_style_example()
        console.input("🎯 Press Enter to see performance expectations...")

        self.display_performance_expectations()

        console.print(
            "\n🎉 Demo complete! The hierarchical decomposition system is ready to handle prompts "
            "from basic geometry to complex character animations."
        )


def main():
    """Main entry point for the demo."""
    try:
        demo = HierarchicalPromptDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        console.print(f"\n❌ Error running demo: {e}")
        raise


if __name__ == "__main__":
    main()
