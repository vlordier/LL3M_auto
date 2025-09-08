"""Integration tests for hierarchical prompt decomposition using YAML test cases."""

from pathlib import Path
from unittest.mock import patch

import pytest
import structlog
import yaml

from src.agents.factory import get_agent_factory
from src.agents.mcts import get_task_complexity_level, sort_tasks_by_complexity
from src.utils.types import TaskType
from tests.utils.builders import WorkflowStateBuilder

logger = structlog.get_logger(__name__)


class TestHierarchicalPrompts:
    """Test hierarchical decomposition using YAML prompt definitions."""

    @classmethod
    def setup_class(cls):
        """Load test prompts from YAML file."""
        yaml_path = Path(__file__).parent.parent / "data" / "hierarchical_prompts.yaml"
        with open(yaml_path) as f:
            cls.test_data = yaml.safe_load(f)

    def test_yaml_structure_validity(self):
        """Validate that the YAML test file has correct structure."""
        assert "test_prompts" in self.test_data
        assert "test_config" in self.test_data
        assert "charmorph_vitruvian_specs" in self.test_data
        assert "performance_benchmarks" in self.test_data

        # Check that we have all complexity levels
        expected_levels = [
            "level_1_basic_geometry",
            "level_2_simple_objects",
            "level_3_textured_objects",
            "level_4_lit_scenes",
            "level_5_animated_complex"
        ]

        for level in expected_levels:
            assert level in self.test_data["test_prompts"], f"Missing {level} in test prompts"

    @pytest.mark.parametrize("level_name,level_data", [
        ("level_1_basic_geometry", None),
        ("level_2_simple_objects", None),
        ("level_3_textured_objects", None),
        ("level_4_lit_scenes", None),
        ("level_5_animated_complex", None),
    ])
    def test_prompt_level_structure(self, level_name, level_data):
        """Test that each complexity level has proper structure."""
        level_data = self.test_data["test_prompts"][level_name]

        assert isinstance(level_data, list), f"{level_name} should be a list of prompts"
        assert len(level_data) > 0, f"{level_name} should have at least one prompt"

        for i, prompt_data in enumerate(level_data):
            assert "prompt" in prompt_data, f"{level_name}[{i}] missing 'prompt' field"
            assert "expected_complexity" in prompt_data, f"{level_name}[{i}] missing 'expected_complexity'"
            assert "expected_tasks" in prompt_data, f"{level_name}[{i}] missing 'expected_tasks'"
            assert "description" in prompt_data, f"{level_name}[{i}] missing 'description'"

            # Validate task types are valid
            for task_type in prompt_data["expected_tasks"]:
                assert hasattr(TaskType, task_type.upper()), f"Invalid task type: {task_type}"

    @pytest.mark.asyncio
    async def test_level_1_basic_geometry_decomposition(self):
        """Test decomposition of Level 1 (basic geometry) prompts."""
        await self._test_level_decomposition("level_1_basic_geometry", 1)

    @pytest.mark.asyncio
    async def test_level_2_simple_objects_decomposition(self):
        """Test decomposition of Level 2 (simple objects) prompts."""
        await self._test_level_decomposition("level_2_simple_objects", 2)

    @pytest.mark.asyncio
    async def test_level_3_textured_objects_decomposition(self):
        """Test decomposition of Level 3 (textured objects) prompts."""
        await self._test_level_decomposition("level_3_textured_objects", 3)

    @pytest.mark.asyncio
    async def test_level_4_lit_scenes_decomposition(self):
        """Test decomposition of Level 4 (lit scenes) prompts."""
        await self._test_level_decomposition("level_4_lit_scenes", 4)

    @pytest.mark.asyncio
    async def test_level_5_animated_complex_decomposition(self):
        """Test decomposition of Level 5 (animated complex) prompts."""
        await self._test_level_decomposition("level_5_animated_complex", 5)

    async def _test_level_decomposition(self, level_name: str, expected_max_complexity: int):
        """Helper method to test decomposition for a specific complexity level."""
        level_prompts = self.test_data["test_prompts"][level_name]

        with patch("src.agents.base.get_settings"), \
             patch("src.agents.base.LMStudioClient"), \
             patch("src.agents.base.OpenAIClient"):

            factory = get_agent_factory()

            # Choose appropriate planner based on complexity
            planner = factory.create_planner() if expected_max_complexity <= 2 else factory.create_mcts()

            # Test a sample of prompts from this level
            for prompt_data in level_prompts[:2]:  # Test first 2 prompts per level
                logger.info(f"Testing prompt: {prompt_data['prompt'][:50]}...")

                state = WorkflowStateBuilder().with_prompt(prompt_data["prompt"]).build()

                # Mock successful response with expected task types
                expected_tasks = prompt_data["expected_tasks"]
                mock_subtasks = self._create_mock_subtasks(expected_tasks)

                with patch.object(planner, 'process') as mock_process:
                    mock_response = type('MockResponse', (), {
                        'success': True,
                        'data': mock_subtasks,
                        'agent_type': planner.agent_type,
                        'execution_time': 1.0
                    })()
                    mock_process.return_value = mock_response

                    response = await planner.process(state)

                    # Validate response structure
                    assert response.success, f"Failed to process: {prompt_data['prompt']}"
                    assert len(response.data) > 0, "No subtasks generated"

                    # Validate hierarchical ordering
                    sorted_tasks = sort_tasks_by_complexity(response.data)
                    complexity_levels = [get_task_complexity_level(task.type) for task in sorted_tasks]

                    # Check that tasks are in hierarchical order
                    assert complexity_levels == sorted(complexity_levels), \
                        f"Tasks not in hierarchical order: {complexity_levels}"

                    # Validate maximum complexity doesn't exceed expected
                    max_complexity = max(complexity_levels) if complexity_levels else 0
                    assert max_complexity <= expected_max_complexity, \
                        f"Max complexity {max_complexity} exceeds expected {expected_max_complexity}"

                    logger.info(f"✓ Validated hierarchical ordering: {complexity_levels}")

    def _create_mock_subtasks(self, task_type_names: list[str]) -> list:
        """Create mock subtasks for testing."""
        from src.utils.types import SubTask, TaskType

        subtasks = []
        for i, task_type_name in enumerate(task_type_names):
            task_type = getattr(TaskType, task_type_name.upper())
            complexity_level = get_task_complexity_level(task_type)

            subtask = SubTask(
                id=f"mock-task-{i}",
                type=task_type,
                description=f"Mock {task_type_name} task",
                priority=complexity_level,  # Use complexity as priority
                dependencies=[]
            )
            subtasks.append(subtask)

        return subtasks

    def test_charmorph_vitruvian_specifications(self):
        """Test CharMorph Vitruvian specifications are properly defined."""
        specs = self.test_data["charmorph_vitruvian_specs"]

        assert specs["base_system"] == "CharMorph for Blender"
        assert specs["model_type"] == "Vitruvian Human"

        # Check required capabilities
        features = specs["supported_features"]
        assert features["realistic_human_proportions"] is True
        assert features["animation_ready"] is True

        # Check animation capabilities
        animations = specs["animation_capabilities"]
        assert "Gangnam Style" in animations["dance_moves"]
        assert "fighting" in animations["action_sequences"]

        # Check customization options
        customization = specs["customization_options"]
        assert "blonde" in customization["hair_colors"]
        assert "swimwear" in customization["clothing"]

    def test_performance_benchmarks_defined(self):
        """Test that performance benchmarks are properly defined."""
        benchmarks = self.test_data["performance_benchmarks"]

        # Check that all levels have time expectations
        expected_keys = [
            "level_1_generation_time",
            "level_2_generation_time",
            "level_3_generation_time",
            "level_4_generation_time",
            "level_5_generation_time"
        ]

        for key in expected_keys:
            assert key in benchmarks, f"Missing benchmark: {key}"
            assert "seconds" in benchmarks[key], f"Time format should include 'seconds': {benchmarks[key]}"

        # Check complexity scaling expectations
        scaling = benchmarks["complexity_scaling"]
        assert "task_count_level_1" in scaling
        assert "task_count_level_5" in scaling

    @pytest.mark.parametrize("complexity_level", [1, 2, 3, 4, 5])
    def test_complexity_progression_validation(self, complexity_level):
        """Test that complexity levels show proper progression in task count and types."""
        level_name = f"level_{complexity_level}_" + [
            "basic_geometry", "simple_objects", "textured_objects", "lit_scenes", "animated_complex"
        ][complexity_level - 1]

        level_prompts = self.test_data["test_prompts"][level_name]

        for prompt_data in level_prompts:
            expected_tasks = prompt_data["expected_tasks"]
            expected_complexity = prompt_data["expected_complexity"]

            # Validate that expected complexity matches level
            assert expected_complexity == complexity_level, \
                f"Complexity mismatch in {level_name}: expected {complexity_level}, got {expected_complexity}"

            # Validate task progression makes sense
            task_types = [getattr(TaskType, task.upper()) for task in expected_tasks]
            complexity_levels = [get_task_complexity_level(task_type) for task_type in task_types]

            # Should not exceed the expected complexity level
            max_task_complexity = max(complexity_levels) if complexity_levels else 0
            assert max_task_complexity <= complexity_level, \
                f"Task complexity {max_task_complexity} exceeds level {complexity_level} in: {prompt_data['prompt']}"

    def test_gangnam_style_prompt_specifications(self):
        """Specifically test the Gangnam Style prompt (most complex test case)."""
        level_5_prompts = self.test_data["test_prompts"]["level_5_animated_complex"]

        gangnam_prompt = None
        for prompt_data in level_5_prompts:
            if "Gangnam Style" in prompt_data["prompt"]:
                gangnam_prompt = prompt_data
                break

        assert gangnam_prompt is not None, "Gangnam Style prompt not found"

        # Validate it uses CharMorph Vitruvian
        assert "CharMorph Vitruvian" in gangnam_prompt["prompt"]
        assert "blonde woman" in gangnam_prompt["prompt"]
        assert "bikini" in gangnam_prompt["prompt"]
        assert "dancing" in gangnam_prompt["prompt"]

        # Should have all 5 task types for maximum complexity
        assert gangnam_prompt["expected_complexity"] == 5
        expected_tasks = set(gangnam_prompt["expected_tasks"])
        all_task_types = {"geometry", "scene_setup", "material", "lighting", "animation"}
        assert expected_tasks == all_task_types, \
            f"Gangnam Style prompt should include all task types, missing: {all_task_types - expected_tasks}"

        # Check CharMorph specifications
        charmorph_spec = gangnam_prompt.get("charmorph_spec", {})
        assert charmorph_spec["base_model"] == "CharMorph Vitruvian"
        assert charmorph_spec["gender"] == "female"
        assert charmorph_spec["hair_color"] == "blonde"
        assert charmorph_spec["clothing"] == "bikini"
        assert charmorph_spec["animation"] == "Gangnam Style dance"
