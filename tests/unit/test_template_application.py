"""Tests for template application functionality in CodingAgent."""

import pytest

from src.agents.coding import CodingAgent
from src.utils.types import SubTask, TaskType


class TestTemplateApplication:
    """Test template application methods in CodingAgent."""

    @pytest.fixture
    def coding_agent(self, agent_config):
        """Create a CodingAgent for testing."""
        return CodingAgent(agent_config)

    def test_determine_template_key_explicit(self, coding_agent):
        """Test template key determination from explicit parameters."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create a shape",
            parameters={"template": "sphere"},
        )
        template = {"cube": "...", "sphere": "...", "cylinder": "..."}

        key = coding_agent._determine_template_key(task, template)
        assert key == "sphere"

    def test_determine_template_key_from_description(self, coding_agent):
        """Test template key determination from description."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create a cylinder for the scene",
            parameters={},
        )
        template = {"cube": "...", "sphere": "...", "cylinder": "..."}

        key = coding_agent._determine_template_key(task, template)
        assert key == "cylinder"

    def test_determine_template_key_default_geometry(self, coding_agent):
        """Test default template key for geometry tasks."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create some object",
            parameters={},
        )
        template = {"cube": "...", "sphere": "...", "cylinder": "..."}

        key = coding_agent._determine_template_key(task, template)
        assert key == "cube"  # Default for geometry

    def test_determine_template_key_none_for_non_geometry(self, coding_agent):
        """Test no template key for non-geometry tasks."""
        task = SubTask(
            id="test-1",
            type=TaskType.MATERIAL,
            description="Create material",
            parameters={},
        )
        template = {"cube": "...", "sphere": "...", "cylinder": "..."}

        key = coding_agent._determine_template_key(task, template)
        assert key is None

    def test_fill_template_with_location(self, coding_agent):
        """Test template filling with location parameters."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create cube",
            parameters={"location": [1, 2, 3]},
        )
        template_code = "bpy.ops.mesh.primitive_cube_add(location=({x}, {y}, {z}))"

        result = coding_agent._fill_template(template_code, task)
        assert "location=(1, 2, 3)" in result

    def test_fill_template_with_color(self, coding_agent):
        """Test template filling with color parameters."""
        task = SubTask(
            id="test-1",
            type=TaskType.MATERIAL,
            description="Create red material",
            parameters={"color": [0.8, 0.1, 0.1]},
        )
        template_code = "color = ({r}, {g}, {b}, 1.0)"

        result = coding_agent._fill_template(template_code, task)
        assert "color = (0.8, 0.1, 0.1, 1.0)" in result

    def test_fill_template_with_defaults(self, coding_agent):
        """Test template filling with default values."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create cube",
            parameters={},
        )
        template_code = "location=({x}, {y}, {z})"

        result = coding_agent._fill_template(template_code, task)
        assert "location=(0, 0, 0)" in result

    def test_fill_template_format_error(self, coding_agent):
        """Test template filling with format error handling."""
        task = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create cube",
            parameters={},
        )
        template_code = "missing_key={missing}"

        # Should return original template on format error
        result = coding_agent._fill_template(template_code, task)
        assert result == template_code

    def test_is_template_already_applied_cube(self, coding_agent):
        """Test detection of already applied cube template."""
        code = "bpy.ops.mesh.primitive_cube_add()"
        assert coding_agent._is_template_already_applied(code, "cube") is True
        assert coding_agent._is_template_already_applied(code, "sphere") is False

    def test_is_template_already_applied_sphere(self, coding_agent):
        """Test detection of already applied sphere template."""
        code = "bpy.ops.mesh.primitive_uv_sphere_add()"
        assert coding_agent._is_template_already_applied(code, "sphere") is True
        assert coding_agent._is_template_already_applied(code, "cube") is False

    def test_integrate_template_code(self, coding_agent):
        """Test template code integration."""
        base_code = "import bpy\n\n# Existing code"
        template_code = "# New template\nbpy.ops.mesh.primitive_cube_add()"

        result = coding_agent._integrate_template_code(base_code, template_code)

        assert "import bpy" in result
        assert "# New template" in result
        assert "bpy.ops.mesh.primitive_cube_add()" in result

    def test_add_task_comment(self, coding_agent):
        """Test task comment addition."""
        code = "import bpy\n\nsome_code()"
        comment = "# Task: Create cube\n"

        result = coding_agent._add_task_comment(code, comment)

        assert "# Task: Create cube" in result
        assert result.count("import bpy") == 1  # Should not duplicate imports
