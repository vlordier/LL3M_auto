"""Tests for fallback documentation functionality in RetrievalAgent."""

import pytest

from src.agents.retrieval import RetrievalAgent
from src.utils.types import SubTask, TaskType


class TestFallbackDocumentation:
    """Test fallback documentation methods in RetrievalAgent."""

    @pytest.fixture
    def retrieval_agent(self, agent_config):
        """Create a RetrievalAgent for testing."""
        return RetrievalAgent(agent_config)

    def test_get_fallback_documentation_geometry_only(self, retrieval_agent):
        """Test fallback documentation for geometry tasks only."""
        subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create a cube",
                parameters={},
            )
        ]

        result = retrieval_agent._get_fallback_documentation(subtasks)

        assert "# Targeted Blender Python API Reference" in result
        assert "## Geometry Creation" in result
        assert "primitive_cube_add" in result

    def test_get_fallback_documentation_material_only(self, retrieval_agent):
        """Test fallback documentation for material tasks only."""
        subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.MATERIAL,
                description="Create red material",
                parameters={"color": [0.8, 0.2, 0.2]},
            )
        ]

        result = retrieval_agent._get_fallback_documentation(subtasks)

        assert "## Material Creation and Assignment" in result
        assert "Base Color" in result
        assert "(0.8, 0.2, 0.2, 1.0)" in result

    def test_get_fallback_documentation_mixed_tasks(self, retrieval_agent):
        """Test fallback documentation for mixed task types."""
        subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create a sphere",
                parameters={},
            ),
            SubTask(
                id="task-2",
                type=TaskType.LIGHTING,
                description="Add lighting",
                parameters={},
            ),
        ]

        result = retrieval_agent._get_fallback_documentation(subtasks)

        assert "## Geometry Creation" in result
        assert "## Lighting Setup" in result
        assert "primitive_uv_sphere_add" in result
        assert "light_add" in result

    def test_extract_shapes_from_tasks_various_shapes(self, retrieval_agent):
        """Test shape extraction from task descriptions."""
        subtasks = [
            SubTask(
                id="1",
                type=TaskType.GEOMETRY,
                description="Create a cube",
                parameters={},
            ),
            SubTask(
                id="2",
                type=TaskType.GEOMETRY,
                description="Add a sphere",
                parameters={},
            ),
            SubTask(
                id="3",
                type=TaskType.GEOMETRY,
                description="Make cylinder",
                parameters={},
            ),
            SubTask(
                id="4",
                type=TaskType.GEOMETRY,
                description="Create plane",
                parameters={},
            ),
        ]

        shapes = retrieval_agent._extract_shapes_from_tasks(subtasks)

        assert "cube" in shapes
        assert "sphere" in shapes
        assert "cylinder" in shapes
        assert "plane" in shapes

    def test_extract_shapes_from_tasks_ball_as_sphere(self, retrieval_agent):
        """Test that 'ball' is recognized as sphere."""
        subtasks = [
            SubTask(
                id="1",
                type=TaskType.GEOMETRY,
                description="Create a ball",
                parameters={},
            )
        ]

        shapes = retrieval_agent._extract_shapes_from_tasks(subtasks)
        assert "sphere" in shapes

    def test_extract_shapes_from_tasks_default_cube(self, retrieval_agent):
        """Test default cube when no shapes recognized."""
        subtasks = [
            SubTask(
                id="1",
                type=TaskType.GEOMETRY,
                description="Create something",
                parameters={},
            )
        ]

        shapes = retrieval_agent._extract_shapes_from_tasks(subtasks)
        assert "cube" in shapes
        assert len(shapes) == 1

    def test_get_shape_docs_cube(self, retrieval_agent):
        """Test cube shape documentation generation."""
        docs = retrieval_agent._get_shape_docs("cube")

        assert len(docs) == 3
        assert any("# Add cube" in doc for doc in docs)
        assert "primitive_cube_add" in " ".join(docs)
        assert "cube = bpy.context.object" in docs

    def test_get_shape_docs_sphere(self, retrieval_agent):
        """Test sphere shape documentation generation."""
        docs = retrieval_agent._get_shape_docs("sphere")

        assert len(docs) == 3
        assert any("# Add sphere" in doc for doc in docs)
        assert "primitive_uv_sphere_add" in " ".join(docs)

    def test_get_shape_docs_cylinder(self, retrieval_agent):
        """Test cylinder shape documentation generation."""
        docs = retrieval_agent._get_shape_docs("cylinder")

        assert len(docs) == 3
        assert any("# Add cylinder" in doc for doc in docs)
        assert "primitive_cylinder_add" in " ".join(docs)

    def test_get_shape_docs_plane(self, retrieval_agent):
        """Test plane shape documentation generation."""
        docs = retrieval_agent._get_shape_docs("plane")

        assert len(docs) == 3
        assert any("# Add plane" in doc for doc in docs)
        assert "primitive_plane_add" in " ".join(docs)

    def test_get_shape_docs_unknown(self, retrieval_agent):
        """Test unknown shape returns empty docs."""
        docs = retrieval_agent._get_shape_docs("unknown")
        assert docs == []

    def test_get_material_fallback_docs_with_color(self, retrieval_agent):
        """Test material fallback docs with specific color."""
        subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.MATERIAL,
                description="Create material",
                parameters={"color": [0.1, 0.9, 0.3]},
            )
        ]

        result = retrieval_agent._get_material_fallback_docs(subtasks)

        assert "## Material Creation and Assignment" in result
        assert "(0.1, 0.9, 0.3, 1.0)" in result

    def test_get_material_fallback_docs_default_color(self, retrieval_agent):
        """Test material fallback docs with default color."""
        subtasks = [
            SubTask(
                id="task-1",
                type=TaskType.MATERIAL,
                description="Create material",
                parameters={},
            )
        ]

        result = retrieval_agent._get_material_fallback_docs(subtasks)

        assert "## Material Creation and Assignment" in result
        assert "(0.8, 0.2, 0.2, 1.0)" in result

    def test_get_lighting_fallback_docs(self, retrieval_agent):
        """Test lighting fallback documentation."""
        result = retrieval_agent._get_lighting_fallback_docs()

        assert "## Lighting Setup" in result
        assert "light_add" in result
        assert "SUN" in result
        assert "AREA" in result

    def test_get_scene_setup_fallback_docs(self, retrieval_agent):
        """Test scene setup fallback documentation."""
        result = retrieval_agent._get_scene_setup_fallback_docs()

        assert "## Scene Setup" in result
        assert "camera_add" in result
        assert "CYCLES" in result
        assert "render.resolution" in result

    def test_get_basic_operations_docs(self, retrieval_agent):
        """Test basic operations fallback documentation."""
        result = retrieval_agent._get_basic_operations_docs()

        assert "## Basic Operations" in result
        assert "select_all" in result
        assert "location" in result
        assert "rotation_euler" in result
