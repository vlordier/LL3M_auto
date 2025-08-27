"""Test Context7 MCP integration."""

import pytest

from src.knowledge.context7_client import Context7MCPClient, Context7RetrievalService
from src.utils.types import AgentType


class TestContext7MCPClient:
    """Test Context7 MCP client."""

    @pytest.mark.asyncio
    async def test_resolve_library_id(self) -> None:
        """Test library ID resolution."""
        client = Context7MCPClient()

        # Test Blender library resolution
        library_id = await client.resolve_library_id("blender")
        assert library_id == "/blender/python-api"

        library_id = await client.resolve_library_id("bpy")
        assert library_id == "/blender/python-api"

    @pytest.mark.asyncio
    async def test_get_library_docs(self) -> None:
        """Test documentation retrieval."""
        client = Context7MCPClient()

        docs = await client.get_library_docs("/blender/python-api")
        assert docs is not None
        assert "Blender Python API" in docs
        assert "bpy.ops.mesh.primitive_cube_add" in docs

    @pytest.mark.asyncio
    async def test_get_library_docs_with_topic(self) -> None:
        """Test documentation retrieval with specific topic."""
        client = Context7MCPClient()

        docs = await client.get_library_docs("/blender/python-api", topic="geometry")
        assert docs is not None
        assert "Geometry Operations" in docs


class TestContext7RetrievalService:
    """Test Context7 retrieval service."""

    @pytest.mark.asyncio
    async def test_retrieve_documentation_success(self) -> None:
        """Test successful documentation retrieval."""
        service = Context7RetrievalService()

        subtasks = ["Create a cube", "Add materials"]
        response = await service.retrieve_documentation(subtasks)

        assert response.agent_type == AgentType.RETRIEVAL
        assert response.success is True
        assert response.data is not None
        assert "Blender Python API" in response.data

    @pytest.mark.asyncio
    async def test_extract_topic_from_subtasks(self) -> None:
        """Test topic extraction from subtasks."""
        service = Context7RetrievalService()

        # Test geometry detection
        geometry_tasks = ["Create mesh geometry", "Modify vertices"]
        topic = service._extract_topic_from_subtasks(geometry_tasks)
        assert topic == "geometry"

        # Test material detection
        material_tasks = ["Apply materials", "Set texture properties"]
        topic = service._extract_topic_from_subtasks(material_tasks)
        assert topic == "material"

        # Test lighting detection
        lighting_tasks = ["Add sun light", "Configure illumination"]
        topic = service._extract_topic_from_subtasks(lighting_tasks)
        assert topic == "lighting"

        # Test no specific topic
        general_tasks = ["Do something general"]
        topic = service._extract_topic_from_subtasks(general_tasks)
        assert topic is None
