"""Context7 MCP client for Blender documentation retrieval."""

import asyncio
from typing import Any

import aiohttp
import structlog

from ..utils.config import get_settings
from ..utils.types import AgentResponse, AgentType

logger = structlog.get_logger(__name__)


class Context7MCPClient:
    """Client for interacting with Context7 MCP server."""

    def __init__(self) -> None:
        """Initialize the Context7 MCP client."""
        self.server_url = get_settings().context7.mcp_server
        self.api_key = get_settings().context7.api_key
        self.timeout = get_settings().context7.timeout
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Context7MCPClient":
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def resolve_library_id(self, library_name: str) -> str | None:
        """Resolve a package name to Context7-compatible library ID."""
        try:
            # For Blender, we'll search for 'blender' or 'bpy'
            if "blender" in library_name.lower() or library_name.lower() == "bpy":
                # This would be the actual MCP call - placeholder implementation
                return "/blender/python-api"

            logger.info("Resolving library ID", library_name=library_name)

            # Placeholder: In real implementation, this would call the MCP server
            # For now, return a default Blender library ID
            return "/blender/python-api"

        except Exception as e:
            logger.error(
                "Failed to resolve library ID", error=str(e), library_name=library_name
            )
            return None

    async def get_library_docs(
        self, library_id: str, topic: str | None = None, tokens: int = 10000
    ) -> str | None:
        """Fetch documentation for a specific library."""
        try:
            logger.info(
                "Fetching library documentation",
                library_id=library_id,
                topic=topic,
                tokens=tokens,
            )

            # Placeholder: In real implementation, this would call the MCP server
            # For now, return sample Blender documentation
            return self._get_sample_blender_docs(topic)

        except Exception as e:
            logger.error(
                "Failed to fetch library docs", error=str(e), library_id=library_id
            )
            return None

    def _get_sample_blender_docs(self, topic: str | None = None) -> str:
        """Return sample Blender documentation for testing."""
        base_docs = """
# Blender Python API Documentation

## Basic Operations

### Creating Objects
```python
import bpy

# Add a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Add a sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))

# Add a cylinder
bpy.ops.mesh.primitive_cylinder_add(location=(4, 0, 0))
```

### Materials
```python
# Create a new material
material = bpy.data.materials.new(name="MyMaterial")
material.use_nodes = True

# Get the material's node tree
nodes = material.node_tree.nodes
bsdf = nodes.get("Principled BSDF")

# Set base color
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
```

### Scene Management
```python
# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Set viewport shading
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'MATERIAL'
```
"""

        if topic:
            topic_docs = {
                "geometry": """
### Geometry Operations
```python
# Modify object geometry
bpy.context.object.data.vertices[0].co = (1, 1, 1)

# Apply modifiers
modifier = bpy.context.object.modifiers.new(name="Subsurf", type='SUBSURF')
modifier.levels = 2
```
""",
                "material": """
### Advanced Materials
```python
# Create node-based material
mat = bpy.data.materials.new(name="NodeMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes

# Add texture node
tex_node = nodes.new('ShaderNodeTexImage')
```
""",
                "lighting": """
### Lighting Setup
```python
# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.object
light.data.energy = 3.0
```
""",
            }

            return base_docs + topic_docs.get(topic, "")

        return base_docs


class Context7RetrievalService:
    """Service for retrieving Blender documentation using Context7."""

    def __init__(self) -> None:
        """Initialize the retrieval service."""
        self.client = Context7MCPClient()

    async def retrieve_documentation(
        self,
        subtasks: list[str],
        context: str | None = None,  # noqa: ARG002
    ) -> AgentResponse:
        """Retrieve relevant Blender documentation for given subtasks."""
        start_time = asyncio.get_event_loop().time()

        try:
            async with self.client as client:
                # Resolve Blender library ID
                library_id = await client.resolve_library_id("blender")
                if not library_id:
                    return AgentResponse(
                        agent_type=AgentType.RETRIEVAL,
                        success=False,
                        data=None,
                        message="Failed to resolve Blender library ID",
                        execution_time=asyncio.get_event_loop().time() - start_time,
                    )

                # Determine topic from subtasks
                topic = self._extract_topic_from_subtasks(subtasks)

                # Fetch documentation
                docs = await client.get_library_docs(library_id, topic=topic)
                if not docs:
                    return AgentResponse(
                        agent_type=AgentType.RETRIEVAL,
                        success=False,
                        data=None,
                        message="Failed to fetch Blender documentation",
                        execution_time=asyncio.get_event_loop().time() - start_time,
                    )

                logger.info(
                    "Successfully retrieved documentation",
                    topic=topic,
                    doc_length=len(docs),
                )

                return AgentResponse(
                    agent_type=AgentType.RETRIEVAL,
                    success=True,
                    data=docs,
                    message=f"Retrieved Blender documentation for topic: {topic}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                    metadata={"topic": topic, "library_id": library_id},
                )

        except Exception as e:
            logger.error("Documentation retrieval failed", error=str(e))
            return AgentResponse(
                agent_type=AgentType.RETRIEVAL,
                success=False,
                data=None,
                message=f"Documentation retrieval failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    def _extract_topic_from_subtasks(self, subtasks: list[str]) -> str | None:
        """Extract the primary topic from subtasks."""
        # Simple keyword matching - can be improved with ML
        subtasks_text = " ".join(subtasks).lower()

        if any(
            keyword in subtasks_text
            for keyword in ["geometry", "mesh", "vertex", "face"]
        ):
            return "geometry"
        elif any(
            keyword in subtasks_text for keyword in ["material", "shader", "texture"]
        ):
            return "material"
        elif any(
            keyword in subtasks_text for keyword in ["light", "illumination", "shadow"]
        ):
            return "lighting"

        return None
