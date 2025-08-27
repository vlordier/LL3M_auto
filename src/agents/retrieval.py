"""Retrieval agent for fetching relevant Blender documentation using Context7 MCP."""

import asyncio
import re
from typing import Any

from ..knowledge.context7_client import Context7RetrievalService
from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent


class RetrievalAgent(EnhancedBaseAgent):
    """Retrieves relevant Blender documentation using Context7 MCP."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize retrieval agent."""
        super().__init__(config)
        self.context7_service = Context7RetrievalService()
        self.documentation_cache: dict[str, str] = {}
        self.topic_extraction_patterns = {
            TaskType.GEOMETRY: [
                r"mesh",
                r"primitive",
                r"vertex",
                r"face",
                r"edge",
                r"cube",
                r"sphere",
                r"cylinder",
                r"plane",
                r"torus",
            ],
            TaskType.MATERIAL: [
                r"material",
                r"shader",
                r"node",
                r"bsdf",
                r"texture",
                r"color",
                r"metallic",
                r"roughness",
                r"emission",
            ],
            TaskType.LIGHTING: [
                r"light",
                r"lamp",
                r"sun",
                r"area",
                r"point",
                r"energy",
                r"color",
                r"shadow",
                r"hdri",
            ],
            TaskType.SCENE_SETUP: [
                r"camera",
                r"render",
                r"scene",
                r"world",
                r"background",
            ],
            TaskType.ANIMATION: [
                r"keyframe",
                r"animation",
                r"timeline",
                r"frame",
            ],
        }

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.RETRIEVAL

    @property
    def name(self) -> str:
        """Return agent name."""
        return "Documentation Retrieval"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Retrieve relevant documentation for subtasks."""
        start_time = time.monotonic()

        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="Invalid input: subtasks are required",
                    execution_time=0.0,
                )

            self.logger.info(
                "Starting documentation retrieval", num_subtasks=len(state.subtasks)
            )

            # Extract topics from subtasks
            topics = self._extract_topics_from_subtasks(state.subtasks)

            # Build search queries
            search_queries = self._build_search_queries(state.subtasks, topics)

            # Retrieve documentation concurrently
            documentation_parts = await self._retrieve_documentation_parallel(
                search_queries
            )

            # Combine and filter documentation
            combined_docs = self._combine_documentation(
                documentation_parts, state.subtasks
            )

            # Enhance with context-specific examples
            enhanced_docs = await self._enhance_with_examples(
                combined_docs, state.subtasks
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Documentation retrieval completed",
                doc_length=len(enhanced_docs),
                topics=topics,
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=enhanced_docs,
                message=f"Retrieved documentation for {len(topics)} topics",
                execution_time=execution_time,
                metadata={
                    "topics": topics,
                    "search_queries": search_queries,
                    "doc_sections": len(documentation_parts),
                },
            )

        except Exception as e:
            self.logger.error("Documentation retrieval failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data="",
                message=f"Retrieval failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    def _extract_topics_from_subtasks(self, subtasks: list[SubTask]) -> list[str]:
        """Extract documentation topics from subtasks."""
        topics = set()

        for subtask in subtasks:
            # Add task type as primary topic
            topics.add(subtask.type.value)

            # Extract specific keywords based on task type and description
            description_lower = subtask.description.lower()

            if subtask.type in self.topic_extraction_patterns:
                patterns = self.topic_extraction_patterns[subtask.type]
                for pattern in patterns:
                    if re.search(pattern, description_lower):
                        topics.add(pattern)

            # Extract from parameters
            for key, value in subtask.parameters.items():
                if isinstance(value, str):
                    topics.add(value.lower())

        return list(topics)

    def _build_search_queries(
        self, subtasks: list[SubTask], topics: list[str]
    ) -> list[str]:
        """Build specific search queries for Context7."""
        queries = []

        # Topic-based queries
        for topic in topics:
            queries.append(f"blender python {topic} api")

        # Task-specific queries
        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                queries.append(f"bpy.ops.mesh.primitive {subtask.description}")
            elif subtask.type == TaskType.MATERIAL:
                queries.append(f"blender material nodes bsdf {subtask.description}")
            elif subtask.type == TaskType.LIGHTING:
                queries.append(f"bpy.ops.object.light_add {subtask.description}")

        return list(set(queries))  # Remove duplicates

    async def _retrieve_documentation_parallel(self, queries: list[str]) -> list[str]:
        """Retrieve documentation for multiple queries in parallel."""
        tasks = []

        for query in queries:
            # Check cache first
            if query in self.documentation_cache:
                tasks.append(asyncio.create_task(self._get_cached_docs(query)))
            else:
                tasks.append(asyncio.create_task(self._fetch_and_cache_docs(query)))

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _get_cached_docs(self, query: str) -> str:
        """Get documentation from cache."""
        return self.documentation_cache[query]

    async def _fetch_and_cache_docs(self, query: str) -> str:
        """Fetch documentation and cache it."""
        try:
            response = await self.context7_service.retrieve_documentation([query])

            if response.success and response.data:
                self.documentation_cache[query] = response.data
                return response.data
            else:
                self.logger.warning("Failed to retrieve docs", query=query)
                return ""
        except Exception as e:
            self.logger.error("Documentation fetch error", query=query, error=str(e))
            return ""

    def _combine_documentation(
        self, doc_parts: list[str], subtasks: list[SubTask]
    ) -> str:
        """Combine retrieved documentation into coherent guide."""
        # Filter out empty or error responses
        valid_docs = [doc for doc in doc_parts if isinstance(doc, str) and doc.strip()]

        if not valid_docs:
            return self._get_fallback_documentation(subtasks)

        # Structure documentation by sections
        sections = {
            "overview": "# Blender Python API Documentation\n\n",
            "basic_operations": "",
            "geometry": "",
            "materials": "",
            "lighting": "",
            "examples": "",
        }

        # Categorize and organize documentation
        for doc in valid_docs:
            if "mesh" in doc.lower() or "primitive" in doc.lower():
                sections["geometry"] += doc + "\n\n"
            elif "material" in doc.lower() or "bsdf" in doc.lower():
                sections["materials"] += doc + "\n\n"
            elif "light" in doc.lower() or "lamp" in doc.lower():
                sections["lighting"] += doc + "\n\n"
            else:
                sections["basic_operations"] += doc + "\n\n"

        # Combine sections
        combined = sections["overview"]

        for section_name, content in sections.items():
            if section_name != "overview" and content.strip():
                combined += f"## {section_name.replace('_', ' ').title()}\n\n{content}"

        return combined

    async def _enhance_with_examples(
        self, documentation: str, subtasks: list[SubTask]
    ) -> str:
        """Enhance documentation with context-specific examples."""
        # Generate examples based on subtasks
        examples_section = "\n\n## Context-Specific Examples\n\n"

        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                examples_section += f"### {subtask.description}\n"
                examples_section += self._generate_geometry_example(subtask)
            elif subtask.type == TaskType.MATERIAL:
                examples_section += f"### {subtask.description}\n"
                examples_section += self._generate_material_example(subtask)

        return documentation + examples_section

    def _generate_geometry_example(self, subtask: SubTask) -> str:
        """Generate geometry creation example."""
        shape = subtask.parameters.get("shape", "cube").lower()
        location = subtask.parameters.get("location", [0, 0, 0])

        return f"""```python
# Create {shape}
bpy.ops.mesh.primitive_{shape}_add(location={location})
obj = bpy.context.active_object
obj.name = "{subtask.id}"
```

"""

    def _generate_material_example(self, subtask: SubTask) -> str:
        """Generate material creation example."""
        color = subtask.parameters.get("color", [0.8, 0.2, 0.2])

        return f"""```python
# Create material for {subtask.description}
material = bpy.data.materials.new(name="{subtask.id}_material")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = {color + [1.0]}

# Assign to active object
bpy.context.active_object.data.materials.append(material)
```

"""

    def _get_fallback_documentation(self, subtasks: list[SubTask]) -> str:
        """Provide fallback documentation when Context7 fails."""
        return """# Basic Blender Python API Reference

## Geometry Creation
```python
import bpy

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Add primitives
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
```

## Material Creation
```python
# Create material
material = bpy.data.materials.new(name="MyMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)
```
"""

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for retrieval agent."""
        return state.subtasks is not None and len(state.subtasks) > 0