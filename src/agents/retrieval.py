"""Retrieval agent for fetching relevant Blender documentation using Context7 MCP."""

import asyncio
import re
import time
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

            execution_time = time.monotonic() - start_time

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
            for _key, value in subtask.parameters.items():
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

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions and return only strings
        return [result for result in results if isinstance(result, str)]

    async def _get_cached_docs(self, query: str) -> str:
        """Get documentation from cache."""
        return self.documentation_cache[query]

    async def _fetch_and_cache_docs(self, query: str) -> str:
        """Fetch documentation and cache it."""
        try:
            response = await self.context7_service.retrieve_documentation([query])

            if response.success and response.data:
                doc_content = str(response.data)
                self.documentation_cache[query] = doc_content
                return doc_content
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

<<<<<<< HEAD
    def _get_fallback_documentation(self, _subtasks: list[SubTask]) -> str:
        """Provide fallback documentation when Context7 fails."""
        return """# Basic Blender Python API Reference
=======
    def _get_fallback_documentation(self, subtasks: list[SubTask]) -> str:
        """Provide targeted fallback documentation based on subtasks."""
        # Analyze subtasks to provide relevant documentation
        task_types = {task.type.value.lower() for task in subtasks}
>>>>>>> origin/master

        docs = ["# Targeted Blender Python API Reference\n"]

        # Add relevant sections based on task types
        if "geometry" in task_types:
            docs.append(self._get_geometry_fallback_docs(subtasks))

        if "material" in task_types:
            docs.append(self._get_material_fallback_docs(subtasks))

        if "lighting" in task_types:
            docs.append(self._get_lighting_fallback_docs())

        if "scene_setup" in task_types:
            docs.append(self._get_scene_setup_fallback_docs())

        # Add basic operations if no specific types found
        if not task_types or len(docs) == 1:
            docs.append(self._get_basic_operations_docs())

        return "\n\n".join(docs)

    def _get_geometry_fallback_docs(self, subtasks: list[SubTask]) -> str:
        """Get geometry-specific fallback documentation."""
        geometry_tasks = [
            task for task in subtasks if task.type.value.lower() == "geometry"
        ]

        docs = ["## Geometry Creation"]
        docs.append("```python\nimport bpy")

        # Get shapes mentioned in tasks
        shapes_mentioned = self._extract_shapes_from_tasks(geometry_tasks)

        docs.append("\n# Clear existing objects")
        docs.append("bpy.ops.object.select_all(action='SELECT')")
        docs.append("bpy.ops.object.delete(use_global=False, confirm=False)")

        # Add shape-specific code
        for shape in shapes_mentioned:
            shape_docs = self._get_shape_docs(shape)
            docs.extend(shape_docs)

        docs.append("```")
        return "\n".join(docs)

    def _extract_shapes_from_tasks(self, geometry_tasks: list[SubTask]) -> set[str]:
        """Extract mentioned shapes from geometry task descriptions."""
        shapes_mentioned = set()
        for task in geometry_tasks:
            description = task.description.lower()
            if "cube" in description:
                shapes_mentioned.add("cube")
            elif "sphere" in description or "ball" in description:
                shapes_mentioned.add("sphere")
            elif "cylinder" in description:
                shapes_mentioned.add("cylinder")
            elif "plane" in description:
                shapes_mentioned.add("plane")

        if not shapes_mentioned:
            shapes_mentioned.add("cube")  # Default

        return shapes_mentioned

    def _get_shape_docs(self, shape: str) -> list[str]:
        """Get documentation for a specific shape."""
        if shape == "cube":
            return [
                "\n# Add cube",
                "bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))",
                "cube = bpy.context.object",
            ]
        elif shape == "sphere":
            return [
                "\n# Add sphere",
                "bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))",
                "sphere = bpy.context.object",
            ]
        elif shape == "cylinder":
            return [
                "\n# Add cylinder",
                "bpy.ops.mesh.primitive_cylinder_add(location=(4, 0, 0))",
                "cylinder = bpy.context.object",
            ]
        elif shape == "plane":
            return [
                "\n# Add plane",
                "bpy.ops.mesh.primitive_plane_add(location=(0, -2, 0))",
                "plane = bpy.context.object",
            ]
        return []

    def _get_material_fallback_docs(self, subtasks: list[SubTask]) -> str:
        """Get material-specific fallback documentation."""
        docs = ["## Material Creation and Assignment"]
        docs.append("```python")
        docs.append("# Create and assign material")
        docs.append("material = bpy.data.materials.new(name='MyMaterial')")
        docs.append("material.use_nodes = True")
        docs.append("bsdf = material.node_tree.nodes['Principled BSDF']")

        # Check if specific colors are mentioned
        material_tasks = [
            task for task in subtasks if task.type.value.lower() == "material"
        ]
        colors_mentioned = False

        for task in material_tasks:
            if task.parameters and "color" in task.parameters:
                colors_mentioned = True
                color = task.parameters["color"]
                if isinstance(color, list | tuple) and len(color) >= 3:
                    color_str = f"({color[0]}, {color[1]}, {color[2]}, 1.0)"
                    docs.append(
                        f"bsdf.inputs['Base Color'].default_value = {color_str}"
                    )
                    break

        if not colors_mentioned:
            docs.append(
                "bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)"
            )

        docs.append("\n# Assign to active object")
        docs.append("obj = bpy.context.active_object")
        docs.append("if obj:")
        docs.append("    obj.data.materials.append(material)")
        docs.append("```")

        return "\n".join(docs)

    def _get_lighting_fallback_docs(self) -> str:
        """Get lighting-specific fallback documentation."""
        return """## Lighting Setup
```python
# Add sun light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun_light = bpy.context.object
sun_light.data.energy = 5.0

# Add area light for fill
bpy.ops.object.light_add(type='AREA', location=(-5, 5, 5))
area_light = bpy.context.object
area_light.data.energy = 10.0
area_light.data.size = 2.0
```"""

    def _get_scene_setup_fallback_docs(self) -> str:
        """Get scene setup fallback documentation."""
        return """## Scene Setup
```python
# Camera setup
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
```"""

    def _get_basic_operations_docs(self) -> str:
        """Get basic operations fallback documentation."""
        return """## Basic Operations
```python
import bpy

# Selection operations
bpy.ops.object.select_all(action='DESELECT')
obj = bpy.data.objects.get('Cube')
if obj:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

# Transform operations
obj.location = (1, 2, 3)
obj.rotation_euler = (0.5, 0, 0)
obj.scale = (2, 1, 1)
```"""

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for retrieval agent."""
        return state.subtasks is not None and len(state.subtasks) > 0
