"""Coding agent for generating executable Blender Python code."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from ..blender.templates import (
    GEOMETRY_TEMPLATES,
    LIGHTING_TEMPLATES,
    MATERIAL_TEMPLATES,
    SCENE_TEMPLATES,
)
from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import check_user_termination


@dataclass
class CodeGenerationPrompt:
    """Templates for code generation prompts."""

    SYSTEM_PROMPT = (
        "You are a master Blender Python developer creating production-quality 3D assets.\n\n"
        "EXPERTISE:\n"
        "- Blender Python API (bpy, bmesh, mathutils)\n"
        "- Procedural modeling techniques\n"
        "- Material node systems (Principled BSDF, textures, shaders)\n"
        "- Lighting and rendering optimization\n"
        "- Scene organization and naming conventions\n\n"
        "CODE QUALITY STANDARDS:\n"
        "- Write clean, modular, well-documented Python code\n"
        "- Use proper error handling and validation\n"
        "- Follow Blender API best practices\n"
        "- Ensure reproducible results across sessions\n"
        "- Optimize for performance and memory usage\n\n"
        "BLENDER-SPECIFIC REQUIREMENTS:\n"
        "- Always clear scene first: bpy.ops.object.select_all(), bpy.ops.object.delete()\n"
        "- Use proper object naming: obj.name = 'descriptive_name'\n"
        "- Set active object after creation: bpy.context.view_layer.objects.active = obj\n"
        "- Apply transformations when needed: bpy.ops.object.transform_apply()\n"
        "- Use bmesh for complex geometry operations\n"
        "- Create materials with proper node setups\n\n"
        "OUTPUT: Generate complete, executable Python code that runs successfully in Blender."
    )

    USER_TEMPLATE = """Generate Blender Python code for these subtasks:

Subtasks:
{subtasks_json}

Available Documentation:
{documentation}

Requirements:
- Generate complete, executable Python code
- Follow the subtask order and dependencies
- Use proper Blender API calls
- Include error handling and logging
- Make code modular and well-commented
- Ensure reproducible results

Return only the Python code, no explanations."""


class CodingAgent(EnhancedBaseAgent):
    """Generates executable Blender Python code from subtasks and documentation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize coding agent."""
        super().__init__(config)
        self.code_generator = CodeGenerationPrompt()
        self.templates = {
            TaskType.GEOMETRY: GEOMETRY_TEMPLATES,
            TaskType.MATERIAL: MATERIAL_TEMPLATES,
            TaskType.LIGHTING: LIGHTING_TEMPLATES,
            TaskType.SCENE_SETUP: SCENE_TEMPLATES,
        }

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.CODING

    @property
    def name(self) -> str:
        """Return agent name."""
        return "Code Generator"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Generate Blender Python code from subtasks and documentation."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Check for user termination request
            if check_user_termination(state):
                self.logger.info("User requested workflow termination")
                # Set workflow to stop continuing
                state.should_continue = False
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    data="",
                    message="Workflow terminated by user request",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="Invalid input: subtasks and documentation required",
                    execution_time=0.0,
                )

            self.logger.info("Starting code generation", num_subtasks=len(state.subtasks))

            # Prepare subtasks data for LLM
            subtasks_json = self._prepare_subtasks_for_llm(state.subtasks)

            # Create generation prompt
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self.code_generator.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.code_generator.USER_TEMPLATE.format(
                        subtasks_json=subtasks_json, documentation=state.documentation
                    ),
                },
            ]

            # Generate code with LLM
            raw_code = await self.make_openai_request(messages)

            # Clean and validate generated code
            clean_code = self._clean_generated_code(raw_code)

            # Enhance code with templates and best practices
            enhanced_code = self._enhance_with_templates(clean_code, state.subtasks)

            # Add standard imports and setup
            final_code = self._add_standard_setup(enhanced_code)

            # Validate code structure
            validation_result = self._validate_code_structure(final_code)

            if not validation_result["valid"]:
                self.logger.warning(
                    "Generated code failed validation",
                    issues=validation_result["issues"],
                )
                # Try fallback template-based generation
                final_code = self._generate_fallback_code(state.subtasks)

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Code generation completed",
                code_length=len(final_code),
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=final_code,
                message=(f"Generated {len(final_code)} characters of Blender Python code"),
                execution_time=execution_time,
                metadata={
                    "code_lines": len(final_code.split("\n")),
                    "validation": validation_result,
                },
            )

        except Exception as code_generation_error:
            AgentErrorHandler.log_error(
                self.logger,
                code_generation_error,
                "code generation",
                subtasks_count=len(state.subtasks) if state.subtasks else 0,
            )
            return AgentErrorHandler.create_error_response(
                self.agent_type, code_generation_error, "code generation", start_time
            )

    def _prepare_subtasks_for_llm(self, subtasks: list[SubTask]) -> str:
        """Prepare subtasks data in JSON format for LLM consumption."""
        subtasks_data = []

        for subtask in subtasks:
            subtasks_data.append(
                {
                    "id": subtask.id,
                    "type": subtask.type.value,
                    "description": subtask.description,
                    "priority": subtask.priority,
                    "dependencies": subtask.dependencies,
                    "parameters": subtask.parameters,
                }
            )

        return json.dumps(subtasks_data, indent=2)

    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean and format generated code."""
        # Remove markdown code blocks
        code = re.sub(r"```python\s*\n?", "", raw_code)
        code = re.sub(r"```\s*$", "", code)

        # Remove explanatory text before/after code
        lines = code.split("\n")
        start_idx = 0
        end_idx = len(lines)

        # Find first import or bpy statement
        for i, line in enumerate(lines):
            if ("import " in line or "bpy." in line) and not line.strip().startswith("#"):
                start_idx = i
                break

        # Take code from first import to end
        cleaned_lines = lines[start_idx:end_idx]

        # Remove trailing explanatory text
        while (
            cleaned_lines
            and not any(
                keyword in cleaned_lines[-1] for keyword in ["bpy.", "import", "def ", "class ", "=", "if ", "for "]
            )
            and not cleaned_lines[-1].strip().startswith("#")
        ):
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    def _enhance_with_templates(self, code: str, subtasks: list[SubTask]) -> str:
        """Enhance generated code with template-based improvements."""
        enhanced_code = code

        # Add template-based enhancements for missing functionality
        for subtask in subtasks:
            if subtask.type in self.templates:
                template_additions = self._get_template_enhancements(subtask)
                if template_additions:
                    enhanced_code += f"\n\n# Template enhancement for {subtask.id}\n"
                    enhanced_code += template_additions

        return enhanced_code

    def _get_template_enhancements(self, subtask: SubTask) -> str:
        """Get template-based code enhancements for a subtask."""
        templates = self.templates.get(subtask.type, {})

        # Extract relevant templates based on subtask description
        relevant_templates = []
        description_lower = subtask.description.lower()

        for template_name, template_code in templates.items():
            if template_name.lower() in description_lower:
                relevant_templates.append(template_code)

        if relevant_templates:
            # Format templates with subtask parameters
            formatted_templates = []
            for template in relevant_templates:
                try:
                    formatted = template.format(**subtask.parameters)
                    formatted_templates.append(formatted)
                except KeyError:
                    # Template requires parameters not available
                    formatted_templates.append(template)

            return "\n".join(formatted_templates)

        return ""

    def _add_standard_setup(self, code: str) -> str:
        """Add standard imports and setup code."""
        setup_code = """import bpy
import bmesh
import mathutils
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Ensure we're in object mode
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

"""

        # Check if code already has imports/setup
        if "import bpy" not in code:
            return setup_code + code
        else:
            return code

    def _validate_code_structure(self, code: str) -> dict[str, Any]:
        """Validate the structure and safety of generated code."""
        issues = []

        # Check for required imports
        if "import bpy" not in code:
            issues.append("Missing bpy import")

        # Check for unsafe operations
        unsafe_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"import\s+os",
            r"import\s+subprocess",
            r"__import__",
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Unsafe operation detected: {pattern}")

        # Check for basic Blender operations
        if "bpy.ops." not in code and "bpy.data." not in code:
            issues.append("No Blender operations found")

        # Validate Python syntax
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            issues.append(f"Syntax error: {e!s}")

        return {"valid": len(issues) == 0, "issues": issues}

    def _generate_fallback_code(self, subtasks: list[SubTask]) -> str:
        """Generate fallback code using templates when LLM generation fails."""
        fallback_code = """import bpy
import bmesh

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

"""

        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                fallback_code += self._generate_geometry_fallback(subtask)
            elif subtask.type == TaskType.MATERIAL:
                fallback_code += self._generate_material_fallback(subtask)
            elif subtask.type == TaskType.LIGHTING:
                fallback_code += self._generate_lighting_fallback(subtask)

            fallback_code += "\n"

        return fallback_code

    def _generate_geometry_fallback(self, subtask: SubTask) -> str:
        """Generate fallback geometry code."""
        shape = subtask.parameters.get("shape", "cube")
        location = subtask.parameters.get("location", [0, 0, 0])

        return f"""
# Create {subtask.description}
bpy.ops.mesh.primitive_{shape}_add(location={location})
obj = bpy.context.active_object
obj.name = "{subtask.id}"
"""

    def _generate_material_fallback(self, subtask: SubTask) -> str:
        """Generate fallback material code."""
        color = subtask.parameters.get("color", [0.8, 0.2, 0.2, 1.0])

        return f"""
# Create material for {subtask.description}
material = bpy.data.materials.new(name="{subtask.id}_material")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = {color}

# Assign to active object
if bpy.context.active_object:
    bpy.context.active_object.data.materials.append(material)
"""

    def _generate_lighting_fallback(self, subtask: SubTask) -> str:
        """Generate fallback lighting code."""
        light_type = subtask.parameters.get("type", "SUN")
        location = subtask.parameters.get("location", [5, 5, 10])
        energy = subtask.parameters.get("energy", 3.0)

        return f"""
# Create light for {subtask.description}
bpy.ops.object.light_add(type='{light_type}', location={location})
light = bpy.context.active_object
light.data.energy = {energy}
light.name = "{subtask.id}_light"
"""

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for coding agent."""
        if not state.subtasks or len(state.subtasks) == 0:
            return False

        if not state.documentation or len(state.documentation.strip()) == 0:
            self.logger.warning("No documentation provided, will use templates")
            # Still valid, will use fallback templates

        return True
