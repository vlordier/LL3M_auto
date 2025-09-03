"""Coding agent for generating executable Blender Python code."""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from ..blender.templates import (
    GEOMETRY_TEMPLATES,
    LIGHTING_TEMPLATES,
    MATERIAL_TEMPLATES,
    MODIFIER_TEMPLATES,
    SCENE_TEMPLATES,
)
from ..utils.types import AgentResponse, AgentType, SubTask, WorkflowState
from .base import EnhancedBaseAgent


@dataclass
class CodeGenerationPrompt:
    """Templates for code generation prompts."""

    SYSTEM_PROMPT = """You are an expert Blender Python programmer specializing in
procedural 3D asset creation.

Your role is to generate clean, efficient, and executable Blender Python code based on:
1. Structured subtasks with specific requirements
2. Relevant Blender API documentation
3. Code templates and best practices

Code Requirements:
- Use only the Blender Python API (bpy module)
- Generate modular, readable code with proper error handling
- Include comments explaining key operations
- Follow Blender best practices for object creation and manipulation
- Ensure objects are properly named and organized
- Handle edge cases and provide fallbacks

Code Structure:
- Import statements at the top
- Scene setup and cleanup
- Object creation and modification
- Material and lighting setup
- Final scene organization"""

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
        self.prompt_template = CodeGenerationPrompt()

        # Initialize template system with available templates
        self.templates = {
            "geometry": GEOMETRY_TEMPLATES,
            "material": MATERIAL_TEMPLATES,
            "lighting": LIGHTING_TEMPLATES,
            "scene": SCENE_TEMPLATES,
            "modifier": MODIFIER_TEMPLATES,
        }

    @property
    def agent_type(self) -> AgentType:
        """Return coding agent type."""
        return AgentType.CODING

    @property
    def name(self) -> str:
        """Return human-readable agent name."""
        return "Code Generator"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Generate Blender Python code from subtasks and documentation."""
        start_time = time.monotonic()

        try:
            # Validate inputs
            if not state.subtasks:
                return AgentResponse(
                    success=False,
                    data=None,
                    message="No subtasks available for code generation",
                    agent_type=self.agent_type,
                    execution_time=time.monotonic() - start_time,
                )

            # Prepare documentation context from previous retrieval step
            documentation = state.documentation or "No documentation available"

            # Convert subtasks to JSON format for the prompt
            subtasks_data = [
                {
                    "id": task.id,
                    "type": task.type.value,
                    "description": task.description,
                    "priority": task.priority,
                    "dependencies": task.dependencies,
                    "parameters": task.parameters,
                }
                for task in state.subtasks
            ]

            # Generate code using OpenAI
            system_prompt = self.prompt_template.SYSTEM_PROMPT
            user_prompt = self.prompt_template.USER_TEMPLATE.format(
                subtasks_json=json.dumps(subtasks_data, indent=2),
                documentation=documentation,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Make API request
            raw_response = await self.make_openai_request(
                messages=cast(list[ChatCompletionMessageParam], messages)
            )

            # Extract and clean generated code
            cleaned_code = self._clean_generated_code(raw_response)

            # Basic validation - check if code contains import bpy
            if "import bpy" not in cleaned_code:
                cleaned_code = "import bpy\n\n" + cleaned_code

            # Apply code templates and optimizations
            final_code = await self._enhance_code_with_templates(
                cleaned_code, state.subtasks
            )

            execution_time = time.monotonic() - start_time

            return AgentResponse(
                success=True,
                data=final_code,
                message=(
                    f"Generated {len(final_code)} characters of Blender Python code"
                ),
                execution_time=execution_time,
                metadata={
                    "subtasks_processed": len(state.subtasks),
                    "code_length": len(final_code),
                    "templates_used": list(self.templates.keys()),
                },
                agent_type=self.agent_type,
            )

        except Exception as e:
            self.logger.error("Code generation failed", error=str(e))
            return AgentResponse(
                success=False,
                data=None,
                message=f"Code generation error: {str(e)}",
                agent_type=self.agent_type,
                execution_time=time.monotonic() - start_time,
            )

    async def _enhance_code_with_templates(
        self, base_code: str, subtasks: list[SubTask]
    ) -> str:
        """Enhance generated code with templates and optimizations."""
        enhanced_code = base_code

        # Apply task-specific templates
        for task in subtasks:
            task_type = task.type.value.lower()
            if task_type in self.templates:
                template = self.templates[task_type]
                if template and task.parameters:
                    # Apply template-specific enhancements
                    enhanced_code = await self._apply_template(
                        enhanced_code, template, task
                    )

        return enhanced_code

    async def _apply_template(
        self, code: str, template: dict[str, str], task: SubTask
    ) -> str:
        """Apply a specific template to enhance code."""
        enhanced_code = code

        # Extract template key from task parameters or description
        template_key = self._determine_template_key(task, template)

        if template_key and template_key in template:
            template_code = template[template_key]

            # Fill template with task parameters
            filled_template = self._fill_template(template_code, task)

            # Add template-based code if it's not already present
            if not self._is_template_already_applied(code, template_key):
                enhanced_code = self._integrate_template_code(code, filled_template)

        # Add task comment
        task_comment = f"# Task: {task.description}\n"
        enhanced_code = self._add_task_comment(enhanced_code, task_comment)

        return enhanced_code

    def _determine_template_key(
        self, task: SubTask, template: dict[str, str]
    ) -> str | None:
        """Determine which template to use based on task parameters and description."""
        # Check task parameters for explicit template
        if "template" in task.parameters:
            return str(task.parameters["template"])

        # Infer from description for common shapes/operations
        description = task.description.lower()

        for template_key in template.keys():
            if template_key in description:
                return template_key

        # Default fallback based on task type
        if task.type.value.lower() == "geometry":
            if "sphere" in description or "ball" in description:
                return "sphere"
            elif "cylinder" in description or "tube" in description:
                return "cylinder"
            elif "plane" in description or "floor" in description:
                return "plane"
            else:
                return "cube"  # Default geometry

        return None

    def _fill_template(self, template_code: str, task: SubTask) -> str:
        """Fill template with task parameters."""
        # Default values
        params = {
            "x": 0,
            "y": 0,
            "z": 0,
            "rx": 0,
            "ry": 0,
            "rz": 0,
            "r": 0.8,
            "g": 0.2,
            "b": 0.2,
            "name": f"Object_{task.id}",
            "object_name": "bpy.context.object",
            "energy": 5.0,
            "size": 2.0,
            "metallic": 0.5,
            "roughness": 0.3,
            "strength": 1.0,
            "samples": 128,
            "width": 1920,
            "height": 1080,
            "levels": 2,
            "segments": 3,
            "count": 3,
            "offset_x": 2.0,
            "offset_y": 0.0,
            "offset_z": 0.0,
        }

        # Update with task parameters
        if task.parameters:
            # Handle location parameter
            if "location" in task.parameters:
                location = task.parameters["location"]
                if isinstance(location, list | tuple) and len(location) >= 3:
                    params["x"] = location[0]
                    params["y"] = location[1]
                    params["z"] = location[2]

            # Handle color parameter
            if "color" in task.parameters:
                color = task.parameters["color"]
                if isinstance(color, list | tuple) and len(color) >= 3:
                    params["r"] = color[0]
                    params["g"] = color[1]
                    params["b"] = color[2]

            # Update any other direct parameter matches
            for key, value in task.parameters.items():
                if key in params:
                    params[key] = value

        # Format template with parameters
        try:
            return template_code.format(**params)
        except KeyError as e:
            # If formatting fails, return original template
            self.logger.warning(f"Template formatting failed: {e}")
            return template_code

    def _is_template_already_applied(self, code: str, template_key: str) -> bool:
        """Check if template-specific code is already present."""
        # Simple heuristics to avoid duplicate template application
        if template_key == "cube" and "primitive_cube_add" in code:
            return True
        elif template_key == "sphere" and "primitive_uv_sphere_add" in code:
            return True
        elif template_key == "cylinder" and "primitive_cylinder_add" in code:
            return True
        elif template_key == "plane" and "primitive_plane_add" in code:
            return True

        return False

    def _integrate_template_code(self, code: str, template_code: str) -> str:
        """Integrate template code into existing code."""
        lines = code.split("\n")

        # Find appropriate insertion point
        insertion_point = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                # Insert at first empty line after imports
                insertion_point = i
                break

        # Insert template code
        template_lines = template_code.strip().split("\n")
        for i, template_line in enumerate(template_lines):
            lines.insert(insertion_point + i, template_line)

        return "\n".join(lines)

    def _add_task_comment(self, code: str, task_comment: str) -> str:
        """Add task comment to code."""
        lines = code.split("\n")
        if lines and lines[0].startswith("import"):
            # Find the end of imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith("import") or line.startswith("from"):
                    import_end = i + 1
                elif line.strip() == "":
                    continue
                else:
                    break

            lines.insert(import_end, f"\n{task_comment}")
        else:
            lines.insert(0, task_comment)

        return "\n".join(lines)

    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean generated code by removing markdown markers and explanations."""
        # Remove code block markers
        code = raw_code.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        # Split into lines and remove explanatory text after code
        lines = code.split("\n")
        code_lines = []

        for line in lines:
            # Skip lines that are clearly markdown markers
            if line.strip() == "```":
                continue

            # Stop processing when we encounter explanatory text
            if line.strip() and not line.strip().startswith("#"):
                # Check if this looks like explanatory text rather than code
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "this code creates",
                        "this creates",
                        "code creates",
                        "this adds",
                        "this script",
                    ]
                ):
                    break
                else:
                    code_lines.append(line)
            else:
                code_lines.append(line)

        return "\n".join(code_lines).strip()

    def _validate_code_structure(self, code: str) -> dict[str, Any]:
        """Validate code structure and return validation results."""
        issues = []

        # Handle escaped newlines in the code
        actual_code = code.replace("\\n", "\n")

        # Check for bpy import
        if "import bpy" not in actual_code:
            issues.append("Missing bpy import")

        # Check basic syntax
        try:
            compile(actual_code, "<string>", "exec")
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Compilation error: {e}")

        return {"valid": len(issues) == 0, "issues": issues}

    def _generate_fallback_code(self, subtasks: list[SubTask]) -> str:
        """Generate fallback code when main generation fails."""
        fallback_lines = ["import bpy", ""]

        for task in subtasks:
            if task.type.value.lower() == "geometry":
                if "cube" in task.description.lower():
                    fallback_lines.append("# Create cube")
                    fallback_lines.append("bpy.ops.mesh.primitive_cube_add()")
                elif "sphere" in task.description.lower():
                    fallback_lines.append("# Create sphere")
                    fallback_lines.append("bpy.ops.mesh.primitive_uv_sphere_add()")
                else:
                    fallback_lines.append(f"# {task.description}")
                    fallback_lines.append("bpy.ops.mesh.primitive_cube_add()")
                fallback_lines.append("")

        return "\n".join(fallback_lines)

    def _validate_code_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax of generated code."""
        try:
            compile(code, "<string>", "exec")
            return True, "Syntax is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Compilation error: {e}"

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input state for code generation."""
        # Check basic requirements from base class
        if not await super().validate_input(state):
            return False

        # Check for subtasks - coding agent needs subtasks to work with
        if not state.subtasks:
            return False

        # Check for documentation - while not strictly required, it's expected
        if not state.documentation:
            self.logger.warning("No documentation provided, will use templates")

        return True

    def _extract_code_metrics(self, code: str) -> dict[str, int]:
        """Extract metrics from generated code."""
        lines = code.split("\n")
        return {
            "total_lines": len(lines),
            "code_lines": len(
                [
                    line
                    for line in lines
                    if line.strip() and not line.strip().startswith("#")
                ]
            ),
            "comment_lines": len(
                [line for line in lines if line.strip().startswith("#")]
            ),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "bpy_calls": len(re.findall(r"bpy\.", code)),
            "function_definitions": len(re.findall(r"def\s+\w+\s*\(", code)),
        }
