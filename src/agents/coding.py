"""Coding agent for generating executable Blender Python code."""

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from ..blender.templates import (
    GEOMETRY_TEMPLATES,
    LIGHTING_TEMPLATES,
    MATERIAL_TEMPLATES,
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
            raw_response = await self.make_openai_request(messages=messages)

            # Extract and validate generated code
            generated_code = raw_response.strip()

            # Basic validation - check if code contains import bpy
            if "import bpy" not in generated_code:
                generated_code = "import bpy\n\n" + generated_code

            # Apply code templates and optimizations
            final_code = await self._enhance_code_with_templates(
                generated_code, state.subtasks
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
        # This would implement template-specific code enhancements
        # For now, just return the original code with comments
        # TODO: Implement actual template usage
        _ = template  # Unused for now, but will be used for template application
        task_comment = f"# Task: {task.description}\n"

        # Insert task comment before the main content
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
            return False

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
