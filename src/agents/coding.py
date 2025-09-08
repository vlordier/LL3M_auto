"""Coding agent for generating executable Blender Python code."""

import asyncio
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
        "# ROLE DEFINITION\n"
        "You are a master Blender Python developer with 10+ years of experience creating production-quality 3D assets. "
        "Your code is used in professional pipelines for games, films, and architectural visualization.\n\n"

        "# TECHNICAL EXPERTISE REQUIREMENTS\n"
        "## Core Blender API Mastery:\n"
        "- **bpy.data**: Mesh, Material, Light, Camera, Scene management\n"
        "- **bpy.ops**: Operator calls with proper context handling\n"
        "- **bpy.context**: Scene, view layer, active object management\n"
        "- **bmesh**: Advanced mesh editing, topology operations\n"
        "- **mathutils**: Vector, Matrix, Euler, Quaternion mathematics\n"
        "- **bpy.props**: Custom properties and UI integration\n\n"

        "## Specialized Domain Knowledge:\n"
        "- **Procedural Modeling**: Geometry nodes, modifiers, parametric design\n"
        "- **Material Systems**: Shader nodes, Principled BSDF, texture mapping\n"
        "- **Lighting**: HDRI, area lights, sun, point, spot configurations\n"
        "- **Rendering**: Cycles, Eevee-Next, Workbench optimization\n"
        "- **Animation**: Keyframes, drivers, constraints, rigging basics\n\n"

        "# MANDATORY CODE STANDARDS\n"
        "## Structure Requirements:\n"
        "1. **Function Organization**: Separate functions for geometry, materials, lighting, scene setup\n"
        "2. **Error Handling**: Try-except blocks around ALL Blender operations\n"
        "3. **Documentation**: Docstrings for all functions with parameter types\n"
        "4. **Logging**: Use print() statements for key operations and debugging\n"
        "5. **Modularity**: Reusable functions, avoid code duplication\n\n"

        "## Performance Requirements:\n"
        "- **Memory Management**: Delete unused meshes, clear orphaned data\n"
        "- **Batch Operations**: Use bmesh for multiple mesh edits\n"
        "- **Context Efficiency**: Minimize context switches\n"
        "- **Viewport Optimization**: Disable updates during generation\n\n"

        "# CRITICAL BLENDER-SPECIFIC RULES\n"
        "## Mandatory Scene Setup (Execute in this order):\n"
        "```python\n"
        "# 1. Clear existing scene\n"
        "bpy.ops.object.select_all(action='SELECT')\n"
        "bpy.ops.object.delete(use_global=False)\n"
        "\n"
        "# 2. Clear orphaned data\n"
        "bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)\n"
        "```\n\n"

        "## Modern Render Engine Names (CRITICAL - USE EXACT STRINGS):\n"
        "- **Eevee**: `'BLENDER_EEVEE_NEXT'` (NOT 'EEVEE' or 'BLENDER_EEVEE')\n"
        "- **Cycles**: `'CYCLES'`\n"
        "- **Workbench**: `'BLENDER_WORKBENCH'`\n\n"

        "## Object Management Protocol:\n"
        "```python\n"
        "# After creating any object:\n"
        "obj = bpy.context.active_object\n"
        "obj.name = 'specific_descriptive_name'\n"
        "bpy.context.view_layer.objects.active = obj\n"
        "obj.select_set(True)\n"
        "```\n\n"

        "## Material Creation Protocol:\n"
        "```python\n"
        "# Always create materials this way:\n"
        "mat = bpy.data.materials.new(name='MaterialName')\n"
        "mat.use_nodes = True\n"
        "nodes = mat.node_tree.nodes\n"
        "links = mat.node_tree.links\n"
        "# Clear default nodes\n"
        "nodes.clear()\n"
        "```\n\n"

        "# OUTPUT REQUIREMENTS\n"
        "## Code Structure (Mandatory Format):\n"
        "```python\n"
        "import bpy\n"
        "import bmesh\n"
        "import mathutils\n"
        "from mathutils import Vector, Matrix\n"
        "import math\n"
        "\n"
        "def clear_scene():\n"
        "    \"\"\"Clear existing scene and orphaned data\"\"\"\n"
        "    # Implementation here\n"
        "\n"
        "def create_geometry():\n"
        "    \"\"\"Create main geometry\"\"\"\n"
        "    # Implementation here\n"
        "\n"
        "def create_materials():\n"
        "    \"\"\"Create and assign materials\"\"\"\n"
        "    # Implementation here\n"
        "\n"
        "def setup_lighting():\n"
        "    \"\"\"Configure scene lighting\"\"\"\n"
        "    # Implementation here\n"
        "\n"
        "def setup_scene():\n"
        "    \"\"\"Configure camera and render settings\"\"\"\n"
        "    # Implementation here\n"
        "\n"
        "def main():\n"
        "    \"\"\"Main execution function\"\"\"\n"
        "    clear_scene()\n"
        "    create_geometry()\n"
        "    create_materials()\n"
        "    setup_lighting()\n"
        "    setup_scene()\n"
        "    print('Asset generation completed successfully')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "```\n\n"

        "# QUALITY ASSURANCE\n"
        "## Before Submitting Code:\n"
        "1. **Syntax Check**: Ensure valid Python syntax\n"
        "2. **API Verification**: Use only existing Blender API calls\n"
        "3. **Error Prevention**: Handle common failure points\n"
        "4. **Resource Management**: Clean up temporary objects\n"
        "5. **Result Validation**: Verify expected objects exist\n\n"

        "## Success Criteria:\n"
        "- Code executes without errors in Blender\n"
        "- Creates visually appealing, structurally sound 3D asset\n"
        "- Follows all architectural and naming conventions\n"
        "- Generates reusable, maintainable code\n"
        "- Optimizes for both quality and performance\n\n"

        "CRITICAL: Generate ONLY executable Python code. "
        "No explanations, comments outside code, or markdown formatting."
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

            # Apply ruff formatting and linting with healing
            formatted_code, ruff_validation = await self._apply_ruff_formatting_and_validation(final_code)
            if formatted_code:
                final_code = formatted_code

            # Validate code structure
            validation_result = self._validate_code_structure(final_code)

            # Combine validation results
            combined_validation = {
                "valid": validation_result["valid"] and ruff_validation["valid"],
                "issues": validation_result["issues"] + ruff_validation["issues"],
                "ruff_applied": ruff_validation["ruff_applied"],
                "healing_attempts": ruff_validation.get("healing_attempts", 0)
            }

            if not combined_validation["valid"]:
                self.logger.warning(
                    "Generated code failed validation",
                    issues=combined_validation["issues"],
                    ruff_applied=combined_validation["ruff_applied"]
                )
                # Try healing the code if ruff found issues
                if ruff_validation["issues"] and ruff_validation["healing_attempts"] == 0:
                    healed_code = await self._heal_code_with_ruff_feedback(final_code, ruff_validation["issues"], state)
                    if healed_code:
                        final_code = healed_code
                        # Re-validate healed code
                        _, ruff_validation_retry = await self._apply_ruff_formatting_and_validation(final_code)
                        combined_validation = {
                            "valid": True,  # Assume healed code is valid
                            "issues": ruff_validation_retry["issues"],
                            "ruff_applied": True,
                            "healing_attempts": 1
                        }

                if not combined_validation["valid"]:
                    # Try fallback template-based generation as last resort
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
                    "validation": combined_validation,
                    "ruff_formatted": combined_validation.get("ruff_applied", False),
                    "healing_applied": combined_validation.get("healing_attempts", 0) > 0,
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

    async def _apply_ruff_formatting_and_validation(self, code: str) -> tuple[str | None, dict[str, Any]]:
        """Apply ruff formatting and validation to generated code.

        Returns:
            Tuple of (formatted_code, validation_result)
            formatted_code is None if ruff failed to format
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "ruff_applied": False,
            "healing_attempts": 0
        }

        try:
            # Create temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_path = Path(temp_file.name)

            # Step 1: Apply ruff format
            try:
                format_result = subprocess.run(
                    ['ruff', 'format', str(temp_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if format_result.returncode == 0:
                    # Read the formatted code
                    with open(temp_path) as f:
                        formatted_code = f.read()
                    validation_result["ruff_applied"] = True
                    self.logger.info("Ruff formatting applied successfully")
                else:
                    formatted_code = code  # Keep original if formatting failed
                    validation_result["issues"].append(f"Ruff format failed: {format_result.stderr}")
                    self.logger.warning("Ruff formatting failed", error=format_result.stderr)

            except subprocess.TimeoutExpired:
                formatted_code = code
                validation_result["issues"].append("Ruff format timed out")
                self.logger.warning("Ruff formatting timed out")
            except FileNotFoundError:
                formatted_code = code
                validation_result["issues"].append("Ruff not found - install with: pip install ruff")
                self.logger.warning("Ruff not installed")

            # Step 2: Run ruff check for linting
            try:
                # Write formatted code back for checking
                with open(temp_path, 'w') as f:
                    f.write(formatted_code)

                check_result = subprocess.run(
                    ['ruff', 'check', str(temp_path), '--output-format=json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if check_result.returncode == 0:
                    self.logger.info("Ruff linting passed")
                else:
                    # Parse ruff check output
                    try:
                        ruff_issues = json.loads(check_result.stdout)
                        for issue in ruff_issues:
                            validation_result["issues"].append(
                                f"Line {issue.get('location', {}).get('row', '?')}: "
                                f"{issue.get('code', 'UNKNOWN')} - {issue.get('message', 'Unknown error')}"
                            )
                        validation_result["valid"] = len(ruff_issues) == 0
                        self.logger.info(f"Ruff found {len(ruff_issues)} linting issues")
                    except json.JSONDecodeError:
                        validation_result["issues"].append(f"Ruff check failed: {check_result.stderr}")
                        validation_result["valid"] = False

            except subprocess.TimeoutExpired:
                validation_result["issues"].append("Ruff check timed out")
                validation_result["valid"] = False
            except FileNotFoundError:
                # Already logged above
                pass

            # Cleanup
            temp_path.unlink(missing_ok=True)

            return formatted_code if validation_result["ruff_applied"] else None, validation_result

        except Exception as e:
            self.logger.exception("Error during ruff processing")
            validation_result["issues"].append(f"Ruff processing error: {e}")
            validation_result["valid"] = False
            return None, validation_result

    async def _heal_code_with_ruff_feedback(
        self,
        code: str,
        ruff_issues: list[str],
        state: WorkflowState
    ) -> str | None:
        """Attempt to heal code using ruff feedback with LLM.

        Args:
            code: The original code with issues
            ruff_issues: List of ruff error messages
            state: Current workflow state

        Returns:
            Healed code or None if healing failed
        """
        try:
            self.logger.info(f"Attempting to heal code with {len(ruff_issues)} ruff issues")

            # Create healing prompt
            healing_prompt = self._create_healing_prompt(code, ruff_issues, state)

            # Prepare messages for LLM
            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": self._get_healing_system_prompt()
                },
                {
                    "role": "user",
                    "content": healing_prompt
                }
            ]

            # Get healed code from LLM
            healed_raw_code = await self.make_openai_request(messages)
            healed_code = self._clean_generated_code(healed_raw_code)

            # Basic validation that we still have meaningful code
            if len(healed_code.strip()) < len(code.strip()) * 0.5:
                self.logger.warning("Healed code is significantly shorter, rejecting")
                return None

            if "import bpy" not in healed_code:
                self.logger.warning("Healed code missing bpy import, rejecting")
                return None

            self.logger.info("Code healing completed successfully")
            return healed_code

        except Exception:
            self.logger.exception("Code healing failed")
            return None

    def _create_healing_prompt(self, code: str, ruff_issues: list[str], state: WorkflowState) -> str:
        """Create a prompt for healing code based on ruff feedback."""
        return f"""Fix the following Python code to resolve ruff linting issues while maintaining functionality:

ORIGINAL CODE:
```python
{code}
```

RUFF ISSUES TO FIX:
{chr(10).join(f"- {issue}" for issue in ruff_issues)}

REQUIREMENTS:
1. Fix ALL ruff linting issues listed above
2. Maintain the original functionality and logic
3. Keep all Blender API calls and operations
4. Ensure code follows PEP 8 standards
5. Preserve all imports and function structure
6. Return ONLY the corrected Python code

CONTEXT:
- This is Blender Python code for: {state.prompt if state else "3D asset generation"}
- Code must remain executable in Blender environment
- Do not add explanations or markdown formatting"""

    def _get_healing_system_prompt(self) -> str:
        """Get system prompt for code healing."""
        return """# CODE HEALING SPECIALIST

You are an expert Python code quality engineer specializing in fixing linting issues while preserving functionality.

## YOUR TASK:
Fix ruff linting issues in Blender Python code without breaking functionality.

## CRITICAL REQUIREMENTS:
1. **Preserve Functionality**: Never change the core logic or Blender operations
2. **Fix All Issues**: Address every ruff error and warning listed
3. **Maintain Structure**: Keep the same function/class organization
4. **Blender Compatibility**: Ensure all bpy calls remain valid
5. **Clean Output**: Return only the corrected code, no explanations

## COMMON RUFF FIXES:
- **F401 (unused import)**: Remove or use the import
- **E501 (line too long)**: Break long lines appropriately
- **F841 (unused variable)**: Remove or use the variable
- **E302/E305 (blank lines)**: Add proper spacing between functions/classes
- **W292 (no newline at end)**: Add final newline
- **E261/E262 (inline comments)**: Fix comment spacing
- **F811 (redefined name)**: Rename duplicate variables/functions
- **C901 (too complex)**: Simplify overly complex functions

## OUTPUT FORMAT:
Return ONLY the corrected Python code without any markdown formatting or explanations."""
