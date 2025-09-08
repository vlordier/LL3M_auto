"""Coding agent for generating executable Blender Python code."""

import asyncio
import json
import math
import re
import secrets
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from ..services import services
from ..utils.types import AgentResponse, AgentType, SubTask, TaskType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import check_user_termination
from ..blender.templates import (
    GEOMETRY_TEMPLATES,
    LIGHTING_TEMPLATES,
    MATERIAL_TEMPLATES,
    SCENE_TEMPLATES,
)
from .mcts import MCTSTaskDecomposer

# Context7 integration for enhanced documentation
CONTEXT7_AVAILABLE = True  # Assume available in this environment


@dataclass
class HealingNode:
    """Node in the MCTS tree for code healing strategies."""
    id: str = field(default_factory=lambda: f"heal_{secrets.token_hex(4)}")
    parent: "HealingNode" = None
    children: list["HealingNode"] = field(default_factory=list)

    # Healing strategy information
    strategy: str = ""
    healed_code: str = ""
    remaining_issues: list[str] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False

    @property
    def average_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    @property
    def ucb1_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.average_reward

        exploration = 1.414 * math.sqrt(math.log(self.parent.visits) / self.visits)
        return self.average_reward + exploration


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

        "# QUALITY ASSURANCE & CODE FORMATTING\n"
        "## Critical Syntax Rules (PREVENT COMMON ERRORS):\n"
        "1. **Complete if/else blocks**: NEVER leave if statements without indented body\n"
        "   - BAD: `if condition:`\n"
        "   - GOOD: `if condition:\\n    pass  # or actual code`\n"
        "2. **Single statement per line**: NO colon-separated statements\n"
        "   - BAD: `x = 1; y = 2`\n"
        "   - GOOD: `x = 1\\ny = 2`\n"
        "3. **Safe imports only**: NO `import os` or system imports\n"
        "   - ALLOWED: `import bpy, bmesh, mathutils, math, random`\n"
        "4. **Proper indentation**: Use 4 spaces consistently\n\n"

        "## Before Submitting Code:\n"
        "1. **Syntax Check**: Ensure ALL if/for/while blocks have indented bodies\n"
        "2. **API Verification**: Use only existing Blender API calls\n"
        "3. **Error Prevention**: Handle common failure points\n"
        "4. **Resource Management**: Clean up temporary objects\n"
        "5. **Result Validation**: Verify expected objects exist\n"
        "6. **Format Compliance**: Each statement on separate line, proper spacing\n\n"

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

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:
        """Initialize coding agent."""
        super().__init__(config, client)
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

    async def _get_enhanced_blender_docs(self, subtasks: list[SubTask]) -> str:
        """Fetch enhanced Blender Python API documentation using Context7."""
        if not CONTEXT7_AVAILABLE:
            return ""

        try:
            # Determine which Blender API topics we need based on subtasks
            topics = []
            for task in subtasks:
                if task.type == TaskType.GEOMETRY:
                    topics.extend(["bpy.ops.mesh", "bmesh", "geometry", "modifiers"])
                elif task.type == TaskType.MATERIAL:
                    topics.extend(["bpy.data.materials", "shader nodes", "principled bsdf"])
                elif task.type == TaskType.LIGHTING:
                    topics.extend(["bpy.data.lights", "world", "hdri", "area light"])
                elif task.type == TaskType.SCENE_SETUP:
                    topics.extend(["bpy.context.scene", "camera", "render"])

            # Get unique topics
            unique_topics = list(set(topics))

            # Fetch Blender Python API documentation
            docs = []
            for topic in unique_topics[:5]:  # Limit to 5 topics to avoid token limits
                try:
                    # This will be handled by the MCP server integration
                    # For now, we'll enhance the existing documentation
                    docs.append(f"Enhanced docs for {topic} would be fetched here")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch docs for {topic}: {e}")
                    continue

            enhanced_docs = "\n".join(docs) if docs else ""
            self.logger.info("Enhanced documentation fetched", topics=len(unique_topics))

            return enhanced_docs

        except Exception as e:
            self.logger.warning(f"Failed to fetch enhanced documentation: {e}")
            return ""

    def _extract_execution_context(self, state: WorkflowState) -> dict[str, Any]:
        """Extract execution context for enhanced healing and generation."""
        context = {
            "has_execution_result": bool(state.execution_result),
            "execution_success": state.execution_result.success if state.execution_result else None,
            "execution_errors": state.execution_result.errors if state.execution_result else [],
            "execution_logs": state.execution_result.logs if state.execution_result else [],
            "previous_attempts": state.refinement_iterations,
            "detected_issues": [issue.description for issue in state.detected_issues],
        }

        if state.execution_result and not state.execution_result.success:
            # Extract common error patterns for better healing
            context["error_patterns"] = self._identify_error_patterns(state.execution_result.errors)

        return context

    def _identify_error_patterns(self, errors: list[str]) -> dict[str, list[str]]:
        """Identify common error patterns to improve code generation."""
        patterns = {
            "blender_api_errors": [],
            "python_syntax_errors": [],
            "import_errors": [],
            "context_errors": [],
            "material_errors": [],
            "geometry_errors": []
        }

        for error in errors:
            error_lower = error.lower()
            if "attributeerror" in error_lower and "bpy" in error_lower:
                patterns["blender_api_errors"].append(error)
            elif "syntaxerror" in error_lower or "indentationerror" in error_lower:
                patterns["python_syntax_errors"].append(error)
            elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
                patterns["import_errors"].append(error)
            elif "context" in error_lower and ("active" in error_lower or "selected" in error_lower):
                patterns["context_errors"].append(error)
            elif "material" in error_lower or "shader" in error_lower:
                patterns["material_errors"].append(error)
            elif "mesh" in error_lower or "geometry" in error_lower:
                patterns["geometry_errors"].append(error)

        return {k: v for k, v in patterns.items() if v}  # Only non-empty patterns

    def _create_error_learning_context(self, execution_context: dict[str, Any]) -> str:
        """Create learning context from execution errors to prevent repeated mistakes."""
        if not execution_context["execution_errors"]:
            return ""

        context_parts = [
            "## PREVIOUS EXECUTION ERRORS TO AVOID:",
            f"Attempt #{execution_context['previous_attempts'] + 1} - Learn from these failures:"
        ]

        error_patterns = execution_context.get("error_patterns", {})

        if error_patterns.get("blender_api_errors"):
            context_parts.append("\n### BLENDER API ERRORS:")
            for error in error_patterns["blender_api_errors"]:
                context_parts.append(f"- AVOID: {error}")
                # Add specific corrections based on common patterns
                if "attributeerror" in error.lower():
                    context_parts.append("  → Check Blender API documentation for correct method names")
                    context_parts.append("  → Ensure objects are properly selected and active")

        if error_patterns.get("context_errors"):
            context_parts.append("\n### CONTEXT MANAGEMENT ERRORS:")
            for error in error_patterns["context_errors"]:
                context_parts.append(f"- AVOID: {error}")
                context_parts.append("  → Always set active object: bpy.context.view_layer.objects.active = obj")
                context_parts.append("  → Ensure object selection: obj.select_set(True)")

        if error_patterns.get("material_errors"):
            context_parts.append("\n### MATERIAL/SHADER ERRORS:")
            for error in error_patterns["material_errors"]:
                context_parts.append(f"- AVOID: {error}")
                context_parts.append("  → Use mat.use_nodes = True before accessing node_tree")
                context_parts.append("  → Clear existing nodes: nodes.clear()")

        if error_patterns.get("geometry_errors"):
            context_parts.append("\n### GEOMETRY/MESH ERRORS:")
            for error in error_patterns["geometry_errors"]:
                context_parts.append(f"- AVOID: {error}")
                context_parts.append("  → Use bmesh for complex mesh operations")
                context_parts.append("  → Update mesh data: mesh.update()")

        # Add general error prevention tips
        context_parts.extend([
            "\n### ERROR PREVENTION STRATEGY:",
            "1. Always check object existence before operations",
            "2. Use try-except blocks around Blender API calls",
            "3. Validate context state before object manipulations",
            "4. Clear scene properly before creating new objects",
            "5. Update Blender data after modifications"
        ])

        return "\n".join(context_parts)

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

            # Extract execution context for trace-based improvements
            execution_context = self._extract_execution_context(state)
            self.logger.debug("Execution context extracted", context=execution_context)

            # Prepare subtasks data for LLM
            subtasks_json = self._prepare_subtasks_for_llm(state.subtasks)

            # Get enhanced Blender documentation
            enhanced_docs = await self._get_enhanced_blender_docs(state.subtasks)

            # Combine documentation sources
            combined_documentation = state.documentation
            if enhanced_docs:
                combined_documentation += f"\n\n# ENHANCED BLENDER API DOCUMENTATION\n{enhanced_docs}"

            # Add execution-based context if available
            if execution_context["execution_errors"]:
                error_context = self._create_error_learning_context(execution_context)
                combined_documentation += f"\n\n# EXECUTION ERROR CONTEXT\n{error_context}"

            # Create enhanced generation prompt with better formatting guidelines
            base_prompt = self.code_generator.USER_TEMPLATE.format(
                subtasks_json=subtasks_json, documentation=combined_documentation
            )
            enhanced_user_prompt = (
                f"{base_prompt}\n\n"
                "CRITICAL FORMATTING REMINDERS:\n"
                "- Every if/else/for/while statement MUST have an indented body (use 'pass' if empty)\n"
                "- Each Python statement on a separate line (no semicolons)\n"
                "- Use only safe imports: bpy, bmesh, mathutils, math, random\n"
                "- 4-space indentation consistently throughout\n"
                "- Include proper error handling with try-except blocks\n\n"
                "Generate clean, properly formatted Blender Python code:"
            )

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": self.code_generator.SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_user_prompt},
            ]

            # Generate code with LLM
            raw_code = await self.make_openai_request(messages)

            # Clean and validate generated code
            clean_code = self._clean_generated_code(raw_code)

            # Enhance code with templates and best practices
            enhanced_code = self._enhance_with_templates(clean_code, state.subtasks)

            # Add standard imports and setup
            pre_final_code = self._add_standard_setup(enhanced_code)

            # Apply pre-validation to fix common issues before Ruff
            final_code, pre_validation_issues = self._pre_validate_syntax(pre_final_code)
            if pre_validation_issues:
                self.logger.info(
                    f"Pre-validation applied {len(pre_validation_issues)} fixes",
                    fixes=pre_validation_issues
                )

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
                    healed_code = await self._heal_code_with_comprehensive_feedback(
                        final_code, ruff_validation["issues"], execution_context, state
                    )
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

            # EXECUTION VALIDATION - inspired by Blender agent pattern
            execution_validation = await self._validate_code_execution(final_code, state)
            if not execution_validation["success"] and execution_validation.get("retry_needed", False):
                # Execution failed - apply healing with execution feedback
                self.logger.info("Code execution failed, applying execution-based healing")
                execution_healed_code = await self._heal_with_execution_feedback(
                    final_code, execution_validation, state
                )
                if execution_healed_code:
                    final_code = execution_healed_code
                    # Re-validate healed code
                    execution_validation = await self._validate_code_execution(final_code, state)
                    # Re-validate healed code
                    execution_validation = await self._validate_code_execution(final_code, state)

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Code generation completed",
                code_length=len(final_code),
                execution_time=execution_time,
                execution_valid=execution_validation["success"]
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
                    "execution_validation": execution_validation,
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

    def _extract_code_from_response(self, response: str) -> str:
        """Extract and clean code from LLM response - alias for _clean_generated_code."""
        return self._clean_generated_code(response)

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

    def _pre_validate_syntax(self, code: str) -> tuple[str, list[str]]:
        """Pre-validate and fix common syntax issues before Ruff processing."""
        issues = []
        lines = code.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for incomplete if/else/for/while statements
            control_keywords = [
                'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except',
                'finally:', 'def ', 'class '
            ]
            if (stripped.endswith(':') and any(stripped.startswith(kw) for kw in control_keywords)):
                # Look ahead to see if there's an indented block
                if i + 1 >= len(lines) or not lines[i + 1].strip() or not lines[i + 1].startswith('    '):
                    # Add 'pass' statement with proper indentation
                    fixed_lines.append(line)
                    indentation = line[:len(line) - len(line.lstrip())]
                    fixed_lines.append(f"{indentation}    pass")
                    issues.append(f"Added 'pass' after incomplete block on line {i + 1}")
                else:
                    fixed_lines.append(line)
            # Check for multiple statements on one line (contains semicolon)
            elif ';' in line and not line.strip().startswith('#'):
                # Split on semicolon and put each statement on separate line
                statements = line.split(';')
                indentation = line[:len(line) - len(line.lstrip())]
                for j, stmt in enumerate(statements):
                    if stmt.strip():
                        if j == 0:
                            fixed_lines.append(stmt.rstrip())
                        else:
                            fixed_lines.append(f"{indentation}{stmt.strip()}")
                issues.append(f"Split multiple statements on line {i + 1}")
            # Check for unsafe imports
            elif re.match(r'^\s*import\s+os\b', line) or re.match(r'^\s*from\s+os\b', line):
                # Comment out unsafe imports
                fixed_lines.append(f"# {line.strip()}  # Unsafe import removed")
                issues.append(f"Removed unsafe import on line {i + 1}")
            else:
                fixed_lines.append(line)

            i += 1

        return '\n'.join(fixed_lines), issues

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

        # Step 0: Pre-validate and fix common syntax issues
        try:
            pre_validated_code, pre_issues = self._pre_validate_syntax(code)
            validation_result["issues"].extend(pre_issues)
            if pre_issues:
                self.logger.info(f"Pre-validation fixed {len(pre_issues)} issues", issues=pre_issues)
            code = pre_validated_code
        except Exception as e:
            validation_result["issues"].append(f"Pre-validation failed: {e}")
            self.logger.warning(f"Pre-validation failed: {e}")

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
                        
                        # Filter out Blender-specific import issues
                        filtered_issues = []
                        for issue in ruff_issues:
                            issue_code = issue.get('code', '')
                            issue_message = issue.get('message', '').lower()
                            
                            # Skip Blender-specific import issues
                            if self._is_blender_import_issue(issue_code, issue_message):
                                continue
                            
                            filtered_issues.append(issue)
                            validation_result["issues"].append(
                                f"Line {issue.get('location', {}).get('row', '?')}: "
                                f"{issue.get('code', 'UNKNOWN')} - {issue.get('message', 'Unknown error')}"
                            )
                        
                        validation_result["valid"] = len(filtered_issues) == 0
                        
                        # Log with different counts for filtered vs total
                        if len(filtered_issues) != len(ruff_issues):
                            ignored_count = len(ruff_issues) - len(filtered_issues)
                            self.logger.info(
                                f"Ruff found {len(filtered_issues)} linting issues "
                                f"({ignored_count} Blender import issues ignored)"
                            )
                        else:
                            self.logger.info(f"Ruff found {len(filtered_issues)} linting issues")
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

    async def _heal_code_with_comprehensive_feedback(
        self,
        code: str,
        ruff_issues: list[str],
        execution_context: dict[str, Any],
        state: WorkflowState
    ) -> str | None:
        """Heal code using MCTS-based exploration of healing strategies."""
        if not ruff_issues and not execution_context["execution_errors"]:
            return code  # Nothing to heal

        self.logger.info(f"Starting MCTS-based healing with {len(ruff_issues)} ruff issues and {len(execution_context['execution_errors'])} execution errors")

        # First, try to fix any incomplete code blocks
        working_code = code
        try:
            all_errors = ruff_issues + execution_context["execution_errors"]
            working_code = await self._fix_incomplete_code_blocks(working_code, all_errors)
        except Exception as e:
            self.logger.debug(f"Incomplete code block fixing failed: {e}")

        # Use MCTS to explore different healing strategies
        try:
            healed_code = await self._mcts_heal_code(working_code, ruff_issues, execution_context, state)
            if healed_code:
                return healed_code
        except Exception as e:
            self.logger.warning(f"MCTS healing failed: {e}, falling back to traditional healing")

        # Fallback to traditional comprehensive healing
        try:
            healing_prompt = self._create_comprehensive_healing_prompt(
                code, ruff_issues, execution_context, state
            )

            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": self._get_comprehensive_healing_system_prompt()
                },
                {
                    "role": "user",
                    "content": healing_prompt
                }
            ]

            healed_raw_code = await self.make_openai_request(messages)
            healed_code = self._clean_generated_code(healed_raw_code)

            if self._validate_healed_code(healed_code, code, execution_context):
                self.logger.info("Traditional comprehensive healing successful")
                return healed_code
            else:
                self.logger.warning("Healed code failed validation")
                return None

        except Exception:
            self.logger.exception("Traditional comprehensive healing failed")
            return None

    def _create_comprehensive_healing_prompt(
        self,
        code: str,
        ruff_issues: list[str],
        execution_context: dict[str, Any],
        state: WorkflowState
    ) -> str:
        """Create a comprehensive healing prompt using both static analysis and execution traces."""
        prompt_parts = [
            "Fix the following Blender Python code using both static analysis feedback and execution trace learning:",
            "",
            "ORIGINAL CODE:",
            "```python",
            code,
            "```",
            ""
        ]

        if ruff_issues:
            prompt_parts.extend([
                "STATIC ANALYSIS ISSUES (RUFF):",
                *[f"- {issue}" for issue in ruff_issues],
                ""
            ])

        if execution_context["execution_errors"]:
            prompt_parts.extend([
                "EXECUTION TRACE ERRORS TO LEARN FROM:",
                *[f"- {error}" for error in execution_context["execution_errors"]],
                ""
            ])

        if execution_context.get("error_patterns"):
            prompt_parts.extend([
                "IDENTIFIED ERROR PATTERNS:",
                *[f"- {pattern}: {len(errors)} occurrences"
                  for pattern, errors in execution_context["error_patterns"].items()],
                ""
            ])

        prompt_parts.extend([
            "COMPREHENSIVE HEALING REQUIREMENTS:",
            "1. Fix ALL ruff linting issues",
            "2. Address execution error patterns from previous attempts",
            "3. Add robust error handling for identified failure points",
            "4. Maintain original functionality and logic",
            "5. Use defensive programming practices",
            "6. Add validation checks before critical Blender operations",
            "7. Ensure proper context management (active object, selection)",
            "8. Return ONLY the corrected Python code",
            "",
            f"CONTEXT: Blender Python code for: {state.prompt if state else '3D asset generation'}",
            f"ATTEMPT: #{execution_context.get('previous_attempts', 0) + 1}",
            ""
        ])

        return "\n".join(prompt_parts)

    def _get_comprehensive_healing_system_prompt(self) -> str:
        """Get system prompt for comprehensive code healing."""
        return (
            "You are an expert Blender Python developer specializing in robust code healing. "
            "Your expertise includes:\n"
            "- Fixing static analysis issues (ruff, pylint)\n"
            "- Learning from execution trace errors to prevent repeated failures\n"
            "- Implementing defensive programming patterns for Blender API\n"
            "- Adding proper error handling and validation\n"
            "- Context management and object lifecycle handling\n\n"

            "CRITICAL HEALING PRINCIPLES:\n"
            "1. Every fix must address both the immediate issue and prevent future occurrences\n"
            "2. Add try-except blocks around all Blender API operations\n"
            "3. Validate object existence and context state before operations\n"
            "4. Use proper object selection and activation patterns\n"
            "5. Handle edge cases that caused previous execution failures\n"
            "6. Maintain code readability while adding robustness\n\n"

            "Return only clean, executable Python code with no explanations."
        )

    def _validate_healed_code(
        self,
        healed_code: str,
        original_code: str,
        execution_context: dict[str, Any]
    ) -> bool:
        """Enhanced validation for healed code using execution context."""
        # Basic length check
        if len(healed_code.strip()) < len(original_code.strip()) * 0.5:
            self.logger.warning("Healed code is significantly shorter, rejecting")
            return False

        # Essential imports check
        if "import bpy" not in healed_code:
            self.logger.warning("Healed code missing bpy import, rejecting")
            return False

        # Check for defensive programming additions based on error patterns
        error_patterns = execution_context.get("error_patterns", {})

        if error_patterns.get("context_errors") and "bpy.context.view_layer.objects.active" not in healed_code:
            self.logger.info("Healed code should include active object management")

        if error_patterns.get("blender_api_errors") and healed_code.count("try:") == 0:
            self.logger.warning("Healed code lacks error handling for API errors")
            return False

        # Check for removal of problematic patterns
        if "import os" in healed_code:
            self.logger.warning("Healed code still contains unsafe imports")
            return False

        return True

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

    # MCTS-based code healing methods

    async def _mcts_heal_code(
        self,
        code: str,
        ruff_issues: list[str],
        execution_context: dict[str, Any],
        state: WorkflowState
    ) -> str | None:
        """Use MCTS to explore different healing strategies and select the best one."""
        import math
        from dataclasses import dataclass, field

        # Initialize MCTS for healing
        root = HealingNode(
            strategy="original",
            healed_code=code,
            remaining_issues=ruff_issues
        )

        best_healing = None
        best_score = -1.0
        max_iterations = 20  # Focused search for healing

        self.logger.info("Starting MCTS healing exploration", max_iterations=max_iterations)

        for iteration in range(max_iterations):
            # Selection: find leaf node using UCB1
            selected_node = self._select_healing_node(root)

            # Expansion: create new healing strategies
            expanded_node = await self._expand_healing_node(selected_node, code, ruff_issues, execution_context, state)

            # Simulation: evaluate the healing strategy
            reward = await self._simulate_healing(expanded_node, code, ruff_issues, execution_context)

            # Backpropagation: update statistics
            self._backpropagate_healing(expanded_node, reward)

            # Track best solution
            if expanded_node.healed_code and reward > best_score:
                best_score = reward
                best_healing = expanded_node.healed_code

                # Early termination for excellent solutions
                if reward > 0.9 and len(expanded_node.remaining_issues) == 0:
                    self.logger.info(f"Early termination: excellent healing found (reward: {reward:.3f})")
                    break

        if best_healing:
            self.logger.info(f"MCTS healing completed with best reward: {best_score:.3f}")
            return best_healing
        else:
            self.logger.warning("MCTS healing found no improvement")
            return None

    def _select_healing_node(self, root: "HealingNode") -> "HealingNode":
        """Select a leaf node using UCB1 for healing exploration."""
        current = root

        while current.children:
            current = max(current.children, key=lambda child: child.ucb1_score)

        return current

    async def _expand_healing_node(
        self,
        node: "HealingNode",
        original_code: str,
        original_issues: list[str],
        execution_context: dict[str, Any],
        state: WorkflowState
    ) -> "HealingNode":
        """Expand a healing node with different strategies."""
        if node.visits < 3 or len(node.children) >= 5:  # Limit expansion: need visits AND limit children
            return node

        # Generate different healing strategies
        strategies = self._generate_healing_strategies(node.remaining_issues, execution_context)

        for strategy in strategies[:2]:  # Limit to 2 new children per expansion
            try:
                healed_code = await self._apply_healing_strategy(
                    node.healed_code, strategy, original_issues, execution_context, state
                )

                if healed_code and healed_code != node.healed_code:
                    # Validate and check remaining issues
                    remaining_issues = await self._check_remaining_issues(healed_code)

                    child = node.__class__(
                        parent=node,
                        strategy=strategy["name"],
                        healed_code=healed_code,
                        remaining_issues=remaining_issues,
                        is_terminal=len(remaining_issues) == 0
                    )
                    node.children.append(child)

            except Exception as e:
                self.logger.debug(f"Healing strategy {strategy.get('name')} failed: {e}")

        # Return a random child for simulation
        if node.children:
            return secrets.choice(node.children)
        return node

    def _generate_healing_strategies(self, issues: list[str], execution_context: dict[str, Any]) -> list[dict]:
        """Generate different healing strategies based on the types of issues."""
        strategies = []

        # Filter out Blender import issues first
        filtered_issues = []
        for issue in issues:
            # Extract issue code and message from the formatted string
            # Format: "Line X: CODE - message"
            parts = issue.split(': ', 1)
            if len(parts) >= 2:
                # Now split the second part on ' - ' to get CODE and message
                code_and_message = parts[1].split(' - ', 1)
                if len(code_and_message) >= 2:
                    issue_code = code_and_message[0]  # This is the CODE part
                    issue_message = code_and_message[1].lower()  # This is the message part
                else:
                    issue_code = ""
                    issue_message = parts[1].lower()
            else:
                issue_code = ""
                issue_message = issue.lower()
            
            # Skip if this is a Blender import issue
            if not self._is_blender_import_issue(issue_code, issue_message):
                filtered_issues.append(issue)
        
        # Categorize the filtered issues (prioritize functional over style)
        import_issues = [i for i in filtered_issues if "F401" in i or "unused" in i.lower()]
        undefined_issues = [i for i in filtered_issues if "F821" in i or "undefined" in i.lower()]
        complexity_issues = [i for i in filtered_issues if "C901" in i or "complex" in i.lower()]
        # De-prioritize pure style issues (focus on functional problems first)  
        style_issues = [i for i in filtered_issues if any(code in i for code in ["E501", "E302", "W292"]) 
                       and not any(code in i for code in ["F821", "F401", "C901"])]

        # Strategy 1: Focus on imports and unused variables
        if import_issues:
            strategies.append({
                "name": "import_cleanup",
                "focus": "imports_and_unused",
                "priority": import_issues,
                "approach": "conservative"
            })

        # Strategy 2: Focus on undefined variables (critical fixes)
        if undefined_issues:
            strategies.append({
                "name": "variable_definition",
                "focus": "undefined_variables",
                "priority": undefined_issues,
                "approach": "define_missing"
            })

        # Strategy 3: Focus on style issues
        if style_issues:
            strategies.append({
                "name": "style_fixes",
                "focus": "formatting_and_style",
                "priority": style_issues,
                "approach": "formatting"
            })

        # Strategy 4: Comprehensive approach (filtered issues only)
        strategies.append({
            "name": "comprehensive",
            "focus": "all_issues",
            "priority": filtered_issues,
            "approach": "holistic"
        })

        # Strategy 5: Execution-context-aware healing
        if execution_context.get("execution_errors"):
            strategies.append({
                "name": "execution_aware",
                "focus": "execution_context", 
                "priority": filtered_issues,
                "approach": "trace_based"
            })

        # Strategy 6: Critic-guided healing (only for quality/structural issues)
        quality_issues = [i for i in filtered_issues if any(code in i for code in ["C901", "PLR", "N8", "D1"])]
        if quality_issues or len(filtered_issues) > 5:  # Use for complex cases or quality issues
            strategies.append({
                "name": "critic_guided",
                "focus": "code_quality_analysis", 
                "priority": quality_issues or filtered_issues[:3],  # Focus on top issues
                "approach": "quality_driven"
            })

        # Strategy 7: Documentation-enhanced healing (only for API-related issues)
        api_issues = [i for i in filtered_issues if any(keyword in i.lower() 
                     for keyword in ["attributeerror", "bpy.", "bmesh", "mathutils"])]
        if api_issues:
            strategies.append({
                "name": "documentation_enhanced",
                "focus": "api_usage_patterns",
                "priority": api_issues,
                "approach": "documentation_driven"
            })

        # Strategy 8: Web-search assisted healing (for complex runtime errors only)
        complex_issues = [i for i in filtered_issues if any(keyword in i.lower() 
                         for keyword in ["attributeerror", "modulenotfounderror", "typeerror", "valueerror"])]
        if complex_issues:
            strategies.append({
                "name": "web_assisted",
                "focus": "complex_error_resolution",
                "priority": complex_issues,
                "approach": "research_driven"
            })

        return strategies

    async def _apply_healing_strategy(
        self,
        code: str,
        strategy: dict,
        original_issues: list[str],
        execution_context: dict[str, Any],
        state: WorkflowState
    ) -> str | None:
        """Apply a specific healing strategy to the code."""
        strategy_name = strategy["name"]
        focus = strategy["focus"]

        # Create strategy-specific healing prompt
        if strategy_name == "import_cleanup":
            prompt = self._create_import_cleanup_prompt(code, strategy["priority"])
        elif strategy_name == "variable_definition":
            prompt = self._create_variable_definition_prompt(code, strategy["priority"], execution_context)
        elif strategy_name == "style_fixes":
            prompt = self._create_style_fixes_prompt(code, strategy["priority"])
        elif strategy_name == "execution_aware":
            prompt = self._create_execution_aware_prompt(code, original_issues, execution_context, state)
        elif strategy_name == "critic_guided":
            try:
                prompt = await self._create_critic_guided_prompt(code, strategy["priority"], execution_context, state)
            except Exception as e:
                self.logger.debug(f"Critic-guided prompt creation failed: {e}, using fallback")
                prompt = self._create_comprehensive_healing_prompt(code, original_issues, execution_context, state)
        elif strategy_name == "documentation_enhanced":
            try:
                prompt = await self._create_documentation_enhanced_prompt(code, strategy["priority"], execution_context, state)
            except Exception as e:
                self.logger.debug(f"Documentation-enhanced prompt creation failed: {e}, using fallback")
                prompt = self._create_comprehensive_healing_prompt(code, original_issues, execution_context, state)
        elif strategy_name == "web_assisted":
            try:
                prompt = await self._create_web_assisted_prompt(code, strategy["priority"], execution_context, state)
            except Exception as e:
                self.logger.debug(f"Web-assisted prompt creation failed: {e}, using fallback")
                prompt = self._create_comprehensive_healing_prompt(code, original_issues, execution_context, state)
        else:  # comprehensive
            prompt = self._create_comprehensive_healing_prompt(code, original_issues, execution_context, state)

        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are a code healing specialist focused on {focus}. Apply {strategy['approach']} healing approach."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = await self.make_openai_request(messages, temperature=0.2)
            healed_code = self._clean_generated_code(response)

            return healed_code if healed_code and healed_code != code else None

        except Exception as e:
            self.logger.debug(f"Strategy {strategy_name} failed: {e}")
            return None

    async def _check_remaining_issues(self, code: str) -> list[str]:
        """Check what issues remain after healing attempt."""
        try:
            # Use ruff to check remaining issues
            _, validation_result = await self._apply_ruff_formatting_and_validation(code)
            return validation_result.get("issues", [])
        except Exception:
            return []

    async def _simulate_healing(
        self,
        node: "HealingNode",
        original_code: str,
        original_issues: list[str],
        execution_context: dict[str, Any]
    ) -> float:
        """Simulate and evaluate a healing strategy."""
        if not node.healed_code:
            return 0.0

        # Calculate improvement metrics
        original_issue_count = len(original_issues)
        remaining_issue_count = len(node.remaining_issues)

        # Issue reduction score (0-1)
        if original_issue_count == 0:
            issue_score = 1.0
        else:
            issue_score = max(0.0, (original_issue_count - remaining_issue_count) / original_issue_count)

        # Code quality score based on syntactic validity
        try:
            import ast
            ast.parse(node.healed_code)
            syntax_score = 1.0
        except SyntaxError:
            syntax_score = 0.0

        # Enhanced execution context improvement
        execution_score = 0.5  # Default neutral
        if execution_context.get("execution_errors"):
            # Check if healing addressed execution context issues
            error_patterns = execution_context.get("error_patterns", {})
            if any(pattern_errors for pattern_errors in error_patterns.values()):
                # Reward strategies that address execution patterns
                if "try:" in node.healed_code or "except" in node.healed_code:
                    execution_score = 0.8
                if "bpy.context" in node.healed_code and "active" in node.healed_code:
                    execution_score = max(execution_score, 0.7)

        # Strategy-specific scoring bonuses
        strategy_bonus = self._calculate_strategy_bonus(node.strategy, node.healed_code, original_code)
        
        # Code quality improvements beyond basic syntax
        quality_score = self._assess_code_quality_improvements(node.healed_code, original_code)
        
        # API usage correctness (Blender-specific)
        api_score = self._assess_blender_api_usage(node.healed_code)

        # Combine scores with enhanced weighting
        final_score = (
            0.40 * issue_score +        # 40% weight on issue reduction
            0.20 * syntax_score +       # 20% weight on syntax validity  
            0.15 * execution_score +    # 15% weight on execution awareness
            0.10 * strategy_bonus +     # 10% weight on strategy-specific improvements
            0.10 * quality_score +      # 10% weight on code quality
            0.05 * api_score           # 5% weight on Blender API correctness
        )

        return min(1.0, max(0.0, final_score))

    def _backpropagate_healing(self, node: "HealingNode", reward: float) -> None:
        """Update statistics up the healing tree."""
        current = node
        while current:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _calculate_strategy_bonus(self, strategy: str, healed_code: str, original_code: str) -> float:
        """Calculate bonus score based on strategy-specific improvements."""
        bonus = 0.5  # Default neutral
        
        if strategy == "critic_guided":
            # Reward improvements in code structure and readability
            if len(healed_code.split('\n')) > len(original_code.split('\n')) * 0.9:
                bonus = 0.8  # Good structural preservation
            if "def " in healed_code and healed_code.count("def ") >= original_code.count("def "):
                bonus = min(1.0, bonus + 0.2)  # Additive bonus for function preservation
                
        elif strategy == "documentation_enhanced":
            # Reward correct API usage patterns
            if "bpy.ops." in healed_code or "bpy.data." in healed_code:
                bonus = 0.8  # Uses documented API patterns
            if "bmesh" in healed_code and "bmesh.new()" in healed_code:
                bonus = max(bonus, 0.7)  # Proper bmesh usage
                
        elif strategy == "web_assisted":
            # Reward defensive programming and error handling
            error_handling_count = healed_code.count("try:") + healed_code.count("if ") + healed_code.count("hasattr")
            original_handling = original_code.count("try:") + original_code.count("if ") + original_code.count("hasattr")
            if error_handling_count > original_handling:
                bonus = 0.9  # Added defensive patterns
                
        elif strategy == "execution_aware":
            # Reward context management improvements
            if "bpy.context.view_layer" in healed_code:
                bonus = 0.8  # Better context management
            if "select_set" in healed_code or "active" in healed_code:
                bonus = max(bonus, 0.7)  # Object selection patterns
                
        return bonus

    def _assess_code_quality_improvements(self, healed_code: str, original_code: str) -> float:
        """Assess general code quality improvements."""
        score = 0.5  # Default neutral
        
        # Check for improved structure
        healed_lines = healed_code.split('\n')
        original_lines = original_code.split('\n')
        
        # Reward maintained or improved line count (not just deletion)
        line_ratio = len(healed_lines) / max(len(original_lines), 1)
        if 0.8 <= line_ratio <= 1.5:  # Reasonable code expansion/contraction
            score = 0.7
            
        # Reward better variable naming
        if any(name in healed_code for name in ['obj', 'mesh', 'material', 'light', 'camera']):
            score = max(score, 0.6)
            
        # Reward proper imports organization
        healed_imports = [line.strip() for line in healed_lines if line.strip().startswith('import')]
        original_imports = [line.strip() for line in original_lines if line.strip().startswith('import')]
        if len(healed_imports) >= len(original_imports):  # Maintained necessary imports
            score = min(1.0, score + 0.3)  # Additive bonus instead of max
            
        # Penalty for overly simplified code
        if len(healed_code) < len(original_code) * 0.5:
            score = 0.3  # Likely over-simplified
            
        return score

    def _assess_blender_api_usage(self, code: str) -> float:
        """Assess correctness of Blender Python API usage patterns."""
        score = 0.5  # Default neutral
        
        # Check for proper Blender imports
        if "import bpy" in code:
            score = 0.7
            
        # Check for proper object handling
        if "bpy.context" in code and ("active" in code or "selected" in code):
            score = max(score, 0.8)
            
        # Check for proper mesh operations
        if "bmesh" in code and ("bmesh.new()" in code or "bmesh.from_mesh" in code):
            score = max(score, 0.8)
            
        # Check for proper material setup
        if "bpy.data.materials.new" in code and "use_nodes" in code:
            score = max(score, 0.7)
            
        # Check for proper cleanup patterns
        if "bm.free()" in code or "bm.clear()" in code:
            score = max(score, 0.9)  # Excellent memory management
            
        # Penalty for problematic patterns
        if "os." in code or "sys." in code:
            score = min(score, 0.3)  # Unsafe system access
            
        if ".delete(" in code and "bpy.ops" not in code:
            score = min(score, 0.4)  # Potentially unsafe deletion
            
        return score

    # Strategy-specific prompt creation methods

    def _create_import_cleanup_prompt(self, code: str, import_issues: list[str]) -> str:
        """Create a focused prompt for import cleanup."""
        return f"""Fix ONLY the import and unused variable issues in this Blender Python code:

ISSUES TO FIX:
{chr(10).join(f'- {issue}' for issue in import_issues)}

CODE TO FIX:
```python
{code}
```

INSTRUCTIONS:
1. Remove or use unused imports
2. Remove unused variables
3. Keep all functional code intact
4. Maintain Blender API compatibility

Return only the corrected code."""

    def _create_variable_definition_prompt(self, code: str, undefined_issues: list[str], execution_context: dict[str, Any]) -> str:
        """Create a focused prompt for fixing undefined variables."""
        return f"""Fix ONLY the undefined variable issues in this Blender Python code:

UNDEFINED VARIABLE ISSUES:
{chr(10).join(f'- {issue}' for issue in undefined_issues)}

CODE TO FIX:
```python
{code}
```

CONTEXT FROM EXECUTION:
{execution_context.get('execution_errors', [])}

INSTRUCTIONS:
1. Define missing variables with appropriate default values
2. Use common Blender patterns for undefined variables
3. Maintain code logic and functionality  
4. Consider execution context when defining variables

COMMON BLENDER VARIABLE PATTERNS:
- x, y, z = 0.0, 0.0, 0.0  # For coordinates
- energy = 1.0  # For light energy
- size = 1.0   # For object scale

Return only the corrected code."""

    def _create_style_fixes_prompt(self, code: str, style_issues: list[str]) -> str:
        """Create a focused prompt for style fixes."""
        return f"""Fix ONLY the style and formatting issues in this Blender Python code:

STYLE ISSUES TO FIX:
{chr(10).join(f'- {issue}' for issue in style_issues)}

CODE TO FIX:
```python
{code}
```

INSTRUCTIONS:
1. Fix docstring formatting (end with period, question mark, or exclamation)
2. Ensure proper line lengths
3. Add proper blank lines between functions
4. Add final newline if missing
5. Fix comment spacing
6. Do NOT change functionality

Return only the corrected code."""

    def _create_execution_aware_prompt(self, code: str, issues: list[str], execution_context: dict[str, Any], state: WorkflowState) -> str:
        """Create a prompt that considers execution context for healing."""
        error_learning = self._create_error_learning_context(execution_context)

        return f"""Fix this Blender Python code considering both static issues and execution context:

STATIC ISSUES:
{chr(10).join(f'- {issue}' for issue in issues[:10])}  # Limit for brevity

EXECUTION CONTEXT & LEARNING:
{error_learning}

CODE TO FIX:
```python
{code}
```

HEALING STRATEGY:
1. Fix all static analysis issues
2. Add defensive programming based on execution context
3. Ensure Blender API compatibility
4. Add error handling where execution failures occurred
5. Maintain original functionality and intent
6. Test logic flow and edge cases mentally
7. Ensure code is self-contained and executable

Return only the corrected, executable Python code."""

    # ENHANCED HEALING STRATEGIES WITH AGENT INTEGRATION

    async def _create_critic_guided_prompt(
        self, 
        code: str, 
        issues: list[str], 
        execution_context: dict[str, Any], 
        state: WorkflowState
    ) -> str:
        """Create a healing prompt guided by critic agent analysis."""
        try:
            # Use the critic agent to analyze code quality
            from ..agents.factory import get_agent_factory
            
            # Create a temporary state for critic analysis
            temp_state = WorkflowState(
                prompt="Code quality analysis for healing",
                generated_code=code,
                execution_result=state.execution_result if state else None
            )
            
            critic_agent = get_agent_factory().create_critic()
            critic_response = await critic_agent.process(temp_state)
            
            critic_insights = ""
            if critic_response.success and critic_response.data:
                analysis = critic_response.data
                critic_insights = f"""
CRITIC AGENT ANALYSIS:
- Overall Quality: {analysis.get('overall_score', 'Unknown')}/10
- Key Issues: {'; '.join(analysis.get('improvement_suggestions', [])[:3])}
- Quality Dimensions: {analysis.get('dimension_scores', {})}
- Critical Problems: {analysis.get('critical_issues', [])}
"""
            
        except Exception as e:
            self.logger.debug(f"Critic analysis failed: {e}")
            critic_insights = "Critic analysis unavailable - proceeding with static analysis only."

        return f"""Fix the following Blender Python code using critic-guided quality analysis:

ORIGINAL CODE:
```python
{code}
```

STATIC ANALYSIS ISSUES:
{chr(10).join(f"- {issue}" for issue in issues)}

{critic_insights}

CRITIC-GUIDED HEALING REQUIREMENTS:
1. Address all quality dimensions identified by the critic
2. Fix critical issues that affect visual output quality
3. Improve code structure based on critic recommendations
4. Maintain functional correctness while enhancing quality
5. Focus on areas that impact 3D asset generation quality

Return ONLY the corrected Python code with no explanations."""

    async def _create_documentation_enhanced_prompt(
        self, 
        code: str, 
        issues: list[str], 
        execution_context: dict[str, Any], 
        state: WorkflowState
    ) -> str:
        """Create a healing prompt enhanced with Context7 documentation search."""
        try:
            # Extract Blender API calls from the code to search for documentation
            api_calls = self._extract_blender_api_calls(code)
            
            documentation_context = ""
            if api_calls:
                # Search for relevant Blender Python API documentation
                docs = await self._search_context7_documentation(api_calls, issues)
                if docs:
                    documentation_context = f"""
RELEVANT BLENDER PYTHON API DOCUMENTATION:
{docs}
"""
            
        except Exception as e:
            self.logger.debug(f"Documentation search failed: {e}")
            documentation_context = "Documentation search unavailable - using fallback patterns."

        return f"""Fix the following Blender Python code using enhanced API documentation:

ORIGINAL CODE:
```python
{code}
```

STATIC ANALYSIS ISSUES:
{chr(10).join(f"- {issue}" for issue in issues)}

{documentation_context}

DOCUMENTATION-ENHANCED HEALING REQUIREMENTS:
1. Use correct Blender Python API patterns from documentation
2. Follow best practices for bpy operations
3. Implement proper error handling for API calls
4. Use recommended parameter combinations
5. Apply defensive programming for Blender context management

Return ONLY the corrected Python code with no explanations."""

    async def _create_web_assisted_prompt(
        self, 
        code: str, 
        issues: list[str], 
        execution_context: dict[str, Any], 
        state: WorkflowState
    ) -> str:
        """Create a healing prompt enhanced with web search for complex issues."""
        try:
            # Search for solutions to complex errors
            complex_issues = [issue for issue in issues if any(keyword in issue.lower() 
                            for keyword in ["attributeerror", "modulenotfounderror", "typeerror", "valueerror"])]
            
            web_solutions = ""
            if complex_issues[:2]:  # Limit to 2 most critical issues
                solutions = await self._search_web_solutions(complex_issues[:2])
                if solutions:
                    web_solutions = f"""
WEB SEARCH SOLUTIONS FOR COMPLEX ISSUES:
{solutions}
"""
            
        except Exception as e:
            self.logger.debug(f"Web search failed: {e}")
            web_solutions = "Web search unavailable - using pattern-based healing."

        return f"""Fix the following Blender Python code using web-researched solutions:

ORIGINAL CODE:
```python
{code}
```

COMPLEX ISSUES TO RESOLVE:
{chr(10).join(f"- {issue}" for issue in issues)}

{web_solutions}

WEB-ASSISTED HEALING REQUIREMENTS:
1. Apply proven solutions from the Blender community
2. Use established error handling patterns
3. Implement workarounds for known Blender Python API limitations
4. Follow community best practices for complex operations
5. Add defensive checks based on common failure patterns

Return ONLY the corrected Python code with no explanations."""

    def _extract_blender_api_calls(self, code: str) -> list[str]:
        """Extract Blender API calls from code for documentation search."""
        import re
        
        # Find bpy.* patterns
        bpy_calls = re.findall(r'bpy\.[a-zA-Z_][a-zA-Z0-9_.]*', code)
        
        # Find bmesh patterns
        bmesh_calls = re.findall(r'bmesh\.[a-zA-Z_][a-zA-Z0-9_.]*', code)
        
        # Find mathutils patterns
        mathutils_calls = re.findall(r'mathutils\.[a-zA-Z_][a-zA-Z0-9_.]*', code)
        
        all_calls = bpy_calls + bmesh_calls + mathutils_calls
        return list(set(all_calls))  # Remove duplicates

    async def _search_context7_documentation(self, api_calls: list[str], issues: list[str]) -> str:
        """Search Context7 for relevant Blender Python API documentation."""
        try:
            # Import MCP tools if available
            from ..mcp_integration import search_blender_docs
            
            # Create search queries based on API calls and issues
            search_queries = []
            
            for call in api_calls[:5]:  # Limit to 5 most important calls
                if 'bpy.ops.mesh' in call:
                    search_queries.append("mesh operations bmesh")
                elif 'bpy.data.materials' in call:
                    search_queries.append("materials shader nodes")
                elif 'bpy.data.lights' in call:
                    search_queries.append("lighting area lights")
                elif 'bpy.context' in call:
                    search_queries.append("context management")
            
            # Add issue-specific queries
            for issue in issues[:3]:
                if "AttributeError" in issue:
                    search_queries.append("python api attributeerror")
                elif "undefined" in issue.lower():
                    search_queries.append("variable scope blender")
            
            docs_results = []
            for query in search_queries[:3]:  # Limit to 3 searches
                try:
                    docs = await search_blender_docs(query)
                    if docs:
                        docs_results.append(f"Query '{query}':\n{docs}\n")
                except Exception as e:
                    self.logger.debug(f"Documentation search for '{query}' failed: {e}")
            
            return '\n'.join(docs_results) if docs_results else ""
            
        except Exception as e:
            self.logger.debug(f"Context7 documentation search failed: {e}")
            return ""

    async def _search_web_solutions(self, complex_issues: list[str]) -> str:
        """Search the web for solutions to complex coding issues."""
        try:
            # Import web search if available
            from ..mcp_integration import search_web_solutions
            
            solutions = []
            for issue in complex_issues:
                # Extract the core error from the issue string
                error_pattern = self._extract_error_pattern(issue)
                if error_pattern:
                    query = f"blender python {error_pattern} solution"
                    try:
                        result = await search_web_solutions(query)
                        if result:
                            solutions.append(f"Issue: {issue}\nSolution: {result}\n")
                    except Exception as e:
                        self.logger.debug(f"Web search for '{query}' failed: {e}")
            
            return '\n'.join(solutions) if solutions else ""
            
        except Exception as e:
            self.logger.debug(f"Web search failed: {e}")
            return ""

    def _extract_error_pattern(self, issue: str) -> str:
        """Extract searchable error pattern from issue string."""
        import re
        
        # Extract error types like AttributeError, TypeError, etc.
        error_match = re.search(r'(AttributeError|TypeError|ValueError|NameError|ModuleNotFoundError)', issue)
        if error_match:
            return error_match.group(1)
        
        # Extract specific API-related errors
        if 'bpy.' in issue:
            bpy_match = re.search(r'bpy\.[a-zA-Z_][a-zA-Z0-9_.]*', issue)
            if bpy_match:
                return f"bpy {bpy_match.group(0)} error"
        
        return ""

    async def _fix_incomplete_code_blocks(self, code: str, validation_errors: list[str]) -> str:
        """Fix incomplete code blocks using MCP integration."""
        try:
            from ..mcp_integration import fix_incomplete_code_blocks
            
            # Check for incomplete block errors in validation
            incomplete_errors = [
                error for error in validation_errors 
                if "incomplete" in error.lower() or "pass after" in error.lower()
            ]
            
            if incomplete_errors:
                self.logger.info(f"Fixing {len(incomplete_errors)} incomplete code blocks")
                fixed_code = await fix_incomplete_code_blocks(code, incomplete_errors)
                return fixed_code
            
            return code
            
        except Exception as e:
            self.logger.debug(f"Failed to fix incomplete code blocks: {e}")
            return code

    # EXECUTION VALIDATION SYSTEM - Inspired by Blender Agent Pattern

    async def _validate_code_execution(self, code: str, state: WorkflowState) -> dict[str, Any]:
        """Validate code by attempting execution in a safe environment."""
        try:
            self.logger.info("Starting code execution validation")
            
            # Create a safe execution environment
            execution_result = await self._execute_code_safely(code, state)
            
            validation_result = {
                "success": execution_result.get("success", False),
                "execution_time": execution_result.get("execution_time", 0.0),
                "output": execution_result.get("output", ""),
                "errors": execution_result.get("errors", []),
                "warnings": execution_result.get("warnings", []),
                "retry_needed": False
            }
            
            # Determine if retry is needed based on error types
            if not validation_result["success"] and validation_result["errors"]:
                retry_needed = self._should_retry_execution(validation_result["errors"])
                validation_result["retry_needed"] = retry_needed
                
            # Log execution validation result
            if validation_result["success"]:
                self.logger.info("Code execution validation successful")
                # Save successful pattern for learning
                await self._save_successful_pattern(code, state, validation_result)
            else:
                self.logger.warning(
                    "Code execution validation failed",
                    error_count=len(validation_result["errors"]),
                    retry_needed=validation_result["retry_needed"]
                )
                
            return validation_result
            
        except Exception as e:
            self.logger.exception("Execution validation failed with exception", error=str(e))
            return {
                "success": False,
                "execution_time": 0.0,
                "output": "",
                "errors": [f"Validation exception: {str(e)}"],
                "warnings": [],
                "retry_needed": True
            }

    async def _execute_code_safely(self, code: str, state: WorkflowState) -> dict[str, Any]:
        """Execute code in a safe, isolated environment with timeout."""
        import tempfile
        import subprocess
        import asyncio
        import os
        
        try:
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # Add safety wrapper around the code
                safe_code = self._wrap_code_for_safe_execution(code)
                temp_file.write(safe_code)
                temp_file.flush()
                temp_path = temp_file.name
            
            # Execute with timeout and capture output
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Run Python with limited permissions and timeout
                process = await asyncio.create_subprocess_exec(
                    'python', temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempfile.gettempdir()  # Run in temp directory
                )
                
                # Wait with timeout (30 seconds max)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=30.0
                )
                
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Parse results
                return {
                    "success": process.returncode == 0,
                    "execution_time": execution_time,
                    "output": stdout.decode('utf-8') if stdout else "",
                    "errors": [stderr.decode('utf-8')] if stderr and stderr.strip() else [],
                    "warnings": [],
                    "return_code": process.returncode
                }
                
            except asyncio.TimeoutError:
                # Kill the process and return timeout error
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                return {
                    "success": False,
                    "execution_time": 30.0,
                    "output": "",
                    "errors": ["Execution timeout after 30 seconds"],
                    "warnings": [],
                    "return_code": -1
                }
                
        except Exception as e:
            return {
                "success": False,
                "execution_time": 0.0,
                "output": "",
                "errors": [f"Execution setup failed: {str(e)}"],
                "warnings": [],
                "return_code": -2
            }
        finally:
            # Clean up temporary file
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass

    def _wrap_code_for_safe_execution(self, code: str) -> str:
        """Wrap code with safety measures for execution validation."""
        safety_wrapper = f'''#!/usr/bin/env python3
"""
Code Execution Validation - Safe Wrapper
Generated code wrapped for validation purposes.
"""

import sys
import traceback
import warnings
from io import StringIO

# Capture warnings
warnings.simplefilter("always")
warning_capture = StringIO()

def capture_warning(message, category, filename, lineno, file=None, line=None):
    warning_capture.write(f"Warning: {{category.__name__}}: {{message}}\\n")

warnings.showwarning = capture_warning

print("=== EXECUTION VALIDATION START ===")

try:
    # Execute the user code
{self._indent_code(code, 4)}
    
    print("=== EXECUTION VALIDATION SUCCESS ===")
    
except Exception as e:
    print(f"=== EXECUTION VALIDATION FAILED ===")
    print(f"Error: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
finally:
    # Print captured warnings
    warnings_output = warning_capture.getvalue()
    if warnings_output.strip():
        print("=== WARNINGS ===")
        print(warnings_output)
'''
        return safety_wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)

    def _is_blender_import_issue(self, issue_code: str, issue_message: str) -> bool:
        """Check if a Ruff issue is related to Blender imports that should be ignored."""
        
        # Blender-specific modules that are commonly imported but may appear unused
        blender_modules = [
            'bpy', 'bmesh', 'mathutils', 'gpu', 'blf', 'bgl',
            'aud', 'freestyle', 'bpy_extras', 'addon_utils'
        ]
        
        # Issue codes to ignore for Blender modules
        blender_ignore_codes = [
            'F401',  # Module imported but unused
            'I001',  # Import block is un-sorted (Blender imports often need specific order)
        ]
        
        # Check if this is a Blender import issue we should ignore
        if issue_code in blender_ignore_codes:
            # First, check if the message mentions any Blender modules (with or without backticks)
            for module in blender_modules:
                if module in issue_message or f"`{module}`" in issue_message:
                    return True
                    
            # For specific Blender import patterns, be more specific
            specific_blender_patterns = [
                'from mathutils import',
                'from bpy import',
                'import bpy',
                'import bmesh',
                'import mathutils',
                # Handle backtick-quoted patterns (only Blender modules)
                '`bpy`',
                '`bmesh`',
                '`mathutils`',
                '`mathutils.vector`',
                '`mathutils.matrix`',
                '`gpu`',
                '`blf`',
                '`bgl`',
                '`bpy_extras`',
                '`addon_utils`'
            ]
            
            for pattern in specific_blender_patterns:
                if pattern in issue_message:
                    return True
                    
            # For I001 (import sorting), check if it's likely related to Blender imports
            if issue_code == 'I001' and ('import block is un-sorted' in issue_message):
                return True
        
        return False

    def _should_retry_execution(self, errors: list[str]) -> bool:
        """Determine if execution should be retried based on error types."""
        retry_indicators = [
            "NameError",           # Undefined variables
            "SyntaxError",         # Syntax issues
            "IndentationError",    # Indentation problems
            "ImportError",         # Import issues
            "AttributeError",      # Method/attribute errors
            "TypeError",           # Type mismatches
            "ModuleNotFoundError"  # Missing modules
        ]
        
        error_text = ' '.join(errors).lower()
        return any(indicator.lower() in error_text for indicator in retry_indicators)

    async def _heal_with_execution_feedback(
        self, 
        code: str, 
        execution_validation: dict[str, Any], 
        state: WorkflowState
    ) -> str | None:
        """Heal code using execution feedback - core of the feedback loop."""
        try:
            self.logger.info("Applying execution-based healing")
            
            # Create execution-focused healing prompt
            healing_prompt = self._create_execution_healing_prompt(code, execution_validation, state)
            
            messages = [
                {
                    "role": "system", 
                    "content": self._get_execution_healing_system_prompt()
                },
                {
                    "role": "user",
                    "content": healing_prompt
                }
            ]
            
            # Get healed code from LLM
            healed_raw_code = await self.make_openai_request(messages, temperature=0.2)
            healed_code = self._extract_code_from_response(healed_raw_code)
            
            if healed_code and healed_code != code:
                self.logger.info("Execution-based healing applied successfully")
                return healed_code
            else:
                self.logger.warning("Execution-based healing produced no changes")
                return None
                
        except Exception as e:
            self.logger.exception("Execution-based healing failed", error=str(e))
            return None

    def _create_execution_healing_prompt(
        self, 
        code: str, 
        execution_validation: dict[str, Any], 
        state: WorkflowState
    ) -> str:
        """Create a healing prompt based on execution feedback."""
        errors = execution_validation.get("errors", [])
        output = execution_validation.get("output", "")
        
        return f"""EXECUTION VALIDATION FAILED - Fix the following code based on runtime feedback:

ORIGINAL CODE:
```python
{code}
```

EXECUTION ERRORS:
{chr(10).join(f'- {error}' for error in errors)}

EXECUTION OUTPUT:
{output}

TASK CONTEXT:
Original Goal: {state.prompt}
Subtasks: {[task.description for task in (state.subtasks or [])]}

HEALING INSTRUCTIONS:
1. Analyze the execution errors and identify the root causes
2. Fix syntax errors, undefined variables, and import issues  
3. Ensure all variables are properly defined before use
4. Add proper error handling where needed
5. Maintain the original functionality and intent
6. Test logic flow and edge cases mentally
7. Ensure code is self-contained and executable

COMMON FIXES NEEDED:
- Define missing variables with appropriate default values
- Fix syntax errors (missing colons, indentation issues)
- Import required modules at the top
- Handle potential None values and edge cases
- Ensure proper object initialization and cleanup

Return only the corrected, executable Python code."""

    def _get_execution_healing_system_prompt(self) -> str:
        """System prompt for execution-based healing."""
        return """You are an expert Blender Python developer specializing in runtime error analysis and code healing.

Your role is to analyze execution failures and fix code to make it run successfully while preserving the original intent.

EXECUTION HEALING PRINCIPLES:
1. **Root Cause Analysis**: Identify the fundamental cause of execution failures
2. **Defensive Programming**: Add checks and safeguards to prevent runtime errors
3. **Variable Initialization**: Ensure all variables are defined before use
4. **Import Management**: Include all necessary imports and handle missing dependencies
5. **Error Handling**: Add try-catch blocks for risky operations
6. **Logic Validation**: Verify the code logic matches the intended behavior

HEALING STRATEGY:
- Focus on making the code EXECUTABLE first, then optimize
- Preserve the original functionality and intent
- Add minimal necessary changes to fix execution issues
- Ensure code is self-contained and doesn't depend on external state
- Test edge cases and boundary conditions mentally

OUTPUT REQUIREMENTS:
- Return ONLY executable Python code
- No explanations, comments, or markdown formatting
- Code must be syntactically correct and logically sound
- All variables must be defined before use
- Include proper imports and error handling"""

    async def _save_successful_pattern(self, code: str, state: WorkflowState, execution_result: dict):
        """Save successful code patterns for future learning."""
        try:
            pattern = {
                "code_length": len(code),
                "goal": state.prompt,
                "subtasks_count": len(state.subtasks or []),
                "execution_time": execution_result.get("execution_time", 0.0),
                "success_indicators": self._extract_success_indicators(code, execution_result),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Log successful pattern for learning
            self.logger.info(
                "Successful execution pattern captured",
                pattern_hash=hash(code) % 10000,
                execution_time=pattern["execution_time"],
                indicators=len(pattern["success_indicators"])
            )
            
        except Exception as e:
            self.logger.debug(f"Pattern saving failed: {e}")

    def _extract_success_indicators(self, code: str, execution_result: dict) -> list[str]:
        """Extract patterns that indicate successful code generation."""
        indicators = []
        
        # Code structure indicators
        if "def " in code:
            indicators.append("contains_functions")
        if "class " in code:
            indicators.append("contains_classes") 
        if "try:" in code:
            indicators.append("has_error_handling")
        if "import " in code:
            indicators.append("proper_imports")
            
        # Execution indicators
        if execution_result.get("success"):
            indicators.append("execution_success")
        if execution_result.get("execution_time", 0) < 5.0:
            indicators.append("fast_execution")
            
        return indicators
