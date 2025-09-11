"""User Proxy Agent for handling human-in-the-loop interactions and feedback processing.

Based on LL3M ArXiv paper: https://arxiv.org/html/2508.08228v1
This agent facilitates seamless integration of user feedback into the 3D asset generation pipeline.
"""

import asyncio
import json
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from ..utils.types import AgentResponse, AgentType, WorkflowState
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler


class UserProxyAgent(EnhancedBaseAgent):
    """Handles user feedback processing and human-in-the-loop interactions."""

    def __init__(self, config: dict[str, Any], client: Any = None) -> None:  # noqa: ARG002
        """Initialize user proxy agent."""
        super().__init__(config)
        self.feedback_templates = {
            "refinement_prompt": (
                "Based on the visual analysis and current asset quality, "
                "what specific improvements would you like to make? "
                "You can:\n"
                "1. Request specific changes (e.g., 'make it taller', 'change color to blue')\n"
                "2. Accept the current result by typing 'ACCEPT'\n"
                "3. Terminate the process by typing 'TERMINATE'\n\n"
                "Current Quality Scores:\n"
                "{quality_summary}\n\n"
                "Identified Issues:\n"
                "{issues_summary}\n\n"
                "Your feedback: "
            ),
            "initial_review_prompt": (
                "Please review the generated 3D asset. "
                "Does it match your expectations for: '{original_prompt}'?\n\n"
                "Quality Assessment:\n"
                "{quality_summary}\n\n"
                "Your feedback (or type 'ACCEPT' to proceed, 'TERMINATE' to end): "
            ),
        }

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.USER_PROXY

    @property
    def name(self) -> str:
        """Return agent name."""
        return "User Proxy Agent"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process user feedback and determine next workflow actions."""
        start_time = asyncio.get_event_loop().time()

        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data={},
                    message="Invalid input: execution result required for user review",
                    execution_time=0.0,
                )

            self.logger.info("Starting user feedback processing")

            # Process current user feedback if available
            feedback_analysis = await self._analyze_user_feedback(state)

            # Determine if additional user input is needed
            needs_user_input = self._should_prompt_user(feedback_analysis)

            # Generate user prompt if needed
            user_prompt = ""
            if needs_user_input:
                user_prompt = await self._generate_user_prompt(state)

            # Update workflow state based on feedback
            workflow_action = self._determine_workflow_action(feedback_analysis)

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "User feedback processing completed",
                action=workflow_action["action"],
                needs_input=needs_user_input,
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data={
                    "feedback_analysis": feedback_analysis,
                    "user_prompt": user_prompt,
                    "workflow_action": workflow_action,
                    "needs_user_input": needs_user_input,
                },
                message=self._generate_response_message(workflow_action, needs_user_input),
                execution_time=execution_time,
                metadata={
                    "action": workflow_action["action"],
                    "priority": workflow_action.get("priority", "medium"),
                },
            )

        except Exception as user_feedback_error:
            AgentErrorHandler.log_error(
                self.logger,
                user_feedback_error,
                "user feedback processing",
                has_user_feedback=bool(state.user_feedback),
            )
            return AgentErrorHandler.create_error_response(
                self.agent_type,
                user_feedback_error,
                "user feedback processing",
                start_time,
            )

    async def _analyze_user_feedback(self, state: WorkflowState) -> dict[str, Any]:
        """Analyze user feedback using LLM to extract actionable instructions."""
        if not state.user_feedback:
            return {
                "feedback_type": "none",
                "action_required": "prompt_user",
                "parsed_instructions": [],
            }

        feedback = state.user_feedback.strip()

        # Handle explicit commands
        if feedback.upper() in ["ACCEPT", "OK", "GOOD", "DONE"]:
            return {
                "feedback_type": "acceptance",
                "action_required": "complete",
                "parsed_instructions": [],
            }

        if feedback.upper() in ["TERMINATE", "STOP", "CANCEL", "END"]:
            return {
                "feedback_type": "termination",
                "action_required": "terminate",
                "parsed_instructions": [],
            }

        # Parse complex feedback using LLM
        analysis_prompt = f"""
        Analyze this user feedback for 3D asset refinement: "{feedback}"

        Original prompt was: "{state.original_prompt or state.prompt}"

        Extract:
        1. Specific changes requested
        2. Priority level (high/medium/low)
        3. Category (geometry/material/lighting/composition)

        Respond in JSON:
        {{
            "feedback_type": "refinement_request",
            "action_required": "refine",
            "parsed_instructions": [
                {{
                    "category": "geometry|material|lighting|composition",
                    "description": "specific change requested",
                    "priority": "high|medium|low"
                }}
            ],
            "refinement_summary": "one sentence summary of requested changes"
        }}
        """

        try:
            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": (
                        "# ROLE & EXPERTISE\\n"
                        "You are an expert user experience analyst and 3D asset feedback interpreter "
                        "with specialized knowledge in:\\n"
                        "- **3D Asset Evaluation**: Geometry, materials, lighting, composition assessment\\n"
                        "- **User Intent Analysis**: Parsing ambiguous requests, identifying core requirements\\n"
                        "- **Technical Translation**: Converting user language to specific Blender operations\\n"
                        "- **Feedback Classification**: Categorizing feedback by type, priority, and complexity\\n\\n"
                        "# ANALYSIS METHODOLOGY\\n"
                        "## Feedback Parsing Framework:\\n"
                        "### 1. INTENT EXTRACTION\\n"
                        "- **Primary Goal**: What is the user ultimately trying to achieve?\\n"
                        "- **Specific Changes**: What exact modifications are requested?\\n"
                        "- **Quality Expectations**: What standards or quality level is expected?\\n\\n"
                        "### 2. TECHNICAL CLASSIFICATION\\n"
                        "- **Geometry Changes**: Shape, topology, proportions, scale modifications\\n"
                        "- **Material Changes**: Color, texture, surface properties, shader adjustments\\n"
                        "- **Lighting Changes**: Illumination, shadows, mood, atmosphere\\n"
                        "- **Composition Changes**: Camera angle, framing, scene organization\\n"
                        "- **Animation Changes**: Motion, dynamics, temporal effects\\n\\n"
                        "### 3. PRIORITY ASSESSMENT\\n"
                        "- **Critical (1)**: Fundamental flaws, unusable without fix\\n"
                        "- **High (2)**: Major issues, significantly impacts quality\\n"
                        "- **Medium (3)**: Notable improvements, enhances quality\\n"
                        "- **Low (4)**: Minor tweaks, cosmetic improvements\\n"
                        "- **Optional (5)**: Nice-to-have, experimental changes\\n\\n"
                        "### 4. COMPLEXITY ESTIMATION\\n"
                        "- **Simple**: Basic parameter changes, color adjustments\\n"
                        "- **Moderate**: Geometry modifications, material replacements\\n"
                        "- **Complex**: Structural changes, topology rebuilding\\n"
                        "- **Advanced**: Complete redesign, architectural changes\\n\\n"
                        "# MANDATORY RESPONSE FORMAT\\n"
                        "## JSON Structure (EXACT FORMAT REQUIRED):\\n"
                        "Example response format:\\n"
                        "{\\n"
                        '  "feedback_analysis": {\\n'
                        '    "user_intent": "clear summary of what user wants",\\n'
                        '    "sentiment": "positive|neutral|negative|mixed",\\n'
                        '    "clarity_score": 0.0-1.0,\\n'
                        '    "actionability_score": 0.0-1.0\\n'
                        "  },\\n"
                        '  "requested_changes": [\\n'
                        "    {\\n"
                        '      "change_id": "unique_identifier",\\n'
                        '      "category": "geometry|material|lighting|composition|animation",\\n'
                        '      "description": "specific_change_description",\\n'
                        '      "priority": 1-5,\\n'
                        '      "complexity": "simple|moderate|complex|advanced",\\n'
                        '      "technical_requirements": ["specific_blender_operations"],\\n'
                        '      "success_criteria": ["measurable_outcomes"],\\n'
                        '      "estimated_impact": "low|medium|high"\\n'
                        "    }\\n"
                        "  ],\\n"
                        '  "clarification_needed": ["questions_for_user"],\\n'
                        '  "potential_conflicts": ["contradictory_requirements"],\\n'
                        '  "implementation_strategy": "recommended_approach"\\n'
                        "}\\n\\n"
                        "# CRITICAL REQUIREMENTS\\n"
                        "## Response Validation:\\n"
                        "- **Valid JSON Only**: No text before or after JSON structure\\n"
                        "- **Complete Structure**: All required fields must be present\\n"
                        "- **Realistic Assessment**: Honest complexity and priority ratings\\n"
                        "- **Actionable Details**: Specific enough for implementation\\n\\n"
                        "## Quality Standards:\\n"
                        "- **Technical Accuracy**: Use correct Blender terminology\\n"
                        "- **User Empathy**: Understand user frustration and intent\\n"
                        "- **Practical Feasibility**: Consider implementation constraints\\n"
                        "- **Clear Communication**: Avoid ambiguous interpretations\\n\\n"
                        "CRITICAL: Your analysis directly impacts user satisfaction. "
                        "Be thorough, accurate, and user-focused."
                    ),
                },
                {"role": "user", "content": analysis_prompt},
            ]

            response = await self.make_openai_request(messages, model="gpt-5-chat-latest")
            return json.loads(response)

        except (json.JSONDecodeError, Exception) as feedback_parsing_error:
            self.logger.warning(
                "Failed to parse user feedback with LLM",
                error=str(feedback_parsing_error),
            )
            # Fallback: basic keyword analysis
            return self._basic_feedback_analysis(feedback)

    def _basic_feedback_analysis(self, feedback: str) -> dict[str, Any]:
        """Basic keyword-based feedback analysis as fallback."""
        feedback_lower = feedback.lower()
        instructions = []

        # Geometry keywords
        geometry_keywords = [
            "bigger",
            "smaller",
            "taller",
            "shorter",
            "thicker",
            "thinner",
            "scale",
            "size",
        ]
        if any(keyword in feedback_lower for keyword in geometry_keywords):
            instructions.append(
                {
                    "category": "geometry",
                    "description": f"Size/scale modification: {feedback}",
                    "priority": "medium",
                }
            )

        # Material keywords
        material_keywords = [
            "color",
            "material",
            "texture",
            "shiny",
            "matte",
            "metallic",
            "glossy",
        ]
        if any(keyword in feedback_lower for keyword in material_keywords):
            instructions.append(
                {
                    "category": "material",
                    "description": f"Material modification: {feedback}",
                    "priority": "medium",
                }
            )

        # Lighting keywords
        lighting_keywords = ["brighter", "darker", "lighting", "shadow", "illumination"]
        if any(keyword in feedback_lower for keyword in lighting_keywords):
            instructions.append(
                {
                    "category": "lighting",
                    "description": f"Lighting modification: {feedback}",
                    "priority": "low",
                }
            )

        return {
            "feedback_type": "refinement_request" if instructions else "unclear",
            "action_required": "refine" if instructions else "prompt_user",
            "parsed_instructions": instructions,
            "refinement_summary": (feedback if instructions else "Unclear feedback, requesting clarification"),
        }

    def _should_prompt_user(self, feedback_analysis: dict[str, Any]) -> bool:
        """Determine if we should prompt the user for input."""
        # Always prompt if no feedback provided
        if feedback_analysis["feedback_type"] == "none":
            return True

        # Return condition directly for unclear feedback
        return feedback_analysis["feedback_type"] == "unclear"

    async def _generate_user_prompt(self, state: WorkflowState) -> str:
        """Generate contextual user prompt based on current state."""
        # Prepare quality summary
        quality_summary = self._format_quality_summary(state)
        issues_summary = self._format_issues_summary(state)

        # Choose appropriate prompt template
        if state.refinement_iterations == 0:
            # First time seeing the asset
            return self.feedback_templates["initial_review_prompt"].format(
                original_prompt=state.original_prompt or state.prompt,
                quality_summary=quality_summary,
            )
        else:
            # Subsequent refinement rounds
            return self.feedback_templates["refinement_prompt"].format(
                quality_summary=quality_summary,
                issues_summary=issues_summary,
            )

    def _format_quality_summary(self, state: WorkflowState) -> str:
        """Format quality assessment results for user display."""
        summary_parts = []

        # Add critic analysis if available
        if hasattr(state, "critic_analysis") and state.critic_analysis:
            overall_score = state.critic_analysis.get("overall_score", "N/A")
            summary_parts.append(f"Overall Score: {overall_score}/10")

            # Add individual scores
            aspects = [
                "visual_quality",
                "geometry",
                "materials",
                "lighting",
                "composition",
            ]
            for aspect in aspects:
                if aspect in state.critic_analysis:
                    aspect_data = state.critic_analysis[aspect]
                    if isinstance(aspect_data, dict) and "score" in aspect_data:
                        score = aspect_data["score"]
                        summary_parts.append(f"{aspect.replace('_', ' ').title()}: {score}/10")

        # Add verification results if available
        if hasattr(state, "verification_result") and state.verification_result:
            verification_score = state.verification_result.get("quality_score", "N/A")
            summary_parts.append(f"Technical Quality: {verification_score}/10")

        return "\n".join(summary_parts) if summary_parts else "Quality assessment pending"

    def _format_issues_summary(self, state: WorkflowState) -> str:
        """Format identified issues for user display."""
        issues = []

        # Add critic issues
        if hasattr(state, "critic_analysis") and state.critic_analysis:
            critic_issues = state.critic_analysis.get("critical_issues", [])
            issues.extend([f"• {issue}" for issue in critic_issues])

        # Add verification issues
        if hasattr(state, "verification_result") and state.verification_result:
            verification_issues = state.verification_result.get("issues", [])
            issues.extend([f"• {issue}" for issue in verification_issues])

        return "\n".join(issues) if issues else "No critical issues identified"

    def _determine_workflow_action(self, feedback_analysis: dict[str, Any]) -> dict[str, Any]:
        """Determine the next workflow action based on feedback analysis."""
        action_required = feedback_analysis.get("action_required", "prompt_user")

        if action_required == "complete":
            return {
                "action": "complete",
                "reason": "User accepted the current result",
            }

        elif action_required == "terminate":
            return {
                "action": "terminate",
                "reason": "User requested termination",
            }

        elif action_required == "refine":
            # Extract refinement instructions
            instructions = feedback_analysis.get("parsed_instructions", [])
            high_priority_count = sum(1 for instr in instructions if instr.get("priority") == "high")

            priority = "high" if high_priority_count > 0 else "medium"

            return {
                "action": "refine",
                "reason": feedback_analysis.get("refinement_summary", "User requested refinements"),
                "instructions": instructions,
                "priority": priority,
            }

        else:
            return {
                "action": "wait_for_input",
                "reason": "Waiting for user feedback",
            }

    def _generate_response_message(self, workflow_action: dict[str, Any], needs_user_input: bool) -> str:
        """Generate human-readable response message."""
        action = workflow_action["action"]

        if action == "complete":
            return "User accepted the asset. Workflow completed successfully."

        elif action == "terminate":
            return "User requested termination. Ending workflow."

        elif action == "refine":
            instruction_count = len(workflow_action.get("instructions", []))
            priority = workflow_action.get("priority", "medium")
            return f"Processing {instruction_count} refinement requests (priority: {priority})"

        elif action == "wait_for_input":
            if needs_user_input:
                return "Waiting for user feedback on the generated asset."
            else:
                return "User feedback processed, ready for next workflow step."

        else:
            return "Processing user feedback..."

    async def validate_input(self, state: WorkflowState) -> bool:  # noqa: ARG002
        """Validate input for user proxy agent."""
        # User proxy can work with various states, but execution result is helpful
        return True  # More permissive validation since this agent handles various scenarios
