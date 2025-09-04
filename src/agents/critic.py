"""Critic agent for visual analysis and quality assessment of 3D assets using GPT-4V."""

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..knowledge.context7_client import Context7RetrievalService
from ..utils.types import (
    AgentResponse,
    AgentType,
    ExecutionResult,
    WorkflowState,
)
from .base import EnhancedBaseAgent


@dataclass
class VisualAnalysisPrompt:
    """Templates for visual analysis prompts."""

    SYSTEM_PROMPT = (
        "You are a professional 3D artist and technical reviewer specializing in "
        "procedural asset creation and quality assessment.\n\n"
        "Your role is to analyze 3D asset screenshots and provide detailed, "
        "constructive feedback focusing on:\n"
        "1. Visual quality and aesthetics\n"
        "2. Geometry accuracy and topology\n"
        "3. Material and texture quality\n"
        "4. Lighting and scene composition\n"
        "5. Technical correctness\n"
        "6. Adherence to original requirements\n\n"
        "Provide specific, actionable recommendations for improvement when issues "
        "are identified."
    )

    ANALYSIS_TEMPLATE = (
        "Analyze this 3D asset screenshot and provide a detailed quality "
        "assessment.\n\n"
        "Original Request: {original_prompt}\n\n"
        "Subtasks Completed:\n{subtasks_summary}\n\n"
        "Please evaluate:\n"
        "1. **Visual Quality**: Overall appearance, aesthetics, professional look\n"
        "2. **Geometry**: Shape accuracy, mesh topology, model complexity\n"
        "3. **Materials**: Surface appearance, texture quality, shader setup\n"
        "4. **Lighting**: Scene illumination, shadows, ambient lighting\n"
        "5. **Composition**: Camera angle, framing, scene organization\n"
        "6. **Requirements Match**: How well does this match the original request?\n\n"
        "Provide your assessment as a JSON response with this structure:\n"
        "{{\n"
        '  "overall_score": <1-10>,\n'
        '  "visual_quality": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "geometry": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "materials": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "lighting": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "composition": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "requirements_match": {{ "score": <1-10>, "notes": "feedback" }},\n'
        '  "needs_refinement": <true/false>,\n'
        '  "critical_issues": ["issue1", "issue2"],\n'
        '  "improvement_suggestions": ["suggestion1", "suggestion2"],\n'
        '  "refinement_priority": "high|medium|low"\n'
        "}}"
    )

    COMPARISON_TEMPLATE = (
        "Compare these before and after 3D asset screenshots to assess improvement.\n\n"
        "Original Request: {original_prompt}\n\n"
        "Refinement Request: {refinement_request}\n\n"
        "Please compare the two versions and provide:\n"
        "1. **Improvement Assessment**: What got better?\n"
        "2. **Issue Resolution**: Were the identified problems fixed?\n"
        "3. **New Issues**: Any new problems introduced?\n"
        "4. **Quality Change**: Overall quality improvement score\n\n"
        "Provide your assessment as a JSON response:\n"
        "{{\n"
        '  "improvement_score": <-5 to +5>,\n'
        '  "issues_resolved": ["resolved_issue1", "resolved_issue2"],\n'
        '  "new_issues": ["new_issue1", "new_issue2"],\n'
        '  "quality_change": "improved|same|degraded",\n'
        '  "recommendations": ["recommendation1", "recommendation2"],\n'
        '  "continue_refinement": <true/false>\n'
        "}}"
    )


class CriticAgent(EnhancedBaseAgent):
    """Provides visual analysis and quality assessment of 3D assets using GPT-4V."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize critic agent."""
        super().__init__(config)
        self.visual_analyzer = VisualAnalysisPrompt()
        self.context7_service = Context7RetrievalService()
        self.quality_thresholds = {
            "overall_score": 7.0,
            "visual_quality": 6.5,
            "geometry": 7.0,
            "materials": 6.0,
            "lighting": 6.0,
            "composition": 6.0,
            "requirements_match": 8.0,
        }

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.CRITIC

    @property
    def name(self) -> str:
        """Return agent name."""
        return "Visual Quality Critic"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Analyze 3D asset screenshots and provide quality assessment."""
        start_time = asyncio.get_event_loop().time()

        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data={},
                    message="Invalid input: execution result with screenshot required",
                    execution_time=0.0,
                )

            self.logger.info("Starting visual quality analysis")

            # Determine analysis type
            if (
                hasattr(state, "previous_screenshot_path")
                and state.previous_screenshot_path
            ):
                # Comparison analysis for refinement
                analysis_result = await self._analyze_refinement_comparison(state)
            else:
                # Initial quality analysis
                analysis_result = await self._analyze_initial_quality(state)

            execution_time = asyncio.get_event_loop().time() - start_time

            if analysis_result.get("error"):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data={},
                    message=f"Analysis failed: {analysis_result['error']}",
                    execution_time=execution_time,
                )

            # Determine if refinement is needed
            needs_refinement = self._evaluate_refinement_need(analysis_result)

            # Get documentation for issues if refinement is needed
            if needs_refinement:
                docs = await self._get_documentation_for_issues(analysis_result)
                analysis_result["documentation_for_issues"] = docs

            self.logger.info(
                "Visual analysis completed",
                overall_score=analysis_result.get("overall_score", 0),
                needs_refinement=needs_refinement,
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=analysis_result,
                message=self._generate_summary_message(analysis_result),
                execution_time=execution_time,
                metadata={
                    "needs_refinement": needs_refinement,
                    "refinement_priority": analysis_result.get(
                        "refinement_priority", "low"
                    ),
                },
            )

        except Exception as e:
            self.logger.exception("Visual analysis failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data={},
                message=f"Analysis failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def _analyze_initial_quality(self, state: WorkflowState) -> dict[str, Any]:
        """Perform initial quality analysis of 3D asset screenshot."""
        screenshot_path = (
            state.execution_result.screenshot_path if state.execution_result else None
        )

        if not screenshot_path or not Path(screenshot_path).exists():
            return {"error": "Screenshot file not found"}

        # Encode screenshot as base64
        screenshot_base64 = await self._encode_image_base64(screenshot_path)
        if not screenshot_base64:
            return {"error": "Failed to encode screenshot"}

        # Prepare subtasks summary
        subtasks_summary = "\n".join(
            [
                f"- {subtask.type.value}: {subtask.description}"
                for subtask in (state.subtasks or [])
            ]
        )

        # Create analysis messages
        messages = [
            {"role": "system", "content": self.visual_analyzer.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.visual_analyzer.ANALYSIS_TEMPLATE.format(
                            original_prompt=state.original_prompt or state.prompt,
                            subtasks_summary=subtasks_summary,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        # Get GPT-4V analysis
        response_text = await self.make_openai_request(messages, model="gpt-4o")

        # Parse JSON response
        try:
            analysis_result = json.loads(response_text)
            return analysis_result
        except json.JSONDecodeError as e:
            self.logger.exception("Failed to parse analysis response", error=str(e))
            return {"error": f"Failed to parse analysis: {str(e)}"}

    async def _analyze_refinement_comparison(
        self, state: WorkflowState
    ) -> dict[str, Any]:
        """Compare before and after screenshots for refinement assessment."""
        current_screenshot = (
            state.execution_result.screenshot_path if state.execution_result else None
        )
        previous_screenshot = getattr(state, "previous_screenshot_path", None)

        if not current_screenshot or not previous_screenshot:
            return {"error": "Both before and after screenshots required"}

        if (
            not Path(current_screenshot).exists()
            or not Path(previous_screenshot).exists()
        ):
            return {"error": "Screenshot files not found"}

        # Encode both screenshots
        current_base64 = await self._encode_image_base64(current_screenshot)
        previous_base64 = await self._encode_image_base64(previous_screenshot)

        if not current_base64 or not previous_base64:
            return {"error": "Failed to encode screenshots"}

        # Create comparison messages
        messages = [
            {"role": "system", "content": self.visual_analyzer.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.visual_analyzer.COMPARISON_TEMPLATE.format(
                            original_prompt=state.original_prompt or state.prompt,
                            refinement_request=getattr(state, "refinement_request", ""),
                        ),
                    },
                    {"type": "text", "text": "BEFORE (Original version):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{previous_base64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "AFTER (Refined version):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        # Get GPT-4V comparison analysis
        response_text = await self.make_openai_request(messages, model="gpt-4o")

        # Parse JSON response
        try:
            analysis_result = json.loads(response_text)
            return analysis_result
        except json.JSONDecodeError as e:
            self.logger.exception("Failed to parse comparison response", error=str(e))
            return {"error": f"Failed to parse comparison: {str(e)}"}

    async def _encode_image_base64(self, image_path: str) -> str | None:
        """Encode image file as base64 string."""
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                self.logger.error("Image file not found", path=image_path)
                return None

            with open(image_file, "rb") as f:
                image_data = f.read()

            return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            self.logger.exception("Failed to encode image", path=image_path, error=str(e))
            return None

    def _evaluate_refinement_need(self, analysis_result: dict[str, Any]) -> bool:
        """Evaluate if refinement is needed based on analysis scores."""
        if "needs_refinement" in analysis_result:
            return analysis_result["needs_refinement"]

        # Check overall score against threshold
        overall_score = analysis_result.get("overall_score", 0)
        if overall_score < self.quality_thresholds["overall_score"]:
            return True

        # Check individual aspect scores
        for aspect, threshold in self.quality_thresholds.items():
            if aspect == "overall_score":
                continue

            aspect_data = analysis_result.get(aspect, {})
            if isinstance(aspect_data, dict):
                score = aspect_data.get("score", 0)
                if score < threshold:
                    return True

        # Check for critical issues
        critical_issues = analysis_result.get("critical_issues", [])
        return bool(critical_issues)

    def _generate_summary_message(self, analysis_result: dict[str, Any]) -> str:
        """Generate human-readable summary message from analysis."""
        if "improvement_score" in analysis_result:
            # Refinement comparison summary
            improvement = analysis_result.get("improvement_score", 0)
            quality_change = analysis_result.get("quality_change", "same")

            if improvement > 0:
                return f"Asset quality improved (+{improvement}) - {quality_change}"
            elif improvement < 0:
                return f"Asset quality degraded ({improvement}) - {quality_change}"
            else:
                return f"No significant quality change - {quality_change}"

        # Initial analysis summary
        overall_score = analysis_result.get("overall_score", 0)
        needs_refinement = analysis_result.get("needs_refinement", False)
        critical_issues = analysis_result.get("critical_issues", [])

        summary = f"Quality score: {overall_score}/10"

        if needs_refinement:
            priority = analysis_result.get("refinement_priority", "medium")
            summary += f" - Refinement needed ({priority} priority)"

        if critical_issues:
            summary += f" - {len(critical_issues)} critical issues"

        return summary

    async def _get_documentation_for_issues(
        self, analysis_result: dict[str, Any]
    ) -> str:
        """Get documentation for identified issues using Context7."""
        issues = analysis_result.get("critical_issues", [])
        suggestions = analysis_result.get("improvement_suggestions", [])

        search_queries = list(set(issues + suggestions))

        if not search_queries:
            return ""

        self.logger.info(
            "Retrieving documentation for critic issues", queries=search_queries
        )

        response = await self.context7_service.retrieve_documentation(search_queries)

        if response.success:
            return response.data

        return ""

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for critic agent."""
        if not state.execution_result:
            return False

        if not isinstance(state.execution_result, ExecutionResult):
            return False

        screenshot_path = state.execution_result.screenshot_path
        if not screenshot_path:
            return False

        return Path(screenshot_path).exists()
