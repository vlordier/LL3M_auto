"""Critic agent for visual analysis and quality assessment of 3D assets using GPT-4V."""

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from ..knowledge.context7_client import Context7RetrievalService
from ..utils.types import (
    AgentResponse,
    AgentType,
    WorkflowState,
)
from .base import EnhancedBaseAgent
from .critic_schemas import RefinementComparisonResult, VisualAnalysisResult
from .error_handling import AgentErrorHandler


@dataclass
class VisualAnalysisPrompt:
    """Templates for visual analysis prompts."""

    SYSTEM_PROMPT = (
        "You are a master 3D artist, technical director, and quality assurance specialist with "
        "15+ years of experience in procedural modeling, architectural visualization, product design, "
        "and photorealistic rendering. You have expert knowledge in Blender, Maya, 3ds Max, "
        "and industry-standard workflows.\n\n"
        "COMPREHENSIVE ANALYSIS FRAMEWORK:\n"
        "🎨 **VISUAL QUALITY DIMENSIONS:**\n"
        "1. **Visual Appeal**: Overall aesthetics, professional appearance, artistic quality\n"
        "2. **Geometry**: Mesh topology, shape accuracy, technical execution\n"
        "3. **Materials**: Surface properties, texture quality, shader complexity\n"
        "4. **Lighting**: Illumination setup, shadow quality, mood\n"
        "5. **Composition**: Camera work, framing, visual hierarchy\n"
        "6. **Requirements Match**: Adherence to original prompt specifications\n\n"
        "🔧 **TECHNICAL ANALYSIS DIMENSIONS:**\n"
        "7. **Technical Quality**: Mesh quality, topology, technical execution\n"
        "8. **Scale & Proportions**: Realistic dimensions, functional proportions\n"
        "9. **Structural Integrity**: Component connections, stability, physics\n\n"
        "🌟 **SCENE & CONTEXT DIMENSIONS:**\n"
        "10. **Scene Setup**: Environment, background, context appropriateness\n"
        "11. **Presentation Quality**: Professional rendering, post-processing\n"
        "12. **Artistic Appeal**: Creative design, aesthetic innovation\n\n"
        "⚙️ **FUNCTIONAL ASSESSMENT DIMENSIONS:**\n"
        "13. **Functional Accuracy**: Real-world usability, practical design\n"
        "14. **Detail Level**: Appropriate complexity for purpose\n"
        "15. **Realism**: Photorealistic quality, believability\n\n"
        "🚨 **CRITICAL ISSUES TO IDENTIFY:**\n"
        "- Disconnected geometry, floating parts, intersections\n"
        "- Unrealistic proportions, impossible scale relationships\n"
        "- Poor mesh topology, non-manifold geometry\n"
        "- Missing materials, incorrect shading\n"
        "- Inadequate lighting, overexposure, underexposure\n"
        "- Compositional problems, poor camera work\n"
        "- Functional impossibilities, structural failures\n\n"
        "Provide specific, actionable, technical feedback for code improvements."
    )

    ANALYSIS_TEMPLATE = (
        "Perform a comprehensive multi-dimensional analysis of this 3D asset screenshot.\n\n"
        "📋 **ASSET CONTEXT:**\n"
        "Original Request: {original_prompt}\n\n"
        "Subtasks Completed:\n{subtasks_summary}\n\n"
        "🔍 **COMPREHENSIVE EVALUATION REQUIRED:**\n\n"
        "**VISUAL QUALITY DIMENSIONS:**\n"
        "1. **Visual Quality**: Overall aesthetics, professional appearance, artistic merit\n"
        "2. **Geometry**: Mesh topology, shape accuracy, technical execution quality\n"
        "3. **Materials**: Surface properties, texture quality, shader complexity\n"
        "4. **Lighting**: Scene illumination, shadow quality, mood and atmosphere\n"
        "5. **Composition**: Camera angle, framing, visual hierarchy\n"
        "6. **Requirements Match**: Adherence to original prompt specifications\n\n"
        "**TECHNICAL ANALYSIS DIMENSIONS:**\n"
        "7. **Technical Quality**: Mesh quality, topology, render execution\n"
        "8. **Scale & Proportions**: Realistic dimensions, functional proportions\n"
        "9. **Structural Integrity**: Component connections, stability, physics\n\n"
        "**SCENE & CONTEXT DIMENSIONS:**\n"
        "10. **Scene Setup**: Environment, background, context appropriateness\n"
        "11. **Presentation Quality**: Professional rendering, post-processing\n"
        "12. **Artistic Appeal**: Creative design, aesthetic innovation\n\n"
        "**FUNCTIONAL ASSESSMENT DIMENSIONS:**\n"
        "13. **Functional Accuracy**: Real-world usability, practical design\n"
        "14. **Detail Level**: Appropriate complexity for intended purpose\n"
        "15. **Realism**: Photorealistic quality, believability factor\n\n"
        "📊 **FOR EACH DIMENSION, PROVIDE:**\n"
        "- Score from 1-10 (1=critical failure, 5=adequate, 10=exceptional)\n"
        "- Detailed notes explaining your professional assessment\n"
        "- Specific issues identified in this dimension\n"
        "- Targeted improvement suggestions with technical details\n"
        "- Critical flag if this dimension has major problems\n\n"
        "🎯 **FINAL ASSESSMENT:**\n"
        "- Overall quality score (1-10)\n"
        "- Refinement priority (critical/high/medium/low)\n"
        "- Asset type classification\n"
        "- Complexity level assessment (1-5)\n"
        "- Primary focus areas for improvement"
    )

    COMPARISON_TEMPLATE = (
        "Compare these before and after 3D asset screenshots to assess improvement.\n\n"
        "Original Request: {original_prompt}\n\n"
        "Refinement Request: {refinement_request}\n\n"
        "Please compare the two versions and provide:\n"
        "1. **Improvement Assessment**: What got better?\n"
        "2. **Issue Resolution**: Were the identified problems fixed?\n"
        "3. **New Issues**: Any new problems introduced?\n"
        "4. **Quality Change**: Overall quality improvement score (-5 to +5)\n\n"
        "Analyze the changes and provide:\n"
        "- Improvement score from -5 to +5 (-5=much worse, 0=no change, +5=much better)\n"
        "- List of issues that were resolved\n"
        "- Any new issues introduced\n"
        "- Whether quality improved, stayed the same, or degraded\n"
        "- Recommendations for further improvement\n"
        "- Whether refinement should continue"
    )


class CriticAgent(EnhancedBaseAgent):
    """Provides visual analysis and quality assessment of 3D assets using GPT-4V."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize critic agent."""
        super().__init__(config)
        self.visual_analyzer = VisualAnalysisPrompt()
        self.context7_service = Context7RetrievalService()
        self.quality_thresholds = {
            # Overall quality standards
            "overall_score": 7.0,

            # Visual quality dimensions
            "visual_quality": 6.5,
            "geometry": 7.0,
            "materials": 6.0,
            "lighting": 6.0,
            "composition": 6.0,
            "requirements_match": 8.0,

            # Technical analysis dimensions
            "technical_quality": 7.0,
            "scale_proportions": 7.5,
            "structural_integrity": 8.0,

            # Scene and context dimensions
            "scene_setup": 6.0,
            "presentation_quality": 6.5,
            "artistic_appeal": 5.5,

            # Functional assessment dimensions
            "functional_accuracy": 7.5,
            "detail_level": 6.0,
            "realism": 6.5,
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

            self.logger.info(
                "Starting comprehensive multi-dimensional visual quality analysis",
                has_previous_screenshot=hasattr(state, "previous_screenshot_path") and bool(getattr(state, "previous_screenshot_path", None)),
                screenshot_path=state.execution_result.screenshot_path if state.execution_result else None,
                analysis_type="refinement_comparison" if (hasattr(state, "previous_screenshot_path") and state.previous_screenshot_path) else "initial_analysis",
                refinement_iteration=getattr(state, "refinement_iterations", 0)
            )

            # Determine analysis type
            if hasattr(state, "previous_screenshot_path") and state.previous_screenshot_path:
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

            # Enhanced logging for comprehensive analysis
            self.logger.info(
                "Comprehensive visual analysis completed",
                overall_score=analysis_result.get("overall_score", 0),
                needs_refinement=needs_refinement,
                execution_time=execution_time,
                refinement_priority=analysis_result.get("refinement_priority", "unknown"),
                critical_issues_count=len(analysis_result.get("critical_issues", [])),
                improvement_suggestions_count=len(analysis_result.get("improvement_suggestions", [])),
                asset_type=analysis_result.get("asset_type_assessment", "unknown"),
                complexity_level=analysis_result.get("complexity_level", 1),
                analysis_dimensions_evaluated=15
            )

            # Log detailed dimension scores
            dimension_scores = {}
            for dimension in ["visual_quality", "geometry", "materials", "lighting", "composition",
                             "requirements_match", "technical_quality", "scale_proportions",
                             "structural_integrity", "scene_setup", "presentation_quality",
                             "artistic_appeal", "functional_accuracy", "detail_level", "realism"]:
                if dimension in analysis_result:
                    dim_data = analysis_result[dimension]
                    if isinstance(dim_data, dict) and "score" in dim_data:
                        dimension_scores[dimension] = dim_data["score"]

                        # Log critical issues for each dimension
                        if dim_data.get("is_critical", False):
                            self.logger.warning(
                                f"Critical issues found in {dimension}",
                                dimension=dimension,
                                score=dim_data["score"],
                                issues=dim_data.get("issues", []),
                                suggestions=dim_data.get("suggestions", [])
                            )

            self.logger.debug(
                "Detailed dimension scoring breakdown",
                dimension_scores=dimension_scores,
                average_score=sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0,
                below_threshold_dimensions=[k for k, v in dimension_scores.items() if v < self.quality_thresholds.get(k, 6.0)]
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=analysis_result,
                message=self._generate_summary_message(analysis_result),
                execution_time=execution_time,
                metadata={
                    "needs_refinement": needs_refinement,
                    "refinement_priority": analysis_result.get("refinement_priority", "low"),
                },
            )

        except Exception as visual_analysis_error:
            AgentErrorHandler.log_error(
                self.logger,
                visual_analysis_error,
                "visual analysis",
                has_execution_result=state.execution_result is not None,
            )
            return AgentErrorHandler.create_error_response(
                self.agent_type, visual_analysis_error, "visual analysis", start_time
            )

    async def _analyze_initial_quality(self, state: WorkflowState) -> dict[str, Any]:
        """Perform initial quality analysis of 3D asset screenshot."""
        screenshot_path = state.execution_result.screenshot_path if state.execution_result else None

        if not screenshot_path or not Path(screenshot_path).exists():
            return {"error": "Screenshot file not found"}

        # Encode screenshot as base64
        screenshot_base64 = await self._encode_image_base64(screenshot_path)
        if not screenshot_base64:
            return {"error": "Failed to encode screenshot"}

        # Prepare subtasks summary
        subtasks_summary = "\n".join(
            [f"- {subtask.type.value}: {subtask.description}" for subtask in (state.subtasks or [])]
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

        # Get structured GPT-4V analysis
        try:
            analysis_result = await self.make_structured_request(
                cast(list[ChatCompletionMessageParam], messages),
                VisualAnalysisResult,
                model="gpt-5-chat-latest"
            )

            self.logger.info(
                "Structured analysis completed successfully",
                overall_score=analysis_result.overall_score,
                needs_refinement=analysis_result.needs_refinement,
                refinement_priority=analysis_result.refinement_priority.value,
                critical_issues_count=len(analysis_result.critical_issues),
                improvement_suggestions_count=len(analysis_result.improvement_suggestions),
                asset_type_assessment=analysis_result.asset_type_assessment,
                complexity_level=analysis_result.complexity_level,
                analysis_focus_areas=analysis_result.analysis_focus
            )

            # Log comprehensive dimension analysis
            dimensions_data = {
                "visual_quality": analysis_result.visual_quality,
                "geometry": analysis_result.geometry,
                "materials": analysis_result.materials,
                "lighting": analysis_result.lighting,
                "composition": analysis_result.composition,
                "requirements_match": analysis_result.requirements_match,
                "technical_quality": analysis_result.technical_quality,
                "scale_proportions": analysis_result.scale_proportions,
                "structural_integrity": analysis_result.structural_integrity,
                "scene_setup": analysis_result.scene_setup,
                "presentation_quality": analysis_result.presentation_quality,
                "artistic_appeal": analysis_result.artistic_appeal,
                "functional_accuracy": analysis_result.functional_accuracy,
                "detail_level": analysis_result.detail_level,
                "realism": analysis_result.realism
            }

            for dim_name, dim_data in dimensions_data.items():
                self.logger.debug(
                    f"Dimension analysis: {dim_name}",
                    dimension=dim_name,
                    score=dim_data.score,
                    is_critical=dim_data.is_critical,
                    issues_count=len(dim_data.issues),
                    suggestions_count=len(dim_data.suggestions),
                    notes_length=len(dim_data.notes)
                )

                # Log critical dimension issues
                if dim_data.is_critical:
                    self.logger.warning(
                        f"CRITICAL DIMENSION: {dim_name} requires immediate attention",
                        dimension=dim_name,
                        score=dim_data.score,
                        critical_issues=dim_data.issues,
                        urgent_suggestions=dim_data.suggestions
                    )

            return analysis_result.model_dump()

        except Exception as e:
            self.logger.exception("Failed to get structured analysis response", error=str(e))
            return {"error": f"Failed to get structured analysis: {e!s}"}

    async def _analyze_refinement_comparison(self, state: WorkflowState) -> dict[str, Any]:
        """Compare before and after screenshots for refinement assessment."""
        current_screenshot = state.execution_result.screenshot_path if state.execution_result else None
        previous_screenshot = getattr(state, "previous_screenshot_path", None)

        if not current_screenshot or not previous_screenshot:
            return {"error": "Both before and after screenshots required"}

        if not Path(current_screenshot).exists() or not Path(previous_screenshot).exists():
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

        # Get structured GPT-4V comparison analysis
        try:
            comparison_result = await self.make_structured_request(
                cast(list[ChatCompletionMessageParam], messages),
                RefinementComparisonResult,
                model="gpt-5-chat-latest"
            )

            self.logger.info(
                "Structured refinement comparison completed successfully",
                improvement_score=comparison_result.improvement_score,
                quality_change=comparison_result.quality_change.value,
                continue_refinement=comparison_result.continue_refinement,
                issues_resolved_count=len(comparison_result.issues_resolved),
                new_issues_count=len(comparison_result.new_issues),
                recommendations_count=len(comparison_result.recommendations)
            )

            # Log detailed comparison analysis
            if comparison_result.issues_resolved:
                self.logger.info(
                    "Issues successfully resolved in refinement",
                    resolved_issues=comparison_result.issues_resolved,
                    resolution_count=len(comparison_result.issues_resolved)
                )

            if comparison_result.new_issues:
                self.logger.warning(
                    "New issues introduced during refinement",
                    new_issues=comparison_result.new_issues,
                    new_issues_count=len(comparison_result.new_issues)
                )

            if comparison_result.recommendations:
                self.logger.info(
                    "Additional refinement recommendations provided",
                    recommendations=comparison_result.recommendations,
                    recommendations_count=len(comparison_result.recommendations)
                )

            return comparison_result.model_dump()

        except Exception as e:
            self.logger.exception("Failed to get structured comparison response", error=str(e))
            return {"error": f"Failed to get structured comparison: {e!s}"}

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

        # Check individual aspect scores across all dimensions
        dimensions_below_threshold = []
        critical_dimensions = []

        for aspect, threshold in self.quality_thresholds.items():
            if aspect == "overall_score":
                continue

            aspect_data = analysis_result.get(aspect, {})
            if isinstance(aspect_data, dict):
                score = aspect_data.get("score", 0)
                is_critical = aspect_data.get("is_critical", False)

                if score < threshold:
                    dimensions_below_threshold.append((aspect, score, threshold))

                if is_critical:
                    critical_dimensions.append(aspect)

        # Log dimension analysis for refinement decision
        if dimensions_below_threshold or critical_dimensions:
            self.logger.info(
                "Quality assessment: dimensions requiring improvement",
                dimensions_below_threshold=[(dim, score) for dim, score, _ in dimensions_below_threshold],
                critical_dimensions=critical_dimensions,
                total_failing_dimensions=len(dimensions_below_threshold),
                refinement_recommended=True
            )
            return True
        else:
            self.logger.info(
                "Quality assessment: all dimensions meet thresholds",
                all_dimensions_passing=True,
                refinement_recommended=False
            )

        # Check for critical issues
        critical_issues = analysis_result.get("critical_issues", [])
        has_critical_issues = bool(critical_issues)
        if has_critical_issues:
            self.logger.warning(
                "Critical issues detected - refinement required",
                critical_issues=critical_issues,
                critical_issues_count=len(critical_issues)
            )

        return has_critical_issues

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

    async def _get_documentation_for_issues(self, analysis_result: dict[str, Any]) -> str:
        """Get documentation for identified issues using Context7."""
        issues = analysis_result.get("critical_issues", [])
        suggestions = analysis_result.get("improvement_suggestions", [])

        search_queries = list(set(issues + suggestions))

        if not search_queries:
            return ""

        self.logger.info("Retrieving documentation for critic issues", queries=search_queries)

        response = await self.context7_service.retrieve_documentation(search_queries)

        if response.success:
            return response.data

        return ""

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for critic agent."""
        if not state.execution_result:
            return False

        # Type system ensures execution_result is ExecutionResult if not None
        screenshot_path = state.execution_result.screenshot_path
        if not screenshot_path:
            return False

        return Path(screenshot_path).exists()
