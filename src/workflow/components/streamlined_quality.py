"""Streamlined quality assessment components with DRY principles."""

from collections.abc import Callable
from typing import Any

import structlog

from ...utils.types import WorkflowState
from .node_factory import NodeFactory
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class QualityAssessment:
    """Simplified quality assessment with standardized agents."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.node_factory = NodeFactory(config)

    def create_quality_pipeline(self) -> Callable:
        """Create comprehensive quality assessment pipeline."""
        # Use standardized assessment nodes
        critic_assessment = self.node_factory.create_assessment_node("critic")
        verification_assessment = self.node_factory.create_assessment_node("verification")

        async def quality_pipeline(state: WorkflowState) -> WorkflowState:
            """Run comprehensive quality assessment."""
            logger.info("Running quality assessment pipeline")

            # Skip if execution failed
            if not state.execution_result or not state.execution_result.success:
                logger.info("Skipping quality assessment - execution failed")
                return state

            try:
                # Run assessments in parallel conceptually (simplified for now)
                assessments = []

                # Agent-based assessments
                critic_result = await critic_assessment(state)
                verification_result = await verification_assessment(state)
                assessments.extend([critic_result, verification_result])

                # Static code quality assessment
                code_quality = self._assess_code_quality(state)
                output_quality = self._assess_output_quality(state)
                assessments.extend([code_quality, output_quality])

                # Aggregate results
                total_score = sum(r["score"] for r in assessments) / len(assessments)
                all_issues = [issue for r in assessments for issue in r["issues"]]

                # Update state
                state.quality_score = total_score
                state.quality_issues = all_issues
                state.quality_feedback = self._aggregate_feedback(assessments)

                logger.info(f"Quality assessment completed - score: {total_score:.1f}")
                return state

            except Exception as e:
                logger.exception(f"Quality assessment failed: {e}")
                state.error_message = f"Quality assessment error: {e}"
                return state

        return quality_pipeline

    def create_quick_quality_check(self) -> Callable:
        """Create quick quality check with bypass logic."""

        async def quick_quality_check(state: WorkflowState) -> WorkflowState:
            """Quick quality check - bypass full assessment for obvious successes."""
            logger.info("Running quick quality check")

            # Bypass condition: successful execution with all outputs
            if (
                state.execution_result
                and state.execution_result.success
                and not state.execution_result.errors
                and state.blend_file_path
                and state.screenshot_path
            ):
                logger.info("Quick quality check passed - skipping full assessment")
                state.quality_score = 4.0
                state.quality_feedback = "Quick assessment: Execution successful with all outputs"
                return state

            # Otherwise run simplified assessment
            logger.info("Quick quality check requires assessment")

            # Simple assessment logic
            if state.execution_result and state.execution_result.success:
                state.quality_score = 3.5 if not state.execution_result.errors else 2.5
                state.quality_feedback = "Quality assessment completed successfully"
            else:
                state.quality_score = 1.0
                state.quality_feedback = "Quality assessment identified issues"

            return state

        return quick_quality_check

    def _assess_code_quality(self, state: WorkflowState) -> dict[str, Any]:
        """Assess code quality with static analysis."""
        issues = []
        score = 3.0

        if state.generated_code:
            # Check for common issues
            if "TODO" in state.generated_code:
                issues.append("Code contains TODO comments")
                score -= 0.5

            if "pass  # TODO" in state.generated_code:
                issues.append("Code contains placeholder pass statements")
                score -= 1.0

            if len(state.generated_code.split("\n")) < 5:
                issues.append("Generated code is very short")
                score -= 0.5

        return {
            "score": max(0, score),
            "issues": issues,
            "feedback": "Code quality assessment completed",
        }

    def _assess_output_quality(self, state: WorkflowState) -> dict[str, Any]:
        """Assess output quality (files generated)."""
        issues = []
        score = 3.0

        # Check for expected outputs
        if not state.blend_file_path:
            issues.append("No blend file generated")
            score -= 1.0

        if not state.screenshot_path:
            issues.append("No screenshot generated")
            score -= 0.5

        return {
            "score": max(0, score),
            "issues": issues,
            "feedback": "Output validation completed",
        }

    def _aggregate_feedback(self, results: list[dict]) -> str:
        """Aggregate feedback from multiple assessment results."""
        feedback_parts = []
        for result in results:
            if result.get("feedback"):
                gate_name = result.get("gate", "assessment")
                feedback_parts.append(f"{gate_name}: {result['feedback']}")

        return " | ".join(feedback_parts) if feedback_parts else "Assessment completed"


# Simplified condition functions
def should_assess_quality(state: WorkflowState) -> str:
    """Determine if quality assessment should be performed."""
    if not state.execution_result or not state.execution_result.success:
        return "end"

    # Skip if already refined successfully
    if state.execution_result.success and not state.execution_result.errors and state.refinement_iterations >= 1:
        logger.info("Skipping quality assessment - execution successful after refinement")
        return "end"

    return "assess"


# Factory functions for backward compatibility
def create_quick_quality_check(config: WorkflowConfig) -> Callable:
    """Factory function for quick quality check."""
    assessment = QualityAssessment(config)
    return assessment.create_quick_quality_check()


def create_comprehensive_quality_check(config: WorkflowConfig) -> Callable:
    """Factory function for comprehensive quality assessment."""
    assessment = QualityAssessment(config)
    return assessment.create_quality_pipeline()


# Simplified quality gate class
class QualityGate:
    """Simplified quality gate with standardized interface."""

    def __init__(self, name: str, assessment_func: Callable, threshold: float = 2.0):
        self.name = name
        self.assessment_func = assessment_func
        self.threshold = threshold

    async def assess(self, state: WorkflowState) -> dict[str, Any]:
        """Run quality assessment with consistent error handling."""
        try:
            result = await self.assessment_func(state)
            passed = result.get("score", 0) >= self.threshold

            return {
                "gate": self.name,
                "score": result.get("score", 0),
                "passed": passed,
                "issues": result.get("issues", []),
                "feedback": result.get("feedback", ""),
            }

        except Exception as e:
            logger.exception(f"Quality gate {self.name} failed: {e}")
            return {
                "gate": self.name,
                "score": 0,
                "passed": False,
                "issues": [str(e)],
                "feedback": f"Assessment failed: {e}",
            }
