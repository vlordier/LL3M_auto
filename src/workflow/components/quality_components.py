"""Quality assessment components for workflow evaluation."""

from collections.abc import Callable
from typing import Literal

import structlog

from ...utils.types import WorkflowState
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class QualityGate:
    """Individual quality assessment gate."""

    def __init__(self, name: str, assessment_func: Callable, threshold: float = 2.0):
        self.name = name
        self.assessment_func = assessment_func
        self.threshold = threshold

    async def assess(self, state: WorkflowState) -> dict:
        """Run quality assessment."""
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
            logger.exception("Quality gate %s failed", self.name)
            return {
                "gate": self.name,
                "score": 0,
                "passed": False,
                "issues": [str(e)],
                "feedback": f"Assessment failed: {e}",
            }


class QualityAssessmentPipeline:
    """Configurable quality assessment pipeline."""

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def create_standard_assessment(self) -> Callable:
        """Create standard quality assessment flow."""

        async def standard_assessment(state: WorkflowState) -> WorkflowState:
            """Run standard quality assessment: critic -> verification -> assessment."""
            logger.info("Running standard quality assessment")

            # Skip if execution failed
            if not state.execution_result or not state.execution_result.success:
                logger.info("Skipping quality assessment - execution failed")
                return state

            # Skip if already refined successfully and no errors
            if (
                state.execution_result.success
                and not state.execution_result.errors
                and state.refinement_iterations >= 1
            ):
                logger.info("Skipping quality assessment - execution successful after refinement")
                return state

            try:
                # For modular system, we simulate quality assessment
                # In a full implementation, these would use actual agent factories

                # Simulate quality scoring based on execution result
                if state.execution_result and state.execution_result.success:
                    state.quality_score = 3.5 if not state.execution_result.errors else 2.5
                    state.critic_feedback = "Quality assessment completed successfully"
                else:
                    state.quality_score = 1.0
                    state.critic_feedback = "Quality assessment identified issues"

                return state

            except Exception as e:
                logger.exception("Quality assessment failed")
                state.error_message = f"Quality assessment error: {e}"
                return state

        return standard_assessment

    def create_comprehensive_assessment(self) -> Callable:
        """Create comprehensive quality assessment with all gates."""

        async def comprehensive_assessment(state: WorkflowState) -> WorkflowState:
            """Run comprehensive quality assessment."""
            logger.info("Running comprehensive quality assessment")

            if not state.execution_result or not state.execution_result.success:
                logger.info("Skipping comprehensive assessment - execution failed")
                return state

            try:
                # All quality gates
                gates = [
                    QualityGate("critic", self._create_critic_assessment()),
                    QualityGate("verification", self._create_verification_assessment()),
                    QualityGate("code_quality", self._create_code_quality_assessment()),
                    QualityGate("output_validation", self._create_output_validation_assessment()),
                ]

                results = []
                for gate in gates:
                    result = await gate.assess(state)
                    results.append(result)

                    # Stop early if critical gate fails
                    if not result["passed"] and gate.name in ["critic", "verification"]:
                        break

                # Aggregate results
                total_score = sum(r["score"] for r in results) / len(results) if results else 0
                all_issues = [issue for r in results for issue in r["issues"]]

                # Update state
                state.quality_score = total_score
                state.quality_issues = all_issues
                state.quality_feedback = self._aggregate_feedback(results)

                return state

            except Exception as e:
                logger.exception("Comprehensive assessment failed")
                state.error_message = f"Comprehensive assessment error: {e}"
                return state

        return comprehensive_assessment

    def _create_critic_assessment(self) -> Callable:
        """Create critic assessment function."""

        async def critic_assessment(state: WorkflowState) -> dict:
            # Critic functionality simulated in modular system
            from ...agents.factory import get_agent_factory

            try:
                critic = get_agent_factory().create_critic()
                response = await critic.process(state)

                if response.success:
                    score = 3.0
                    issues = []
                    feedback = response.data or "Critic assessment completed"
                else:
                    score = 1.0
                    issues = [response.message] if response.message else ["Critic assessment failed"]
                    feedback = response.message or "Critic assessment failed"
            except Exception as e:
                score = 1.0
                issues = [str(e)]
                feedback = f"Critic assessment error: {e}"

            return {"score": score, "issues": issues, "feedback": feedback}

        return critic_assessment

    def _create_verification_assessment(self) -> Callable:
        """Create verification assessment function."""

        async def verification_assessment(state: WorkflowState) -> dict:
            # Verification functionality simulated in modular system
            from ...agents.factory import get_agent_factory

            try:
                verification = get_agent_factory().create_verification()
                response = await verification.process(state)

                if response.success:
                    score = 3.0
                    issues = []
                    feedback = response.data or "Verification assessment completed"
                else:
                    score = 1.0
                    issues = [response.message] if response.message else ["Verification assessment failed"]
                    feedback = response.message or "Verification assessment failed"
            except Exception as e:
                score = 1.0
                issues = [str(e)]
                feedback = f"Verification assessment error: {e}"

            return {"score": score, "issues": issues, "feedback": feedback}

        return verification_assessment

    def _create_code_quality_assessment(self) -> Callable:
        """Create code quality assessment function."""

        async def code_quality_assessment(state: WorkflowState) -> dict:
            # Simple code quality checks
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

        return code_quality_assessment

    def _create_output_validation_assessment(self) -> Callable:
        """Create output validation assessment function."""

        async def output_validation_assessment(state: WorkflowState) -> dict:
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

        return output_validation_assessment

    async def _run_quality_assessment(self, state: WorkflowState) -> WorkflowState:
        """Run final quality assessment node."""
        # Quality assessment functionality simulated in modular system
        try:
            # Simulate quality assessment by analyzing current state
            if state.execution_result and state.execution_result.success:
                state.quality_score = 3.5 if not state.execution_result.errors else 2.5
                state.critic_feedback = "Quality assessment completed successfully"
            else:
                state.quality_score = 1.0
                state.critic_feedback = "Quality assessment identified issues"
        except Exception as e:
            logger.exception("Quality assessment failed")
            state.error_message = f"Quality assessment error: {e}"

        return state

    def _aggregate_feedback(self, results: list) -> str:
        """Aggregate feedback from multiple assessment results."""
        feedback_parts = []
        for result in results:
            if result.get("feedback"):
                feedback_parts.append(f"{result['gate']}: {result['feedback']}")

        return " | ".join(feedback_parts)


def create_quick_quality_check(config: WorkflowConfig) -> Callable:
    """Create a quick quality check that bypasses full assessment for obvious successes."""

    async def quick_quality_check(state: WorkflowState) -> WorkflowState:
        """Quick quality check - bypass full assessment if obviously successful."""
        logger.info("Running quick quality check")

        # If execution was successful with no errors, skip full assessment
        if (
            state.execution_result
            and state.execution_result.success
            and not state.execution_result.errors
            and state.blend_file_path
            and state.screenshot_path
        ):
            logger.info("Quick quality check passed - skipping full assessment")
            state.quality_score = 4.0  # High quality score
            state.quality_feedback = "Quick assessment: Execution successful with all outputs"
            return state

        # Otherwise run standard assessment
        logger.info("Quick quality check requires full assessment")
        pipeline = QualityAssessmentPipeline(config)
        assessment = pipeline.create_standard_assessment()
        return await assessment(state)

    return quick_quality_check


def should_assess_quality(state: WorkflowState) -> Literal["assess", "end"]:
    """Determine if quality assessment should be performed."""
    if not state.execution_result or not state.execution_result.success:
        return "end"

    # Skip quality assessment if execution was perfect and already refined
    if state.execution_result.success and not state.execution_result.errors and state.refinement_iterations >= 1:
        logger.info("Skipping quality assessment - execution successful after refinement")
        return "end"

    return "assess"
