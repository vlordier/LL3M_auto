"""LL3M Orchestrator for coordinating 3D asset generation workflows."""

from typing import Any

import structlog
from langgraph.graph.state import CompiledStateGraph

from ..utils.types import WorkflowState
from .workflow import create_ll3m_workflow

logger = structlog.get_logger(__name__)


class LL3MOrchestrator:
    """Main orchestrator for LL3M asset generation workflows."""

    def __init__(self) -> None:
        """Initialize the orchestrator with the comprehensive workflow."""
        self._workflow: CompiledStateGraph[WorkflowState] | None = None

    def _get_workflow(self) -> CompiledStateGraph[WorkflowState]:
        """Get or create the workflow graph."""
        if self._workflow is None:
            self._workflow = create_ll3m_workflow()
            logger.info("Created comprehensive workflow with full quality assessment pipeline")
        return self._workflow

    async def generate_asset(
        self, prompt: str, export_formats: list[str] | None = None, asset_name: str = "asset"
    ) -> dict[str, Any]:
        """Generate a 3D asset from a text prompt.

        Args:
            prompt: Text description of the asset to generate
            export_formats: List of export formats (e.g., ['blend', 'obj'])
            asset_name: Name for the generated asset

        Returns:
            Dictionary containing generation results and metadata
        """
        logger.info("Starting asset generation", prompt=prompt, formats=export_formats)

        # Create initial state
        initial_state = WorkflowState(
            prompt=prompt,
            user_feedback=None,
            documentation="",
            generated_code="",
            execution_result=None,
            refinement_request="",
            asset_metadata=None,
            error_message=None,
            previous_screenshot_path=None,
            critic_analysis=None,
            verification_result=None,
            user_prompt="",
            original_prompt=prompt,
        )

        try:
            workflow = self._get_workflow()

            # Execute the workflow
            result_dict = await workflow.ainvoke(
                initial_state, config={"configurable": {"thread_id": f"asset_{asset_name}"}}
            )
            result = WorkflowState(**result_dict)

            # Extract results
            if result.execution_result and result.execution_result.success:
                return {
                    "success": True,
                    "asset_path": result.execution_result.asset_path,
                    "screenshot_path": result.execution_result.screenshot_path,
                    "generation_time": result.execution_result.execution_time,
                    "exported_files": getattr(result.execution_result, "exported_files", []),
                    "metadata": result.asset_metadata,
                    "code": result.generated_code,
                }
            else:
                error_msg = result.error_message or "Asset generation failed"
                logger.error("Asset generation failed", error=error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "code": result.generated_code,
                }

        except Exception as e:
            logger.exception("Unexpected error during asset generation")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    async def refine_asset(self, original_prompt: str, feedback: str, asset_name: str = "asset") -> dict[str, Any]:
        """Refine an existing asset based on user feedback.

        Args:
            original_prompt: Original prompt used to generate the asset
            feedback: User feedback for refinement
            asset_name: Name of the asset being refined

        Returns:
            Dictionary containing refinement results and metadata
        """
        logger.info("Starting asset refinement", original_prompt=original_prompt, feedback=feedback)

        # Create state for refinement
        refinement_state = WorkflowState(
            prompt=original_prompt,
            user_feedback=feedback,
            documentation="",
            generated_code="",
            execution_result=None,
            refinement_request="",
            asset_metadata=None,
            error_message=None,
            previous_screenshot_path=None,
            critic_analysis=None,
            verification_result=None,
            user_prompt="",
            original_prompt=original_prompt,
            refinement_iterations=1,  # Start with 1 since this is a refinement
        )

        try:
            workflow = self._get_workflow()

            # Execute the workflow with refinement
            result_dict = await workflow.ainvoke(
                refinement_state, config={"configurable": {"thread_id": f"refine_{asset_name}"}}
            )
            result = WorkflowState(**result_dict)

            # Extract results
            if result.execution_result and result.execution_result.success:
                return {
                    "success": True,
                    "asset_path": result.execution_result.asset_path,
                    "screenshot_path": result.execution_result.screenshot_path,
                    "generation_time": result.execution_result.execution_time,
                    "refinement_iterations": result.refinement_iterations,
                    "metadata": result.asset_metadata,
                    "code": result.generated_code,
                }
            else:
                error_msg = result.error_message or "Asset refinement failed"
                logger.error("Asset refinement failed", error=error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "refinement_iterations": result.refinement_iterations,
                    "code": result.generated_code,
                }

        except Exception as e:
            logger.exception("Unexpected error during asset refinement")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }
