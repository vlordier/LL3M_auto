"""LL3M Orchestrator - Main coordinator for multi-agent 3D asset generation workflows."""

import asyncio
import uuid
from typing import Any

import structlog

from ..assets.manager import AssetManager
from ..utils.config import get_settings
from ..utils.types import ExecutionResult, WorkflowState
from .main_graph import create_ll3m_workflow

logger = structlog.get_logger(__name__)


class LL3MOrchestrator:
    """Main orchestrator for LL3M multi-agent workflows."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the orchestrator."""
        self.config = config or {}
        self.settings = get_settings()
        self.asset_manager = AssetManager(
            repository_path=str(self.settings.app.output_directory / "assets")
        )
        self.workflow = create_ll3m_workflow(self.config)

        logger.info("LL3M Orchestrator initialized")

    async def generate_asset(
        self,
        prompt: str,
        export_format: str = "blend",
        skip_refinement: bool = False,
        tags: list[str] | None = None,
    ) -> ExecutionResult:
        """Generate a 3D asset from a text prompt.

        Args:
            prompt: Text description of the desired 3D asset
            export_format: Export format (blend, gltf, obj, fbx)
            skip_refinement: Skip automatic refinement phase
            tags: Optional tags for asset categorization

        Returns:
            ExecutionResult with asset generation details
        """
        start_time = asyncio.get_event_loop().time()
        session_id = str(uuid.uuid4())

        logger.info(
            "Starting asset generation",
            prompt=prompt,
            session_id=session_id,
            export_format=export_format,
            skip_refinement=skip_refinement,
        )

        try:
            # Create initial workflow state
            workflow_state = WorkflowState(
                prompt=prompt,
                session_id=session_id,
                export_format=export_format,
                refinement_iterations=0,
                should_continue=True,
                error_message="",
                needs_refinement=False,
            )

            # Configure workflow for refinement settings
            if skip_refinement:
                workflow_state.max_refinement_iterations = 0
            else:
                workflow_state.max_refinement_iterations = (
                    self.settings.workflow.max_refinement_iterations
                )

            # Execute the workflow
            result = await self._run_workflow(workflow_state)
            execution_time = asyncio.get_event_loop().time() - start_time

            if result.execution_result and result.execution_result.success:
                # Create managed asset
                try:
                    managed_asset = self.asset_manager.create_from_workflow_state(
                        result, tags=tags
                    )
                    if managed_asset:
                        logger.info(
                            "Asset stored successfully",
                            asset_id=managed_asset.id,
                            session_id=session_id,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to store asset",
                        error=str(e),
                        session_id=session_id,
                    )

                return ExecutionResult(
                    success=True,
                    asset_path=result.execution_result.asset_path,
                    screenshot_path=result.execution_result.screenshot_path,
                    logs=result.execution_result.logs,
                    errors=result.execution_result.errors,
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "refinement_iterations": result.refinement_iterations,
                        "quality_score": getattr(
                            result.verification_result, "quality_score", None
                        )
                        if hasattr(result, "verification_result")
                        else None,
                        "needs_refinement": result.needs_refinement,
                        "managed_asset_id": managed_asset.id if managed_asset else None,
                    },
                )
            else:
                error_msg = result.error_message or "Unknown workflow error"
                logger.error(
                    "Asset generation failed",
                    error=error_msg,
                    session_id=session_id,
                    execution_time=execution_time,
                )

                return ExecutionResult(
                    success=False,
                    asset_path=None,
                    screenshot_path=None,
                    logs=[],
                    errors=[error_msg],
                    execution_time=execution_time,
                    metadata={"session_id": session_id},
                )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(
                "Asset generation exception",
                error=str(e),
                session_id=session_id,
                execution_time=execution_time,
            )

            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Generation failed: {str(e)}"],
                execution_time=execution_time,
                metadata={"session_id": session_id},
            )

    async def refine_asset(
        self,
        asset_id: str,
        user_feedback: str,
        tags: list[str] | None = None,
    ) -> ExecutionResult:
        """Refine an existing asset based on user feedback.

        Args:
            asset_id: ID of the asset to refine
            user_feedback: User feedback for refinement
            tags: Optional additional tags

        Returns:
            ExecutionResult with refinement details
        """
        start_time = asyncio.get_event_loop().time()
        session_id = str(uuid.uuid4())

        logger.info(
            "Starting asset refinement",
            asset_id=asset_id,
            feedback=user_feedback,
            session_id=session_id,
        )

        try:
            # Get existing asset
            managed_asset = self.asset_manager.repository.get_asset(asset_id)
            if not managed_asset:
                return ExecutionResult(
                    success=False,
                    asset_path=None,
                    screenshot_path=None,
                    logs=[],
                    errors=[f"Asset not found: {asset_id}"],
                    execution_time=0.0,
                    metadata={"session_id": session_id},
                )

            # Create refinement workflow state based on original prompt + feedback
            refinement_prompt = (
                f"{managed_asset.original_prompt}\n\nRefinement: {user_feedback}"
            )

            workflow_state = WorkflowState(
                prompt=refinement_prompt,
                original_prompt=managed_asset.original_prompt,
                refinement_request=user_feedback,
                session_id=session_id,
                refinement_iterations=0,
                max_refinement_iterations=2,  # Limit refinement-of-refinement
                should_continue=True,
                error_message="",
                needs_refinement=False,
                subtasks=managed_asset.subtasks,  # Reuse original subtasks as starting point
            )

            # Execute refinement workflow
            result = await self._run_workflow(workflow_state)
            execution_time = asyncio.get_event_loop().time() - start_time

            if result.execution_result and result.execution_result.success:
                # Add new version to existing asset
                try:
                    new_version = self.asset_manager.add_refinement_version(
                        asset_id, result, user_feedback
                    )
                    if new_version:
                        logger.info(
                            "Asset refined successfully",
                            asset_id=asset_id,
                            new_version=new_version.version,
                            session_id=session_id,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to store refined asset",
                        error=str(e),
                        asset_id=asset_id,
                        session_id=session_id,
                    )

                return ExecutionResult(
                    success=True,
                    asset_path=result.execution_result.asset_path,
                    screenshot_path=result.execution_result.screenshot_path,
                    logs=result.execution_result.logs,
                    errors=result.execution_result.errors,
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "original_asset_id": asset_id,
                        "refinement_iterations": result.refinement_iterations,
                        "quality_score": getattr(
                            result.verification_result, "quality_score", None
                        )
                        if hasattr(result, "verification_result")
                        else None,
                        "new_version": new_version.version if new_version else None,
                    },
                )
            else:
                error_msg = result.error_message or "Unknown refinement error"
                logger.error(
                    "Asset refinement failed",
                    error=error_msg,
                    asset_id=asset_id,
                    session_id=session_id,
                    execution_time=execution_time,
                )

                return ExecutionResult(
                    success=False,
                    asset_path=None,
                    screenshot_path=None,
                    logs=[],
                    errors=[error_msg],
                    execution_time=execution_time,
                    metadata={
                        "session_id": session_id,
                        "original_asset_id": asset_id,
                    },
                )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(
                "Asset refinement exception",
                error=str(e),
                asset_id=asset_id,
                session_id=session_id,
                execution_time=execution_time,
            )

            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=[],
                errors=[f"Refinement failed: {str(e)}"],
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "original_asset_id": asset_id,
                },
            )

    async def _run_workflow(self, initial_state: WorkflowState) -> WorkflowState:
        """Execute the LangGraph workflow with the given initial state.

        Args:
            initial_state: Initial workflow state

        Returns:
            Final workflow state after execution
        """
        try:
            # Run the workflow synchronously since LangGraph handles async internally
            config = {"configurable": {"thread_id": initial_state.session_id}}

            # Get the compiled graph
            compiled_graph = self.workflow.compile()

            # Execute the workflow
            final_state = None
            async for state in compiled_graph.astream(initial_state, config=config):
                final_state = state

            if final_state:
                # LangGraph returns dict with node names as keys
                # Extract the actual state from the last node output
                for _node_name, state_data in final_state.items():
                    if isinstance(state_data, WorkflowState):
                        return state_data

                # If no WorkflowState found, create error state
                logger.error("Workflow returned invalid state format")
                error_state = initial_state
                error_state.error_message = (
                    "Workflow execution failed - invalid state format"
                )
                error_state.should_continue = False
                return error_state
            else:
                logger.error("Workflow returned no final state")
                error_state = initial_state
                error_state.error_message = "Workflow execution failed - no final state"
                error_state.should_continue = False
                return error_state

        except Exception as e:
            logger.error("Workflow execution failed", error=str(e))
            error_state = initial_state
            error_state.error_message = f"Workflow execution failed: {str(e)}"
            error_state.should_continue = False
            return error_state

    def get_asset_statistics(self) -> dict[str, Any]:
        """Get statistics about generated assets."""
        return self.asset_manager.repository.get_asset_statistics()

    def list_assets(
        self,
        tags: list[str] | None = None,
        min_quality_score: float | None = None,
        limit: int | None = None,
    ):
        """List assets with optional filtering."""
        return self.asset_manager.repository.list_assets(
            tags=tags,
            min_quality_score=min_quality_score,
            limit=limit,
        )

    def get_best_assets(self, limit: int = 10):
        """Get the highest quality assets."""
        return self.asset_manager.get_best_assets(limit=limit)

    def cleanup_low_quality_assets(self, min_quality: float = 5.0) -> int:
        """Remove assets with quality scores below threshold."""
        return self.asset_manager.cleanup_low_quality_assets(min_quality=min_quality)
