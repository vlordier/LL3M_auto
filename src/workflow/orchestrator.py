"""LL3M Orchestrator for coordinating 3D asset generation workflows."""

from typing import Any

import structlog
from langgraph.graph.state import CompiledStateGraph

from ..cache.retrieval_service import get_retrieval_service
from ..utils.types import WorkflowState
from .workflow import create_ll3m_workflow

logger = structlog.get_logger(__name__)


class LL3MOrchestrator:
    """Main orchestrator for LL3M asset generation workflows."""

    def __init__(self) -> None:
        """Initialize the orchestrator with the comprehensive workflow."""
        self._workflow: CompiledStateGraph[WorkflowState] | None = None
        self.retrieval_service = get_retrieval_service()
        self.enable_caching = True

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

        # Try cache retrieval first if enabled
        cache_result = None
        if self.enable_caching:
            try:
                # Detect complexity level from prompt for better cache matching
                complexity_level = self._detect_complexity_level(prompt)
                cache_result = await self.retrieval_service.retrieve_cached_code(
                    prompt=prompt,
                    complexity_level=complexity_level,
                    include_alternatives=True
                )
                
                if cache_result.cache_hit:
                    logger.info("Cache hit - using cached code", 
                               entry_id=cache_result.entry.id,
                               similarity=cache_result.similarity_score,
                               retrieval_time=cache_result.retrieval_time_ms)
                    
                    # Return cached result with generated metadata
                    return {
                        "success": True,
                        "cached": True,
                        "cache_entry_id": cache_result.entry.id,
                        "similarity_score": cache_result.similarity_score,
                        "retrieval_time_ms": cache_result.retrieval_time_ms,
                        "code": cache_result.entry.code,
                        "code_files": [cache_result.entry.code],  # For compatibility
                        "output_files": [],  # Will be populated after execution
                        "metadata": cache_result.entry.metadata
                    }
                else:
                    logger.info("Cache miss - proceeding with generation", 
                               alternatives=len(cache_result.alternatives),
                               retrieval_time=cache_result.retrieval_time_ms)
                    
            except Exception as e:
                logger.warning(f"Cache retrieval failed, proceeding with generation: {e}")

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
                generation_result = {
                    "success": True,
                    "cached": False,
                    "asset_path": result.execution_result.asset_path,
                    "screenshot_path": result.execution_result.screenshot_path,
                    "generation_time": result.execution_result.execution_time,
                    "exported_files": getattr(result.execution_result, "exported_files", []),
                    "metadata": result.asset_metadata,
                    "code": result.generated_code,
                    "code_files": [result.generated_code] if result.generated_code else [],
                    "output_files": [result.execution_result.screenshot_path] if result.execution_result.screenshot_path else []
                }
                
                # Store successful generation in cache
                if self.enable_caching and result.generated_code:
                    try:
                        complexity_level = self._detect_complexity_level(prompt)
                        quality_score = None
                        
                        # Extract quality score from critic analysis if available
                        if result.critic_analysis and hasattr(result.critic_analysis, 'overall_score'):
                            quality_score = result.critic_analysis.overall_score
                            
                        await self.retrieval_service.store_generated_code(
                            prompt=prompt,
                            generated_code=result.generated_code,
                            complexity_level=complexity_level,
                            quality_score=quality_score,
                            workflow_state=result
                        )
                        logger.info("Stored generated code in cache", prompt=prompt[:50])
                        
                    except Exception as e:
                        logger.warning(f"Failed to cache generated code: {e}")
                
                return generation_result
            else:
                error_msg = result.error_message or "Asset generation failed"
                logger.error("Asset generation failed", error=error_msg)
                return {
                    "success": False,
                    "cached": False,
                    "error": error_msg,
                    "code": result.generated_code,
                }

        except Exception as e:
            logger.exception("Unexpected error during asset generation")
            return {
                "success": False,
                "cached": False,
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
            
    def _detect_complexity_level(self, prompt: str) -> int:
        """Detect complexity level from prompt for cache matching.
        
        Args:
            prompt: The generation prompt
            
        Returns:
            Complexity level (1-5)
        """
        prompt_lower = prompt.lower()
        
        # Keywords for different complexity levels
        complexity_indicators = {
            5: ["animation", "character", "dance", "movement", "rigged", "animated", "sequence"],
            4: ["lighting", "scene", "environment", "multiple", "room", "interior", "exterior", "landscape"],
            3: ["texture", "material", "wood", "metal", "fabric", "realistic", "detailed", "shiny", "rough"],
            2: ["furniture", "object", "table", "chair", "simple", "basic", "house", "building"],
            1: ["cube", "sphere", "cylinder", "plane", "primitive", "shape", "geometry", "ovoid"]
        }
        
        for level in range(5, 0, -1):
            if any(keyword in prompt_lower for keyword in complexity_indicators[level]):
                return level
                
        return 2  # Default complexity
        
    def set_caching_enabled(self, enabled: bool):
        """Enable or disable caching.
        
        Args:
            enabled: Whether to enable caching
        """
        self.enable_caching = enabled
        logger.info(f"RAG caching {'enabled' if enabled else 'disabled'}")
        
    async def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics and performance metrics.
        
        Returns:
            Cache statistics
        """
        return await self.retrieval_service.get_service_statistics()
