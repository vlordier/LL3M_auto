"""LL3M Orchestrator for coordinating 3D asset generation workflows."""

import time
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
        self,
        prompt: str,
        export_formats: list[str] | None = None,
        asset_name: str = "asset",
        output_directory: str | None = None,
    ) -> dict[str, Any]:
        """Generate a 3D asset from a text prompt.

        Args:
            prompt: Text description of the asset to generate
            export_formats: List of export formats (e.g., ['blend', 'obj'])
            asset_name: Name for the generated asset
            output_directory: Directory where outputs should be saved

        Returns:
            Dictionary containing generation results and metadata
        """
        generation_start_time = time.time()
        logger.info(
            "🚀 Starting asset generation",
            prompt=prompt,
            formats=export_formats,
            asset_name=asset_name,
            output_directory=output_directory,
            caching_enabled=self.enable_caching,
        )

        # Input validation
        if not prompt or not prompt.strip():
            error_msg = "Prompt cannot be empty or whitespace only"
            logger.error("❌ Generation failed - invalid input", error=error_msg)
            return {
                "success": False,
                "cached": False,
                "error": error_msg,
                "generation_time": 0,
            }

        if len(prompt) > 1000:
            logger.warning("⚠️ Very long prompt detected", prompt_length=len(prompt))

        logger.debug(
            "📋 Generation parameters validated",
            prompt_length=len(prompt),
            asset_name_valid=bool(asset_name and asset_name.strip()),
            output_dir_specified=output_directory is not None,
        )

        # Try cache retrieval first if enabled
        cache_result = None
        cache_retrieval_time = 0
        if self.enable_caching:
            cache_start_time = time.time()
            try:
                logger.debug("🔍 Starting cache retrieval phase")
                # Detect complexity level from prompt for better cache matching
                complexity_level = self._detect_complexity_level(prompt)
                logger.debug(
                    "📊 Complexity analysis completed",
                    complexity_level=complexity_level,
                    detected_keywords=self._get_complexity_keywords(prompt),
                )

                cache_result = await self.retrieval_service.retrieve_cached_code(
                    prompt=prompt,
                    complexity_level=complexity_level,
                    include_alternatives=True,
                )
                cache_retrieval_time = time.time() - cache_start_time

                logger.debug(
                    "🗂️ Cache query completed",
                    retrieval_time_s=f"{cache_retrieval_time:.3f}",
                    cache_hit=cache_result.cache_hit if cache_result else False,
                    alternatives_found=(len(cache_result.alternatives) if cache_result else 0),
                )

                if cache_result.cache_hit:
                    logger.info(
                        "✅ Cache hit - using cached code, but still executing in Blender",
                        entry_id=cache_result.entry.id,
                        similarity=f"{cache_result.similarity_score:.3f}",
                        retrieval_time_ms=cache_result.retrieval_time_ms,
                        code_length=len(cache_result.entry.code),
                        original_prompt=cache_result.entry.prompt[:100] + "...",
                        cache_usage_count=getattr(cache_result.entry, "usage_count", "unknown"),
                    )

                    # Use cached code but execute it to generate actual files
                    generated_code = cache_result.entry.code
                    is_cached = True
                else:
                    logger.info(
                        "❌ Cache miss - proceeding with full generation",
                        alternatives=len(cache_result.alternatives),
                        retrieval_time_ms=cache_result.retrieval_time_ms,
                        best_similarity=max(
                            [alt.similarity_score for alt in cache_result.alternatives],
                            default=0,
                        ),
                    )
                    if cache_result.alternatives:
                        logger.debug(
                            "🔄 Alternative cache entries found",
                            alternatives=[
                                {
                                    "id": alt.entry.id,
                                    "similarity": f"{alt.similarity_score:.3f}",
                                    "prompt": alt.entry.prompt[:50] + "...",
                                }
                                for alt in cache_result.alternatives[:3]  # Show top 3
                            ],
                        )
                    is_cached = False

            except Exception as e:
                cache_retrieval_time = time.time() - cache_start_time
                logger.exception(
                    "💥 Cache retrieval failed, proceeding with generation",
                    error=str(e),
                    error_type=type(e).__name__,
                    retrieval_time_s=f"{cache_retrieval_time:.3f}",
                    stack_trace=(logger.exception if logger.isEnabledFor(logger.DEBUG) else None),
                )

        # Create initial state with comprehensive setup
        output_dir = output_directory or "outputs"
        logger.debug(
            "🏗️ Creating initial workflow state",
            output_directory=output_dir,
            has_cached_code="generated_code" in locals(),
            prompt_hash=hash(prompt),
        )

        initial_state = WorkflowState(
            prompt=prompt,
            original_prompt=prompt,
            output_directory=output_dir,
        )

        # If we have cached code, inject it into the state to skip generation
        if "generated_code" in locals():
            logger.info(
                "📋 Injecting cached code into workflow state",
                code_length=len(generated_code),
                is_cached=is_cached,
                cache_entry_id=(cache_result.entry.id if cache_result and cache_result.cache_hit else None),
            )
            initial_state.generated_code = generated_code
            initial_state.is_cached = is_cached
        else:
            logger.debug("🆕 No cached code available, will generate fresh code")

        try:
            logger.debug("⚙️ Initializing workflow graph")
            workflow = self._get_workflow()
            logger.debug("✅ Workflow graph ready for execution")

            # Execute the workflow with comprehensive monitoring
            thread_id = f"asset_{asset_name}_{int(time.time())}"
            execution_start = time.time()

            logger.info(
                "🚀 Starting workflow execution",
                thread_id=thread_id,
                initial_state_keys=list(vars(initial_state).keys()),
                workflow_config={"thread_id": thread_id},
            )

            try:
                result_dict = await workflow.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
                execution_time = time.time() - execution_start
                logger.info(
                    "✅ Workflow execution completed",
                    execution_time_s=f"{execution_time:.3f}",
                    result_keys=(list(result_dict.keys()) if isinstance(result_dict, dict) else "unknown"),
                    thread_id=thread_id,
                )
                result = WorkflowState(**result_dict)

            except Exception as workflow_error:
                execution_time = time.time() - execution_start
                logger.exception(
                    "💥 Workflow execution failed",
                    error=str(workflow_error),
                    error_type=type(workflow_error).__name__,
                    execution_time_s=f"{execution_time:.3f}",
                    thread_id=thread_id,
                    initial_state_summary={
                        "prompt_length": len(initial_state.prompt),
                        "has_generated_code": bool(initial_state.generated_code),
                        "is_cached": initial_state.is_cached,
                        "output_directory": initial_state.output_directory,
                    },
                )
                raise

            # Extract and validate results with comprehensive logging
            total_generation_time = time.time() - generation_start_time

            logger.debug(
                "📊 Analyzing workflow execution results",
                has_execution_result=result.execution_result is not None,
                execution_success=(result.execution_result.success if result.execution_result else None),
                has_asset_metadata=result.asset_metadata is not None,
                generated_code_length=(len(result.generated_code) if result.generated_code else 0),
                refinement_iterations=result.refinement_iterations,
                total_time_s=f"{total_generation_time:.3f}",
            )

            if result.execution_result and result.execution_result.success:
                logger.info(
                    "🎉 Asset generation successful!",
                    asset_path=result.execution_result.asset_path,
                    screenshot_path=result.execution_result.screenshot_path,
                    blender_execution_time=result.execution_result.execution_time,
                    total_generation_time_s=f"{total_generation_time:.3f}",
                    cache_retrieval_time_s=(f"{cache_retrieval_time:.3f}" if cache_retrieval_time > 0 else "N/A"),
                    was_cached=is_cached if "is_cached" in locals() else False,
                )

                # Validate generated files exist
                generated_files = []
                if result.execution_result.asset_path:
                    from pathlib import Path

                    asset_file = Path(result.execution_result.asset_path)
                    if asset_file.exists():
                        generated_files.append(
                            {
                                "path": str(asset_file),
                                "size_bytes": asset_file.stat().st_size,
                                "type": "asset",
                            }
                        )
                        logger.debug(f"✅ Asset file verified: {asset_file} ({asset_file.stat().st_size} bytes)")
                    else:
                        logger.warning(f"⚠️ Asset file missing: {asset_file}")

                if result.execution_result.screenshot_path:
                    screenshot_file = Path(result.execution_result.screenshot_path)
                    if screenshot_file.exists():
                        generated_files.append(
                            {
                                "path": str(screenshot_file),
                                "size_bytes": screenshot_file.stat().st_size,
                                "type": "screenshot",
                            }
                        )
                        logger.debug(
                            f"✅ Screenshot verified: {screenshot_file} ({screenshot_file.stat().st_size} bytes)"
                        )
                    else:
                        logger.warning(f"⚠️ Screenshot file missing: {screenshot_file}")

                generation_result = {
                    "success": True,
                    "cached": is_cached if "is_cached" in locals() else False,
                    "asset_path": result.execution_result.asset_path,
                    "screenshot_path": result.execution_result.screenshot_path,
                    "generation_time": result.execution_result.execution_time,
                    "total_time": total_generation_time,
                    "cache_time": cache_retrieval_time,
                    "exported_files": getattr(result.execution_result, "exported_files", []),
                    "generated_files": generated_files,
                    "metadata": result.asset_metadata,
                    "code": result.generated_code,
                    "code_files": ([result.generated_code] if result.generated_code else []),
                    "output_files": (
                        [result.execution_result.screenshot_path] if result.execution_result.screenshot_path else []
                    ),
                    "final_state": result,
                    "performance_metrics": {
                        "total_time_s": total_generation_time,
                        "cache_time_s": cache_retrieval_time,
                        "execution_time_s": result.execution_result.execution_time,
                        "refinement_iterations": result.refinement_iterations,
                        "was_cached": is_cached if "is_cached" in locals() else False,
                    },
                }

                # Store successful generation in cache with detailed logging
                if (
                    self.enable_caching
                    and result.generated_code
                    and not (is_cached if "is_cached" in locals() else False)
                ):
                    cache_store_start = time.time()
                    try:
                        logger.debug("💾 Starting cache storage process")
                        complexity_level = self._detect_complexity_level(prompt)
                        quality_score = None

                        # Extract quality score from critic analysis if available
                        if result.critic_analysis and hasattr(result.critic_analysis, "overall_score"):
                            quality_score = result.critic_analysis.overall_score
                            logger.debug(f"📊 Quality score extracted: {quality_score}")
                        else:
                            logger.debug("📊 No quality score available")

                        await self.retrieval_service.store_generated_code(
                            prompt=prompt,
                            generated_code=result.generated_code,
                            complexity_level=complexity_level,
                            quality_score=quality_score,
                            workflow_state=result,
                        )
                        cache_store_time = time.time() - cache_store_start
                        logger.info(
                            "✅ Generated code stored in cache successfully",
                            prompt_preview=prompt[:50] + "...",
                            code_length=len(result.generated_code),
                            complexity_level=complexity_level,
                            quality_score=quality_score,
                            storage_time_s=f"{cache_store_time:.3f}",
                        )

                    except Exception as e:
                        cache_store_time = time.time() - cache_store_start
                        logger.exception(
                            "💥 Failed to cache generated code",
                            error=str(e),
                            error_type=type(e).__name__,
                            storage_time_s=f"{cache_store_time:.3f}",
                            prompt_preview=prompt[:50] + "...",
                        )
                elif is_cached if "is_cached" in locals() else False:
                    logger.debug("💾 Skipping cache storage - code was from cache")
                elif not self.enable_caching:
                    logger.debug("💾 Skipping cache storage - caching disabled")
                elif not result.generated_code:
                    logger.warning("💾 Skipping cache storage - no generated code available")

                return generation_result
            else:
                error_msg = result.error_message or "Asset generation failed"
                total_generation_time = time.time() - generation_start_time

                # Comprehensive failure analysis
                failure_details = {
                    "execution_result_exists": result.execution_result is not None,
                    "execution_success": (result.execution_result.success if result.execution_result else None),
                    "execution_errors": (result.execution_result.errors if result.execution_result else []),
                    "execution_logs": (result.execution_result.logs if result.execution_result else []),
                    "detected_issues": [
                        issue.dict() if hasattr(issue, "dict") else str(issue) for issue in result.detected_issues
                    ],
                    "refinement_iterations": result.refinement_iterations,
                    "generated_code_length": (len(result.generated_code) if result.generated_code else 0),
                    "has_critic_analysis": result.critic_analysis is not None,
                    "total_time_s": total_generation_time,
                }

                logger.error("❌ Asset generation failed", error=error_msg, **failure_details)

                return {
                    "success": False,
                    "cached": is_cached if "is_cached" in locals() else False,
                    "error": error_msg,
                    "code": result.generated_code,
                    "failure_details": failure_details,
                    "total_time": total_generation_time,
                }

        except Exception as e:
            total_generation_time = time.time() - generation_start_time
            logger.exception(
                "💥 Unexpected error during asset generation",
                error=str(e),
                error_type=type(e).__name__,
                total_time_s=f"{total_generation_time:.3f}",
                prompt_preview=prompt[:100] + "...",
                stack_trace=True,  # This will include the full stack trace
            )
            return {
                "success": False,
                "cached": is_cached if "is_cached" in locals() else False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__,
                "total_time": total_generation_time,
                "debug_info": {
                    "prompt_length": len(prompt),
                    "asset_name": asset_name,
                    "output_directory": output_directory,
                    "export_formats": export_formats,
                    "caching_enabled": self.enable_caching,
                },
            }

    async def refine_asset(
        self,
        original_prompt: str,
        feedback: str,
        asset_name: str = "asset",
        output_directory: str | None = None,
    ) -> dict[str, Any]:
        """Refine an existing asset based on user feedback.

        Args:
            original_prompt: Original prompt used to generate the asset
            feedback: User feedback for refinement
            asset_name: Name of the asset being refined
            output_directory: Optional directory for refined outputs

        Returns:
            Dictionary containing refinement results and metadata
        """
        logger.info(
            "Starting asset refinement",
            original_prompt=original_prompt,
            feedback=feedback,
        )

        # Create state for refinement
        refinement_state = WorkflowState(
            prompt=original_prompt,
            user_feedback=feedback,
            documentation="",
            generated_code="",
            execution_result=None,
            output_directory=output_directory or "outputs",
            refinement_request="",
            asset_metadata=None,
            error_message=None,
            previous_screenshot_path=None,
            critic_analysis=None,
            verification_result=None,
            user_prompt="",
            original_prompt=original_prompt,
            refinement_iterations=1,  # Start with 1 since this is a refinement
            verification_score=0.0,
            has_critical_issues=False,
        )

        try:
            workflow = self._get_workflow()

            # Execute the workflow with refinement
            result_dict = await workflow.ainvoke(
                refinement_state,
                config={"configurable": {"thread_id": f"refine_{asset_name}"}},
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
            5: [
                "animation",
                "character",
                "dance",
                "movement",
                "rigged",
                "animated",
                "sequence",
            ],
            4: [
                "lighting",
                "scene",
                "environment",
                "multiple",
                "room",
                "interior",
                "exterior",
                "landscape",
            ],
            3: [
                "texture",
                "material",
                "wood",
                "metal",
                "fabric",
                "realistic",
                "detailed",
                "shiny",
                "rough",
            ],
            2: [
                "furniture",
                "object",
                "table",
                "chair",
                "simple",
                "basic",
                "house",
                "building",
            ],
            1: [
                "cube",
                "sphere",
                "cylinder",
                "plane",
                "primitive",
                "shape",
                "geometry",
                "ovoid",
            ],
        }

        for level in range(5, 0, -1):
            if any(keyword in prompt_lower for keyword in complexity_indicators[level]):
                return level

        return 2  # Default complexity

    def _get_complexity_keywords(self, prompt: str) -> list[str]:
        """Extract complexity keywords from prompt for debugging.

        Args:
            prompt: The generation prompt

        Returns:
            List of detected complexity keywords
        """
        prompt_lower = prompt.lower()
        complexity_indicators = {
            5: [
                "animation",
                "character",
                "dance",
                "movement",
                "rigged",
                "animated",
                "sequence",
            ],
            4: [
                "lighting",
                "scene",
                "environment",
                "multiple",
                "room",
                "interior",
                "exterior",
                "landscape",
            ],
            3: [
                "texture",
                "material",
                "wood",
                "metal",
                "fabric",
                "realistic",
                "detailed",
                "shiny",
                "rough",
            ],
            2: [
                "furniture",
                "object",
                "table",
                "chair",
                "simple",
                "basic",
                "house",
                "building",
            ],
            1: [
                "cube",
                "sphere",
                "cylinder",
                "plane",
                "primitive",
                "shape",
                "geometry",
                "ovoid",
            ],
        }

        detected = []
        for level, keywords in complexity_indicators.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    detected.append(f"L{level}:{keyword}")
        return detected

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
