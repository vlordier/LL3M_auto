"""RAG retrieval service for cached prompt-code pairs."""

from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from ..utils.types import WorkflowState
from .vector_db import CacheEntry, VectorDatabase, get_vector_db

logger = structlog.get_logger(__name__)


class CacheRetrievalResult(BaseModel):
    """Result of cache retrieval operation."""

    cache_hit: bool = Field(..., description="Whether cache was hit")
    entry: CacheEntry | None = Field(None, description="Retrieved cache entry")
    similarity_score: float | None = Field(None, description="Similarity score if found")
    alternatives: list[tuple[CacheEntry, float]] = Field(default_factory=list, description="Alternative similar entries")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class RAGRetrievalService:
    """Service for retrieving cached code based on prompt similarity."""

    def __init__(self, vector_db: VectorDatabase | None = None):
        """Initialize the retrieval service.

        Args:
            vector_db: Vector database instance (uses global if None)
        """
        self.vector_db = vector_db or get_vector_db()

        # Retrieval configuration
        self.similarity_threshold = 0.8  # High threshold for exact matches
        self.alternative_threshold = 0.7  # Lower threshold for alternatives
        self.max_alternatives = 3

    async def retrieve_cached_code(
        self,
        prompt: str,
        complexity_level: int = 1,
        include_alternatives: bool = True
    ) -> CacheRetrievalResult:
        """Retrieve cached code for a prompt.

        Args:
            prompt: The generation prompt
            complexity_level: Target complexity level
            include_alternatives: Include alternative similar entries

        Returns:
            Cache retrieval result
        """
        start_time = datetime.now()

        logger.debug(f"Retrieving cache for: {prompt[:50]}...", complexity=complexity_level)

        try:
            # First try exact match
            exact_match = await self.vector_db.get_by_prompt(prompt, complexity_level)

            if exact_match:
                # Update usage statistics
                await self.vector_db.update_usage(exact_match.id, success=True)

                retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

                logger.info(f"Exact cache hit for: {prompt[:50]}...",
                           entry_id=exact_match.id, usage_count=exact_match.usage_count + 1)

                return CacheRetrievalResult(
                    cache_hit=True,
                    entry=exact_match,
                    similarity_score=1.0,
                    retrieval_time_ms=retrieval_time
                )

            # No exact match, try similarity search
            similar_entries = await self.vector_db.find_similar(
                prompt=prompt,
                top_k=5,  # Get more for filtering
                min_similarity=self.alternative_threshold,
                complexity_level=complexity_level
            )

            # Filter by quality and success rate
            filtered_entries = self._filter_by_quality(similar_entries)

            cache_hit = False
            best_entry = None
            best_similarity = 0.0
            alternatives = []

            if filtered_entries:
                # Check if best match meets our threshold for direct use
                best_entry, best_similarity = filtered_entries[0]

                if best_similarity >= self.similarity_threshold:
                    cache_hit = True
                    # Update usage for the retrieved entry
                    await self.vector_db.update_usage(best_entry.id, success=True)

                    logger.info(f"High similarity cache hit: {best_similarity:.3f} for: {prompt[:50]}...",
                               entry_id=best_entry.id, original_prompt=best_entry.prompt[:50])
                else:
                    # Include as alternatives
                    if include_alternatives:
                        alternatives = filtered_entries[:self.max_alternatives]

                    logger.debug(f"Cache miss - best similarity: {best_similarity:.3f} < {self.similarity_threshold}")

            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

            if not cache_hit:
                logger.info(f"Cache miss for: {prompt[:50]}...", alternatives_found=len(alternatives))

            return CacheRetrievalResult(
                cache_hit=cache_hit,
                entry=best_entry if cache_hit else None,
                similarity_score=best_similarity if cache_hit else None,
                alternatives=alternatives if not cache_hit else [],
                retrieval_time_ms=retrieval_time
            )

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

            return CacheRetrievalResult(
                cache_hit=False,
                retrieval_time_ms=retrieval_time
            )

    async def store_generated_code(
        self,
        prompt: str,
        generated_code: str,
        complexity_level: int,
        quality_score: float | None = None,
        workflow_state: WorkflowState | None = None
    ) -> CacheEntry:
        """Store newly generated code in the cache.

        Args:
            prompt: Original prompt
            generated_code: Generated Blender code
            complexity_level: Complexity level
            quality_score: Quality assessment score
            workflow_state: Optional workflow state for metadata

        Returns:
            Created cache entry
        """
        # Extract metadata from workflow state
        metadata = {}
        if workflow_state:
            metadata.update({
                "generation_time": getattr(workflow_state, 'generation_time', None),
                "model_used": getattr(workflow_state, 'model_name', 'unknown'),
                "workflow_version": getattr(workflow_state, 'workflow_version', '1.0'),
                "agent_sequence": getattr(workflow_state, 'agent_sequence', [])
            })

        logger.info(f"Storing generated code for: {prompt[:50]}...",
                   complexity=complexity_level, quality=quality_score)

        try:
            entry = await self.vector_db.add_entry(
                prompt=prompt,
                code=generated_code,
                complexity_level=complexity_level,
                quality_score=quality_score,
                metadata=metadata
            )

            logger.info(f"Cached entry stored: {entry.id}")
            return entry

        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
            raise

    def _filter_by_quality(
        self,
        similar_entries: list[tuple[CacheEntry, float]]
    ) -> list[tuple[CacheEntry, float]]:
        """Filter and rank similar entries by quality metrics.

        Args:
            similar_entries: List of (entry, similarity) tuples

        Returns:
            Filtered and ranked entries
        """
        if not similar_entries:
            return []

        # Apply quality filters
        quality_filtered = []

        for entry, similarity in similar_entries:
            # Skip entries with very low success rates (if they have enough usage data)
            if entry.usage_count >= 3 and entry.success_rate < 0.4:
                continue

            # Skip entries with very low quality scores
            if entry.quality_score is not None and entry.quality_score < 4.0:
                continue

            quality_filtered.append((entry, similarity))

        # Rank by composite score: similarity * quality_factor * success_factor
        ranked_entries = []

        for entry, similarity in quality_filtered:
            # Quality factor (0.5-1.5, default 1.0)
            if entry.quality_score is not None:
                quality_factor = min(1.5, max(0.5, entry.quality_score / 7.0))
            else:
                quality_factor = 1.0

            # Success rate factor (0.5-1.5, default 1.0)
            if entry.usage_count >= 2:  # Only apply if we have enough data
                success_factor = min(1.5, max(0.5, entry.success_rate * 1.5))
            else:
                success_factor = 1.0

            # Usage popularity factor (slight boost for frequently used)
            usage_factor = min(1.2, 1.0 + (entry.usage_count * 0.02))

            composite_score = similarity * quality_factor * success_factor * usage_factor
            ranked_entries.append((entry, similarity, composite_score))

        # Sort by composite score
        ranked_entries.sort(key=lambda x: x[2], reverse=True)

        # Return original format (entry, similarity)
        return [(entry, similarity) for entry, similarity, _ in ranked_entries]

    def get_cache_recommendations(
        self,
        prompt: str,
        complexity_level: int = 1
    ) -> dict[str, Any]:
        """Get recommendations about using cache for this prompt.

        Args:
            prompt: The prompt to analyze
            complexity_level: Target complexity level

        Returns:
            Recommendations and analysis
        """
        # Simple analysis based on prompt characteristics
        prompt_lower = prompt.lower()

        recommendations = {
            "should_use_cache": True,
            "confidence": "medium",
            "reasoning": [],
            "suggestions": []
        }

        # Check for common patterns that cache well
        if any(word in prompt_lower for word in ["cube", "sphere", "cylinder", "plane"]):
            recommendations["confidence"] = "high"
            recommendations["reasoning"].append("Simple geometric shapes typically have good cache coverage")

        elif any(word in prompt_lower for word in ["chair", "table", "furniture", "house"]):
            recommendations["confidence"] = "high"
            recommendations["reasoning"].append("Common objects usually have good cache matches")

        elif any(word in prompt_lower for word in ["animation", "dance", "movement", "character"]):
            recommendations["confidence"] = "low"
            recommendations["reasoning"].append("Complex animations have unique requirements, cache may be less helpful")
            recommendations["suggestions"].append("Consider using cache for base geometry, then applying animations")

        elif any(word in prompt_lower for word in ["unique", "custom", "specific", "special"]):
            recommendations["confidence"] = "low"
            recommendations["reasoning"].append("Unique requirements may not match existing cache entries well")

        # Length-based analysis
        if len(prompt.split()) > 15:
            recommendations["confidence"] = "low"
            recommendations["reasoning"].append("Very specific prompts may need custom generation")

        elif len(prompt.split()) < 5:
            recommendations["confidence"] = "high"
            recommendations["reasoning"].append("Simple prompts often have good cache matches")

        return recommendations

    async def get_service_statistics(self) -> dict[str, Any]:
        """Get retrieval service statistics.

        Returns:
            Service statistics
        """
        db_stats = self.vector_db.get_statistics()

        return {
            "database_stats": db_stats,
            "configuration": {
                "similarity_threshold": self.similarity_threshold,
                "alternative_threshold": self.alternative_threshold,
                "max_alternatives": self.max_alternatives
            },
            "thresholds": {
                "exact_match": 1.0,
                "high_similarity": self.similarity_threshold,
                "alternative_similarity": self.alternative_threshold
            }
        }


# Singleton instance
_retrieval_service_instance: RAGRetrievalService | None = None


def get_retrieval_service() -> RAGRetrievalService:
    """Get the global retrieval service instance."""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RAGRetrievalService()
    return _retrieval_service_instance
