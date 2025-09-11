"""OpenAI-based embedding generation for RAG caching system."""

import asyncio
import hashlib
import time
from typing import Any

import numpy as np
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..utils.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""

    text: str = Field(..., description="Text to embed")
    context_type: str = Field(default="code", description="Type of content (code, prompt, description)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""

    text: str = Field(..., description="Original text")
    embedding: list[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Embedding model used")
    dimensions: int = Field(..., description="Vector dimensions")
    text_hash: str = Field(..., description="SHA-256 hash of the text")
    context_type: str = Field(..., description="Content type")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")


class OpenAIEmbedder:
    """OpenAI-based text embedding service for RAG caching."""

    def __init__(self):
        """Initialize the embedder with OpenAI client."""
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)
        self.model = "text-embedding-3-large"  # Best OpenAI embedding model
        self.dimensions = 3072  # Default dimensions for text-embedding-3-large

        # Embedding cache for session-level caching
        self._embedding_cache: dict[str, EmbeddingResult] = {}

        # Rate limiting
        self.requests_per_minute = 3000  # OpenAI limit for embeddings
        self.request_timestamps: list[float] = []

    async def embed_text(
        self,
        text: str,
        context_type: str = "code",
        metadata: dict[str, Any] | None = None,
        use_cache: bool = True
    ) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            context_type: Type of content (code, prompt, description)
            metadata: Additional metadata
            use_cache: Whether to use session cache

        Returns:
            EmbeddingResult with vector and metadata
        """
        # Create hash for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Check session cache first
        if use_cache and text_hash in self._embedding_cache:
            logger.debug(f"Using cached embedding for: {text[:50]}...")
            return self._embedding_cache[text_hash]

        # Rate limiting check
        await self._ensure_rate_limit()

        try:
            logger.debug(f"Generating embedding for: {text[:50]}...")

            # Generate embedding using OpenAI
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )

            embedding_vector = response.data[0].embedding

            # Create result
            result = EmbeddingResult(
                text=text,
                embedding=embedding_vector,
                model=self.model,
                dimensions=len(embedding_vector),
                text_hash=text_hash,
                context_type=context_type,
                metadata=metadata or {}
            )

            # Cache the result
            if use_cache:
                self._embedding_cache[text_hash] = result

            logger.info(f"Generated embedding: {len(embedding_vector)}D vector for {len(text)} chars")
            return result

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(
        self,
        texts: list[str],
        context_type: str = "code",
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 100,
        use_cache: bool = True
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            context_type: Content type for all texts
            metadata: List of metadata dicts (same length as texts)
            batch_size: Batch size for API calls
            use_cache: Whether to use session cache

        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")

        results = []
        metadata = metadata or [{}] * len(texts)

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            # Check which texts need embeddings (not in cache)
            texts_to_embed = []
            cache_results = []
            indices_to_embed = []

            for j, text in enumerate(batch_texts):
                text_hash = hashlib.sha256(text.encode()).hexdigest()

                if use_cache and text_hash in self._embedding_cache:
                    cache_results.append((i + j, self._embedding_cache[text_hash]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i + j)

            # Generate embeddings for non-cached texts
            if texts_to_embed:
                await self._ensure_rate_limit()

                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts_to_embed,
                        dimensions=self.dimensions
                    )

                    # Process batch results
                    for j, (text, embedding_data) in enumerate(zip(texts_to_embed, response.data, strict=False)):
                        original_index = indices_to_embed[j]
                        text_hash = hashlib.sha256(text.encode()).hexdigest()

                        result = EmbeddingResult(
                            text=text,
                            embedding=embedding_data.embedding,
                            model=self.model,
                            dimensions=len(embedding_data.embedding),
                            text_hash=text_hash,
                            context_type=context_type,
                            metadata=batch_metadata[original_index - i]
                        )

                        # Cache result
                        if use_cache:
                            self._embedding_cache[text_hash] = result

                        cache_results.append((original_index, result))

                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Fallback to individual embeddings
                    for j, text in enumerate(texts_to_embed):
                        try:
                            result = await self.embed_text(
                                text,
                                context_type,
                                batch_metadata[indices_to_embed[j] - i],
                                use_cache
                            )
                            cache_results.append((indices_to_embed[j], result))
                        except Exception as individual_e:
                            logger.error(f"Individual embedding failed for text {j}: {individual_e}")

            # Sort results by original index
            cache_results.sort(key=lambda x: x[0])
            batch_results = [result for _, result in cache_results]
            results.extend(batch_results)

            # Progress update
            logger.debug(f"Completed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

        logger.info(f"Generated {len(results)} embeddings successfully")
        return results

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
        method: str = "cosine"
    ) -> float:
        """Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method (cosine, dot, euclidean)

        Returns:
            Similarity score (higher = more similar for cosine/dot)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        if method == "cosine":
            # Cosine similarity: ranges from -1 to 1, higher is more similar
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        elif method == "dot":
            # Dot product similarity
            return float(np.dot(vec1, vec2))

        elif method == "euclidean":
            # Euclidean distance (lower = more similar)
            distance = np.linalg.norm(vec1 - vec2)
            # Convert to similarity (higher = more similar)
            return float(1.0 / (1.0 + distance))

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def find_most_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[tuple[str, list[float]]],
        top_k: int = 5,
        min_similarity: float = 0.7
    ) -> list[tuple[str, float]]:
        """Find most similar embeddings to a query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of (id, embedding) tuples
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (id, similarity_score) tuples, sorted by similarity (desc)
        """
        if not candidate_embeddings:
            return []

        similarities = []

        for candidate_id, candidate_embedding in candidate_embeddings:
            similarity = self.compute_similarity(query_embedding, candidate_embedding)

            if similarity >= min_similarity:
                similarities.append((candidate_id, similarity))

        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def _ensure_rate_limit(self):
        """Ensure we don't exceed OpenAI rate limits."""
        current_time = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < 60.0
        ]

        # Check if we're approaching the rate limit
        if len(self.request_timestamps) >= self.requests_per_minute - 10:  # Leave some buffer
            # Calculate time to wait
            oldest_request = min(self.request_timestamps)
            time_to_wait = 60.0 - (current_time - oldest_request)

            if time_to_wait > 0:
                logger.info(f"Rate limit protection: waiting {time_to_wait:.1f}s")
                await asyncio.sleep(time_to_wait)

        # Record this request
        self.request_timestamps.append(current_time)

    def clear_cache(self):
        """Clear the session embedding cache."""
        logger.info(f"Clearing embedding cache ({len(self._embedding_cache)} entries)")
        self._embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cached_embeddings": len(self._embedding_cache),
            "cache_memory_usage": sum(
                len(result.embedding) * 4 + len(result.text)  # Approximate bytes
                for result in self._embedding_cache.values()
            ),
            "recent_requests": len(self.request_timestamps),
            "model": self.model,
            "dimensions": self.dimensions
        }


# Singleton instance for application-wide use
_embedder_instance: OpenAIEmbedder | None = None


def get_embedder() -> OpenAIEmbedder:
    """Get the global embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = OpenAIEmbedder()
    return _embedder_instance
