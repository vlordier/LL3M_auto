"""Vector database for RAG caching system."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from .embedder import get_embedder

logger = structlog.get_logger(__name__)


class CacheEntry(BaseModel):
    """A cached prompt-code entry."""

    id: str | None = Field(None, description="Unique identifier")
    prompt: str = Field(..., description="Original prompt")
    code: str = Field(..., description="Generated Blender code")
    embedding: list[float] = Field(..., description="Prompt embedding vector")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    complexity_level: int = Field(default=1, description="Complexity level 1-5")
    quality_score: float | None = Field(None, description="Quality score 0-10")
    success_rate: float = Field(default=0.0, description="Historical success rate")
    usage_count: int = Field(default=0, description="Times this cache entry was used")
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: datetime | None = Field(None)


class VectorDatabase:
    """SQLite-based vector database for RAG caching."""

    def __init__(self, db_path: str = "cache/vector_cache.db"):
        """Initialize the vector database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        self.embedder = get_embedder()
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    code TEXT NOT NULL,
                    embedding TEXT NOT NULL,  -- JSON-encoded vector
                    metadata TEXT DEFAULT '{}',  -- JSON-encoded metadata
                    complexity_level INTEGER DEFAULT 1,
                    quality_score REAL,
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    UNIQUE(prompt, complexity_level)
                )
            """)

            # Create indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON cache_entries(complexity_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON cache_entries(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage ON cache_entries(usage_count DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache_entries(created_at DESC)")

            conn.commit()

        logger.info(f"Vector database initialized: {self.db_path}")

    async def add_entry(
        self,
        prompt: str,
        code: str,
        complexity_level: int = 1,
        quality_score: float | None = None,
        metadata: dict[str, Any] | None = None
    ) -> CacheEntry:
        """Add a new cache entry.

        Args:
            prompt: The prompt text
            code: The generated Blender code
            complexity_level: Complexity level 1-5
            quality_score: Quality score 0-10
            metadata: Additional metadata

        Returns:
            The created cache entry
        """
        # Generate embedding for the prompt
        embedding_result = await self.embedder.embed_text(
            prompt, context_type="prompt"
        )

        # Create cache entry
        entry_id = f"cache_{hash(prompt + str(complexity_level)) % 10000000:07d}"
        entry = CacheEntry(
            id=entry_id,
            prompt=prompt,
            code=code,
            embedding=embedding_result.embedding,
            complexity_level=complexity_level,
            quality_score=quality_score,
            metadata=metadata or {}
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (id, prompt, code, embedding, metadata, complexity_level,
                     quality_score, success_rate, usage_count, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.prompt,
                    entry.code,
                    json.dumps(entry.embedding),
                    json.dumps(entry.metadata),
                    entry.complexity_level,
                    entry.quality_score,
                    entry.success_rate,
                    entry.usage_count,
                    entry.created_at.isoformat(),
                    entry.last_used.isoformat() if entry.last_used else None
                ))
                conn.commit()

            logger.info(f"Added cache entry: {entry_id} for prompt: {prompt[:50]}...")
            return entry

        except sqlite3.IntegrityError as e:
            logger.warning(f"Cache entry already exists for prompt: {prompt[:50]}... - {e}")
            # Return existing entry
            return await self.get_by_prompt(prompt, complexity_level)

    async def find_similar(
        self,
        prompt: str,
        top_k: int = 5,
        min_similarity: float = 0.75,
        complexity_level: int | None = None
    ) -> list[tuple[CacheEntry, float]]:
        """Find similar cached entries using embedding similarity.

        Args:
            prompt: Query prompt
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            complexity_level: Filter by complexity level

        Returns:
            List of (CacheEntry, similarity_score) tuples
        """
        # Generate embedding for query prompt
        query_embedding = await self.embedder.embed_text(
            prompt, context_type="prompt"
        )

        # Get all cached entries (or filtered by complexity)
        with sqlite3.connect(self.db_path) as conn:
            if complexity_level:
                cursor = conn.execute("""
                    SELECT id, prompt, code, embedding, metadata, complexity_level,
                           quality_score, success_rate, usage_count, created_at, last_used
                    FROM cache_entries
                    WHERE complexity_level = ?
                    ORDER BY usage_count DESC, quality_score DESC
                """, (complexity_level,))
            else:
                cursor = conn.execute("""
                    SELECT id, prompt, code, embedding, metadata, complexity_level,
                           quality_score, success_rate, usage_count, created_at, last_used
                    FROM cache_entries
                    ORDER BY usage_count DESC, quality_score DESC
                """)

            rows = cursor.fetchall()

        if not rows:
            logger.info("No cached entries found")
            return []

        # Calculate similarities
        candidates = []
        for row in rows:
            try:
                cached_embedding = json.loads(row[3])  # embedding column
                similarity = self.embedder.compute_similarity(
                    query_embedding.embedding, cached_embedding
                )

                if similarity >= min_similarity:
                    entry = CacheEntry(
                        id=row[0],
                        prompt=row[1],
                        code=row[2],
                        embedding=cached_embedding,
                        metadata=json.loads(row[4] or '{}'),
                        complexity_level=row[5],
                        quality_score=row[6],
                        success_rate=row[7],
                        usage_count=row[8],
                        created_at=datetime.fromisoformat(row[9]),
                        last_used=datetime.fromisoformat(row[10]) if row[10] else None
                    )
                    candidates.append((entry, similarity))

            except Exception as e:
                logger.warning(f"Error processing cached entry {row[0]}: {e}")
                continue

        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = candidates[:top_k]

        logger.info(f"Found {len(results)} similar entries for: {prompt[:50]}...")
        return results

    async def get_by_prompt(
        self,
        prompt: str,
        complexity_level: int = 1
    ) -> CacheEntry | None:
        """Get exact match by prompt and complexity level.

        Args:
            prompt: The exact prompt
            complexity_level: Complexity level

        Returns:
            Matching cache entry or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, prompt, code, embedding, metadata, complexity_level,
                       quality_score, success_rate, usage_count, created_at, last_used
                FROM cache_entries
                WHERE prompt = ? AND complexity_level = ?
            """, (prompt, complexity_level))

            row = cursor.fetchone()

        if not row:
            return None

        return CacheEntry(
            id=row[0],
            prompt=row[1],
            code=row[2],
            embedding=json.loads(row[3]),
            metadata=json.loads(row[4] or '{}'),
            complexity_level=row[5],
            quality_score=row[6],
            success_rate=row[7],
            usage_count=row[8],
            created_at=datetime.fromisoformat(row[9]),
            last_used=datetime.fromisoformat(row[10]) if row[10] else None
        )

    async def update_usage(self, entry_id: str, success: bool = True):
        """Update usage statistics for a cache entry.

        Args:
            entry_id: Cache entry ID
            success: Whether the cached code was successful
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute("""
                SELECT usage_count, success_rate FROM cache_entries WHERE id = ?
            """, (entry_id,))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Cache entry not found: {entry_id}")
                return

            usage_count, success_rate = row
            new_usage_count = usage_count + 1

            # Calculate new success rate
            total_successes = success_rate * usage_count
            if success:
                total_successes += 1
            new_success_rate = total_successes / new_usage_count

            # Update database
            conn.execute("""
                UPDATE cache_entries
                SET usage_count = ?, success_rate = ?, last_used = ?
                WHERE id = ?
            """, (
                new_usage_count,
                new_success_rate,
                datetime.now().isoformat(),
                entry_id
            ))
            conn.commit()

        logger.debug(f"Updated usage for {entry_id}: count={new_usage_count}, success={new_success_rate:.2f}")

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total_entries = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

            # By complexity level
            complexity_stats = {}
            cursor = conn.execute("""
                SELECT complexity_level, COUNT(*), AVG(quality_score), AVG(success_rate)
                FROM cache_entries
                GROUP BY complexity_level
                ORDER BY complexity_level
            """)

            for row in cursor:
                complexity_stats[row[0]] = {
                    "count": row[1],
                    "avg_quality": row[2] or 0,
                    "avg_success_rate": row[3] or 0
                }

            # Top performing entries
            top_entries = []
            cursor = conn.execute("""
                SELECT prompt, complexity_level, quality_score, success_rate, usage_count
                FROM cache_entries
                ORDER BY success_rate DESC, quality_score DESC, usage_count DESC
                LIMIT 5
            """)

            for row in cursor:
                top_entries.append({
                    "prompt": row[0][:50] + "..." if len(row[0]) > 50 else row[0],
                    "complexity": row[1],
                    "quality": row[2],
                    "success_rate": row[3],
                    "usage_count": row[4]
                })

        return {
            "total_entries": total_entries,
            "complexity_breakdown": complexity_stats,
            "top_performing": top_entries,
            "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
        }

    async def cleanup_low_quality(
        self,
        min_success_rate: float = 0.3,
        min_usage_count: int = 3
    ) -> int:
        """Remove low-quality cache entries.

        Args:
            min_success_rate: Minimum success rate to keep
            min_usage_count: Minimum usage count required for filtering

        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find entries to remove
            cursor = conn.execute("""
                SELECT COUNT(*) FROM cache_entries
                WHERE usage_count >= ? AND success_rate < ?
            """, (min_usage_count, min_success_rate))

            count_to_remove = cursor.fetchone()[0]

            if count_to_remove > 0:
                # Remove low-quality entries
                conn.execute("""
                    DELETE FROM cache_entries
                    WHERE usage_count >= ? AND success_rate < ?
                """, (min_usage_count, min_success_rate))
                conn.commit()

                logger.info(f"Cleaned up {count_to_remove} low-quality cache entries")

        return count_to_remove

    async def clear_cache(self):
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()

        logger.info("Cache cleared")


# Singleton instance
_vector_db_instance: VectorDatabase | None = None


def get_vector_db() -> VectorDatabase:
    """Get the global vector database instance."""
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDatabase()
    return _vector_db_instance
