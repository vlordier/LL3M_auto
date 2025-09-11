"""Abstract interface for knowledge retrieval services."""

from abc import ABC, abstractmethod

from ..utils.types import AgentResponse


class KnowledgeService(ABC):
    """Abstract base class for knowledge retrieval services."""

    @abstractmethod
    async def resolve_library_id(self, library_name: str) -> str | None:
        """Resolve a package name to a compatible library ID."""
        pass

    @abstractmethod
    async def get_library_docs(self, library_id: str, topic: str | None = None, tokens: int = 10000) -> str | None:
        """Fetch documentation for a specific library."""
        pass

    @abstractmethod
    async def search(self, query: str, library_id: str) -> AgentResponse:
        """Search for relevant information within a library."""
        pass
